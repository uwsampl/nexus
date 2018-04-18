#include <glog/logging.h>

#include "nexus/common/model_db.h"
#include "nexus/scheduler/backend_rpc_client.h"
#include "nexus/scheduler/scheduler.h"

namespace nexus {
namespace scheduler {

BackendRpcClient::BackendRpcClient(Scheduler* sch, uint32_t node_id,
                                   const std::string& server_addr,
                                   const std::string& rpc_addr,
                                   const std::string& gpu_device,
                                   size_t gpu_available_memory,
                                   int beacon_sec, int epoch_sec):
    scheduler_(sch),
    node_id_(node_id),
    server_address_(server_addr),
    rpc_address_(rpc_addr),
    gpu_device_(gpu_device),
    gpu_available_memory_(gpu_available_memory),
    beacon_sec_(beacon_sec),
    epoch_sec_(epoch_sec),
    timeout_ms_(beacon_sec * 2 * 1000),
    workload_id_(-1),
    exec_cycle_us_(0.),
    duty_cycle_us_(0.),
    dirty_model_table_(false) {
  auto channel = grpc::CreateChannel(rpc_addr,
                                     grpc::InsecureChannelCredentials());
  stub_ = BackendCtrl::NewStub(channel);
  last_time_ = std::chrono::system_clock::now();
}

void BackendRpcClient::GetInfo(BackendInfo* info) {
  info->set_node_id(node_id_);
  info->set_server_address(server_address_);
}

std::time_t BackendRpcClient::LastAliveTime() {
  return std::chrono::system_clock::to_time_t(last_time_);
}

bool BackendRpcClient::Assign(const BackendRpcClient& other) {
  CHECK(IsIdle()) << "Backend is not idle";
  if (gpu_device_ == other.gpu_device_) {
    workload_id_ = other.workload_id_;
    model_rps_ = other.model_rps_;
    model_table_config_ = other.model_table_config_;
    exec_cycle_us_ = other.exec_cycle_us_;
    duty_cycle_us_ = other.exec_cycle_us_;
    return true;
  }
  // TODO: support assign for two backend that have different GPU devices
  return false;
}

void BackendRpcClient::PrepareLoadModel(
    const ModelSession& model_sess, float workload,
    ModelInstanceConfig* config, float* occupancy) {
  std::string profile_id = ModelSessionToProfileID(model_sess);
  auto profile = ModelDatabase::Singleton().GetModelProfile(gpu_device_,
                                                            profile_id);
  if (profile == nullptr) {
    config->set_batch(0);
    return;
  }
  config->mutable_model_session()->CopyFrom(model_sess);

  // 1. Compute the max batch and throughput to saturate an empty GPU
  float latency_sla_us = model_sess.latency_sla() * 1000;
  uint32_t max_batch;
  uint32_t max_throughput;
  std::tie(max_batch, max_throughput) = profile->GetMaxThroughput(
      model_sess.latency_sla());

  if (exec_cycle_us_ == 0) {
    // empty GPU 
    if (workload == 0 || max_throughput <= workload) {
      // workload can saturate the gpu
      float fwd_latency = profile->GetForwardLatency(max_batch);
      uint32_t memory_usage = profile->GetMemoryUsage(max_batch);
      config->set_batch(max_batch);
      config->set_max_batch(max_batch);
      config->set_forward_latency(fwd_latency);
      config->set_memory_usage(fwd_latency);
      config->set_throughput(max_throughput);
      config->set_workload(max_throughput);
      *occupancy = 1.0;
    } else {
      // 2. Compute the max batch for residue load
      uint32_t preprocess = profile->GetPreprocessLatency();
      uint32_t postprocess = profile->GetPostprocessLatency();
      uint32_t batch = 1;
      for (; batch <= max_batch; ++batch) {
        float fwd_lat = profile->GetForwardLatency(batch);
        // because batch = ceil(workload * duty_cycle),
        // duty_cycle >= (batch - 1) / workload
        float min_duty_cycle = (batch - 1) * 1e6 / workload;
        if (min_duty_cycle + fwd_lat + preprocess + postprocess >
            latency_sla_us) {
          break;
        }
      }
      --batch;
      if (batch == 0) {
        // execution latency of batch size 1 is even too large for latency_sla
        config->set_batch(0);
      } else {
        float fwd_lat = profile->GetForwardLatency(batch);
        uint32_t memory_usage = profile->GetMemoryUsage(batch);
        float duty_cycle = latency_sla_us - fwd_lat - preprocess - postprocess;
        float throughput = batch * 1e6 / duty_cycle;
        config->set_batch(batch);
        config->set_max_batch(max_batch);
        config->set_forward_latency(fwd_lat);
        config->set_memory_usage(memory_usage);
        config->set_throughput(throughput);
        config->set_workload(workload);
        *occupancy = fwd_lat / duty_cycle;
      }
    }
  } else {
    if (workload == 0) {
      config->set_batch(0);
      return;
    }
    // TODO
    config->set_batch(0);
  }
}

void BackendRpcClient::LoadModel(const ModelInstanceConfig& config) {
  if (exec_cycle_us_ > 0) {
    LOG(ERROR) << "Backend is not idle. Don't support multi-batching now.";
  } else {
    exec_cycle_us_ = config.forward_latency();
    duty_cycle_us_ = config.model_session().latency_sla() * 1e3 -
                     exec_cycle_us_;
    auto model_sess_id = ModelSessionToString(config.model_session());
    model_table_config_.emplace(model_sess_id, config);
    model_rps_.emplace(model_sess_id, EWMA(1, epoch_sec_));
    dirty_model_table_ = true;

    LOG(INFO) << "Backend " << node_id_ << " loads " << config.DebugString();
    LOG(INFO) << "Backend " << node_id_ << ": exec cycle " << exec_cycle_us_ <<
        " us, duty cycle: " << duty_cycle_us_ << " us";
  }
}

void BackendRpcClient::LoadModel(const YAML::Node& model_info) {
  ModelInstanceConfig config;
  auto sess = config.mutable_model_session();
  sess->set_framework(model_info["framework"].as<std::string>());
  sess->set_model_name(model_info["model_name"].as<std::string>());
  sess->set_version(model_info["version"].as<uint32_t>());
  sess->set_latency_sla(model_info["latency_sla"].as<uint32_t>());
  if (model_info["image_height"]) {
    sess->set_image_height(model_info["image_height"].as<uint32_t>());
    sess->set_image_width(model_info["image_width"].as<uint32_t>());
  }
  std::string model_sess_id = ModelSessionToString(*sess);
  std::string profile_id = ModelSessionToProfileID(*sess);
  auto profile = ModelDatabase::Singleton().GetModelProfile(gpu_device_,
                                                            profile_id);
  uint32_t batch = model_info["batch"].as<uint32_t>();
  uint32_t max_batch = batch;
  //uint32_t max_batch = profile->GetMaxBatch(sess->latency_sla());
  uint32_t memory_usage = profile->GetMemoryUsage(max_batch);
  float fwd_latency = profile->GetForwardLatency(batch);
  config.set_batch(batch);
  config.set_max_batch(max_batch);
  config.set_memory_usage(memory_usage);
  config.set_forward_latency(fwd_latency);

  // update execution and batch cycles and throughput
  model_table_config_.emplace(model_sess_id, config);
  model_rps_.emplace(model_sess_id, EWMA(1, epoch_sec_));
  exec_cycle_us_ += fwd_latency;
  duty_cycle_us_ += fwd_latency;
  for (auto& iter : model_table_config_) {
    auto& cfg = iter.second;
    float throughput = cfg.batch() * 1e6 / duty_cycle_us_;
    cfg.set_throughput(throughput);
    cfg.set_workload(throughput);
  }
  dirty_model_table_ = true;

  LOG(INFO) << "Backend " << node_id_ << " loads " << config.DebugString();
  LOG(INFO) << "Backend " << node_id_ << ": exec cycle " << exec_cycle_us_ <<
        " us, duty cycle: " << duty_cycle_us_ << " us";
}

void BackendRpcClient::UnloadModel(const std::string& model_sess_id) {
  LOG(INFO) << "Backend " << node_id_ << " unload model: " << model_sess_id;
  auto config = model_table_config_.at(model_sess_id);
  model_table_config_.erase(model_sess_id);
  model_rps_.erase(model_sess_id);
  if (model_table_config_.empty()) {
    exec_cycle_us_ = 0;
    duty_cycle_us_ = 0;
  } else {
    // TODO: support unload model for multi batching
  }
  dirty_model_table_ = true;
}

CtrlStatus BackendRpcClient::UpdateModelTable() {
  if (!dirty_model_table_) {
    return CTRL_OK;
  }
  ModelTableConfig request;
  RpcReply reply;
  for (auto iter : model_table_config_) {
    request.add_model_instance_config()->CopyFrom(iter.second);
  }
  
  // Invoke UpdateModelTable RPC
  grpc::ClientContext context;
  grpc::Status status = stub_->UpdateModelTable(&context, request, &reply);
  if (!status.ok()) {
    LOG(ERROR) << status.error_code() << ": " << status.error_message();
    return CTRL_SERVER_UNREACHABLE;
  }
  last_time_ = std::chrono::system_clock::now();
  if (reply.status() == CTRL_OK) {
    dirty_model_table_ = false;
  }
  return reply.status();
}

void BackendRpcClient::GetModelSessions(std::vector<std::string>* sessions) {
  for (auto iter : model_table_config_) {
    sessions->push_back(iter.first);
  }
}

void BackendRpcClient::UpdateStats(const BackendStatsProto& backend_stats) {
  last_time_ = std::chrono::system_clock::now();
  for (auto model_stats : backend_stats.model_stats()) {
    auto& rps = model_rps_.at(model_stats.model_session_id());
    for (auto num_requests : model_stats.num_requests()) {
      rps.AddSample(num_requests);
    }
  }
}

float BackendRpcClient::GetModelThroughput(const std::string& model_sess_id)
    const {
  return model_table_config_.at(model_sess_id).throughput();
}

float BackendRpcClient::GetModelRps(const std::string& model_sess_id) const {
  return model_rps_.at(model_sess_id).rate();
}

bool BackendRpcClient::IsAlive() {
  auto elapse = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::system_clock::now() - last_time_).count();
  if (elapse < timeout_ms_) {
    return true;
  }
  CheckAliveRequest request;
  RpcReply reply;
  request.set_node_type(BACKEND_NODE);
  request.set_node_id(node_id_);

  // Invoke CheckAlive RPC
  grpc::ClientContext context;
  grpc::Status status = stub_->CheckAlive(&context, request, &reply);
  if (!status.ok()) {
    LOG(ERROR) << status.error_code() << ": " << status.error_message();
    return false;
  }
  last_time_ = std::chrono::system_clock::now();
  return true;
}

bool BackendRpcClient::IsIdle() {
  return exec_cycle_us_ == 0;
}

} // namespace scheduler
} // namespace nexus
