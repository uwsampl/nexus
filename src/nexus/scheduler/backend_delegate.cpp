#include <glog/logging.h>

#include "nexus/common/model_db.h"
#include "nexus/scheduler/backend_delegate.h"
#include "nexus/scheduler/scheduler.h"

namespace nexus {
namespace scheduler {

BackendDelegate::BackendDelegate(uint32_t node_id,
                                 const std::string& server_addr,
                                 const std::string& rpc_addr,
                                 const std::string& gpu_device,
                                 size_t gpu_available_memory,
                                 int beacon_sec, int epoch_sec):
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

float BackendDelegate::Occupancy() const {
  if (exec_cycle_us_ == 0) {
    return 0.;
  }
  return exec_cycle_us_ / duty_cycle_us_;
}

void BackendDelegate::GetInfo(BackendInfo* info) const {
  info->set_node_id(node_id_);
  info->set_server_address(server_address_);
}

std::time_t BackendDelegate::LastAliveTime() const {
  return std::chrono::system_clock::to_time_t(last_time_);
}

void BackendDelegate::Tick() {
  last_time_ = std::chrono::system_clock::now();
}

bool BackendDelegate::Assign(const BackendDelegate& other) {
  CHECK(IsIdle()) << "Backend is not idle";
  if (gpu_device_ == other.gpu_device_) {
    workload_id_ = other.workload_id_;
    model_rps_ = other.model_rps_;
    model_table_config_ = other.model_table_config_;
    exec_cycle_us_ = other.exec_cycle_us_;
    duty_cycle_us_ = other.exec_cycle_us_;
    dirty_model_table_ = true;
    return true;
  }
  // TODO: support assign for two backend that have different GPU devices
  return false;
}

bool BackendDelegate::PrepareLoadModel(
    const ModelSession& model_sess, float workload,
    ModelInstanceConfig* config, float* occupancy) const {
  if (workload_id_ >= 0) {
    // Static configured backend doesn't support loading other models
    return false;
  }
  
  std::string model_sess_id = ModelSessionToString(model_sess);
  if (Occupancy() == 1.0) {
    return false;
  }
  if (model_table_config_.find(model_sess_id) != model_table_config_.end()) {
    return false;
  }
  std::string profile_id = ModelSessionToProfileID(model_sess);
  auto profile = ModelDatabase::Singleton().GetModelProfile(gpu_device_,
                                                            profile_id);
  if (profile == nullptr) {
    return false;
  }
  config->mutable_model_session()->CopyFrom(model_sess);

  // 1. Compute the max batch and throughput to saturate an empty GPU
  double latency_sla_us = model_sess.latency_sla() * 1000;
  uint32_t max_batch;
  uint32_t max_throughput;
  std::tie(max_batch, max_throughput) = profile->GetMaxThroughput(
      model_sess.latency_sla());

  if (workload == 0 || workload >= max_throughput) {
    // Workload can saturate entire gpu
    if (exec_cycle_us_ > 0) {
      // We should always allocate an entire gpu in this case
      config->set_batch(0);
    } else {
      double fwd_latency = profile->GetForwardLatency(max_batch);
      uint32_t memory_usage = profile->GetMemoryUsage(max_batch);
      config->set_batch(max_batch);
      config->set_max_batch(max_batch);
      config->set_forward_latency(fwd_latency);
      config->set_memory_usage(fwd_latency);
      config->set_throughput(max_throughput);
      *occupancy = 1.0;
    }
    return true;
  }

  // 2. Compute the max batch for residue load
  float preprocess = profile->GetPreprocessLatency();
  float postprocess = profile->GetPostprocessLatency();
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
    return false;
  }
  float fwd_lat = profile->GetForwardLatency(batch);
  uint32_t memory_usage = profile->GetMemoryUsage(batch);
  // duty_cycle are constrainted by the following condition:
  // (1) throughput = batch / duty_cycle >= workload
  //     => duty_cycle <= batch / workload
  // (2) duty_cycle + preprocess + postprocess + fwd_lat <= latency_sla
  double duty_cycle = std::min(
      latency_sla_us - fwd_lat - preprocess - postprocess,
      batch * 1e6 / workload);
  float throughput = batch * 1e6 / duty_cycle;

  if (exec_cycle_us_ == 0) {
    config->set_batch(batch);
    config->set_max_batch(max_batch);
    config->set_forward_latency(fwd_lat);
    config->set_memory_usage(memory_usage);
    config->set_throughput(throughput);
    *occupancy = fwd_lat / duty_cycle;
    return true;
  }

  float new_duty_cycle, new_exec_cycle;
  if (duty_cycle < duty_cycle_us_) {
    new_duty_cycle = duty_cycle;
    new_exec_cycle = 0.;
    for (auto iter : model_table_config_) {
      auto& cfg = iter.second;
      uint32_t new_b = std::ceil(duty_cycle * cfg.throughput() / 1e6);
      std::string pid = ModelSessionToProfileID(cfg.model_session());
      new_exec_cycle += ModelDatabase::Singleton().GetModelForwardLatency(
          gpu_device_, pid, new_b);
    }
    new_exec_cycle += fwd_lat;
  } else { // duty_cycle >= duty_cycle_us_
    new_duty_cycle = duty_cycle_us_;
    batch = std::ceil(new_duty_cycle * workload / 1e6);
    fwd_lat = profile->GetForwardLatency(batch);
    memory_usage = profile->GetMemoryUsage(batch);
    throughput = batch * 1e6 / new_duty_cycle;
    new_exec_cycle = exec_cycle_us_ + fwd_lat;
  }
  if (new_exec_cycle > new_duty_cycle) {
    // Failed to load residue load on this gpu
    return false;
  }
  
  config->set_batch(batch);
  config->set_max_batch(max_batch);
  config->set_forward_latency(fwd_lat);
  config->set_memory_usage(memory_usage);
  config->set_throughput(throughput);
  *occupancy = new_exec_cycle / new_duty_cycle;

  return true;
}

void BackendDelegate::LoadModel(const ModelInstanceConfig& config) {
  auto model_sess_id = ModelSessionToString(config.model_session());
  model_table_config_.emplace(model_sess_id, config);
  model_rps_.emplace(model_sess_id, EWMA(1, epoch_sec_));
  if (exec_cycle_us_ == 0) {
    exec_cycle_us_ = config.forward_latency();
    duty_cycle_us_ = config.batch() * 1e6 / config.throughput();
  } else {
    // Multi-batching
    float duty_cycle = config.batch() * 1e6 / config.throughput();
    if (duty_cycle >= duty_cycle_us_) {
      // duty cycle remains the same
      exec_cycle_us_ += config.forward_latency();
    } else {
      duty_cycle_us_ = duty_cycle;
      exec_cycle_us_ = 0;
      for (auto& iter : model_table_config_) {
        auto& cfg = iter.second;
        uint32_t new_batch = std::ceil(duty_cycle * cfg.throughput() / 1e6);
        auto profile = ModelDatabase::Singleton().GetModelProfile(
            gpu_device_, ModelSessionToProfileID(cfg.model_session()));
        float fwd_lat = profile->GetForwardLatency(new_batch);
        exec_cycle_us_ += fwd_lat;
        cfg.set_batch(new_batch);
        cfg.set_forward_latency(fwd_lat);
      }
    }
  }
  dirty_model_table_ = true;

  LOG(INFO) << "Backend " << node_id_ << " loads " << config.DebugString();
  LOG(INFO) << "Backend " << node_id_ << ": exec cycle " << exec_cycle_us_ <<
      " us, duty cycle: " << duty_cycle_us_ << " us";
}

void BackendDelegate::LoadModel(const YAML::Node& model_info) {
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
  }
  dirty_model_table_ = true;

  LOG(INFO) << "Backend " << node_id_ << " loads " << config.DebugString();
  LOG(INFO) << "Backend " << node_id_ << ": exec cycle " << exec_cycle_us_ <<
        " us, duty cycle: " << duty_cycle_us_ << " us";
}

void BackendDelegate::UnloadModel(const std::string& model_sess_id) {
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

float BackendDelegate::UpdateModelThroughput(const std::string& model_sess_id,
                                             float workload) {
  auto& cfg = model_table_config_.at(model_sess_id);
  uint32_t batch = std::ceil(workload * duty_cycle_us_ / 1e6);
  if (batch > cfg.max_batch()) {
    batch = cfg.max_batch();
  }
  if (batch != cfg.batch()) {
    ModelSession model_sess;
    ParseModelSession(model_sess_id, &model_sess);
    std::string profile_id = ModelSessionToProfileID(model_sess);
    auto profile = ModelDatabase::Singleton().GetModelProfile(gpu_device_,
                                                              profile_id);
    exec_cycle_us_ -= profile->GetForwardLatency(cfg.batch());
    // Update batch and throughput
    cfg.set_batch(batch);
    float throughput = batch * 1e6 / duty_cycle_us_;
    cfg.set_throughput(throughput);
    exec_cycle_us_ += profile->GetForwardLatency(batch);
    dirty_model_table_ = true;
  }
  return std::max(workload - cfg.throughput(), 0.f);
}

void BackendDelegate::SpillOutWorkload(
    std::vector<std::pair<std::string, float> >* spillout) {
  if (Occupancy() <= 1.0 || workload_id_ >= 0) {
    return;
  }
  std::vector<std::pair<std::string, float> > workloads;
  for (auto iter : model_table_config_) {
    workloads.emplace_back(iter.first, iter.second.forward_latency());
  }
  std::sort(workloads.begin(), workloads.end(),
            [](std::pair<std::string, float> a,
               std::pair<std::string, float> b) {
              return a.second > b.second;
            });
  // Recompute exec cycle and duty cycle
  exec_cycle_us_ = 0.;
  duty_cycle_us_ = 0.;
  for (auto iter : workloads) {
    auto& model_sess_id = iter.first;
    float fwd_lat = iter.second;
    auto& cfg = model_table_config_.at(model_sess_id);
    auto profile = ModelDatabase::Singleton().GetModelProfile(
        gpu_device_, ModelSessionToProfileID(cfg.model_session()));
    float latency_sla_us = cfg.model_session().latency_sla() * 1000.;
    float preprocess = profile->GetPreprocessLatency();
    float postprocess = profile->GetPostprocessLatency();
    float new_duty_cycle = std::min(
        latency_sla_us - fwd_lat - preprocess - postprocess,
        cfg.batch() * 1e6f / cfg.throughput());
    if (duty_cycle_us_ > 0 && duty_cycle_us_ < new_duty_cycle) {
      new_duty_cycle = duty_cycle_us_;
    }
    if (exec_cycle_us_ + fwd_lat < new_duty_cycle) {
      exec_cycle_us_ += fwd_lat;
      duty_cycle_us_ = new_duty_cycle;
    } else {
      spillout->emplace_back(model_sess_id, cfg.throughput());
      model_table_config_.erase(model_sess_id);
    }
  }
  CHECK_LE(exec_cycle_us_, duty_cycle_us_);
}

CtrlStatus BackendDelegate::UpdateModelTableRpc() {
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

void BackendDelegate::UpdateStats(const BackendStatsProto& backend_stats) {
  last_time_ = std::chrono::system_clock::now();
  for (auto model_stats : backend_stats.model_stats()) {
    auto& rps = model_rps_.at(model_stats.model_session_id());
    for (auto num_requests : model_stats.num_requests()) {
      rps.AddSample(num_requests);
    }
  }
}

void BackendDelegate::AllModelSessions(std::vector<std::string>* sessions)
    const {
  for (auto iter : model_table_config_) {
    sessions->push_back(iter.first);
  }
}

const ModelInstanceConfig* BackendDelegate::GetModelConfig(
    const std::string& model_sess_id) const {
  auto iter = model_table_config_.find(model_sess_id);
  if (iter == model_table_config_.end()) {
    return nullptr;
  }
  return &iter->second;
}

float BackendDelegate::GetModelThroughput(const std::string& model_sess_id)
    const {
  return model_table_config_.at(model_sess_id).throughput();
}

float BackendDelegate::GetModelRps(const std::string& model_sess_id) const {
  return model_rps_.at(model_sess_id).rate();
}

bool BackendDelegate::IsAlive() {
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

bool BackendDelegate::IsIdle() const {
  return (exec_cycle_us_ == 0);
}

} // namespace scheduler
} // namespace nexus
