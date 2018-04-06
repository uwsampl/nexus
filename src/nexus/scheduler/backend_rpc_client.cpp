#include <glog/logging.h>

#include "nexus/common/model_profile.h"
#include "nexus/scheduler/backend_rpc_client.h"
#include "nexus/scheduler/scheduler.h"

namespace nexus {
namespace scheduler {

BackendRpcClient::BackendRpcClient(Scheduler* sch, uint32_t node_id,
                                   const std::string& server_addr,
                                   const std::string& rpc_addr,
                                   const std::string& gpu_device,
                                   size_t gpu_available_memory,
                                   std::chrono::milliseconds timeout):
    scheduler_(sch),
    node_id_(node_id),
    server_address_(server_addr),
    rpc_address_(rpc_addr),
    gpu_device_(gpu_device),
    gpu_available_memory_(gpu_available_memory),
    timeout_(timeout),
    workload_id_(-1),
    exec_cycle_(0.),
    duty_cycle_(0.),
    dirty_model_table_(false) {
  auto channel = grpc::CreateChannel(rpc_addr,
                                     grpc::InsecureChannelCredentials());
  stub_ = BackendCtrl::NewStub(channel);
  last_time_ = std::chrono::system_clock::now();
}

void BackendRpcClient::Tick() {
  std::lock_guard<std::mutex> lock(mutex_);
  last_time_ = std::chrono::system_clock::now();
}

std::time_t BackendRpcClient::LastTime() {
  std::lock_guard<std::mutex> lock(mutex_);
  return std::chrono::system_clock::to_time_t(last_time_);
}

void BackendRpcClient::PrepareLoadModel(
    const ModelSession& model_sess, float workload,
    ModelInstanceDesc* model_desc, float* occupancy) {
  if (workload == 0) {
    model_desc->set_batch(0);
    return;
  }
  std::string model_id = ModelSessionToString(model_sess, false);
  auto profile = ModelProfileTable::Singleton().GetModelProfile(
      gpu_device_, model_id);
  if (profile == nullptr) {
    model_desc->set_batch(0);
    return;
  }
  model_desc->mutable_model_session()->CopyFrom(model_sess);
  
  // lock protected below
  std::lock_guard<std::mutex> lock(mutex_);

  // 1. Compute the max batch and throughput to saturate an empty GPU
  float latency_sla = model_sess.latency_sla();
  uint32_t max_batch;
  uint32_t max_throughput;
  std::tie(max_batch, max_throughput) = profile->GetMaxThroughput(latency_sla);

  if (exec_cycle_ == 0) {
    // empty GPU 
    if (max_throughput <= workload) {
      // workload can saturate the gpu
      float fwd_latency = profile->GetForwardLatency(max_batch);
      uint32_t memory_usage = profile->GetMemoryUsage(max_batch);
      model_desc->set_batch(max_batch);
      model_desc->set_max_batch(max_batch);
      model_desc->set_forward_latency(fwd_latency);
      model_desc->set_memory_usage(fwd_latency);
      model_desc->set_throughput(max_throughput);
      model_desc->set_workload(max_throughput);
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
        float min_duty_cycle = (batch - 1) * 1000. / workload;
        if (min_duty_cycle + fwd_lat + preprocess + postprocess > latency_sla) {
          break;
        }
      }
      --batch;
      if (batch == 0) {
        // execution latency of batch size 1 is even too large for latency_sla
        model_desc->set_batch(0);
      } else {
        float fwd_lat = profile->GetForwardLatency(batch);
        uint32_t memory_usage = profile->GetMemoryUsage(batch);
        float duty_cycle = latency_sla - fwd_lat - preprocess - postprocess;
        float throughput = batch * 1000. / duty_cycle;
        model_desc->set_batch(batch);
        model_desc->set_max_batch(max_batch);
        model_desc->set_forward_latency(fwd_lat);
        model_desc->set_memory_usage(memory_usage);
        model_desc->set_throughput(throughput);
        model_desc->set_workload(workload);
        *occupancy = fwd_lat / duty_cycle;
      }
    }
  } else {
    // TODO
    model_desc->set_batch(0);
  }
}

void BackendRpcClient::LoadModel(const ModelInstanceDesc& model_desc) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (exec_cycle_ > 0) {
    LOG(ERROR) << "Backend is not idle. Don't support multi-batching now.";
  } else {
    exec_cycle_ = model_desc.forward_latency();
    duty_cycle_ = model_desc.model_session().latency_sla() - exec_cycle_;
    model_table_.push_back(model_desc);
    dirty_model_table_ = true;

    LOG(INFO) << "Backend " << node_id_ << " loads " <<
        ModelSessionToString(model_desc.model_session()) <<
        ", batch: " << model_desc.batch() << ", max batch: " <<
        model_desc.max_batch() << ", throughput: " <<
        model_desc.throughput() << ", workload: " << model_desc.workload() <<
        ", exec cycle: " << exec_cycle_ << ", duty cycle: " << duty_cycle_;
  }
}

void BackendRpcClient::LoadModel(const ModelId& model,
                                 const YAML::Node& model_info) {
  ModelInstanceDesc model_desc;
  auto sess = model_desc.mutable_model_session();
  sess->set_framework(model.first);
  sess->set_model_name(model.second);
  sess->set_latency_sla(model_info["latency_slo"].as<float>());
  if (model_info["image_height"]) {
    sess->set_image_height(model_info["image_height"].as<uint32_t>());
    sess->set_image_width(model_info["image_width"].as<uint32_t>());
  }

  std::string model_id = ModelSessionToString(*sess, false);
  std::string sess_id = ModelSessionToString(*sess);
  auto profile = ModelProfileTable::Singleton().GetModelProfile(
      gpu_device_, model_id);
  uint32_t batch = model_info["max_batch"].as<uint32_t>();
  uint32_t max_batch = profile->GetMaxBatch(sess->latency_sla());
  uint32_t memory_usage = profile->GetMemoryUsage(max_batch);
  float fwd_latency = profile->GetForwardLatency(batch);
  model_desc.set_batch(batch);
  model_desc.set_max_batch(max_batch);
  model_desc.set_memory_usage(memory_usage);
  model_desc.set_forward_latency(fwd_latency);

  // update execution and batch cycles and throughput
  std::lock_guard<std::mutex> lock(mutex_);
  model_table_.push_back(model_desc);
  exec_cycle_ += fwd_latency;
  duty_cycle_ += duty_cycle_;
  for (auto& md : model_table_) {
    float throughput = md.batch() * 1000. / duty_cycle_;
    md.set_throughput(throughput);
    md.set_workload(throughput);
  }
  dirty_model_table_ = true;

  LOG(INFO) << "Backend " << node_id_ << " loads " << sess_id << ", batch: " <<
      batch << ", fwd lat: " << fwd_latency << ", exec cycle: " <<
      exec_cycle_ << ", duty cycle: " << duty_cycle_;
}

CtrlStatus BackendRpcClient::UpdateModelTable() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!dirty_model_table_) {
    return CTRL_OK;
  }
  ModelTable model_table;
  RpcReply reply;
  GetModelTableNoLock(&model_table);
  grpc::Status ret = UpdateModelTableRpc(model_table, &reply);
  if (!ret.ok()) {
    return CTRL_SERVER_UNREACHABLE;
  }
  if (reply.status() == CTRL_OK) {
    dirty_model_table_ = false;
  }
  return reply.status();
}

void BackendRpcClient::GetModelTable(ModelTable* model_table) {
  std::lock_guard<std::mutex> lock(mutex_);
  GetModelTableNoLock(model_table);
}

bool BackendRpcClient::IsAlive() {
  std::lock_guard<std::mutex> lock(mutex_);
  auto now = std::chrono::system_clock::now();
  std::chrono::duration<double> elapse = now - last_time_;
  if (elapse < timeout_) {
    return true;
  }
  CheckAliveRequest request;
  RpcReply reply;
  request.set_node_type(BACKEND_NODE);
  request.set_node_id(node_id_);
  grpc::Status ret = CheckAliveRpc(request, &reply);
  if (!ret.ok()) {
    return false;
  }
  return true;
}

bool BackendRpcClient::IsIdle() {
  std::lock_guard<std::mutex> lock(mutex_);
  return exec_cycle_ == 0;
}

grpc::Status BackendRpcClient::UpdateModelTableRpc(
    const ModelTable& request, RpcReply* reply) {
  grpc::ClientContext context;
  grpc::Status ret = stub_->UpdateModelTable(&context, request, reply);
  if (ret.ok()) {
    last_time_ = std::chrono::system_clock::now();
  } else {
    LOG(ERROR) << ret.error_code() << ": " << ret.error_message();
  }
  return ret;
}

grpc::Status BackendRpcClient::CheckAliveRpc(
    const CheckAliveRequest& request, RpcReply* reply) {
  grpc::ClientContext context;
  grpc::Status ret = stub_->CheckAlive(&context, request, reply);
  if (ret.ok()) {
    last_time_ = std::chrono::system_clock::now();
  } else {
    LOG(ERROR) << ret.error_code() << ": " << ret.error_message();
  }
  return ret;
}

void BackendRpcClient::GetModelTableNoLock(ModelTable* model_table) {
  for (auto model_desc : model_table_) {
    model_table->add_model_instance_desc()->CopyFrom(model_desc);
  }
}

} // namespace scheduler
} // namespace nexus
