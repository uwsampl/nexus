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
    overload_(false),
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
    model_instances_ = other.model_instances_;
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
    InstanceInfo* inst_info, float* occupancy) const {
  if (workload_id_ >= 0 || Occupancy() == 1.0) {
    // Static configured backend or fully occupied backend cannot load a new
    // model
    return false;
  }
  std::string model_sess_id = ModelSessionToString(model_sess);
  if (model_instances_.find(model_sess_id) != model_instances_.end()) {
    // Already load this model session
    return false;
  }
  std::string profile_id = ModelSessionToProfileID(model_sess);
  auto profile = ModelDatabase::Singleton().GetModelProfile(gpu_device_,
                                                            profile_id);
  if (profile == nullptr) {
    // Cannot find model profile for this GPU
    return false;
  }
  inst_info->model_session.CopyFrom(model_sess);
  inst_info->profile = profile;
  // Compute the best batch size for the workload
  ComputeBatchSize(inst_info, workload);
  if (inst_info->batch == 0) {
    return false;
  }
  // Compute new duty cycle and new exec cycle if we load this model
  float new_duty_cycle, new_exec_cycle;
  if (duty_cycle_us_ == 0 || inst_info->max_duty_cycle_us < duty_cycle_us_) {
    new_duty_cycle = inst_info->max_duty_cycle_us;
    new_exec_cycle = 0.;
    for (auto iter : model_instances_) {
      auto& info = iter.second;
      uint32_t new_b = std::ceil(new_duty_cycle * info.throughput / 1e6);
      new_exec_cycle += info.profile->GetForwardLatency(new_b);
    }
    new_exec_cycle += inst_info->fwd_latency_us;
  } else { // max_duty_cycle >= duty_cycle_us_
    new_duty_cycle = duty_cycle_us_;
    inst_info->batch = std::ceil(new_duty_cycle * inst_info->throughput / 1e6);
    inst_info->fwd_latency_us = profile->GetForwardLatency(inst_info->batch);
    inst_info->throughput = inst_info->batch * 1e6 / new_duty_cycle;
    new_exec_cycle = exec_cycle_us_ + inst_info->fwd_latency_us;
  }
  if (new_exec_cycle > new_duty_cycle) {
    // Doesn't have enough spare cycles to load this workload
    return false;
  }
  *occupancy = new_exec_cycle / new_duty_cycle;
  return true;
}

void BackendDelegate::LoadModel(const InstanceInfo& inst_info) {
  auto model_sess_id = ModelSessionToString(inst_info.model_session);
  model_instances_.emplace(model_sess_id, inst_info);
  model_rps_.emplace(model_sess_id, EWMA(1, epoch_sec_));
  UpdateCycle();
  auto const& info = model_instances_.at(model_sess_id);
  LOG(INFO) << "Backend " << node_id_ << " loads " << model_sess_id <<
      ", batch " << info.batch << ", max batch " << info.max_batch <<
      ", max duty cycle " << info.max_duty_cycle_us << " us, " <<
      "throughput " << info.throughput << " req/s. Backend exec cycle " <<
      exec_cycle_us_ << " us, duty cycle: " << duty_cycle_us_ << " us";
}

void BackendDelegate::LoadModel(const YAML::Node& model_info) {
  InstanceInfo inst_info;
  ModelSession& sess = inst_info.model_session;
  sess.set_framework(model_info["framework"].as<std::string>());
  sess.set_model_name(model_info["model_name"].as<std::string>());
  sess.set_version(model_info["version"].as<uint32_t>());
  sess.set_latency_sla(model_info["latency_sla"].as<uint32_t>());
  if (model_info["image_height"]) {
    sess.set_image_height(model_info["image_height"].as<uint32_t>());
    sess.set_image_width(model_info["image_width"].as<uint32_t>());
  }
  std::string model_sess_id = ModelSessionToString(sess);
  std::string profile_id = ModelSessionToProfileID(sess);
  auto profile = ModelDatabase::Singleton().GetModelProfile(gpu_device_,
                                                            profile_id);
  inst_info.batch = model_info["batch"].as<uint32_t>();
  inst_info.max_batch = inst_info.batch;
  //inst_info.max_batch = profile->GetMaxBatch(sess.latency_sla());
  inst_info.fwd_latency_us = profile->GetForwardLatency(inst_info.batch);
  inst_info.memory_usage = profile->GetMemoryUsage(inst_info.max_batch);

  // update execution and batch cycles and throughput
  model_instances_.emplace(model_sess_id, inst_info);
  model_rps_.emplace(model_sess_id, EWMA(1, epoch_sec_));
  exec_cycle_us_ += inst_info.fwd_latency_us;
  duty_cycle_us_ += inst_info.fwd_latency_us;
  for (auto& iter : model_instances_) {
    auto& info = iter.second;
    info.throughput = info.batch * 1e6 / duty_cycle_us_;
  }
  dirty_model_table_ = true;

  LOG(INFO) << "Backend " << node_id_ << " loads " << model_sess_id <<
      ", batch " << inst_info.batch << ", exec cycle " << exec_cycle_us_ <<
      " us, duty cycle: " << duty_cycle_us_ << " us";
}

void BackendDelegate::UnloadModel(const std::string& model_sess_id) {
  if (workload_id_ >= 0) {
    return;
  }
  LOG(INFO) << "Backend " << node_id_ << " unload model: " << model_sess_id;
  auto config = model_instances_.at(model_sess_id);
  model_instances_.erase(model_sess_id);
  model_rps_.erase(model_sess_id);
  UpdateCycle();
}

float BackendDelegate::UpdateModelThroughput(const std::string& model_sess_id,
                                             float workload) {
  InstanceInfo* inst_info = &model_instances_.at(model_sess_id);
  uint32_t prev_throughput = inst_info->throughput;
  ComputeBatchSize(inst_info, workload);
  if (prev_throughput == inst_info->throughput) {
    LOG(INFO) << "No change";
  } else {
    UpdateCycle();
    LOG(INFO) << "Backend " << node_id_ << " updates " << model_sess_id <<
        ", batch " << inst_info->batch << ", max batch " <<
        inst_info->max_batch << ", new throughput " << inst_info->throughput;
  }
  return inst_info->throughput;
}

void BackendDelegate::SpillOutWorkload(
    std::vector<std::pair<std::string, float> >* spillout) {
  if (!overload_ || workload_id_ >= 0) {
    return;
  }
  std::vector<std::tuple<std::string, float, float> > workloads;
  for (auto iter : model_instances_) {
    workloads.emplace_back(iter.first, iter.second.fwd_latency_us,
                           iter.second.throughput);
  }
  // Sort workload based on exec latency
  std::sort(workloads.begin(), workloads.end(),
            [](std::tuple<std::string, float, float> a,
               std::tuple<std::string, float, float> b) {
              return std::get<1>(a) > std::get<1>(b);
            });
  // Recompute exec cycle and duty cycle
  model_instances_.clear();
  exec_cycle_us_ = 0.;
  duty_cycle_us_ = 0.;
  for (auto iter : workloads) {
    auto model_sess_id = std::get<0>(iter);
    ModelSession model_sess;
    ParseModelSession(model_sess_id, &model_sess);
    float workload = std::get<2>(iter);
    InstanceInfo inst_info;
    float occupancy;
    bool ret = PrepareLoadModel(model_sess, workload, &inst_info, &occupancy);
    if (!ret) {
      spillout->emplace_back(model_sess_id, workload);
    } else {
      LoadModel(inst_info);
    }
  }
  CHECK_LE(exec_cycle_us_, duty_cycle_us_);
  overload_ = false;
}

CtrlStatus BackendDelegate::UpdateModelTableRpc() {
  if (!dirty_model_table_) {
    return CTRL_OK;
  }
  ModelTableConfig request;
  RpcReply reply;
  for (auto iter : model_instances_) {
    auto const& inst_info = iter.second;
    auto cfg = request.add_model_instance_config();
    cfg->mutable_model_session()->CopyFrom(inst_info.model_session);
    cfg->set_batch(inst_info.batch);
    cfg->set_max_batch(inst_info.max_batch);
    cfg->set_memory_usage(inst_info.memory_usage);
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
      if (rps.rate() < 0 && num_requests == 0) {
        continue;
      }
      rps.AddSample(num_requests);
    }
  }
}

void BackendDelegate::AllModelSessions(std::vector<std::string>* sessions)
    const {
  for (auto iter : model_instances_) {
    sessions->push_back(iter.first);
  }
}

const InstanceInfo* BackendDelegate::GetInstanceInfo(
    const std::string& model_sess_id) const {
  auto iter = model_instances_.find(model_sess_id);
  if (iter == model_instances_.end()) {
    return nullptr;
  }
  return &iter->second;
}

float BackendDelegate::GetModelThroughput(const std::string& model_sess_id)
    const {
  return model_instances_.at(model_sess_id).throughput;
}

float BackendDelegate::GetModelRps(const std::string& model_sess_id) const {
  return std::max(0., model_rps_.at(model_sess_id).rate());
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

void BackendDelegate::ComputeBatchSize(InstanceInfo* inst_info,
                                       float workload) const {
  // 1. Compute the max batch and throughput to saturate an empty GPU
  uint32_t batch, max_batch;
  float max_throughput;
  max_batch = inst_info->profile->GetMaxBatch(
      inst_info->model_session.latency_sla());
  max_throughput = max_batch * 1e6 / inst_info->profile->GetForwardLatency(
      max_batch);
  // std::tie(batch, max_throughput) = inst_info->profile->GetMaxThroughput(
  //     inst_info->model_session.latency_sla());
  if (workload == 0 || workload >= max_throughput) {
    inst_info->batch = max_batch;
    inst_info->max_batch = max_batch;
    inst_info->fwd_latency_us = inst_info->profile->GetForwardLatency(
        max_batch);
    inst_info->max_duty_cycle_us = inst_info->fwd_latency_us;
    inst_info->throughput = max_throughput;
    inst_info->memory_usage = inst_info->profile->GetMemoryUsage(max_batch);
    return;
  }

  // 2. Compute the max batch for residue load
  double latency_sla_us = inst_info->model_session.latency_sla() * 1000;
  float preprocess = inst_info->profile->GetPreprocessLatency();
  float postprocess = inst_info->profile->GetPostprocessLatency();
  batch = 1;
  for (; batch <= max_batch; ++batch) {
    float fwd_lat = inst_info->profile->GetForwardLatency(batch);
    // because batch = ceil(workload * duty_cycle),
    // duty_cycle >= (batch - 1) / workload
    float min_duty_cycle = (batch - 1) * 1e6 / workload;
    if (min_duty_cycle + fwd_lat + preprocess + postprocess > latency_sla_us) {
      break;
    }
  }
  --batch;
  if (batch == 0) {
    // This GPU is too slow so that exec latency of batch 1 is too large to
    // satisfy latency_sla
    inst_info->batch = 0;
    return;
  }
  inst_info->batch = batch;
  inst_info->max_batch = max_batch;
  inst_info->fwd_latency_us = inst_info->profile->GetForwardLatency(batch);
  // duty_cycle are constrainted by the following condition:
  // (1) throughput = batch / duty_cycle >= workload
  //     => duty_cycle <= batch / workload
  // (2) duty_cycle + preprocess + postprocess + fwd_lat <= latency_sla
  inst_info->max_duty_cycle_us = std::min(
      latency_sla_us - inst_info->fwd_latency_us - preprocess - postprocess,
      batch * 1e6 / workload);
  inst_info->throughput = batch * 1e6 / inst_info->max_duty_cycle_us;
  inst_info->memory_usage = inst_info->profile->GetMemoryUsage(max_batch);
}

void BackendDelegate::UpdateCycle() {
  exec_cycle_us_ = 0.;
  duty_cycle_us_ = 0.;
  for (auto iter : model_instances_) {
    auto& inst_info = iter.second;
    if (duty_cycle_us_ == 0 || inst_info.max_duty_cycle_us < duty_cycle_us_) {
      duty_cycle_us_ = inst_info.max_duty_cycle_us;
    }
  }
  for (auto& iter : model_instances_) {
    auto& inst_info = iter.second;
    inst_info.batch = std::ceil(duty_cycle_us_ * inst_info.throughput / 1e6);
    if (inst_info.batch > inst_info.max_batch) {
      overload_ = true;
      inst_info.batch = 0;
      inst_info.fwd_latency_us = inst_info.profile->GetForwardLatency(
          inst_info.max_batch);
    } else {
      inst_info.fwd_latency_us = inst_info.profile->GetForwardLatency(
          inst_info.batch);
    }
    exec_cycle_us_ += inst_info.fwd_latency_us;
  }
  dirty_model_table_ = true;
  if (exec_cycle_us_ > duty_cycle_us_) {
    overload_ = true;
  }
}

} // namespace scheduler
} // namespace nexus
