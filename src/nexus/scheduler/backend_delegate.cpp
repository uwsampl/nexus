#include <cmath>
#include <sstream>
#include <glog/logging.h>

#include "nexus/common/model_db.h"
#include "nexus/scheduler/backend_delegate.h"
#include "nexus/scheduler/scheduler.h"

uint32_t BatchSizeCeilEps(double x, double eps) {
  double floor = std::floor(x);
  if (x - floor < eps && floor != 0)
    return static_cast<uint32_t>(floor);
  return static_cast<uint32_t>(std::ceil(x));
}

namespace nexus {
namespace scheduler {

struct CalcCycleResult {
  struct InstanceInfoChange {
    uint32_t batch;
    double fwd_latency_us;
  };
  double exec_cycle_us;
  double duty_cycle_us;
  bool overload;
  std::vector<InstanceInfoChange> changes;
};

CalcCycleResult calc_cycle(const std::vector<InstanceInfoPtr> &models) {
  double exec_cycle_us = 0;
  double duty_cycle_us = 0;
  bool overload = false;
  std::vector<CalcCycleResult::InstanceInfoChange> changes;
  for (const auto &inst_info : models) {
    if (duty_cycle_us == 0 || inst_info->max_duty_cycle_us < duty_cycle_us) {
      duty_cycle_us = inst_info->max_duty_cycle_us;
    }
  }
  for (const auto &inst_info : models) {
    double fwd_latency_us;
    uint32_t batch = BatchSizeCeilEps(duty_cycle_us * inst_info->workload / 1e6, 1e-3);
    if (batch > inst_info->max_batch) {
      overload = true;
      batch = 0;
      fwd_latency_us = inst_info->profile->GetForwardLatency(inst_info->max_batch);
    } else {
      CHECK_NE(batch, 0);
      fwd_latency_us = inst_info->profile->GetForwardLatency(batch);
    }
    exec_cycle_us += fwd_latency_us;
    changes.push_back({batch, fwd_latency_us});
  }
  if (exec_cycle_us > duty_cycle_us) {
    overload = true;
  }
  return {exec_cycle_us, duty_cycle_us, overload, changes};
}

BackendDelegate::BackendDelegate(uint32_t node_id, const std::string& ip,
                                 const std::string& server_port,
                                 const std::string& rpc_port,
                                 const std::string& gpu_device,
                                 size_t gpu_available_memory, int beacon_sec):
    node_id_(node_id),
    ip_(ip),
    server_port_(server_port),
    rpc_port_(rpc_port),
    gpu_device_(gpu_device),
    gpu_available_memory_(gpu_available_memory),
    beacon_sec_(beacon_sec),
    timeout_ms_(beacon_sec * 3 * 1000),
    workload_id_(-1),
    exec_cycle_us_(0.),
    duty_cycle_us_(0.),
    overload_(false),
    dirty_model_table_(false) {
  std::stringstream rpc_addr;
  rpc_addr << ip_ << ":" << rpc_port_;
  auto channel = grpc::CreateChannel(rpc_addr.str(),
                                     grpc::InsecureChannelCredentials());
  stub_ = BackendCtrl::NewStub(channel);
  last_time_ = std::chrono::system_clock::now();
}

double BackendDelegate::Occupancy() const {
  if (exec_cycle_us_ == 0) {
    return 0.;
  }
  return exec_cycle_us_ / duty_cycle_us_;
}

void BackendDelegate::GetInfo(BackendInfo* info) const {
  info->set_node_id(node_id_);
  info->set_ip(ip_);
  info->set_server_port(server_port_);
  info->set_rpc_port(rpc_port_);
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
    models_ = other.models_;
    backup_models_ = other.backup_models_;
    session_model_map_ = other.session_model_map_;
    exec_cycle_us_ = other.exec_cycle_us_;
    duty_cycle_us_ = other.exec_cycle_us_;
    overload_ = other.overload_;
    dirty_model_table_ = true;
    return true;
  }
  // TODO: support assign for two backend that have different GPU devices
  return false;
}

bool BackendDelegate::PrepareLoadModel(
    const ModelSession& model_sess, double workload,
    InstanceInfo* inst_info, double* occupancy) const {
  if (workload_id_ >= 0 || Occupancy() >= 1.0) {
    // Static configured backend or fully occupied backend cannot load a new
    // model
    return false;
  }
  std::string model_sess_id = ModelSessionToString(model_sess);
  if (session_model_map_.count(model_sess_id) > 0) {
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
  inst_info->model_sessions.push_back(model_sess);
  inst_info->profile = profile;
  // Compute the best batch size for the workload
  ComputeBatchSize(inst_info, workload);
  if (inst_info->batch == 0) {
    return false;
  }
  // Compute new duty cycle and new exec cycle if we load this model
  auto new_models = models_;
  new_models.push_back(std::make_shared<InstanceInfo>(*inst_info));
  auto res = calc_cycle(new_models);
  if (duty_cycle_us_ != 0 && inst_info->max_duty_cycle_us >= duty_cycle_us_) {
    const auto &change = res.changes.back();
    inst_info->batch = change.batch;
    inst_info->fwd_latency_us = change.fwd_latency_us;
    inst_info->throughput = inst_info->batch * 1e6 / duty_cycle_us_;
    CHECK_NE(inst_info->batch, 0);
  }

  if (res.exec_cycle_us > res.duty_cycle_us) {
    // Doesn't have enough spare cycles to load this workload
    return false;
  }
  *occupancy = res.exec_cycle_us / res.duty_cycle_us;
  CHECK_LE(*occupancy, 1.0 + 1e-3) << "Backend is overloaded";
  return true;
}

void BackendDelegate::LoadModel(const InstanceInfo& inst_info) {
  auto model_session_id = ModelSessionToString(inst_info.model_sessions[0]);
  CHECK_EQ(session_model_map_.count(model_session_id), 0) <<
      "Model session " << model_session_id << " already exists.";
  auto info = std::make_shared<InstanceInfo>(inst_info);
  models_.push_back(info);
  session_model_map_.emplace(model_session_id, info);
  UpdateCycle();
  LOG(INFO) << "Backend " << node_id_ << " loads " << model_session_id <<
      ", batch " << info->batch << ", max batch " << info->max_batch <<
      ", max duty cycle " << info->max_duty_cycle_us << " us, " <<
      "throughput " << info->throughput << " req/s. Backend exec cycle " <<
      exec_cycle_us_ << " us, duty cycle: " << duty_cycle_us_ << " us";
}

void BackendDelegate::LoadModel(const YAML::Node& model_info) {
  ModelSession sess;
  if (model_info["share_prefix"]) {
    auto base_model_info = model_info["share_prefix"][0];
    sess.set_framework(base_model_info["framework"].as<std::string>());
    sess.set_model_name(base_model_info["model_name"].as<std::string>());
    sess.set_version(base_model_info["version"].as<uint32_t>());
    sess.set_latency_sla(base_model_info["latency_sla"].as<uint32_t>());
  } else {
    sess.set_framework(model_info["framework"].as<std::string>());
    sess.set_model_name(model_info["model_name"].as<std::string>());
    sess.set_version(model_info["version"].as<uint32_t>());
    sess.set_latency_sla(model_info["latency_sla"].as<uint32_t>());
  }
  if (model_info["image_height"]) {
    sess.set_image_height(model_info["image_height"].as<uint32_t>());
    sess.set_image_width(model_info["image_width"].as<uint32_t>());
  }
  std::string model_session_id = ModelSessionToString(sess);
  std::string profile_id = ModelSessionToProfileID(sess);
  auto profile = ModelDatabase::Singleton().GetModelProfile(gpu_device_,
                                                            profile_id);
  auto inst_info = std::make_shared<InstanceInfo>();
  inst_info->model_sessions.push_back(sess);
  inst_info->profile = profile;
  inst_info->batch = model_info["batch"].as<uint32_t>();
  inst_info->max_batch = inst_info->batch;
  //inst_info->max_batch = profile->GetMaxBatch(sess.latency_sla());
  inst_info->fwd_latency_us = profile->GetForwardLatency(inst_info->batch);
  inst_info->memory_usage = profile->GetMemoryUsage(inst_info->max_batch);

  CHECK_EQ(session_model_map_.count(model_session_id), 0) <<
      "Model session " << model_session_id << " already exists.";
  session_model_map_.emplace(model_session_id, inst_info);

  if (model_info["share_prefix"]) {
    for (int i = 1; i < model_info["share_prefix"].size(); ++i) {
      auto share_model_info = model_info["share_prefix"][i];
      ModelSession share_sess;
      share_sess.set_framework(share_model_info["framework"].as<std::string>());
      share_sess.set_model_name(share_model_info["model_name"].
                                as<std::string>());
      share_sess.set_version(share_model_info["version"].as<uint32_t>());
      share_sess.set_latency_sla(share_model_info["latency_sla"].
                                 as<uint32_t>());
      if (model_info["image_height"]) {
        share_sess.set_image_height(model_info["image_height"].as<uint32_t>());
        share_sess.set_image_width(model_info["image_width"].as<uint32_t>());
      }
      std::string share_sess_id = ModelSessionToString(share_sess);
      CHECK_EQ(session_model_map_.count(share_sess_id), 0) <<
          "Model session " << share_sess_id << " already exists.";
      inst_info->model_sessions.push_back(share_sess);
      session_model_map_.emplace(share_sess_id, inst_info);
    }
  }
  if (model_info["backup"]) {
    inst_info->backup = model_info["backup"].as<bool>();
  }
  if (model_info["weight"]) {
    inst_info->weight = model_info["weight"].as<double>();
  }

  if (inst_info->backup) {
    backup_models_.push_back(inst_info);
  } else {
    models_.push_back(inst_info);
    // update execution and batch cycles and throughput
    exec_cycle_us_ += inst_info->fwd_latency_us;
    duty_cycle_us_ += inst_info->fwd_latency_us;
    for (auto other_info : models_) {
      other_info->throughput = other_info->batch * 1e6 / duty_cycle_us_;
    }
  }
  dirty_model_table_ = true;

  LOG(INFO) << "Backend " << node_id_ << " loads " << model_session_id <<
      ", batch " << inst_info->batch << ", exec cycle " << exec_cycle_us_ <<
      " us, duty cycle: " << duty_cycle_us_ << " us, backup: " <<
      inst_info->backup;
}

void BackendDelegate::LoadPrefixModel(const ModelSession& model_session,
                                      const ModelSession& shared_session) {
  std::string model_session_id = ModelSessionToString(model_session);
  std::string shared_session_id = ModelSessionToString(shared_session);
  CHECK_EQ(session_model_map_.count(model_session_id), 0) <<
      "Model session " << model_session_id << " already exists.";
  CHECK_GT(session_model_map_.count(shared_session_id), 0) <<
      "Model session " << shared_session_id << " doesn't exist.";
  auto inst_info = session_model_map_.at(shared_session_id);
  inst_info->model_sessions.push_back(model_session);
  session_model_map_.emplace(model_session_id, inst_info);
  dirty_model_table_ = true;
}

void BackendDelegate::UnloadModel(const std::string& model_sess_id) {
  if (workload_id_ >= 0) {
    return;
  }
  LOG(INFO) << "Backend " << node_id_ << " unload model: " << model_sess_id;
  auto inst_info = session_model_map_.at(model_sess_id);
  session_model_map_.erase(model_sess_id);
  // Remove model session from instance info
  for (auto iter = inst_info->model_sessions.begin();
       iter != inst_info->model_sessions.end(); ++iter) {
    if (ModelSessionToString(*iter) == model_sess_id) {
      inst_info->model_sessions.erase(iter);
      break;
    }
  }
  if (inst_info->model_sessions.empty()) {
    for (auto iter = models_.begin(); iter != models_.end();
         ++iter) {
      if (*iter == inst_info) {
        models_.erase(iter);
        break;
      }
    }
    UpdateCycle();
  }
  dirty_model_table_ = true;
}

void BackendDelegate::AddBackupForModel(const std::string& model_sess_id,
                                        const BackendInfo& info) {
  auto inst_info = session_model_map_.at(model_sess_id);
  if (inst_info->backup_backends.count(info.node_id()) > 0) {
    return;
  }
  LOG(INFO) << "Backend " << node_id_ << " add backup server " <<
      info.node_id() << " for " << model_sess_id;
  inst_info->backup_backends.emplace(info.node_id(), info);
  dirty_model_table_ = true;
}

void BackendDelegate::RemoveBackupForModel(const std::string& model_sess_id,
                                           uint32_t backend_id) {
  auto inst_info = session_model_map_.at(model_sess_id);
  if (inst_info->backup_backends.erase(backend_id) > 0) {
    dirty_model_table_ = true;
  }
}

double BackendDelegate::UpdateModelThroughput(const std::string& model_sess_id,
                                              double workload) {
  auto inst_info = session_model_map_.at(model_sess_id);
  double prev_throughput = inst_info->throughput;
  ComputeBatchSize(inst_info.get(), workload);
  if (std::abs(prev_throughput - inst_info->throughput) > 1e-3) {
    UpdateCycle();
    LOG(INFO) << "Backend " << node_id_ << " updates " << model_sess_id <<
        ", batch " << inst_info->batch << ", max batch " <<
        inst_info->max_batch << ", new throughput " << inst_info->throughput;
  }
  return inst_info->throughput;
}

void BackendDelegate::SpillOutWorkload(
    std::vector<std::pair<SessionGroup, double> >* spillout) {
  if (!overload_ || workload_id_ >= 0) {
    return;
  }
  LOG(INFO) << "Backend " << node_id_ << " is overloaded (" << Occupancy() <<
      ")";
  std::vector<std::tuple<SessionGroup, double, double> > workloads;
  for (auto inst_info : models_) {
    workloads.emplace_back(inst_info->model_sessions, inst_info->fwd_latency_us,
                           inst_info->workload);
  }
  models_.clear();
  // Sort workload based on exec latency
  std::sort(workloads.begin(), workloads.end(),
            [](std::tuple<SessionGroup, double, double> a,
               std::tuple<SessionGroup, double, double> b) {
              return std::get<1>(a) > std::get<1>(b);
            });
  // Recompute exec cycle and duty cycle
  models_.clear();
  session_model_map_.clear();
  exec_cycle_us_ = 0.;
  duty_cycle_us_ = 0.;
  for (auto iter : workloads) {
    SessionGroup sessions = std::get<0>(iter);
    double workload = std::get<2>(iter);
    InstanceInfo inst_info;
    double occupancy;
    bool ret = PrepareLoadModel(sessions[0], workload, &inst_info, &occupancy);
    if (!ret) {
      spillout->emplace_back(sessions, workload);
    } else {
      LoadModel(inst_info);
      for (int i = 1; i < sessions.size(); ++i) {
        LoadPrefixModel(sessions[i], sessions[0]);
      }
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
  request.set_duty_cycle_us(duty_cycle_us_);
  for (auto inst_info : models_) {
    auto cfg = request.add_model_instance_config();
    for (auto& model_sess : inst_info->model_sessions) {
      cfg->add_model_session()->CopyFrom(model_sess);
    }
    CHECK_NE(inst_info->batch, 0);
    cfg->set_batch(inst_info->batch);
    cfg->set_max_batch(inst_info->max_batch);
    cfg->set_memory_usage(inst_info->memory_usage);
    cfg->set_backup(inst_info->backup);
    for (auto iter : inst_info->backup_backends) {
      cfg->add_backup_backend()->CopyFrom(iter.second);
    }
  }
  for (auto inst_info : backup_models_) {
    auto cfg = request.add_model_instance_config();
    for (auto& model_sess : inst_info->model_sessions) {
      cfg->add_model_session()->CopyFrom(model_sess);
    }
    cfg->set_batch(inst_info->batch);
    cfg->set_max_batch(inst_info->max_batch);
    cfg->set_memory_usage(inst_info->memory_usage);
    cfg->set_backup(inst_info->backup);
  }
  // LOG(INFO) << "Backend " << node_id_ << " update model table: " <<
  //     request.DebugString();
  
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

std::vector<std::string> BackendDelegate::GetModelSessions() const {
  std::vector<std::string> sessions;
  for (auto const& iter : session_model_map_) {
    if (!iter.second->backup) {
      sessions.push_back(iter.first);
    }
  }
  return sessions;
}

std::vector<std::string> BackendDelegate::GetBackupModelSessions() const {
  std::vector<std::string> sessions;
  for (auto const& iter : session_model_map_) {
    if (iter.second->backup) {
      sessions.push_back(iter.first);
    }
  }
  return sessions;
}

const InstanceInfo* BackendDelegate::GetInstanceInfo(
    const std::string& model_sess_id) const {
  auto iter = session_model_map_.find(model_sess_id);
  if (iter == session_model_map_.end()) {
    return nullptr;
  }
  return iter->second.get();
}

double BackendDelegate::GetModelThroughput(const std::string& model_sess_id)
    const {
  return session_model_map_.at(model_sess_id)->throughput;
}

double BackendDelegate::GetModelGPUShare(const std::string& model_sess_id)
    const {
  auto iter = session_model_map_.find(model_sess_id);
  if (iter == session_model_map_.end()) {
    return 0.;
  }
  auto model_inst = iter->second;
  double model_exec_cycle = model_inst->fwd_latency_us;
  return model_exec_cycle / exec_cycle_us_;
}

double BackendDelegate::GetModelWeight(const std::string& model_sess_id)
    const {
  return session_model_map_.at(model_sess_id)->GetWeight();
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
                                       double workload) const {
  // 1. Compute the max batch and throughput to saturate an empty GPU
  uint32_t batch, max_batch;
  double max_throughput;
  max_batch = inst_info->profile->GetMaxBatch(
      inst_info->model_sessions[0].latency_sla());
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
    inst_info->workload = max_throughput;
    CHECK_GE(inst_info->workload, 1e-3);
    inst_info->memory_usage = inst_info->profile->GetMemoryUsage(max_batch);
    return;
  }

  // 2. Compute the max batch for residue load
  double latency_sla_us = inst_info->model_sessions[0].latency_sla() * 1000;
  double preprocess = inst_info->profile->GetPreprocessLatency();
  double postprocess = inst_info->profile->GetPostprocessLatency();
  batch = 1;
  for (; batch <= max_batch; ++batch) {
    double fwd_lat = inst_info->profile->GetForwardLatency(batch);
    // because batch = ceil(workload * duty_cycle),
    // duty_cycle >= (batch - 1) / workload
    double min_duty_cycle = (batch - 1) * 1e6 / workload;
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
  CHECK_GE(inst_info->throughput, workload - 1e-3) << "Throughput is less " <<
      "than workload";
  inst_info->workload = workload;
  CHECK_GE(inst_info->workload, 1e-3);
  inst_info->memory_usage = inst_info->profile->GetMemoryUsage(max_batch);
}

void BackendDelegate::UpdateCycle() {
  auto res = calc_cycle(models_);
  exec_cycle_us_ = res.exec_cycle_us;
  duty_cycle_us_ = res.duty_cycle_us;
  overload_ = res.overload;
  for (size_t i = 0; i < res.changes.size(); ++i) {
    models_[i]->batch = res.changes[i].batch;
    models_[i]->fwd_latency_us = res.changes[i].fwd_latency_us;
  }

  dirty_model_table_ = true;
}

} // namespace scheduler
} // namespace nexus
