#include <boost/filesystem.hpp>
#include <glog/logging.h>
#include <unordered_set>

#include "nexus/common/config.h"
#include "nexus/common/model_db.h"
#include "nexus/scheduler/scheduler.h"

namespace fs = boost::filesystem;

namespace nexus {
namespace scheduler {

INSTANTIATE_RPC_CALL(AsyncService, Register, RegisterRequest, RegisterReply);
INSTANTIATE_RPC_CALL(AsyncService, Unregister, UnregisterRequest, RpcReply);
INSTANTIATE_RPC_CALL(AsyncService, LoadModel, LoadModelRequest, LoadModelReply);
INSTANTIATE_RPC_CALL(AsyncService, UnloadModel, ModelSession, RpcReply);
INSTANTIATE_RPC_CALL(AsyncService, UpdateBackendStats, BackendStatsProto,
                     RpcReply);
INSTANTIATE_RPC_CALL(AsyncService, KeepAlive, KeepAliveRequest, RpcReply);

Scheduler::Scheduler(std::string port, size_t nthreads,
                     std::string model_db_root, int beacon_interval,
                     int epoch_interval) :
    AsyncRpcServiceBase(port, nthreads),
    beacon_interval_sec_(beacon_interval),
    epoch_interval_sec_(epoch_interval) {
  ModelDatabase::Singleton().Init(model_db_root);
}

void Scheduler::LoadWorkloadFile(const std::string& workload_file) {
  LOG(INFO) << "Load workload file from " << workload_file;
  YAML::Node config = YAML::LoadFile(workload_file);
  if (!config["backends"]) {
    return;
  }
  // Load static workload configuration
  for (uint i = 0; i < config["backends"].size(); ++i) {
    const YAML::Node& backend_info = config["backends"][i];
    std::vector<YAML::Node> workload;
    LOG(INFO) << "Workload " << i << ":";
    for (uint j = 0; j < backend_info["workloads"].size(); ++j) {
      const YAML::Node& model_info = backend_info["workloads"][j];
      if (!model_info["framework"]) {
        LOG(FATAL) << "Missing framework in the workload config";
      }
      if (!model_info["model_name"]) {
        LOG(FATAL) << "Missing model_name in the workload config";
      }
      if (!model_info["version"]) {
        LOG(FATAL) << "Missing version in the workload config";
      }
      if (!model_info["batch"]) {
        LOG(FATAL) << "Missing batch in the workload config";
      }
      if (!model_info["latency_sla"]) {
        LOG(FATAL) << "Missing latency_sla in the workload config";
      }
      LOG(INFO) << "- " << model_info["framework"] << ":" <<
          model_info["model_name"] << ":" << model_info["version"] <<
          ", batch " << model_info["max_batch"];
      workload.push_back(model_info);
    }
    workloads_.push_back(workload);
  }
}

void Scheduler::Run() {
  // Start RPC service first
  Start();
  // main scheduler login
  uint64_t elapse_sec = 0;
  uint64_t last_beacon = 0;
  uint64_t last_epoch = 0;
  while (running_) {
    auto now = std::chrono::system_clock::now();
    if (elapse_sec > 0 && elapse_sec % beacon_interval_sec_ == 0) {
      last_beacon = elapse_sec;
      BeaconCheck();
    }
    if (elapse_sec > 0 && elapse_sec % epoch_interval_sec_ == 0) {
      last_epoch = elapse_sec;
      EpochSchedule();
    }
    int next_sec = std::min(last_beacon + beacon_interval_sec_,
                            last_epoch + epoch_interval_sec_);
    std::this_thread::sleep_until(
        now + std::chrono::seconds(next_sec - elapse_sec));
    elapse_sec = next_sec;
  }
}

void Scheduler::Register(RpcCallBase* call, const RegisterRequest& request,
                         RegisterReply* reply) {
  std::vector<std::string> tokens;
  SplitString(call->PeerAddress(), ':', &tokens);
  std::string server_addr = tokens[1] + ':' + request.server_port();
  std::string rpc_addr = tokens[1] + ':' + request.rpc_port();
  LOG(INFO) << "Register server: " << request.DebugString();
  if (request.node_type() == BACKEND_NODE) {
    auto backend = std::make_shared<BackendRpcClient>(
        request.node_id(), server_addr, rpc_addr, request.gpu_device_name(),
        request.gpu_available_memory(), beacon_interval_sec_,
        epoch_interval_sec_);
    RegisterBackend(std::move(backend), reply);
  } else { // FRONTEND_NODE
    auto frontend = std::make_shared<FrontendRpcClient>(
        request.node_id(), server_addr, rpc_addr, beacon_interval_sec_);
    RegisterFrontend(std::move(frontend), reply);
  }
}

void Scheduler::Unregister(RpcCallBase* call, const UnregisterRequest& request,
                           RpcReply* reply) {
  LOG(INFO) << "Unregister " << NodeType_Name(request.node_type()) << " " <<
      request.node_id();
  if (request.node_type() == BACKEND_NODE) {
    UnregisterBackend(request.node_id());
  } else { // FRONTEND_NODE
    UnregisterFrontend(request.node_id());
  }
  reply->set_status(CTRL_OK);
}

void Scheduler::LoadModel(RpcCallBase* call, const LoadModelRequest& request,
                          LoadModelReply* reply) {
  // TODO: support multi batching
  ModelSession model_sess(request.model_session());
  auto info = ModelDatabase::Singleton().GetModelInfo(
      ModelSessionToModelID(model_sess));
  if (info["resizable"] && info["resizable"].as<bool>()) {
    if (model_sess.image_height() == 0) {
      // Set default image size for resizable CNN
      model_sess.set_image_height(info["image_height"].as<uint32_t>());
      model_sess.set_image_width(info["image_width"].as<uint32_t>());
    }
  }
  std::string model_sess_id = ModelSessionToString(model_sess);
  float workload = request.estimate_workload();

  std::lock_guard<std::mutex> lock(mutex_);
  std::vector<std::pair<BackendRpcClientPtr, ModelInstanceConfig> >
      assign_backends;
  
  // TODO: check if model_sess_id already exists
  uint32_t frontend_id = request.node_id();
  auto frontend = GetFrontend(frontend_id);
  if (frontend == nullptr) {
    reply->set_status(CTRL_SERVER_NOT_REGISTERED);
    return;
  }
  // Find backends to serve the workload
  for (auto iter : backends_) {
    auto backend = iter.second;
    if (!backend->IsAlive() || !backend->IsIdle()) {
      continue;
    }
    ModelInstanceConfig config;
    float occupancy;
    bool ret = backend->PrepareLoadModel(model_sess, workload, &config,
                                         &occupancy);
    if (!ret) {
      continue;
    }
    assign_backends.emplace_back(backend, config);
    if (workload == 0) {
      break;
    }
    workload -= config.throughput();
    if (workload <= 0) {
      break;
    }
  }
  // Couldn't find enough backends to satisfy the workload
  if (workload > 0 || assign_backends.size() == 0) {
    reply->set_status(CTRL_NOT_ENOUGH_BACKENDS);
    return;
  }
  // Load the model and
  auto model_route = reply->mutable_model_route();
  ModelInfo model_info;
  model_route->mutable_model_session()->CopyFrom(model_sess);
  for (auto item : assign_backends) {
    auto backend = item.first;
    const auto& config = item.second;
    backend->LoadModel(item.second);
    auto backend_rate = model_route->add_backend_rate();
    backend->GetInfo(backend_rate->mutable_info());
    backend_rate->set_throughput(config.throughput());
    model_info.backend_throughputs.emplace(backend->node_id(),
                                           config.throughput());
  }
  model_table_.emplace(model_sess_id, model_info);
  model_subscribers_.emplace(
      model_sess_id, ServerList{request.node_id()});
  frontend->SubscribeModel(model_sess_id);
  reply->set_status(CTRL_OK);

  for (auto item : assign_backends) {
    auto backend = item.first;
    CtrlStatus ret = backend->UpdateModelTable();
    if (ret != CTRL_OK) {
      // TODO
    }
  }
  reply->set_status(CTRL_OK);
}

void Scheduler::UpdateBackendStats(RpcCallBase*,
                                   const BackendStatsProto& request,
                                   RpcReply* reply) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto backend = GetBackend(request.node_id());
  if (backend == nullptr) {
    reply->set_status(CTRL_SERVER_NOT_REGISTERED);
    return;
  }
  backend->UpdateStats(request);
  reply->set_status(CTRL_OK);
}

void Scheduler::KeepAlive(RpcCallBase*, const KeepAliveRequest& request,
                          RpcReply* reply) {
  LOG(INFO) << "KeepAlive: " << request.DebugString();
  auto frontend = GetFrontend(request.node_id());
  if (frontend == nullptr) {
    reply->set_status(CTRL_SERVER_NOT_REGISTERED);
    return;
  }
  frontend->Tick();
  reply->set_status(CTRL_OK);
}

void Scheduler::HandleRpcs() {
  using namespace std::placeholders;
  new Register_Call(&service_, cq_.get(),
                    std::bind(&Scheduler::Register, this, _1, _2, _3));
  new Unregister_Call(&service_, cq_.get(),
                      std::bind(&Scheduler::Unregister, this, _1, _2, _3));
  new LoadModel_Call(&service_, cq_.get(),
                     std::bind(&Scheduler::LoadModel, this, _1, _2, _3));
  // new UnloadModel_Call(&service_, cq_.get(),
  //                      std::bind(&Scheduler::UnloadModel, this, _1, _2, _3));
  new UpdateBackendStats_Call(
      &service_, cq_.get(),
      std::bind(&Scheduler::UpdateBackendStats, this, _1, _2, _3));
  new KeepAlive_Call(&service_, cq_.get(),
                     std::bind(&Scheduler::KeepAlive, this, _1, _2, _3));
  void* tag;
  bool ok;
  while (running_) {
    cq_->Next(&tag, &ok);
    if (ok) {
      static_cast<RpcCallBase*>(tag)->Proceed();
    }
  }
}

void Scheduler::RegisterFrontend(FrontendRpcClientPtr frontend,
                                 RegisterReply* reply) {
  // lock protected
  std::lock_guard<std::mutex> lock(mutex_);
  if (frontends_.find(frontend->node_id()) != frontends_.end()) {
    reply->set_status(CTRL_FRONTEND_NODE_ID_CONFLICT);
    return;
  }
  // add the frontend client in the frontend map
  frontends_[frontend->node_id()] = frontend;
  reply->set_status(CTRL_OK);
  reply->set_beacon_interval_sec(BEACON_INTERVAL_SEC);
}

void Scheduler::RegisterBackend(BackendRpcClientPtr backend,
                                RegisterReply* reply) {
  // lock protected
  std::lock_guard<std::mutex> lock(mutex_);
  if (backends_.find(backend->node_id()) != backends_.end()) {
    reply->set_status(CTRL_BACKEND_NODE_ID_CONFLICT);
    return;
  }
  // Add the backend client in the backend map
  backends_[backend->node_id()] = backend;
  reply->set_status(CTRL_OK);
  reply->set_beacon_interval_sec(BEACON_INTERVAL_SEC);
  // Update workload to new backend
  AddBackend(backend);
}

void Scheduler::UnregisterFrontend(uint32_t node_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto frontend = GetFrontend(node_id);
  if (frontend == nullptr) {
    LOG(ERROR) << "Cannot find frontend " << node_id;
    return;
  }
  frontends_.erase(frontend->node_id());
  LOG(INFO) << "Remove frontend " << node_id;
  RemoveFrontend(frontend);
}

void Scheduler::UnregisterBackend(uint32_t node_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto backend = GetBackend(node_id);
  if (backend == nullptr) {
    LOG(ERROR) << "Cannot find backend " << node_id;
    return;
  }
  backends_.erase(backend->node_id());
  LOG(INFO) << "Remove backend " << node_id;
  RemoveBackend(backend);
}

void Scheduler::AddBackend(BackendRpcClientPtr backend) {
  // 1. Check if there is any static configured workload to assign
  int assign_load_id = -1;
  for (uint id = 0; id < workloads_.size(); ++id) {
    if (assigned_workloads_.find(id) == assigned_workloads_.end()) {
      assign_load_id = id;
      assigned_workloads_[id] = backend->node_id();
      break;
    }
  }
  if (assign_load_id >= 0) {
    LOG(INFO) << "Assign workload " << assign_load_id << " to backend " <<
        backend->node_id();
    auto workload = workloads_[assign_load_id];
    for (auto model_info : workload) {
      backend->LoadModel(model_info);
    }
    backend->set_workload_id(assign_load_id);
    backend->UpdateModelTable();
  } else {
    // 2. Check if there is any model session that needs extra backend
    // TODO
  }

  // 3. Update model table
  std::vector<std::string> model_sessions;
  backend->GetModelSessions(&model_sessions);
  for (auto& model_sess_id : model_sessions) {
    model_table_.at(model_sess_id).backend_throughputs.emplace(
        backend->node_id(), backend->GetModelThroughput(model_sess_id));
  }
}

void Scheduler::RemoveBackend(BackendRpcClientPtr backend) {
  if (backend->IsIdle()) {
    return;
  }
  // 1. Remove backend from ModelInfo
  std::vector<std::string> old_model_sessions;
  backend->GetModelSessions(&old_model_sessions);
  for (auto& model_sess_id : old_model_sessions) {
    model_table_.at(model_sess_id).backend_throughputs.erase(
        backend->node_id());
  }
  
  // 2. Try to re-assign backend's workload to another idle one
  BackendRpcClientPtr assigned;
  for (auto iter : backends_) {
    if (iter.second->IsIdle()) {
      if (iter.second->Assign(*backend)) {
        assigned = iter.second;
        break;
      }
    }
  }
  
  if (assigned != nullptr) {
    std::vector<std::string> new_model_sessions;
    assigned->GetModelSessions(&new_model_sessions);
    for (auto& model_sess_id : new_model_sessions) {
      model_table_.at(model_sess_id).backend_throughputs.emplace(
          assigned->node_id(), assigned->GetModelThroughput(model_sess_id));
    }
    return;
  }
  
  // Failed to find an idle backend to assign the workload
  if (backend->workload_id() >= 0) {
    assigned_workloads_.erase(backend->workload_id());
    LOG(INFO) << "Remove workload " << backend->workload_id();
    return;
  }
  
  // TODO: 3. If it's not static configured workload, allocate model instances
  // to other backends
}

void Scheduler::RemoveFrontend(FrontendRpcClientPtr frontend) {
  // Update subscribed model sessions
  std::vector<std::string> remove_sessions;
  std::unordered_set<BackendRpcClientPtr> update_backends;
  for (auto model_sess_id : frontend->subscribe_models()) {
    model_subscribers_.at(model_sess_id).erase(frontend->node_id());
    if (model_subscribers_.at(model_sess_id).empty()) {
      remove_sessions.push_back(model_sess_id);
      auto& model_info = model_table_.at(model_sess_id);
      for (auto iter : model_info.backend_throughputs) {
        auto backend = GetBackend(iter.first);
        backend->UnloadModel(model_sess_id);
        update_backends.insert(backend);
      }
    }
  }
  // Remove model sessions
  for (auto& model_sess_id : remove_sessions) {
    LOG(INFO) << "Remove model session: " << model_sess_id;
    model_subscribers_.erase(model_sess_id);
    model_table_.erase(model_sess_id);
  }
  for (auto backend : update_backends) {
    backend->UpdateModelTable();
  }
}

BackendRpcClientPtr Scheduler::GetBackend(uint32_t node_id) {
  auto iter = backends_.find(node_id);
  if (iter == backends_.end()) {
    LOG(ERROR) << "Cannot find backend " << node_id;
    return nullptr;
  }
  return iter->second;
}

FrontendRpcClientPtr Scheduler::GetFrontend(uint32_t node_id) {
  auto iter = frontends_.find(node_id);
  if (iter == frontends_.end()) {
    LOG(ERROR) << "Cannot find frontend " << node_id;
    return nullptr;
  }
  return iter->second;
}

void Scheduler::BeaconCheck() {
  std::lock_guard<std::mutex> lock(mutex_);
  // 1. Remove dead frontends
  std::vector<FrontendRpcClientPtr> dead_frontends;
  for (auto iter : frontends_) {
    auto frontend = iter.second;
    if (frontend->IsAlive()) {
      continue;
    }
    dead_frontends.push_back(frontend);
  }
  for (auto frontend : dead_frontends) {
    frontends_.erase(frontend->node_id());
    std::time_t last_time = frontend->LastAliveTime();
    LOG(INFO) << "Remove frontend " << frontend->node_id() <<
        ", last alive time: " << std::ctime(&last_time);
    RemoveFrontend(frontend);
  }
  
  // 2. Aggregate model session rps
  for (auto& model_iter : model_table_) {
    const auto& model_sess_id = model_iter.first;
    auto& model_info = model_iter.second;
    double rps = 0.;
    for (auto backend_iter : model_info.backend_throughputs) {
      rps += backends_.at(backend_iter.first)->GetModelRps(model_sess_id);
    }
    model_info.rps_history.push_back(rps);
    LOG(INFO) << "Model " << model_sess_id << " rps: " << rps <<
        " req/s (avg over " << epoch_interval_sec_ << " seconds)";
  }
  
  // 3. Remove dead backend
  std::vector<BackendRpcClientPtr> dead_backends;
  for (auto iter : backends_) {
    auto backend = iter.second;
    if (backend->IsAlive()) {
      continue;
    }
    dead_backends.push_back(backend);
  }
  for (auto backend : dead_backends) {
    std::time_t last_time = backend->LastAliveTime();
    LOG(INFO) << "Remove backend " << backend->node_id() <<
        ", last alive time: " << std::ctime(&last_time);
    backends_.erase(backend->node_id());
  }
  // Reassign workload of dead backends
  for (auto backend : dead_backends) {
    RemoveBackend(backend);
  }
}

void Scheduler::EpochSchedule() {
}

} // namespace scheduler
} // namespace nexus
