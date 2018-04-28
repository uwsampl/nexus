#include <boost/filesystem.hpp>
#include <glog/logging.h>
#include <unordered_set>

#include "nexus/common/config.h"
#include "nexus/common/model_db.h"
#include "nexus/scheduler/scheduler.h"

namespace fs = boost::filesystem;

namespace nexus {
namespace scheduler {

DEFINE_bool(epoch_schedule, true, "Enable epoch scheduling");

INSTANTIATE_RPC_CALL(AsyncService, Register, RegisterRequest, RegisterReply);
INSTANTIATE_RPC_CALL(AsyncService, Unregister, UnregisterRequest, RpcReply);
INSTANTIATE_RPC_CALL(AsyncService, LoadModel, LoadModelRequest, LoadModelReply);
INSTANTIATE_RPC_CALL(AsyncService, UpdateBackendStats, BackendStatsProto,
                     RpcReply);
INSTANTIATE_RPC_CALL(AsyncService, KeepAlive, KeepAliveRequest, RpcReply);

Scheduler::Scheduler(std::string port, size_t nthreads,
                     std::string model_db_root, uint32_t beacon_interval,
                     uint32_t epoch_interval) :
    AsyncRpcServiceBase(port, nthreads),
    beacon_interval_sec_(beacon_interval),
    epoch_interval_sec_(epoch_interval) {
  min_history_len_ = (epoch_interval + beacon_interval - 1) / beacon_interval;
  history_len_ = min_history_len_ * 2;
  ModelDatabase::Singleton().Init(model_db_root);
  enable_epoch_schedule_ = FLAGS_epoch_schedule;
  if (!enable_epoch_schedule_) {
    LOG(INFO) << "Epoch scheduling is off";
  }
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
    static_workloads_.push_back(workload);
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
      if (enable_epoch_schedule_) {
        EpochSchedule();
      }
    }
    int next_sec = std::min(last_beacon + beacon_interval_sec_,
                            last_epoch + epoch_interval_sec_);
    std::this_thread::sleep_until(
        now + std::chrono::seconds(next_sec - elapse_sec));
    elapse_sec = next_sec;
  }
}

void Scheduler::Register(const grpc::ServerContext& ctx,
                         const RegisterRequest& request, RegisterReply* reply) {
  std::vector<std::string> tokens;
  SplitString(ctx.peer(), ':', &tokens);
  std::string server_addr = tokens[1] + ':' + request.server_port();
  std::string rpc_addr = tokens[1] + ':' + request.rpc_port();
  LOG(INFO) << "Register server: " << request.DebugString();
  if (request.node_type() == BACKEND_NODE) {
    auto backend = std::make_shared<BackendDelegate>(
        request.node_id(), server_addr, rpc_addr, request.gpu_device_name(),
        request.gpu_available_memory(), beacon_interval_sec_,
        epoch_interval_sec_);
    RegisterBackend(std::move(backend), reply);
  } else { // FRONTEND_NODE
    auto frontend = std::make_shared<FrontendDelegate>(
        request.node_id(), server_addr, rpc_addr, beacon_interval_sec_);
    RegisterFrontend(std::move(frontend), reply);
  }
}

void Scheduler::Unregister(const grpc::ServerContext& ctx,
                           const UnregisterRequest& request, RpcReply* reply) {
  LOG(INFO) << "Unregister " << NodeType_Name(request.node_type()) << " " <<
      request.node_id();
  if (request.node_type() == BACKEND_NODE) {
    UnregisterBackend(request.node_id());
  } else { // FRONTEND_NODE
    UnregisterFrontend(request.node_id());
  }
  reply->set_status(CTRL_OK);
}

void Scheduler::LoadModel(const grpc::ServerContext& ctx,
                          const LoadModelRequest& request,
                          LoadModelReply* reply) {
  ModelSession model_sess(request.model_session());
  auto info = ModelDatabase::Singleton().GetModelInfo(
      ModelSessionToModelID(model_sess));
  if (info == nullptr) {
    reply->set_status(MODEL_NOT_FOUND);
    return;
  }
  if ((*info)["resizable"] && (*info)["resizable"].as<bool>()) {
    if (model_sess.image_height() == 0) {
      // Set default image size for resizable CNN
      model_sess.set_image_height((*info)["image_height"].as<uint32_t>());
      model_sess.set_image_width((*info)["image_width"].as<uint32_t>());
    }
  }
  std::string model_sess_id = ModelSessionToString(model_sess);
  float workload = request.estimate_workload();

  std::lock_guard<std::mutex> lock(mutex_);
  auto frontend = GetFrontend(request.node_id());
  if (frontend == nullptr) {
    reply->set_status(CTRL_SERVER_NOT_REGISTERED);
    return;
  }
  if (model_table_.find(model_sess_id) != model_table_.end()) {
    // TODO: For now, if model session that is already loaded, don't allocate
    // new backends, just rely on epoch scheduling
    reply->set_status(CTRL_OK);
    GetModelRoute(model_sess_id, reply->mutable_model_route());
    frontend->SubscribeModel(model_sess_id);
    model_table_.at(model_sess_id).subscribers.insert(request.node_id());
    return;
  }

  // Find best-fit backends to serve the workload
  std::vector<std::pair<BackendDelegatePtr, InstanceInfo> > assign_backends;
  std::unordered_set<uint32_t> used;
  if (workload == 0) {
    BackendDelegatePtr backend;
    InstanceInfo inst_info;
    FindBestBackend(model_sess, workload, used, &backend, &inst_info);
    if (backend == nullptr) {
      reply->set_status(NOT_ENOUGH_BACKENDS);
      return;
    }
    assign_backends.emplace_back(backend, inst_info);
  } else {
    while (workload > 0) {
      BackendDelegatePtr backend;
      InstanceInfo inst_info;
      FindBestBackend(model_sess, workload, used, &backend, &inst_info);
      if (backend == nullptr) {
        reply->set_status(NOT_ENOUGH_BACKENDS);
        return;
      }
      assign_backends.emplace_back(backend, inst_info);
      used.insert(backend->node_id());
      workload -= inst_info.throughput;
    }
  }

  // Load models
  ModelInfo model_info;
  model_info.subscribers.insert(request.node_id());
  for (auto iter : assign_backends) {
    auto backend = iter.first;
    auto const& inst_info = iter.second;
    backend->LoadModel(inst_info);
    backend->UpdateModelTableRpc();
    model_info.backend_throughputs.emplace(backend->node_id(),
                                           inst_info.throughput);
  }
  model_table_.emplace(model_sess_id, model_info);
  frontend->SubscribeModel(model_sess_id);
  
  // Fill route table in the reply
  reply->set_status(CTRL_OK);
  GetModelRoute(model_sess_id, reply->mutable_model_route());
}

void Scheduler::UpdateBackendStats(const grpc::ServerContext& ctx,
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

void Scheduler::KeepAlive(const grpc::ServerContext& ctx,
                          const KeepAliveRequest& request, RpcReply* reply) {
  std::lock_guard<std::mutex> lock(mutex_);
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

void Scheduler::RegisterFrontend(FrontendDelegatePtr frontend,
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

void Scheduler::RegisterBackend(BackendDelegatePtr backend,
                                RegisterReply* reply) {
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

void Scheduler::AddBackend(BackendDelegatePtr backend) {
  // 1. Check if there is any static configured workload to assign
  int assign_load_id = -1;
  for (uint id = 0; id < static_workloads_.size(); ++id) {
    if (assigned_static_workloads_.find(id) ==
        assigned_static_workloads_.end()) {
      assign_load_id = id;
      assigned_static_workloads_[id] = backend->node_id();
      break;
    }
  }
  if (assign_load_id >= 0) {
    LOG(INFO) << "Assign workload " << assign_load_id << " to backend " <<
        backend->node_id();
    auto workload = static_workloads_[assign_load_id];
    for (auto model_info : workload) {
      backend->LoadModel(model_info);
    }
    backend->set_workload_id(assign_load_id);
  } else {
    // 2. Check if there are unassigned workloads
    std::vector<std::pair<std::string, float> > unassigned_workloads;
    for (auto const& iter : model_table_) {
      auto const& model_info = iter.second;
      if (model_info.unassigned_workload > 0) {
        unassigned_workloads.emplace_back(iter.first,
                                          model_info.unassigned_workload);
      }
    }
    std::sort(unassigned_workloads.begin(), unassigned_workloads.end(),
              [](std::pair<std::string, float> a,
                 std::pair<std::string, float> b) {
                return a.second > b.second;
              });
    for (auto iter : unassigned_workloads) {
      std::string model_sess_id = iter.first;
      float request_rate = iter.second;
      ModelSession model_sess;
      ParseModelSession(model_sess_id, &model_sess);
      InstanceInfo inst_info;
      float occupancy;
      if (backend->PrepareLoadModel(model_sess, request_rate, &inst_info,
                                    &occupancy)) {
        backend->LoadModel(inst_info);
        request_rate -= inst_info.throughput;
        model_table_.at(model_sess_id).unassigned_workload = std::max(
            0.f, request_rate);
      }
    }
  }

  // 3. Update backend model table
  backend->UpdateModelTableRpc();
  
  // 4. Update model info and route
  std::vector<std::string> model_sessions;
  backend->AllModelSessions(&model_sessions);
  for (auto& model_sess_id : model_sessions) {
    if (model_table_.find(model_sess_id) == model_table_.end()) {
      model_table_.emplace(model_sess_id, ModelInfo());
    }
    auto& model_info = model_table_.at(model_sess_id);
    model_info.backend_throughputs.emplace(
        backend->node_id(), backend->GetModelThroughput(model_sess_id));
  }
  std::unordered_set<std::string> changed_routes(model_sessions.begin(),
                                                 model_sessions.end());
  UpdateModelRoutes(changed_routes);
}

void Scheduler::RemoveBackend(BackendDelegatePtr backend) {
  if (backend->IsIdle()) {
    return;
  }
  // 1. Remove backend from ModelInfo
  std::vector<std::string> model_sessions;
  backend->AllModelSessions(&model_sessions);
  for (auto& model_sess_id : model_sessions) {
    model_table_.at(model_sess_id).backend_throughputs.erase(
        backend->node_id());
  }
  std::unordered_set<std::string> changed_routes(model_sessions.begin(),
                                                 model_sessions.end());
  
  // 2. Try to re-assign backend's workload to another idle one
  BackendDelegatePtr assigned;
  for (auto iter : backends_) {
    if (iter.second->IsIdle() && iter.second->Assign(*backend)) {
      assigned = iter.second;
      break;
    }
  }
  if (assigned != nullptr) {
    for (auto& model_sess_id : model_sessions) {
      model_table_.at(model_sess_id).backend_throughputs.emplace(
          assigned->node_id(), assigned->GetModelThroughput(model_sess_id));
    }
    assigned->UpdateModelTableRpc();
  } else {
    // Failed to find an idle backend to assign the workload
    if (backend->workload_id() >= 0) {
      assigned_static_workloads_.erase(backend->workload_id());
      LOG(INFO) << "Remove workload " << backend->workload_id();
    } else {
      // 3. If it's not static configured workload, try to allocate model
      // instances to other backends
      for (auto& model_sess_id : model_sessions) {
        float tp = backend->GetModelThroughput(model_sess_id);
        model_table_.at(model_sess_id).unassigned_workload += tp;
      }
      std::unordered_set<BackendDelegatePtr> changed_backends;
      AllocateUnassignedWorkloads(&changed_routes, &changed_backends);
      for (auto backend : changed_backends) {
        backend->UpdateModelTableRpc();
      }
    }
  }

  // 4. Update changed routes;
  UpdateModelRoutes(changed_routes);
}

void Scheduler::RemoveFrontend(FrontendDelegatePtr frontend) {
  // Update subscribed model sessions
  std::vector<std::string> remove_sessions;
  std::unordered_set<BackendDelegatePtr> update_backends;
  for (auto model_sess_id : frontend->subscribe_models()) {
    auto& model_info = model_table_.at(model_sess_id);
    model_info.subscribers.erase(frontend->node_id());
    if (model_info.subscribers.empty()) {
      remove_sessions.push_back(model_sess_id);
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
    model_table_.erase(model_sess_id);
  }
  for (auto backend : update_backends) {
    backend->UpdateModelTableRpc();
  }
}

BackendDelegatePtr Scheduler::GetBackend(uint32_t node_id) {
  auto iter = backends_.find(node_id);
  if (iter == backends_.end()) {
    LOG(ERROR) << "Cannot find backend " << node_id;
    return nullptr;
  }
  return iter->second;
}

FrontendDelegatePtr Scheduler::GetFrontend(uint32_t node_id) {
  auto iter = frontends_.find(node_id);
  if (iter == frontends_.end()) {
    LOG(ERROR) << "Cannot find frontend " << node_id;
    return nullptr;
  }
  return iter->second;
}

void Scheduler::GetModelRoute(const std::string& model_sess_id,
                              ModelRouteProto* route) {
  route->set_model_session_id(model_sess_id);
  for (auto iter : model_table_.at(model_sess_id).backend_throughputs) {
    auto backend_rate = route->add_backend_rate();
    backends_.at(iter.first)->GetInfo(backend_rate->mutable_info());
    backend_rate->set_throughput(iter.second);
  }
}

void Scheduler::FindBestBackend(
    const ModelSession& model_sess, float request_rate,
    const std::unordered_set<uint32_t>& skips,
    BackendDelegatePtr* best_backend, InstanceInfo* inst_info) {
  using ModelLoad = std::tuple<BackendDelegatePtr, InstanceInfo, float>;
  ModelLoad max_tp_load;
  ModelLoad max_occ_load;
  for (auto iter : backends_) {
    auto backend = iter.second;
    if (skips.find(backend->node_id()) != skips.end()) {
      continue;
    }
    if (!backend->IsAlive() || backend->workload_id() >= 0) {
      continue;
    }
    if (request_rate == 0 && !backend->IsIdle()) {
      continue;
    }
    InstanceInfo inst_info;
    float occupancy;
    bool ret = backend->PrepareLoadModel(model_sess, request_rate, &inst_info,
                                         &occupancy);
    if (!ret) {
      continue;
    }
    if (std::get<0>(max_tp_load) == nullptr ||
        inst_info.throughput > std::get<1>(max_tp_load).throughput) {
      max_tp_load = std::make_tuple(backend, inst_info, occupancy);
    }
    if (std::get<0>(max_occ_load) == nullptr ||
        occupancy > std::get<2>(max_occ_load)) {
      max_occ_load = std::make_tuple(backend, inst_info, occupancy);
    }
  }
  if (request_rate == 0) {
    // for request rate = 0, return backend that provides highest throughput
    *best_backend = std::get<0>(max_tp_load);
    *inst_info = std::get<1>(max_tp_load);
  } else if (std::get<1>(max_tp_load).throughput < request_rate) {
    // If no backend can achieve request rate, return backend that provides
    // highest throughput
    *best_backend = std::get<0>(max_tp_load);
    *inst_info = std::get<1>(max_tp_load);
  } else {
    // Otherwise, return backend that has highest occupancy
    *best_backend = std::get<0>(max_occ_load);
    *inst_info = std::get<1>(max_occ_load);
  }
}

void Scheduler::BeaconCheck() {
  std::lock_guard<std::mutex> lock(mutex_);
  // 1. Remove dead frontends
  std::vector<FrontendDelegatePtr> dead_frontends;
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
    VLOG(1) << model_sess_id << " rps: ";
    for (auto backend_iter : model_info.backend_throughputs) {
      auto backend = backends_.at(backend_iter.first);
      VLOG(1) << "- backend " << backend->node_id() << ": " <<
          backend->GetModelRps(model_sess_id);
      rps += backend->GetModelRps(model_sess_id);
    }
    if (model_info.rps_history.size() > 0 || rps > 0) {
      // Don't push 0 in the begining
      model_info.rps_history.push_back(rps);
    }
    if (model_info.rps_history.size() > history_len_) {
      model_info.rps_history.pop_front();
    }
    VLOG(1) << "Model " << model_sess_id << " rps: " << rps <<
        " req/s (avg over " << epoch_interval_sec_ << " seconds)";
  }
  
  // 3. Remove dead backend
  std::vector<BackendDelegatePtr> dead_backends;
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
  std::lock_guard<std::mutex> lock(mutex_);
  std::unordered_set<std::string> changed_routes;
  std::vector<BackendDelegatePtr> overload_backends;

  // 1. Adjust the GPU allocation based on the workload
  for (auto& iter : model_table_) {
    auto& model_sess_id = iter.first;
    auto& model_info = iter.second;
    double throughput = model_info.total_throughput();
    // Compute the workload mean and std
    uint32_t n = model_info.rps_history.size();
    if (n < min_history_len_) {
      continue;
    }
    double rps_mean = 0., rps_std = 0.;
    for (double rps : model_info.rps_history) {
      rps_mean += rps;
    }
    rps_mean /= n;
    for (double rps : model_info.rps_history) {
      rps_std += (rps - rps_mean) * (rps - rps_mean);
    }
    rps_std = sqrt(rps_std / (n - 1));
    double estimate_rps = std::max(
        model_info.rps_history[n - 1] + rps_std, 0.1);
    model_info.unassigned_workload = std::max(0., estimate_rps - throughput);
    VLOG(1) << model_sess_id << " estimate rps: " << estimate_rps <<
        " (last: " << model_info.rps_history[n - 1] << ", mean: " <<
        rps_mean << ", std: " << rps_std << "), throughput: " << throughput;

    if (estimate_rps < throughput * 0.97) {
      // Workload is smaller than throughput, can release some GPUs
      std::vector<std::pair<uint32_t, double> > adjust_backends;
      // Backends with static configured workload are still fixed
      for (auto iter : model_info.backend_throughputs) {
        auto backend = backends_.at(iter.first);
        if (backend->workload_id() >= 0) {
          estimate_rps -= iter.second;
        } else {
          adjust_backends.push_back(iter);
        }
      }
      // Sort the backends based on throughput
      std::sort(adjust_backends.begin(), adjust_backends.end(),
                [](std::pair<uint32_t, double> a,
                   std::pair<uint32_t, double> b) {
                  return a.second > b.second;
                });
      for (auto iter : adjust_backends) {
        if (estimate_rps <= 0) {
          auto backend = backends_.at(iter.first);
          backend->UnloadModel(model_sess_id);
          model_info.backend_throughputs.erase(iter.first);
        } else if (iter.second > estimate_rps) {
          auto backend = backends_.at(iter.first);
          float new_tp = backend->UpdateModelThroughput(model_sess_id,
                                                        estimate_rps);
          model_info.backend_throughputs[iter.first] = new_tp;
          estimate_rps -= new_tp;
        } else {
          estimate_rps -= iter.second;
        }
      }
      changed_routes.insert(model_sess_id);
    } else if (estimate_rps > throughput) {
      // Workload is larger than throughput, need to allocate more gpus
      std::vector<std::pair<uint32_t, double> > adjust_backends;
      // Backends with static configured workload are still fix
      for (auto iter : model_info.backend_throughputs) {
        auto backend = backends_.at(iter.first);
        if (backend->workload_id() >= 0) {
          estimate_rps -= iter.second;
        } else {
          adjust_backends.push_back(iter);
        }
      }
      // Second sort the backends based on throughput
      std::sort(adjust_backends.begin(), adjust_backends.end(),
                [](std::pair<uint32_t, double> a,
                   std::pair<uint32_t, double> b) {
                  return a.second > b.second;
                });
      for (auto iter : adjust_backends) {
        auto backend = backends_.at(iter.first);
        float new_tp = backend->UpdateModelThroughput(model_sess_id,
                                                      estimate_rps);
        model_info.backend_throughputs[iter.first] = new_tp;
        estimate_rps -= new_tp;
        if (backend->overload()) {
          overload_backends.push_back(backend);
        }
      }
      if (estimate_rps > 0) {
        model_info.unassigned_workload = estimate_rps;
      }
      changed_routes.insert(model_sess_id);
    }
  }

  // 2. Adjust overloaded backends
  for (auto backend : overload_backends) {
    std::vector<std::pair<std::string, float> > spillout;
    backend->SpillOutWorkload(&spillout);
    for (auto iter : spillout) {
      auto& model_info = model_table_.at(iter.first);
      model_info.backend_throughputs.erase(backend->node_id());
      model_info.unassigned_workload += iter.second;
    }
  }
  
  // 3. Allocate the unassigned workloads to backends that still have space
  AllocateUnassignedWorkloads(&changed_routes);

  // 4. Update model table to backends and model routes to frontends
  for (auto iter : backends_) {
    iter.second->UpdateModelTableRpc();
  }
  UpdateModelRoutes(changed_routes);

  DisplayModelTable();
}

void Scheduler::AllocateUnassignedWorkloads(
    std::unordered_set<std::string>* changed_routes,
    std::unordered_set<BackendDelegatePtr>* changed_backends) {
  // TODO: what if more than one unassigned workloads belong to same model session
  // Sort unassigned workloads by request rate
  std::vector<std::pair<std::string, float> > unassigned_workloads;
  for (auto const& iter : model_table_) {
    auto const& model_info = iter.second;
    if (model_info.unassigned_workload > 0) {
      unassigned_workloads.emplace_back(iter.first,
                                        model_info.unassigned_workload);
    }
  }
  if (unassigned_workloads.empty()) {
    return;
  }
  std::sort(unassigned_workloads.begin(), unassigned_workloads.end(),
            [](std::pair<std::string, float> a,
               std::pair<std::string, float> b) {
                  return a.second > b.second;
            });
  for (auto const& iter : unassigned_workloads) {
    std::string model_sess_id = iter.first;
    float request_rate = iter.second;
    // LOG(INFO) << "Try to assign workload " << model_sess_id << ", " <<
    //     request_rate << " req/s";
    ModelSession model_sess;
    ParseModelSession(model_sess_id, &model_sess);
    while (request_rate > 0) {
      BackendDelegatePtr backend;
      InstanceInfo inst_info;
      FindBestBackend(model_sess, request_rate, {}, &backend, &inst_info);
      if (backend == nullptr) {
        LOG(INFO) << "Unassigned workload " << model_sess_id << ", " <<
            request_rate << " req/s";
        break;
      } else {
        request_rate -= inst_info.throughput;
        backend->LoadModel(inst_info);
        model_table_.at(model_sess_id).backend_throughputs.emplace(
            backend->node_id(), inst_info.throughput);
        changed_routes->insert(model_sess_id);
        if (changed_backends != nullptr) {
          changed_backends->insert(backend);
        }
      }
    }
    model_table_.at(model_sess_id).unassigned_workload = std::max(
        0.f, request_rate);
  }
}

void Scheduler::UpdateModelRoutes(
    std::unordered_set<std::string> model_sessions) {
  std::unordered_map<uint32_t, ModelRouteUpdates> frontend_updates;
  for (auto const& model_sess_id : model_sessions) {
    for (auto frontend_id : model_table_.at(model_sess_id).subscribers) {
      if (frontend_updates.find(frontend_id) == frontend_updates.end()) {
        frontend_updates.emplace(frontend_id, ModelRouteUpdates());
      }
      GetModelRoute(model_sess_id,
                    frontend_updates.at(frontend_id).add_model_route());
    }
  }
  for (auto iter : frontend_updates) {
    auto frontend = frontends_.at(iter.first);
    frontend->UpdateModelRoutesRpc(iter.second);
  }
}

void Scheduler::DisplayModelTable() {
  std::stringstream ss;
  for (auto iter : model_table_) {
    auto& model_sess_id = iter.first;
    auto& model_info = iter.second;
    ss << model_sess_id << ":";
    for (auto backend_iter : model_info.backend_throughputs) {
      auto backend = GetBackend(backend_iter.first);
      double throughput = backend_iter.second;
      auto info = backend->GetInstanceInfo(model_sess_id);
      ss << " " << backend_iter.first << "/" << throughput << "/" <<
          info->batch;
    }
    ss << "\n";
  }
  VLOG(1) << "Model table: \n" << ss.str();
}

} // namespace scheduler
} // namespace nexus
