#include <boost/filesystem.hpp>
#include <glog/logging.h>
#include <unordered_set>

#include "scheduler/scheduler.h"

namespace fs = boost::filesystem;

namespace nexus {
namespace scheduler {

INSTANTIATE_RPC_CALL(AsyncService, Register, RegisterRequest, RegisterReply);
INSTANTIATE_RPC_CALL(AsyncService, Unregister, UnregisterRequest, RpcReply);
INSTANTIATE_RPC_CALL(AsyncService, LoadModel, LoadModelRequest, LoadModelReply);
INSTANTIATE_RPC_CALL(AsyncService, UnloadModel, ModelSession, RpcReply);

Scheduler::Scheduler(std::string port, size_t nthreads, double epoch) :
    AsyncRpcServiceBase(port, nthreads),
    epoch_(static_cast<unsigned long>(epoch * 1000)),
    backend_pool_version_(0) {
}

void Scheduler::LoadConfigFile(const std::string& config_file) {
  LOG(INFO) << "Load config file from " << config_file;
  YAML::Node config = YAML::LoadFile(config_file);
  CHECK(config["model_profile_dir"]) << " is missing";
  ModelProfileTable::Singleton().Init(
      config["model_profile_dir"].as<std::string>());
  if (!config["backends"]) {
    return;
  }
  // Load static workload configuration
  for (uint i = 0; i < config["backends"].size(); ++i) {
    const YAML::Node& backend_info = config["backends"][i];
    std::unordered_map<ModelId, YAML::Node> models;
    LOG(INFO) << "backend " << i << " to load models:";
    for (uint j = 0; j < backend_info["models"].size(); ++j) {
      const YAML::Node& model_info = backend_info["models"][j];
      if (!model_info["framework"]) {
        LOG(FATAL) << "Missing framework in the model config";
      }
      if (!model_info["model_name"]) {
      LOG(FATAL) << "Missing model_name in the model config";
      }
      if (!model_info["max_batch"]) {
        LOG(FATAL) << "Missing max_batch in the model config";
      }
      Framework framework = get_Framework(
          model_info["framework"].as<std::string>());
      std::string model_name = model_info["model_name"].as<std::string>();
      size_t max_batch = model_info["max_batch"].as<size_t>();
      ModelId model_id(framework, model_name);
      LOG(INFO) << "- " << model_id << ", " << max_batch;
      models.emplace(model_id, model_info);
    }
    workloads_.push_back(models);
  }
}

void Scheduler::Run() {
  // Start RPC service first
  Start();
  // main scheduler login
  while (running_) {
    /*
    auto now = std::chrono::system_clock::now();
    auto next_epoch = now + epoch_;
    std::vector<BackendRpcClientPtr> dead_backends;
    std::vector<BackendRpcClientPtr> update_backends;
    std::vector<FrontendRpcClientPtr> dead_frontends;
    {
      // lock protected region begins
      std::lock_guard<std::mutex> lock(mutex_);
      std::vector<BackendRpcClientPtr> idle_backends;
      // check if backends are alive
      for (auto it : backends_) {
        auto backend = it.second;
        if (backend->IsAlive()) {
          if (backend->IsIdle()) {
            idle_backends.push_back(backend);
          }
          continue;
        }
        std::time_t last_time = backend->LastTime();
        LOG(INFO) << "Remove backend " << backend->node_id() <<
            ", last time: " << std::ctime(&last_time);
        // remove the assigned workload
        int workload_id = backend->workload_id();
        if (workload_id >= 0) {
          assigned_workloads_.erase(workload_id);
          LOG(INFO) << "Remove workload " << workload_id;
        }
        // add backend to remove list
        dead_backends.push_back(backend);
      }
      // remove the dead backends
      for (auto backend : dead_backends) {
        backends_.erase(backend->node_id());
      }
      // reassign workload to idle backends
      if (idle_backends.size() > 0) {
        for (uint id = 0; id < workloads_.size(); ++id) {
          if (assigned_workloads_.find(id) == assigned_workloads_.end()) {
            auto backend = idle_backends.back();
            idle_backends.pop_back();
            LOG(INFO) << "Assign workload " << id << " to backend " <<
                backend->node_id();
            auto toload_models = workloads_[id];
            backend->InitModelTable(toload_models);
            backend->UpdateModelTable();
            backend->set_workload_id(id);
            update_backends.push_back(backend);
            if (idle_backends.empty()) {
              break;
            }
          }
        }
      }
      // check if frontends are alive
      for (auto it : frontends_) {
        auto frontend = it.second;
        if (frontend->IsAlive()) {
          continue;
        }
        std::time_t last_time = frontend->LastTime();
        LOG(INFO) << "Remove frontend " << frontend->node_id() <<
            ", last time: " << std::ctime(&last_time);
        // add backend to remove list
        dead_frontends.push_back(frontend);
      }
      // remove the dead backends
      for (auto frontend : dead_frontends) {
        frontends_.erase(frontend->node_id());
      }
      // lock protection region ends
    }
    if (update_backends.size() > 0 || dead_backends.size() > 0) {
      // update route table to all frontend
      UpdateRouteTable(update_backends, dead_backends);
      BroadcastRouteTableChange();
    }
    std::this_thread::sleep_until(next_epoch);*/
  }
}

void Scheduler::Register(RpcCallBase* call, const RegisterRequest& request,
                         RegisterReply* reply) {
  std::vector<std::string> tokens;
  SplitString(call->PeerAddress(), ':', &tokens);
  std::string server_addr = tokens[1] + ':' + request.server_port();
  std::string rpc_addr = tokens[1] + ':' + request.rpc_port();
  LOG(INFO) << "Register " << NodeType_Name(request.node_type()) << " " <<
      request.node_id() << " : " << server_addr << ", " << rpc_addr;
  if (request.node_type() == BACKEND_NODE) {
    auto backend = std::make_shared<BackendRpcClient>(
        this, request.node_id(), server_addr, rpc_addr,
        request.gpu_device_name(), request.gpu_available_memory(),
        BackendTimeout());
    RegisterBackend(std::move(backend), reply);
  } else {
    // frontend node
    auto frontend = std::make_shared<FrontendRpcClient>(
        this, request.node_id(), server_addr, rpc_addr, FrontendTimeout());
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
  // TODO: relax the backend that has free cycle for this model session
  float workload = request.estimate_workload();
  if (workload == 0) {
    reply->set_status(CTRL_INVALID_LOAD_MODEL_REQUEST);
    return;
  }
  std::string model_sess_id = ModelSessionToString(request.model_session());
  std::vector<std::pair<BackendRpcClientPtr, ModelInstanceDesc>> assign_backends;
  {
    // TODO: check if model_sess_id already exists
    // lock protection region
    std::lock_guard<std::mutex> lock(mutex_);
    uint32_t frontend_id = request.node_id();
    std::shared_ptr<FrontendRpcClient> frontend = nullptr;
    auto iter = frontends_.find(frontend_id);
    if (iter == frontends_.end()) {
      reply->set_status(CTRL_NOT_REGISTERED);
      return;
    }
    frontend = iter->second;

    for (auto it : backends_) {
      auto backend = it.second;
      if (!backend->IsAlive() || !backend->IsIdle()) {
        continue;
      }
      ModelInstanceDesc model_desc;
      float occupancy;
      backend->PrepareLoadModel(request.model_session(), workload, &model_desc,
                                &occupancy);
      if (model_desc.batch() == 0) {
        continue;
      }
      assign_backends.emplace_back(backend, model_desc);
      workload -= model_desc.workload();
      if (workload == 0) {
        break;
      }
    }
    if (workload > 0) {
      reply->set_status(CTRL_NOT_ENOUGH_BACKENDS);
      return;
    }
    ModelRoute route;
    route.mutable_model_session()->CopyFrom(request.model_session());
    for (auto item : assign_backends) {
      auto backend = item.first;
      const auto& model_desc = item.second;
      backend->LoadModel(model_desc);
      auto backend_rate = route.add_backend_rate();
      backend_rate->set_node_id(backend->node_id());
      backend_rate->set_rate(model_desc.throughput());
    }
    model_routes_.emplace(model_sess_id, route);
    model_subscribers_.emplace(
        model_sess_id, std::vector<uint32_t>({request.node_id()}));
    frontend->SubscribeModel(model_sess_id);
    reply->set_status(CTRL_OK);
    reply->mutable_model_route()->CopyFrom(model_routes_.at(model_sess_id));
  }

  for (auto item : assign_backends) {
    auto backend = item.first;
    CtrlStatus ret = backend->UpdateModelTable();
    if (ret != CTRL_OK) {
      // TODO
    }
  }
  reply->set_status(CTRL_OK);
  //LOG(INFO) << "load model finished";
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
  //                      std::bind(&Scheduler::UnloadModel, this, _1, _2));
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
  {
    // lock protection region
    std::lock_guard<std::mutex> lock(mutex_);
    if (frontends_.find(frontend->node_id()) != frontends_.end()) {
      reply->set_status(CTRL_FRONTEND_NODE_ID_CONFLICT);
      return;
    }
    // add the frontend client in the frontend map
    frontends_[frontend->node_id()] = frontend;
  }
  reply->set_status(CTRL_OK);
  reply->set_epoch_time(epoch_.count());
  GetBackendPool(reply->mutable_init_backend_pool());
}

void Scheduler::RegisterBackend(BackendRpcClientPtr backend,
                                RegisterReply* reply) {
  int assign_load_id = -1;
  {
    // lock protection region
    std::lock_guard<std::mutex> lock(mutex_);
    if (backends_.find(backend->node_id()) != backends_.end()) {
      reply->set_status(CTRL_BACKEND_NODE_ID_CONFLICT);
      return;
    }
    // add the backend client in the backend map
    backends_[backend->node_id()] = backend;
    // assign workload to the new backend node
    for (uint id = 0; id < workloads_.size(); ++id) {
      if (assigned_workloads_.find(id) == assigned_workloads_.end()) {
        assign_load_id = id;
        assigned_workloads_[id] = backend->node_id();
        break;
      }
    }
    reply->set_status(CTRL_OK);
    reply->set_epoch_time(epoch_.count());
    if (assign_load_id >= 0) {
      LOG(INFO) << "Assign workload " << assign_load_id << " to backend " <<
          backend->node_id();
      auto workload = workloads_[assign_load_id];
      for (auto iter : workload) {
        backend->LoadModel(iter.first, iter.second);
      }
      backend->set_workload_id(assign_load_id);
      backend->GetModelTable(reply->mutable_init_model_table());
    }
  }
  onBackendsUpdate({backend}, {});
}

void Scheduler::UnregisterFrontend(uint32_t node_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = frontends_.find(node_id);
  if (it == frontends_.end()) {
    LOG(ERROR) << "Cannot find frontend " << node_id;
    return;
  }
  // TODO: need to remove frontend from its subscribed model sessions
  /*auto frontend = it->second;
  for (auto model_sess_id : frontend->subscribe_models()) {
  }*/
  frontends_.erase(it);
}

void Scheduler::UnregisterBackend(uint32_t node_id) {
  BackendRpcClientPtr backend;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = backends_.find(node_id);
    if (it == backends_.end()) {
      LOG(ERROR) << "Cannot find backend " << node_id;
      return;
    }
    backend = it->second;
    int workload_id = backend->workload_id();
    if (workload_id >= 0) {
      assigned_workloads_.erase(workload_id);
      LOG(INFO) << "Remove workload " << workload_id;
    }
    backends_.erase(it);
    LOG(INFO) << "Remove backend " << node_id;
  }
  // TODO: need to update model routes
  onBackendsUpdate({}, {backend});
}

void Scheduler::onBackendsUpdate(
    const std::vector<BackendRpcClientPtr>& adds,
    const std::vector<BackendRpcClientPtr>& removes) {
  if (adds.size() == 0 && removes.size() == 0) {
    return;
  }
  std::lock_guard<std::mutex> lock(mutex_);
  BackendsUpdate update;
  update.set_base_version(backend_pool_version_);
  ++backend_pool_version_;
  update.set_curr_version(backend_pool_version_);
  for (auto backend : adds) {
    auto backend_info = update.add_add_backend();
    backend_info->set_node_id(backend->node_id());
    backend_info->set_server_address(backend->server_address());
  }
  for (auto backend : removes) {
    auto backend_info = update.add_remove_backend();
    backend_info->set_node_id(backend->node_id());
  }
  backends_updates_.emplace(backend_pool_version_, update);
  LOG(INFO) << "Backend pool version: " << backend_pool_version_;

  // broadcast to all frontends
  for (auto it : frontends_) {
    auto frontend = it.second;
    CtrlStatus ret = frontend->UpdateBackends(backend_pool_version_,
                                              backends_updates_);
    if (ret != CTRL_OK) {
      LOG(INFO) << "Failed to update backend pool to frontend " <<
          frontend->node_id() << ": " << CtrlStatus_Name(ret);
    }
  }
}

void Scheduler::GetBackendPool(BackendsUpdate* update) {
  std::lock_guard<std::mutex> lock(mutex_);
  update->set_base_version(0);
  update->set_curr_version(backend_pool_version_);
  for (auto iter : backends_) {
    auto backend = iter.second;
    auto backend_info = update->add_add_backend();
    backend_info->set_node_id(backend->node_id());
    backend_info->set_server_address(backend->server_address());
  }
}

} // namespace scheduler
} // namespace nexus
