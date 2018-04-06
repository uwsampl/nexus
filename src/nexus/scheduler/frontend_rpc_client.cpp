#include "nexus/scheduler/frontend_rpc_client.h"
#include "nexus/scheduler/scheduler.h"

namespace nexus {
namespace scheduler {

FrontendRpcClient::FrontendRpcClient(Scheduler* sch, uint32_t node_id,
                                     const std::string& server_addr,
                                     const std::string& rpc_addr,
                                     std::chrono::milliseconds timeout):
    scheduler_(sch),
    node_id_(node_id),
    server_address_(server_addr),
    rpc_address_(rpc_addr),
    timeout_(timeout),
    backend_pool_version_(0) {
  auto channel = grpc::CreateChannel(rpc_addr,
                                     grpc::InsecureChannelCredentials());
  stub_ = FrontendCtrl::NewStub(channel);
  last_time_ = std::chrono::system_clock::now();
}

void FrontendRpcClient::Tick() {
  std::lock_guard<std::mutex> lock(mutex_);
  last_time_ = std::chrono::system_clock::now();
}

std::time_t FrontendRpcClient::LastTime() {
  std::lock_guard<std::mutex> lock(mutex_);
  return std::chrono::system_clock::to_time_t(last_time_);
}

CtrlStatus FrontendRpcClient::UpdateBackends(
    uint32_t curr_version,
    const std::unordered_map<uint32_t, BackendsUpdate>& history) {
  std::lock_guard<std::mutex> lock(mutex_);
  uint32_t base_version = backend_pool_version_;
  if (curr_version == base_version) {
    return CTRL_OK;
  }
  grpc::Status ret;
  CtrlStatus status;
  // try update at most 2 times
  for (int i = 0; i < 2; ++i) {
    BackendsUpdate request;
    BackendsUpdateReply reply;
    MergeBackendsUpdate(base_version, curr_version, history, &request);
    ret = UpdateBackendsRpc(request, &reply);
    if (!ret.ok()) {
      return CTRL_SERVER_UNREACHABLE;
    }
    status = reply.status();
    if (status == CTRL_OK) {
      backend_pool_version_ = curr_version;
      return CTRL_OK;
    }
    if (status != CTRL_ROUTE_TABLE_VERSION_MISMATCH) {
      // Unknown error
      LOG(ERROR) << "Frontend " << node_id_ << " error: " <<
          CtrlStatus_Name(status);
      return status;
    }
    base_version = reply.version();
  }
  return status;
}

bool FrontendRpcClient::IsAlive() {
  std::lock_guard<std::mutex> lock(mutex_);
  auto now = std::chrono::system_clock::now();
  std::chrono::duration<double> elapse = now - last_time_;
  if (elapse < timeout_) {
    return true;
  }
  CheckAliveRequest request;
  request.set_node_type(FRONTEND_NODE);
  request.set_node_id(node_id_);
  RpcReply reply;
  grpc::Status ret = CheckAliveRpc(request, &reply);
  if (!ret.ok()) {
    return false;
  }
  return true;
}

void FrontendRpcClient::SubscribeModel(const std::string& model_session_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  subscribe_models_.insert(model_session_id);
}

const std::unordered_set<std::string>& FrontendRpcClient::subscribe_models() {
  std::lock_guard<std::mutex> lock(mutex_);
  return subscribe_models_;
}

grpc::Status FrontendRpcClient::UpdateBackendsRpc(
    const BackendsUpdate& request, BackendsUpdateReply* reply) {
  grpc::ClientContext context;
  grpc::Status ret = stub_->UpdateBackends(&context, request, reply);
  if (ret.ok()) {
    last_time_ = std::chrono::system_clock::now();
  } else {
    LOG(ERROR) << ret.error_code() << ": " << ret.error_message();
  }
  return ret;
}

grpc::Status FrontendRpcClient::CheckAliveRpc(
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

void FrontendRpcClient::MergeBackendsUpdate(
    uint32_t base_version, uint32_t curr_version,
    const std::unordered_map<uint32_t, BackendsUpdate>& history,
    BackendsUpdate* req) const {
  req->set_curr_version(curr_version);
  req->set_base_version(base_version);
  std::unordered_map<uint32_t, BackendInfo&> add_backends;
  std::unordered_map<uint32_t, BackendInfo&> remove_backends;
  for (uint32_t version = base_version + 1; version <= curr_version; ++version) {
    auto update = history.at(version);
    for (auto backend_info : update.add_backend()) {
      uint32_t node_id = backend_info.node_id();
      add_backends.emplace(node_id, backend_info);
    }
    for (auto backend_info : update.remove_backend()) {
      uint32_t node_id = backend_info.node_id();
      auto iter = add_backends.find(node_id);
      if (iter != add_backends.end()) {
        add_backends.erase(iter);
      } else {
        remove_backends.emplace(node_id, backend_info);
      }
    }
  }
  for (auto iter : add_backends) {
    auto add = req->add_add_backend();
    add->CopyFrom(iter.second);
  }
  for (auto iter : remove_backends) {
    auto remove = req->add_remove_backend();
    remove->CopyFrom(iter.second);
  }
}

} // namespace scheduler
} // namespace nexus
