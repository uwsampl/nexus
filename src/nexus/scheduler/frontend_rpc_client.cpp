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
    timeout_(timeout) {
  auto channel = grpc::CreateChannel(rpc_addr,
                                     grpc::InsecureChannelCredentials());
  stub_ = FrontendCtrl::NewStub(channel);
  last_time_ = std::chrono::system_clock::now();
}

std::time_t FrontendRpcClient::LastAliveTime() {
  std::lock_guard<std::mutex> lock(mutex_);
  return std::chrono::system_clock::to_time_t(last_time_);
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

  // Inovke RPC CheckAlive
  grpc::ClientContext context;
  grpc::Status status = stub_->CheckAlive(&context, request, &reply);
  if (!status.ok()) {
    LOG(ERROR) << status.error_code() << ": " << status.error_message();
    return false;
  }
  last_time_ = std::chrono::system_clock::now();
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

} // namespace scheduler
} // namespace nexus
