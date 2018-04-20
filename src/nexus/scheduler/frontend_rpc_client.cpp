#include "nexus/scheduler/frontend_rpc_client.h"
#include "nexus/scheduler/scheduler.h"

namespace nexus {
namespace scheduler {

FrontendRpcClient::FrontendRpcClient(uint32_t node_id,
                                     const std::string& server_addr,
                                     const std::string& rpc_addr,
                                     int beacon_sec):
    node_id_(node_id),
    server_address_(server_addr),
    rpc_address_(rpc_addr),
    beacon_sec_(beacon_sec),
    timeout_ms_(beacon_sec * 2 * 1000) {
  auto channel = grpc::CreateChannel(rpc_addr,
                                     grpc::InsecureChannelCredentials());
  stub_ = FrontendCtrl::NewStub(channel);
  last_time_ = std::chrono::system_clock::now();
}

std::time_t FrontendRpcClient::LastAliveTime() {
  return std::chrono::system_clock::to_time_t(last_time_);
}

void FrontendRpcClient::Tick() {
  last_time_ = std::chrono::system_clock::now();
}

bool FrontendRpcClient::IsAlive() {
  long elapse = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::system_clock::now() - last_time_).count();
  if (elapse < timeout_ms_) {
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
  subscribe_models_.insert(model_session_id);
}

} // namespace scheduler
} // namespace nexus
