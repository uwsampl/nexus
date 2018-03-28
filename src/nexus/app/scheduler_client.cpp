#include <glog/logging.h>
#include <google/protobuf/text_format.h>

#include "app/frontend.h"
#include "app/scheduler_client.h"

namespace nexus {
namespace app {

SchedulerClient::SchedulerClient(Frontend* frontend,
                                 const std::string& scheduler_address):
    frontend_(frontend) {
  auto channel = grpc::CreateChannel(scheduler_address,
                                     grpc::InsecureChannelCredentials());
  stub_ = SchedulerCtrl::NewStub(channel);
}

CtrlStatus SchedulerClient::Register() {
  RegisterRequest request;
  request.set_node_type(FRONTEND_NODE);
  request.set_node_id(frontend_->node_id());
  request.set_server_port(frontend_->port());
  request.set_rpc_port(frontend_->rpc_port());
  grpc::ClientContext context;
  RegisterReply reply;
  grpc::Status status = stub_->Register(&context, request, &reply);
  if (!status.ok()) {
    LOG(ERROR) << status.error_code() << ": " << status.error_message();
    return CTRL_SERVER_UNREACHABLE;
  }
  CtrlStatus ret = reply.status();
  if (ret == CTRL_OK) {
    BackendsUpdateReply dummy;
    frontend_->UpdateBackends(reply.init_backend_pool(), &dummy);
  }
  return ret;
}
    
void SchedulerClient::Unregister() {
  UnregisterRequest request;
  request.set_node_type(FRONTEND_NODE);
  request.set_node_id(frontend_->node_id());
  grpc::ClientContext context;
  RpcReply reply;
  grpc::Status status = stub_->Unregister(&context, request, &reply);
  if (!status.ok()) {
    LOG(ERROR) << status.error_code() << ": " << status.error_message();
    return;
  }
  CtrlStatus ret = reply.status();
  if (ret != CTRL_OK) {
    LOG(ERROR) << "Unregister error: " << CtrlStatus_Name(ret);
  }
}

CtrlStatus SchedulerClient::LoadModelRpc(const LoadModelRequest& request,
                                         LoadModelReply* reply) {
  grpc::ClientContext context;
  grpc::Status ret = stub_->LoadModel(&context, request, reply);
  if (!ret.ok()) {
    LOG(ERROR) << ret.error_code() << ": " << ret.error_message();
    return CTRL_SERVER_UNREACHABLE;
  }
  return CTRL_OK;
}

} // namespace app
} // namespace nexus
