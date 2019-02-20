#include "nexus/app/frontend.h"
#include "nexus/app/rpc_service.h"

namespace nexus {
namespace app {

INSTANTIATE_RPC_CALL(AsyncService, UpdateModelRoutes, ModelRouteUpdates,
                     RpcReply);
INSTANTIATE_RPC_CALL(AsyncService, CheckAlive, CheckAliveRequest, RpcReply);

RpcService::RpcService(Frontend* frontend, std::string port, size_t nthreads):
    AsyncRpcServiceBase(port, nthreads),
    frontend_(frontend) {
}

void RpcService::HandleRpcs() {
  rpc_handlers_.emplace_back(new UpdateModelRoutes_Call(
      &service_, cq_.get(),
      [this](const grpc::ServerContext&, const ModelRouteUpdates& req,
             RpcReply* reply) {
        frontend_->UpdateModelRoutes(req, reply);
      }));
  rpc_handlers_.emplace_back(new CheckAlive_Call(
      &service_, cq_.get(),
      [](const grpc::ServerContext&, const CheckAliveRequest&,
         RpcReply* reply) {
        reply->set_status(CTRL_OK);
      }));
  void* tag;
  bool ok;
  while (running_) {
    cq_->Next(&tag, &ok);
    if (ok) {
      static_cast<RpcCallBase*>(tag)->Proceed();
    }
  }
}

} // namespace app
} // namespace nexus
