#include "app/frontend.h"
#include "app/rpc_service.h"

namespace nexus {
namespace app {

INSTANTIATE_RPC_CALL(AsyncService, UpdateBackends, BackendsUpdate,
                     BackendsUpdateReply);
INSTANTIATE_RPC_CALL(AsyncService, UpdateModelRoutes, ModelRouteList, RpcReply);
INSTANTIATE_RPC_CALL(AsyncService, CheckAlive, CheckAliveRequest, RpcReply);

RpcService::RpcService(Frontend* frontend, std::string port, size_t nthreads):
    AsyncRpcServiceBase(port, nthreads),
    frontend_(frontend) {
}

void RpcService::HandleRpcs() {
  new UpdateBackends_Call(
      &service_, cq_.get(),
      [this](RpcCallBase*, const BackendsUpdate& req,
             BackendsUpdateReply* reply) {
        frontend_->UpdateBackends(req, reply);
      });
  new UpdateModelRoutes_Call(
      &service_, cq_.get(),
      [this](RpcCallBase*, const ModelRouteList& req, RpcReply* reply) {
        frontend_->UpdateModelRoutes(req, reply);
      });
  new CheckAlive_Call(
      &service_, cq_.get(),
      [](RpcCallBase*, const CheckAliveRequest&, RpcReply* reply) {
        reply->set_status(CTRL_OK);
      });
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
