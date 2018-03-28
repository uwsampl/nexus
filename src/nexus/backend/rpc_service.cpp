#include "backend/backend_server.h"
#include "backend/rpc_service.h"
#include "common/rpc_call.h"

namespace nexus {
namespace backend {

INSTANTIATE_RPC_CALL(AsyncService, UpdateModelTable, ModelTable, RpcReply);
INSTANTIATE_RPC_CALL(AsyncService, CheckAlive, CheckAliveRequest, RpcReply);

BackendRpcService::BackendRpcService(BackendServer* backend, std::string port,
                                     size_t nthreads):
    AsyncRpcServiceBase(port, nthreads),
    backend_(backend) {
}

void BackendRpcService::HandleRpcs() {
  new UpdateModelTable_Call(
      &service_, cq_.get(),
      [this](RpcCallBase*, const ModelTable& req, RpcReply* reply) {
        backend_->UpdateModelTable(req, reply);
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

} // namespace backend
} // namespace nexus
