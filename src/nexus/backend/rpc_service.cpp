#include <gflags/gflags.h>

#include "nexus/backend/backend_server.h"
#include "nexus/backend/rpc_service.h"
#include "nexus/common/rpc_call.h"

DECLARE_int32(occupancy_valid);

namespace nexus {
namespace backend {

INSTANTIATE_RPC_CALL(AsyncService, UpdateModelTable, ModelTableConfig,
                     RpcReply);
INSTANTIATE_RPC_CALL(AsyncService, CheckAlive, CheckAliveRequest, RpcReply);
INSTANTIATE_RPC_CALL(AsyncService, CurrentUtilization, UtilizationRequest,
                     UtilizationReply);

BackendRpcService::BackendRpcService(BackendServer* backend, std::string port,
                                     size_t nthreads):
    AsyncRpcServiceBase(port, nthreads),
    backend_(backend) {
}

void BackendRpcService::HandleRpcs() {
  new UpdateModelTable_Call(
      &service_, cq_.get(),
      [this](const grpc::ServerContext&, const ModelTableConfig& req,
             RpcReply* reply) {
        backend_->UpdateModelTable(req, reply);
      });
  new CheckAlive_Call(
      &service_, cq_.get(),
      [](const grpc::ServerContext&, const CheckAliveRequest&,
         RpcReply* reply) {
        reply->set_status(CTRL_OK);
      });
  new CurrentUtilization_Call(
      &service_, cq_.get(),
      [this](const grpc::ServerContext&, const UtilizationRequest&,
         UtilizationReply* reply) {
        reply->set_node_id(backend_->node_id());
        reply->set_utilization(backend_->CurrentUtilization());
        reply->set_valid_ms(FLAGS_occupancy_valid);
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
