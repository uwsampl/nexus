#ifndef NEXUS_BACKEND_RPC_SERVICE_H_
#define NEXUS_BACKEND_RPC_SERVICE_H_

#include <grpc++/grpc++.h>

#include "nexus/common/rpc_service_base.h"
#include "nexus/proto/control.grpc.pb.h"

namespace nexus {
namespace backend {

using AsyncService = nexus::BackendCtrl::AsyncService;

class BackendServer;

class BackendRpcService : public AsyncRpcServiceBase<AsyncService> {
 public:
  BackendRpcService(BackendServer* backend, std::string port,
                    size_t nthreads = 1);

 protected:
  void HandleRpcs() final;

 private:
  BackendServer* backend_;
};

} // namespace backend
} // namespace nexus

#endif // NEXUS_BACKEND_RPC_SERVICE_H_
