#ifndef NEXUS_APP_RPC_SERVICE_H_
#define NEXUS_APP_RPC_SERVICE_H_

#include <grpc++/grpc++.h>

#include "nexus/common/rpc_call.h"
#include "nexus/common/rpc_service_base.h"
#include "nexus/proto/control.grpc.pb.h"

namespace nexus {
namespace app {

using AsyncService = nexus::FrontendCtrl::AsyncService;

class Frontend;

class RpcService : public AsyncRpcServiceBase<AsyncService> {
 public:
  RpcService(Frontend* frontend, std::string port, size_t nthreads = 1);

 protected:
  void HandleRpcs() final;

 private:
  Frontend* frontend_;
};

} // namespace app
} // namespace nexus

#endif // NEXUS_APP_RPC_SERVICE_H_
