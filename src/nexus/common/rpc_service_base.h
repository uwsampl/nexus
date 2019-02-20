#ifndef NEXUS_COMMON_RPC_SERVICE_BASE_H_
#define NEXUS_COMMON_RPC_SERVICE_BASE_H_

#include <glog/logging.h>
#include <grpc++/grpc++.h>
#include <thread>
#include <vector>
#include "nexus/common/rpc_call.h"

namespace nexus {

template<class ServiceType>
class AsyncRpcServiceBase {
 public:
  AsyncRpcServiceBase(std::string port, size_t nthreads):
      AsyncRpcServiceBase("0.0.0.0", port, nthreads) {}
  
  AsyncRpcServiceBase(std::string ip, std::string port, size_t nthreads):
      ip_(ip),
      port_(port),
      nthreads_(nthreads),
      running_(false) {
  }

  virtual ~AsyncRpcServiceBase() {
  	if (running_) {
      Stop();
    }
  }

  std::string port() const { return port_; }

  std::string address() const { return ip_ + ":" + port_; }

  void Start() {
    grpc::ServerBuilder builder;
    std::string addr = ip_ + ":" + port_;
    builder.AddListeningPort(addr, grpc::InsecureServerCredentials());
    builder.RegisterService(&service_);
    cq_ = builder.AddCompletionQueue();
    server_ = builder.BuildAndStart();
    running_ = true;
    for (size_t i = 0; i < nthreads_; ++i) {
      thread_pool_.emplace_back(&AsyncRpcServiceBase::HandleRpcs, this);
    }
    LOG(INFO) << "RPC service is listening on " << addr;
  }

  void Stop() {
    running_ = false;
    server_->Shutdown();
    cq_->Shutdown();

    void *tag;
    bool ok;
    while (cq_->Next(&tag, &ok)) {
      LOG(WARNING) << "There is a event in the grpc::ServerCompletionQueue not handled at " << tag;
    }

    for (auto& thread : thread_pool_) {
      thread.join();
    }
    for (auto rpc_call : rpc_handlers_)
      delete rpc_call;
    rpc_handlers_.clear();

    LOG(INFO) << "RPC service stopped";
  }

 protected:
  virtual void HandleRpcs() = 0;
  std::vector<RpcCallBase*> rpc_handlers_;

 protected:
  std::string ip_;
  std::string port_;
  size_t nthreads_;
  volatile bool running_;
  std::vector<std::thread> thread_pool_;
  ServiceType service_;
  std::unique_ptr<grpc::ServerCompletionQueue> cq_;
  std::unique_ptr<grpc::Server> server_;
};

} // namespace nexus

#endif // NEXUS_COMMON_RPC_SERVICE_BASE_H_
