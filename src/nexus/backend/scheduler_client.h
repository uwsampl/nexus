#ifndef NEXUS_BACKEND_SCHEDULER_CLIENT_H_
#define NEXUS_BACKEND_SCHEDULER_CLIENT_H_

#include <atomic>
#include <grpc++/grpc++.h>
#include <memory>
#include <thread>

#include "nexus/proto/control.grpc.pb.h"

namespace nexus {
namespace backend {

class BackendServer;

class SchedulerClient {
 public:
  SchedulerClient(BackendServer* backend, const std::string& scheduler_address);

  bool Start();

  void Stop();

  void Run();

  CtrlStatus Register();
  
  void Unregister();

 private:
  BackendServer* backend_;
  std::unique_ptr<SchedulerCtrl::Stub> stub_;
  volatile std::atomic_bool running_;
  std::thread thread_;
};

} // namespace backend
} // namespace nexus

#endif // NEXUS_BACKEND_SCHEDULER_CLIENT_H_
