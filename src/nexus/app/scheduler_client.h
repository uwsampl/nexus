#ifndef NEXUS_APP_SCHEDULER_CLIENT_H_
#define NEXUS_APP_SCHEDULER_CLIENT_H_

#include <grpc++/grpc++.h>

#include "nexus/proto/control.grpc.pb.h"
#include "nexus/proto/nnquery.pb.h"

namespace nexus {
namespace app {

class Frontend;

class SchedulerClient {
 public:
  SchedulerClient(Frontend* frontend, const std::string& scheduler_address);

  CtrlStatus Register();

  void Unregister();

  CtrlStatus LoadModelRpc(const LoadModelRequest& request,
                          LoadModelReply* reply);

 private:
  Frontend* frontend_;
  std::unique_ptr<SchedulerCtrl::Stub> stub_;
};

} // namespace app
} // namespace nexus

#endif // NEXUS_APP_SCHEDULER_CLIENT_H_
