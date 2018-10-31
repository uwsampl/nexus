#ifndef NEXUS_SCHEDULER_FRONTEND_DELEGATE_H_
#define NEXUS_SCHEDULER_FRONTEND_DELEGATE_H_

#include <chrono>
#include <grpc++/grpc++.h>
#include <mutex>
#include <unordered_map>
#include <unordered_set>


#include "nexus/proto/control.grpc.pb.h"
#include "nexus/scheduler/complex_query.h"

namespace nexus {
namespace scheduler {

class Scheduler;

class FrontendDelegate {
 public:
  FrontendDelegate(uint32_t node_id, const std::string& ip,
                   const std::string& server_port, const std::string& rpc_addr,
                   int beacon_sec);

  uint32_t node_id() const { return node_id_; }

  std::time_t LastAliveTime();

  void Tick();

  bool IsAlive();

  void SubscribeModel(const std::string& model_session_id);

  const std::unordered_set<std::string>& subscribe_models() const {
    return subscribe_models_;
  }

  CtrlStatus UpdateModelRoutesRpc(const ModelRouteUpdates& request);
  
  CtrlStatus LoadDependency(const LoadDependencyProto& request);
  
  CtrlStatus CurrentRps(const CurRpsProto& request);
  
  bool containComplexQuery() {return complexQuery_;}
 private:
  uint32_t node_id_;
  std::string ip_;
  std::string server_port_;
  std::string rpc_port_;
  int beacon_sec_;
  long timeout_ms_;
  bool complexQuery_;
  ComplexQuery query_;
  std::unique_ptr<FrontendCtrl::Stub> stub_;
  std::chrono::time_point<std::chrono::system_clock> last_time_;
  std::unordered_set<std::string> subscribe_models_;
  RpsRecord rpsRecord_;
  std::string common_gpu_;
};

} // namespace scheduler
} // namespace nexus

#endif // NEXUS_SCHEDULER_FRONTEND_DELEGATE_H_
