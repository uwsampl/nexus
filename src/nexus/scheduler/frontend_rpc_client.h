#ifndef NEXUS_SCHEDULER_FRONTEND_RPC_CLIENT_H_
#define NEXUS_SCHEDULER_FRONTEND_RPC_CLIENT_H_

#include <chrono>
#include <grpc++/grpc++.h>
#include <mutex>
#include <unordered_map>
#include <unordered_set>

#include "nexus/proto/control.grpc.pb.h"

namespace nexus {
namespace scheduler {

class Scheduler;

class FrontendRpcClient {
 public:
  FrontendRpcClient(uint32_t node_id, const std::string& server_addr,
                    const std::string& rpc_addr, int beacon_sec);

  uint32_t node_id() const { return node_id_; }

  std::string server_address() const { return server_address_; }

  std::string rpc_address() const { return rpc_address_; }

  std::time_t LastAliveTime();

  void Tick();

  bool IsAlive();

  void SubscribeModel(const std::string& model_session_id);

  const std::unordered_set<std::string>& subscribe_models() const {
    return subscribe_models_;
  }

 private:
  uint32_t node_id_;
  std::string server_address_;
  std::string rpc_address_;
  int beacon_sec_;
  long timeout_ms_;
  std::unique_ptr<FrontendCtrl::Stub> stub_;
  std::chrono::time_point<std::chrono::system_clock> last_time_;
  std::unordered_set<std::string> subscribe_models_;
};

} // namespace scheduler
} // namespace nexus

#endif // NEXUS_SCHEDULER_FRONTEND_RPC_CLIENT_H_
