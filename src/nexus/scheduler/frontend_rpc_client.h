#ifndef NEXUS_SCHEDULER_FRONTEND_RPC_CLIENT_H_
#define NEXUS_SCHEDULER_FRONTEND_RPC_CLIENT_H_

#include <chrono>
#include <grpc++/grpc++.h>
#include <mutex>
#include <unordered_map>
#include <unordered_set>

#include "proto/control.grpc.pb.h"

namespace nexus {
namespace scheduler {

class Scheduler;

class FrontendRpcClient {
 public:
  FrontendRpcClient(Scheduler* sch, uint32_t node_id,
                    const std::string& server_addr, const std::string& rpc_addr,
                    std::chrono::milliseconds timeout);

  uint32_t node_id() const { return node_id_; }

  std::string server_address() const { return server_address_; }

  std::string rpc_address() const { return rpc_address_; }

  uint32_t backend_pool_version() const { return backend_pool_version_; }

  void Tick();

  std::time_t LastTime();

  CtrlStatus UpdateBackends(
      uint32_t curr_version,
      const std::unordered_map<uint32_t, BackendsUpdate>& history);

  bool IsAlive();

  void SubscribeModel(const std::string& model_session_id);

  const std::unordered_set<std::string>& subscribe_models();

 private:
  grpc::Status UpdateBackendsRpc(const BackendsUpdate& request,
                                 BackendsUpdateReply* reply);

  grpc::Status CheckAliveRpc(const CheckAliveRequest& reqeust,
                             RpcReply* reply);
  
  void MergeBackendsUpdate(
    uint32_t base_version, uint32_t curr_version,
    const std::unordered_map<uint32_t, BackendsUpdate>& history,
    BackendsUpdate* req) const;

 private:
  Scheduler* scheduler_;
  uint32_t node_id_;
  std::string server_address_;
  std::string rpc_address_;
  std::chrono::milliseconds timeout_;
  uint32_t backend_pool_version_;
  std::mutex mutex_;
  std::unique_ptr<FrontendCtrl::Stub> stub_;
  std::chrono::time_point<std::chrono::system_clock> last_time_;
  std::unordered_set<std::string> subscribe_models_;
};

} // namespace scheduler
} // namespace nexus

#endif // NEXUS_SCHEDULER_FRONTEND_RPC_CLIENT_H_
