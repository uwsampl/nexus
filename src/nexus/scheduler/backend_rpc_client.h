#ifndef NEXUS_SCHEDULER_BACKEND_RPC_CLIENT_H_
#define NEXUS_SCHEDULER_BACKEND_RPC_CLIENT_H_

#include <chrono>
#include <grpc++/grpc++.h>
#include <mutex>
#include <unordered_map>
#include <vector>
#include <yaml-cpp/yaml.h>

#include "nexus/common/metric.h"
#include "nexus/common/model_def.h"
#include "nexus/proto/control.grpc.pb.h"

namespace nexus {
namespace scheduler {

class Scheduler;

class BackendRpcClient {
 public:
  BackendRpcClient(
      Scheduler* sch, uint32_t node_id, const std::string& server_addr,
      const std::string& rpc_addr, const std::string& gpu_device,
      size_t gpu_available_memory, int beacon_sec, int epoch_sec);

  uint32_t node_id() const { return node_id_; }

  std::string server_address() const { return server_address_; }

  std::string rpc_address() const { return rpc_address_; }
 
  std::string gpu_device() const { return gpu_device_; }

  size_t gpu_available_memory() const { return gpu_available_memory_; }

  int workload_id() const { return workload_id_; }

  void set_workload_id(int id) { workload_id_ = id; }

  void GetInfo(BackendInfo* info);

  std::time_t LastAliveTime();

  bool Assign(const BackendRpcClient& other);

  void PrepareLoadModel(const ModelSession& model_sess, float workload,
                        ModelInstanceConfig* config, float* occupancy);

  void LoadModel(const ModelInstanceConfig& config);
  
  void LoadModel(const YAML::Node& model_info);

  void UnloadModel(const std::string& model_sess_id);

  CtrlStatus UpdateModelTable();

  void GetModelSessions(std::vector<std::string>* sessions);

  void UpdateStats(const BackendStatsProto& backend_stats);

  float GetModelThroughput(const std::string& model_sess_id) const;

  float GetModelRps(const std::string& model_sess_id) const;

  bool IsAlive();

  bool IsIdle();

 private:
  Scheduler* scheduler_;
  uint32_t node_id_;
  std::string server_address_;
  std::string rpc_address_;
  std::string gpu_device_;
  size_t gpu_available_memory_;
  int beacon_sec_;
  int epoch_sec_;
  long timeout_ms_;
  int workload_id_;
  std::unique_ptr<BackendCtrl::Stub> stub_;
  /*! \brief Mapping from model session id to instance config */
  std::unordered_map<std::string, ModelInstanceConfig> model_table_config_;
  /*! \brief Mapping from model session id to incoming request rate */
  std::unordered_map<std::string, EWMA> model_rps_;
  float exec_cycle_us_;
  float duty_cycle_us_;
  /*! \brief Indicates whether model table is dirty. */
  bool dirty_model_table_;
  std::chrono::time_point<std::chrono::system_clock> last_time_;
};

} // namespace scheduler
} // namespace nexus

#endif // NEXUS_SCHEDULER_BACKEND_RPC_CLIENT_H_
