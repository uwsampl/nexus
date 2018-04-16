#ifndef NEXUS_SCHEDULER_BACKEND_RPC_CLIENT_H_
#define NEXUS_SCHEDULER_BACKEND_RPC_CLIENT_H_

#include <chrono>
#include <grpc++/grpc++.h>
#include <mutex>
#include <unordered_map>
#include <vector>
#include <yaml-cpp/yaml.h>

#include "nexus/common/model_def.h"
#include "nexus/proto/control.grpc.pb.h"

namespace nexus {
namespace scheduler {

class Scheduler;

struct ModelStats {
  std::vector<uint32_t> rps;
  float avg_rps;
  float rps_std;

  ModelStats() : avg_rps(0.), rps_std(0.) {}
};

class BackendRpcClient {
 public:
  BackendRpcClient(
      Scheduler* sch, uint32_t node_id, const std::string& server_addr,
      const std::string& rpc_addr, const std::string& gpu_device,
      size_t gpu_available_memory, std::chrono::seconds timeout);

  uint32_t node_id() const { return node_id_; }

  std::string server_address() const { return server_address_; }

  std::string rpc_address() const { return rpc_address_; }
 
  std::string gpu_device() const { return gpu_device_; }

  size_t gpu_available_memory() const { return gpu_available_memory_; }

  int workload_id() const { return workload_id_; }

  void set_workload_id(int id) { workload_id_ = id; }

  std::time_t LastAliveTime();

  void PrepareLoadModel(const ModelSession& model_sess, float workload,
                        ModelInstanceConfig* config, float* occupancy);

  void LoadModel(const ModelInstanceConfig& config);
  
  void LoadModel(const YAML::Node& model_info);

  CtrlStatus UpdateModelTable();

  void GetModelTable(ModelTableConfig* model_table_config);

  void UpdateStats(const BackendStatsProto& stats);

  bool IsAlive();

  bool IsIdle();

 private:
  void GetModelTableNoLock(ModelTableConfig* model_table_config);

 private:
  Scheduler* scheduler_;
  uint32_t node_id_;
  std::string server_address_;
  std::string rpc_address_;
  std::string gpu_device_;
  size_t gpu_available_memory_;
  std::chrono::seconds timeout_;
  int workload_id_;
  std::unique_ptr<BackendCtrl::Stub> stub_;
  /*! \brief map from model session id to incoming request rate */
  std::unordered_map<std::string, float> workloads_;
  //std::unordered_map<std::string, 
  float exec_cycle_us_;
  float duty_cycle_us_;
  /*! \brief List of models that loaded in the backend. */
  std::vector<ModelInstanceConfig> model_table_config_;
  /*! \brief Indicates whether model table is dirty. */
  bool dirty_model_table_;
  /*! \brief Mutex for entire class */
  std::mutex mutex_;
  std::chrono::time_point<std::chrono::system_clock> last_time_;
};

} // namespace scheduler
} // namespace nexus

#endif // NEXUS_SCHEDULER_BACKEND_RPC_CLIENT_H_
