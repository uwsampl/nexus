#ifndef NEXUS_SCHEDULER_BACKEND_DELEGATE_H_
#define NEXUS_SCHEDULER_BACKEND_DELEGATE_H_

#include <chrono>
#include <grpc++/grpc++.h>
#include <mutex>
#include <unordered_map>
#include <vector>
#include <yaml-cpp/yaml.h>

#include "nexus/common/metric.h"
#include "nexus/common/model_db.h"
#include "nexus/common/model_def.h"
#include "nexus/proto/control.grpc.pb.h"
#include "nexus/scheduler/sch_info.h"

namespace nexus {
namespace scheduler {

class Scheduler;

using InstanceInfoPtr = std::shared_ptr<InstanceInfo>;

class BackendDelegate {
 public:
  BackendDelegate(uint32_t node_id, const std::string& server_addr,
                  const std::string& rpc_addr, const std::string& gpu_device,
                  size_t gpu_available_memory, int beacon_sec, int epoch_sec);

  uint32_t node_id() const { return node_id_; }

  std::string server_address() const { return server_address_; }

  std::string rpc_address() const { return rpc_address_; }
 
  std::string gpu_device() const { return gpu_device_; }

  size_t gpu_available_memory() const { return gpu_available_memory_; }

  int workload_id() const { return workload_id_; }

  void set_workload_id(int id) { workload_id_ = id; }

  float overload() const { return overload_; }

  float Occupancy() const;

  void GetInfo(BackendInfo* info) const;

  std::time_t LastAliveTime() const;

  void Tick();

  bool Assign(const BackendDelegate& other);

  bool PrepareLoadModel(const ModelSession& model_sess, float workload,
                        InstanceInfo* inst_info, float* occupancy) const;

  void LoadModel(const InstanceInfo& inst_info);
  
  void LoadModel(const YAML::Node& model_info);

  void LoadPrefixModel(const ModelSession& model_session,
                       const ModelSession& shared_session);

  void UnloadModel(const std::string& model_sess_id);
  /*!
   * \brief Update model throughput given model session id and throughput.
   * \param model_sess_id Model session ID.
   * \param throughput Expected throughput to be achieved.
   * \return Left over throughput if expected throughput is not achieved,
   *   otherwise 0.
   */
  float UpdateModelThroughput(const std::string& model_sess_id,
                              float throughput);

  void SpillOutWorkload(std::vector<std::pair<SessionGroup, float> >* spillout);

  CtrlStatus UpdateModelTableRpc();

  void UpdateStats(const BackendStatsProto& backend_stats);

  void AllModelSessions(std::vector<std::string>* sessions) const;

  const InstanceInfo* GetInstanceInfo(const std::string& model_sess_id) const;

  float GetModelThroughput(const std::string& model_sess_id) const;

  float GetModelRps(const std::string& model_sess_id) const;

  bool IsAlive();

  bool IsIdle() const;

 private:
  void ComputeBatchSize(InstanceInfo* inst_info, float workload) const;
  
  void UpdateCycle();
  
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

  std::vector<InstanceInfoPtr> model_instances_;
  /*!
   * \brief Mapping from model session id to instance information.
   * It's possible that multiple model session ids mapping to same instance
   * info due to prefix batching.
   */
  std::unordered_map<std::string, InstanceInfoPtr> session_instance_map_;
  /*!
   * \brief Mapping from model session id to incoming request rate.
   * Mapping could be multiple to one.
   */
  std::unordered_map<std::string, std::shared_ptr<EWMA> > model_rps_;
  float exec_cycle_us_;
  float duty_cycle_us_;
  bool overload_;
  /*! \brief Indicates whether model table is dirty. */
  bool dirty_model_table_;
  std::chrono::time_point<std::chrono::system_clock> last_time_;
};

} // namespace scheduler
} // namespace nexus

#endif // NEXUS_SCHEDULER_BACKEND_DELEGATE_H_
