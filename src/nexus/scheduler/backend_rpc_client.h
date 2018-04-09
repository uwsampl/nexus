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

class BackendRpcClient {
 public:
  BackendRpcClient(
      Scheduler* sch, uint32_t node_id, const std::string& server_addr,
      const std::string& rpc_addr, const std::string& gpu_device,
      size_t gpu_available_memory, std::chrono::milliseconds timeout);

  uint32_t node_id() const { return node_id_; }

  std::string server_address() const { return server_address_; }

  std::string rpc_address() const { return rpc_address_; }
 
  std::string gpu_device() const { return gpu_device_; }

  size_t gpu_available_memory() const { return gpu_available_memory_; }

  int workload_id() const { return workload_id_; }

  void set_workload_id(int id) { workload_id_ = id; }

  void Tick();

  std::time_t LastTime();

  void PrepareLoadModel(const ModelSession& model_sess, float workload,
                        ModelInstanceDesc* model_desc, float* occupancy);

  void LoadModel(const ModelInstanceDesc& model_desc);

  void LoadModel(const YAML::Node& model_info);

  CtrlStatus UpdateModelTable();

  void GetModelTable(ModelTable* model_table);

  bool IsAlive();

  bool IsIdle();

 private:
  grpc::Status UpdateModelTableRpc(const ModelTable& request, RpcReply* reply);

  grpc::Status CheckAliveRpc(const CheckAliveRequest& request, RpcReply* reply);

  void GetModelTableNoLock(ModelTable* model_table);

 private:
  Scheduler* scheduler_;
  uint32_t node_id_;
  std::string server_address_;
  std::string rpc_address_;
  std::string gpu_device_;
  size_t gpu_available_memory_;
  std::chrono::milliseconds timeout_;
  int workload_id_;
  std::unique_ptr<BackendCtrl::Stub> stub_;
  /*! \brief map from model session id to incoming request rate */
  std::unordered_map<std::string, float> workloads_;
  float exec_cycle_;
  float duty_cycle_;
  /*! \brief List of models that loaded in the backend. */
  std::vector<ModelInstanceDesc> model_table_;
  /*! \brief Indicates whether model table is dirty. */
  bool dirty_model_table_;
  /*! \brief Mutex for entire class */
  std::mutex mutex_;
  std::chrono::time_point<std::chrono::system_clock> last_time_;
};

} // namespace scheduler
} // namespace nexus

#endif // NEXUS_SCHEDULER_BACKEND_RPC_CLIENT_H_
