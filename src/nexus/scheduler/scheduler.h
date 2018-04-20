#ifndef NEXUS_SCHEDULER_SCHEDULER_H_
#define NEXUS_SCHEDULER_SCHEDULER_H_

#include <chrono>
#include <deque>
#include <grpc++/grpc++.h>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#include <yaml-cpp/yaml.h>

#include "nexus/common/rpc_call.h"
#include "nexus/common/rpc_service_base.h"
#include "nexus/proto/control.grpc.pb.h"
#include "nexus/scheduler/backend_rpc_client.h"
#include "nexus/scheduler/frontend_rpc_client.h"

namespace nexus {
namespace scheduler {

using AsyncService = nexus::SchedulerCtrl::AsyncService;
using BackendRpcClientPtr = std::shared_ptr<BackendRpcClient>;
using FrontendRpcClientPtr = std::shared_ptr<FrontendRpcClient>;
using ServerList = std::unordered_set<uint32_t>;

struct ModelInfo {
  std::unordered_map<uint32_t, double> backend_throughputs;
  std::vector<double> rps_history;

  double total_throughput() const {
    double total = 0.;
    for (auto iter : backend_throughputs) {
      total += iter.second;
    }
    return total;
  }
};

/*! \brief Scheduler acts as a global centralized scheduler server. */
class Scheduler : public AsyncRpcServiceBase<AsyncService> {
 public:
  /*!
   * \brief Constructor for Scheduler object.
   * \param address IP address and port, e.g., 127.0.0.1:1234.
   * \param nthreads Number of threads that handle the RPC calls.
   * \param beacon_interval Beacon interval in seconds.
   * \param epoch_interval Epoch interval in seconds.
   */
  Scheduler(std::string port, size_t nthreads, std::string db_root_dir,
            int beacon_interval, int epoch_interval);
  /*!
   * \brief Loads the workload configuation for backends from config file.
   * \param config_file Config file path.
   */
  void LoadWorkloadFile(const std::string& workload_file);
  /*! \brief Starts the scheduler main thread that monitors the server
   *     aliveness and changes in workload. */
  void Run();

  void Register(RpcCallBase* call, const RegisterRequest& request,
                RegisterReply* reply);

  void Unregister(RpcCallBase* call, const UnregisterRequest& request,
                  RpcReply* reply);
  
  void LoadModel(RpcCallBase* call, const LoadModelRequest& request,
                 LoadModelReply* reply);

  //void UnloadModel(

  void UpdateBackendStats(RpcCallBase* call, const BackendStatsProto& request,
                          RpcReply* reply);

  void KeepAlive(RpcCallBase* call, const KeepAliveRequest& request,
                 RpcReply* reply);

 private:
  /*! \brief Initializes RPC handlers. */
  void HandleRpcs() final;
  /*!
   * \brief Registers frontend RPC client and fills in the register reply.
   *
   *    This function acquires mutex_.
   * \param frontend Frontend RPC client pointer.
   * \param reply Register reply.
   */
  void RegisterFrontend(FrontendRpcClientPtr frontend, RegisterReply* reply);
  /*!
   * \brief Registers backend RPC client and fills in the register reply.
   *
   *    This function acquires mutex_.
   * \param backend Backend RPC client pointer.
   * \param reply Register reply.
   * \note 
   */
  void RegisterBackend(BackendRpcClientPtr backend, RegisterReply* reply);
  /*!
   * \brief Unregister frontend RPC client and fills in the register reply
   *
   *    This function acquires mutex_.
   * \param node_id Frontend node ID.
   */
  void UnregisterFrontend(uint32_t node_id);
  /*!
   * \brief Unregister frontend RPC client and fills in the register reply
   *
   *    This function acquires mutex_.
   * \param node_id Backend node ID.
   */
  void UnregisterBackend(uint32_t node_id);
  /*!
   * \brief Update workload to the new added backend
   *
   *   This function doesn't acquire mutex_.
   * \param backend Backend client pointer
   */
  void AddBackend(BackendRpcClientPtr backend);
  /*!
   * \brief Assign the workload of the removed backend to other idle ones.
   *
   *   This function doesn't acquire mutex_.
   * \param backend Backend client pointer
   */
  void RemoveBackend(BackendRpcClientPtr backend);
  /*!
   * \brief Update the model subscribers, and potentially remove the model
   *   sessions if no one subscribes it.
   *
   *   This function doesn't acquire mutex_.
   * \param backend Backend client pointer
   */
  void RemoveFrontend(FrontendRpcClientPtr frontend);
  /*!
   * \brief Get backend rpc client given the node id.
   *
   *   This function doesn't acquire mutex_.
   * \param node_id Backend node id.
   * \return BackendRpcClient pointer if found, otherwise nullptr
   */
  BackendRpcClientPtr GetBackend(uint32_t node_id);
  /*!
   * \brief Get frontend rpc client given the node id.
   *
   *   This function doesn't acquire mutex_.
   * \param node_id Frontend node id.
   * \return FrontendRpcClient pointer if found, otherwise nullptr
   */
  FrontendRpcClientPtr GetFrontend(uint32_t node_id);

  void GetModelRoute(const std::string& model_session_id,
                     ModelRouteProto* route);


  void FindBestBackend(const ModelSession& model_sess, float workload,
                       const std::unordered_set<uint32_t>& skips,
                       BackendRpcClientPtr* best_backend,
                       ModelInstanceConfig* inst_cfg);
  /*!
   * \brief At each beacon cycle, check whether frontends and backends are
   *   alive, and aggregate model session request rates from backends.
   *
   *   This function acquires mutex_.
   */
  void BeaconCheck();
  /*!
   * \brief At each epoch cycle, re-schedule the resources for all model
   *   sessions based on the request rates during last epoch
   *
   *   This function acquires mutex_.
   */
  void EpochSchedule();

 private:
  /*! \brief Beacon interval in seconds */
  int beacon_interval_sec_;
  /*! \brief Epoch duration in seconds */
  int epoch_interval_sec_;
  /*! \brief Static workload configuration */
  std::vector<std::vector<YAML::Node> > workloads_;
  /*! \brief Mapping from frontend node id to frontend client */
  std::unordered_map<uint32_t, FrontendRpcClientPtr> frontends_;
  /*! \brief Mapping from backend node id to backend client */
  std::unordered_map<uint32_t, BackendRpcClientPtr> backends_;
  /*! \brief Mapping from workload id to backend node id */
  std::unordered_map<int, uint32_t> assigned_workloads_;
  /*! \brief Mapping from model session ID to model information */
  std::unordered_map<std::string, ModelInfo> model_table_;
  /*! \brief Mapping from model session ID to subscribed frontends */
  std::unordered_map<std::string, ServerList> model_subscribers_;
  /*! \brief Mutex for accessing internal data */
  std::mutex mutex_;
};

} // namespace scheduler
} // namespace nexus


#endif // NEXUS_SCHEDULER_SCHEDULER_H_
