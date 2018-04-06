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

#include "nexus/common/model_def.h"
#include "nexus/common/model_profile.h"
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

/*! \brief Scheduler acts as a global centralized scheduler server. */
class Scheduler : public AsyncRpcServiceBase<AsyncService> {
 public:
  /*!
   * \brief Constructor for Scheduler object.
   * \param address IP address and port, e.g., 127.0.0.1:1234.
   * \param nthreads Number of threads that handle the RPC calls.
   * \param epoch Epoch time for scheduling in seconds.
   */
  Scheduler(std::string port, size_t nthreads, double epoch);
  /*!
   * \brief Loads the workload configuation for backends from config file.
   * \param config_file Config file path.
   */
  void LoadConfigFile(const std::string& config_file);
  /*!
   * \brief Returns timeout duration for backend server.
   * \return timeout duration in milliseconds
   */
  std::chrono::milliseconds BackendTimeout() { return epoch_ * 2; }
  /*!
   * \brief Returns timeout duration for frontend server.
   * \return timeout duration in milliseconds
   */
  std::chrono::milliseconds FrontendTimeout() { return epoch_ * 2; }
  /*! \brief starts the scheduler server */
  void Run();

  void Register(RpcCallBase* call, const RegisterRequest& request,
                RegisterReply* reply);

  void Unregister(RpcCallBase* call, const UnregisterRequest& request,
                  RpcReply* reply);
  
  void LoadModel(RpcCallBase* call, const LoadModelRequest& request,
                 LoadModelReply* reply);

  //void UnloadModel(

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
   * \brief Updates backends change history and broadcasts changes to all
   *   frontends.
   *
   * This function acquires mutex_.
   * \param adds Added backends.
   * \param removes Removed backends.
   */
  void onBackendsUpdate(const std::vector<BackendRpcClientPtr>& adds,
                        const std::vector<BackendRpcClientPtr>& removes);
  /*!
   * \brief Gets the snapshot of current backend pool.
   *
   * This function acquires mutex_.
   * \param update Protobuf message to fill in the backend pool.
   */
  void GetBackendPool(BackendsUpdate* update);

 private:
  /*! \brief Epoch duration in milliseconds */
  std::chrono::milliseconds epoch_;
  /*! \brief Static workload configuration */
  std::vector<std::unordered_map<ModelId, YAML::Node> > workloads_;
  /*! \brief Mapping from frontend node id to frontend client */
  std::unordered_map<uint32_t, FrontendRpcClientPtr> frontends_;
  /*! \brief Mapping from backend node id to backend client */
  std::unordered_map<uint32_t, BackendRpcClientPtr> backends_;
  /*! \brief Mapping from workload id to backend node id */
  std::unordered_map<int, uint32_t> assigned_workloads_;

  std::unordered_map<std::string, ModelRoute> model_routes_;
  std::unordered_map<std::string, std::vector<uint32_t> > model_subscribers_;
  /*! \brief Route table version number */
  std::uint32_t backend_pool_version_;
  /*! \brief Route table change list */
  std::unordered_map<uint32_t, BackendsUpdate> backends_updates_;
  /*! \brief Mutex for changing internal data */
  std::mutex mutex_;
};

} // namespace scheduler
} // namespace nexus


#endif // NEXUS_SCHEDULER_SCHEDULER_H_
