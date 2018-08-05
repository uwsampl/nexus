#ifndef NEXUS_BACKEND_BACKEND_SERVER_H_
#define NEXUS_BACKEND_BACKEND_SERVER_H_

#include <atomic>
#include <memory>
#include <random>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>
#include <yaml-cpp/yaml.h>

#include "nexus/backend/backup_client.h"
#include "nexus/backend/gpu_executor.h"
#include "nexus/backend/model_exec.h"
#include "nexus/backend/rpc_service.h"
#include "nexus/backend/task.h"
#include "nexus/backend/worker.h"
#include "nexus/common/backend_pool.h"
#include "nexus/common/block_queue.h"
#include "nexus/common/model_def.h"
#include "nexus/common/server_base.h"
#include "nexus/common/spinlock.h"
#include "nexus/proto/control.grpc.pb.h"

namespace nexus {
namespace backend {

/*!
 * \brief Backend server runs on top of a GPU, handles queries from frontends,
 *   and executes model instances on GPU.
 */
class BackendServer : public ServerBase, public MessageHandler {
 public:
  using ModelTable = std::unordered_map<std::string, ModelExecutorPtr>;
  
  /*!
   * \brief Constructs a backend server
   * \param port Port number for receiving requests
   * \param rpc_port Port number for RPC server and control messages
   * \param sch_addr Scheduler IP address, if no port specified, use default port 10001
   * \param num_workers Number of worker threads
   * \param gpu_id GPU device ID
   * \param model_db_root Model database root directory path
   */
  BackendServer(std::string port, std::string rpc_port, std::string sch_addr,
                size_t num_workers, int gpu_id);
  /*! \brief Deconstructs backend server */
  ~BackendServer();
  /*! \brief Get backend node ID */
  uint32_t node_id() const { return node_id_; }
  /*! \brief Get GPU device ID */
  int gpu_id() const { return gpu_id_; }
  /*! \brief Starts the backend server */
  void Run() final;
  /*! \brief Stops the backend server */
  void Stop() final;
  /*! \brief Accepts a new connection */
  void HandleAccept() final;
  /*!
   * \brief Handles a new message
   * \param conn Connection that receives the message
   * \param message Received message
   */
  void HandleMessage(std::shared_ptr<Connection> conn,
                     std::shared_ptr<Message> message) final;
  /*!
   * \brief Handles error in connection
   * \param conn Connection that encounters an error
   * \param ec Boost error code
   */
  void HandleError(std::shared_ptr<Connection> conn,
                   boost::system::error_code ec) final;
  /*!
   * \brief Updates model table
   * \param req Update model table requests
   * \param reply Replies to update model tabel requests
   */
  void UpdateModelTable(const ModelTableConfig& req, RpcReply* reply);
  /*!
   * \brief Gets the model instance given model session ID
   * \param model_session_id Model session ID
   * \return Model instance pointer
   */
  ModelExecutorPtr GetModel(const std::string& model_session_id);
  /*!
   * \brief Gets all model instances loaded in the backend server
   * \return All model instances
   */
  ModelTable GetModelTable();
  /*!
   * \brief Get backup client given backend id.
   * \param backend_id Node id of backup backend
   * \return Backup client
   */
  std::shared_ptr<BackupClient> GetBackupClient(uint32_t backend_id);
  /*! \brief Returns the current server utilization. */
  inline double CurrentUtilization() const {
    return gpu_executor_->CurrentUtilization();
  }

 private:
  /*! \brief Daemon thread that sends stats to scheduler periodically. */
  void Daemon();
  /*! \brief Register this backend server to global scheduler. */
  void Register();
  /*! \brief Unregister this backend server to global scheduler. */
  void Unregister();
  /*!
   * \brief Send model workload history to global scheduler.
   * \param request Workload history protobuf.
   */
  void UpdateBackendStats(const BackendStatsProto& request);

 private:
  /*! \brief GPU device index */
  int gpu_id_;
  /*! \brief Interval to update stats to scheduler in seconds */
  uint32_t beacon_interval_sec_;
  /*! \brief Flag for whether backend and daemon thread is running */
  std::atomic_bool running_;
  /*! \brief Backend node id */
  uint32_t node_id_;
  /*! \brief Backend RPC service */
  BackendRpcService rpc_service_;
  /*! \brief RPC client for sending requests to scheduler */
  std::unique_ptr<SchedulerCtrl::Stub> sch_stub_;
  /*! \brief Daemon thread */
  std::thread daemon_thread_;
  /*! \brief Frontend connection pool. Guraded by frontend_mutex_. */
  std::set<std::shared_ptr<Connection> > frontend_connections_;
  /*! \brief Mutex for frontend_connections_ */
  std::mutex frontend_mutex_;
  /*! \brief Task queue for workers to work on */
  BlockPriorityQueue<Task> task_queue_;
  /*! \brief Worker thread pool */
  std::vector<std::unique_ptr<Worker> > workers_;
  /*! \brief GPU executor */
  std::unique_ptr<GpuExecutor> gpu_executor_;
  /*!
   * \brief Mapping from model session ID to model instance.
   * Guarded by model_table_mu_.p
   */
  ModelTable model_table_;
  /*! \brief Mutex for accessing model_table_ */
  std::mutex model_table_mu_;
  /*! \brief Backend pool for backup servers. */
  BackendPool backend_pool_;
  /*! \brief Random number genertor */
  std::random_device rd_;
  std::mt19937 rand_gen_;
};

} // namespace backend
} // namespace nexus

#endif // NEXUS_BACKEND_BACKEND_SERVER_H_
