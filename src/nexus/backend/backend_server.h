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

#include "backend/gpu_executor.h"
#include "backend/model_ins.h"
#include "backend/rpc_service.h"
#include "backend/scheduler_client.h"
#include "backend/task.h"
#include "backend/worker.h"
#include "common/block_queue.h"
#include "common/model_def.h"
#include "common/server_base.h"
#include "common/spinlock.h"

namespace nexus {
namespace backend {

// Backend server
class BackendServer : public ServerBase, public MessageHandler {
 public:
  BackendServer(std::string port, std::string rpc_port, std::string sch_addr,
                size_t num_workers, int gpu_id);

  ~BackendServer();

  uint32_t node_id() const { return node_id_; }

  int gpu_id() const { return gpu_id_; }

  std::string rpc_port() const { return rpc_service_.port(); }

  void Run() final;

  void Stop() final;

  void HandleAccept() final;

  void HandleMessage(std::shared_ptr<Connection> conn,
                     std::shared_ptr<Message> message) final;

  void HandleError(std::shared_ptr<Connection> conn,
                   boost::system::error_code ec) final;

  void LoadModelDb(const std::string& db_file);

  void LoadConfigFromFile(const std::string& config_file);

  void UpdateModelTable(const ModelTable& req, RpcReply* reply);

  ModelInstancePtr GetModelInstance(const std::string& model_session_id);

  std::vector<ModelInstancePtr> GetAllModelInstances();

 private:
  /*! \brief indicator whether backend is running */
  std::atomic_bool running_;
  // Backend node id
  uint32_t node_id_;
  // Backend RPC service
  BackendRpcService rpc_service_;
  // Backend controller
  SchedulerClient sch_client_;
  // frontend connections
  std::set<std::shared_ptr<Connection> > frontend_connections_;
  std::mutex frontend_mutex_;
  // GPU device index
  int gpu_id_;
  // BlockQueue that connections append new tasks and workers pull tasks
  BlockPriorityQueue<Task> task_queue_;
  // Vector of workers
  std::vector<std::unique_ptr<Worker> > workers_;
  // Vector of GPU executors
  std::unique_ptr<GpuExecutor> gpu_executor_;
  // Lock for modifying this object
  Spinlock model_table_lock_;
  // Map from framework to default root directory
  std::unordered_map<Framework, std::string> framework_rootdir_;
  // Map from (framework, model_name) to metadata of a model
  std::unordered_map<ModelId, YAML::Node> model_database_;
  // Map from (framework, model_name) to a loaded model
  std::vector<ModelInstancePtr> model_instances_;
  std::unordered_map<std::string, ModelInstancePtr> model_session_map_;
  // random number genertor
  std::random_device rd_;
  std::mt19937 rand_gen_;
};

} // namespace backend
} // namespace nexus

#endif // NEXUS_BACKEND_BACKEND_SERVER_H_
