#ifndef NEXUS_APP_FRONTEND_H_
#define NEXUS_APP_FRONTEND_H_

#include <atomic>
#include <memory>
#include <mutex>
#include <random>
#include <unordered_map>
#include <unordered_set>

#include "nexus/app/model_handler.h"
#include "nexus/app/query_processor.h"
#include "nexus/app/request_context.h"
#include "nexus/app/rpc_service.h"
#include "nexus/app/user_session.h"
#include "nexus/app/worker.h"
#include "nexus/common/backend_pool.h"
#include "nexus/common/block_queue.h"
#include "nexus/common/connection.h"
#include "nexus/common/model_def.h"
#include "nexus/common/server_base.h"
#include "nexus/common/spinlock.h"
#include "nexus/proto/control.grpc.pb.h"
#include "nexus/proto/nnquery.pb.h"

namespace nexus {
namespace app {

class Frontend : public ServerBase, public MessageHandler {
 public:
  Frontend(std::string port, std::string rpc_port, std::string sch_addr);

  ~Frontend();

  //virtual void Process(const RequestProto& request, ReplyProto* reply) = 0;

  uint32_t node_id() const { return node_id_; }

  std::string rpc_port() const { return rpc_service_.port(); }

  void Run(QueryProcessor* qp, size_t nthreads);

  void Stop();
  /*! \brief Accepts new user connection */
  void HandleAccept() final;
  /*!
   * \brief Handles new messages from user or backend connections
   * \param conn Shared pointer of Connection
   * \param message Received message
   */
  void HandleMessage(std::shared_ptr<Connection> conn,
                     std::shared_ptr<Message> message) final;
  /*!
   * \brief Handles connection error
   * \param conn Shared pointer of Connection
   * \param ec Boost error code
   */
  void HandleError(std::shared_ptr<Connection> conn,
                   boost::system::error_code ec) final;

  void UpdateModelRoutes(const ModelRouteUpdates& request, RpcReply* reply);

  std::shared_ptr<UserSession> GetUserSession(uint32_t uid);

  std::shared_ptr<ModelHandler> LoadModel(const LoadModelRequest& req);

 private:
  void Register();

  void Unregister();

  void KeepAlive();

  bool UpdateBackendPoolAndModelRoute(const ModelRouteProto& route);

  void RegisterUser(std::shared_ptr<UserSession> user_sess,
                    const RequestProto& request, ReplyProto* reply);

  void Daemon();

  void ReportWorkload(const WorkloadStatsProto& request);

 private:
  /*! \brief Indicator whether backend is running */
  std::atomic_bool running_;
  /*! \brief Interval to update stats to scheduler in seconds */
  uint32_t beacon_interval_sec_;
  /*! \brief Frontend node ID */
  uint32_t node_id_;
  /*! \brief RPC service */
  RpcService rpc_service_;
  /*! \brief RPC client connected to scheduler */
  std::unique_ptr<SchedulerCtrl::Stub> sch_stub_;
  /*! \brief Backend pool */
  BackendPool backend_pool_;
  /*!
   * \brief Map from backend ID to model sessions servered at this backend.
   * Guarded by model_pool_mu_
   */
  std::unordered_map<uint32_t,
                     std::unordered_set<std::string> > backend_sessions_;
  /*! \brief Request pool */
  RequestPool request_pool_;
  /*! \brief Worker pool for processing requests */
  std::vector<std::unique_ptr<Worker> > workers_;
  /*! \brief User connection pool. Guarded by user_mutex_. */
  std::unordered_set<std::shared_ptr<Connection> > connection_pool_;
  /*! \brief Map from user id to user session. Guarded by user_mutex_. */
  std::unordered_map<uint32_t, std::shared_ptr<UserSession> > user_sessions_;
  /*!
   * \brief Map from model session ID to model handler.
   */
  std::unordered_map<std::string, std::shared_ptr<ModelHandler> > model_pool_;

  std::thread daemon_thread_;
  /*! \brief Mutex for connection_pool_ and user_sessions_ */
  std::mutex user_mutex_;
  /*! \brief Random number generator */
  std::random_device rd_;
  std::mt19937 rand_gen_;
};

} // namespace app
} // namespace nexus

#endif // NEXUS_APP_APP_BASE_H_
