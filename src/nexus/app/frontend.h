#ifndef NEXUS_APP_FRONTEND_H_
#define NEXUS_APP_FRONTEND_H_

#include <atomic>
#include <memory>
#include <mutex>
#include <random>
#include <unordered_map>
#include <unordered_set>

#include "app/rpc_service.h"
#include "app/scheduler_client.h"
#include "app/user_session.h"
#include "app/worker.h"
#include "common/backend_pool.h"
#include "common/block_queue.h"
#include "common/connection.h"
#include "common/model_def.h"
#include "common/model_handler.h"
#include "common/server_base.h"
#include "common/spinlock.h"
#include "proto/control.pb.h"
#include "proto/nnquery.pb.h"

namespace nexus {
namespace app {

class Frontend : public ServerBase, public MessageHandler {
 public:
  Frontend(std::string port, std::string rpc_port, std::string sch_addr,
           size_t nthreads);

  ~Frontend();

  virtual void Process(const RequestProto& request, ReplyProto* reply) = 0;

  uint32_t node_id() const { return node_id_; }

  std::string rpc_port() const { return rpc_service_.port(); }

  void Run();

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

  void UpdateBackends(const BackendsUpdate& request,
                      BackendsUpdateReply* reply);

  void UpdateModelRoutes(const ModelRouteList& routes, RpcReply* reply);

  std::shared_ptr<UserSession> GetUserSession(uint32_t uid);

  std::shared_ptr<ModelHandler> LoadModel(const LoadModelRequest& req);

 private:
  void RegisterUser(std::shared_ptr<UserSession> user_sess,
                    const RequestProto& request, ReplyProto* reply);

 private:
  /*! \brief Indicator whether backend is running */
  std::atomic_bool running_;
  /*! \brief Node id */
  uint32_t node_id_;
  /*! \brief RPC service */
  RpcService rpc_service_;
  /*! \brief RPC client connected to scheduler */
  SchedulerClient sch_client_;
  /*! \brief Backend pool */
  BackendPool backend_pool_;
  /*! \brief Blocking queue for requests */
  BlockQueue<Message> request_queue_;
  /*! \brief Worker pool for processing requests */
  std::vector<std::unique_ptr<Worker> > workers_;
  /*! \brief User connection pool, protected by user_mutex_ */
  std::unordered_set<std::shared_ptr<Connection> > connection_pool_;
  /*! \brief Map from user id to user session, protected by user_mutex_ */
  std::unordered_map<uint32_t, std::shared_ptr<UserSession> > user_sessions_;
  /*! \brief Mutex for connection_pool_ and user_sessions_ */
  std::mutex user_mutex_;
  /*! \brief Model handlers that loaded by the app */
  std::unordered_map<std::string, std::shared_ptr<ModelHandler> > model_pool_;
  /*! \brief Mutex for model_pool_ */
  std::mutex model_pool_mu_;
  /*! \brief Random number generator */
  std::random_device rd_;
  std::mt19937 rand_gen_;
};

} // namespace app
} // namespace nexus

#endif // NEXUS_APP_APP_BASE_H_
