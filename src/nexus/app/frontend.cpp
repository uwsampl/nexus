#include <limits>

#include "nexus/app/frontend.h"

namespace nexus {
namespace app {

Frontend::Frontend(std::string port, std::string rpc_port, std::string sch_addr,
                   size_t nthreads):
    ServerBase(port),
    rpc_service_(this, rpc_port),
    sch_client_(this, sch_addr),
    backend_pool_(io_service_, this),
    rand_gen_(rd_()) {
  // Start RPC service
  rpc_service_.Start();
  // Generate a random Node ID and register itself to scheduler
  std::uniform_int_distribution<uint32_t> dis(
      1, std::numeric_limits<uint32_t>::max());
  node_id_ = dis(rand_gen_);
  while (true) {
    CtrlStatus ret = sch_client_.Register();
    if (ret == CTRL_OK) {
      break;
    }
    if (ret != CTRL_FRONTEND_NODE_ID_CONFLICT) {
      LOG(FATAL) << "Failed to register frontend server to scheduler: " <<
          CtrlStatus_Name(ret);
    }
    node_id_ = dis(rand_gen_);
  }
  // Init message processors
  for (size_t i = 0; i < nthreads; ++i) {
    std::unique_ptr<Worker> worker(new Worker(this, request_queue_));
    worker->Start();
    workers_.push_back(std::move(worker));
  }
}

Frontend::~Frontend() {
  if (running_) {
    Stop();
  }
}

void Frontend::Run() {
  running_ = true;
  LOG(INFO) << "Frontend server (id: " << node_id_ << ") is listening on " <<
      address();
  io_service_.run();
}

void Frontend::Stop() {
  running_ = false;
  // Unregister frontend
  sch_client_.Unregister();;
  // Stop all accept new connections
  ServerBase::Stop();
  // Stop all frontend connections
  for (auto conn: connection_pool_) {
    conn->Stop();
  }
  connection_pool_.clear();
  user_sessions_.clear();
  // Stop all backend connections
  backend_pool_.StopAll();
  // Stop message processors
  for (auto& worker : workers_) {
    worker->Stop();
  }
  // Stop RPC service
  rpc_service_.Stop();
  LOG(INFO) << "Frontend server stopped";
}

void Frontend::HandleAccept() {
  auto conn = std::make_shared<UserSession>(std::move(socket_), this);
  connection_pool_.insert(conn);
  conn->Start();
}

void Frontend::HandleMessage(std::shared_ptr<Connection> conn,
                                   std::shared_ptr<Message> message) {
  switch (message->type()) {
    case kUserRegister: {
      auto user_sess = std::dynamic_pointer_cast<UserSession>(conn);
      if (user_sess == nullptr) {
        LOG(ERROR) << "UserRequest message comes from non-user connection";
        break;
      }
      RequestProto request;
      ReplyProto reply;
      message->DecodeBody(&request);
      RegisterUser(user_sess, request, &reply);
      auto reply_msg = std::make_shared<Message>(kUserReply,
                                                 reply.ByteSizeLong());
      reply_msg->EncodeBody(reply);
      user_sess->Write(reply_msg);
      break;
    }
    case kUserRequest: {
      auto user_sess = std::dynamic_pointer_cast<UserSession>(conn);
      if (user_sess == nullptr) {
        LOG(ERROR) << "UserRequest message comes from non-user connection";
        break;
      }
      request_queue_.push(std::move(message));
      break;
    }
    case kBackendReply: {
      QueryResultProto result;
      message->DecodeBody(&result);
      std::string model_session_id = result.model_session_id();
      auto itr = model_pool_.find(model_session_id);
      if (itr == model_pool_.end()) {
        LOG(ERROR) << "Cannot find model handler for " << model_session_id;
        break;
      }
      itr->second->HandleResult(result);
      break;
    }
    default: {
      LOG(ERROR) << "Wrong message type: " << message->type();
      // TODO: handle wrong type
      break;
    }
  }
}

void Frontend::HandleError(std::shared_ptr<Connection> conn,
                                 boost::system::error_code ec) {
  if (auto backend_conn = std::dynamic_pointer_cast<BackendSession>(conn)) {
    if (ec == boost::asio::error::eof ||
        ec == boost::asio::error::connection_reset) {
      // backend disconnects
    } else {
      LOG(ERROR) << "Backend connection error (" << ec << "): " << ec.message();
    }
    backend_pool_.RemoveBackend(backend_conn->node_id());
  } else { // user_connection
    if (ec == boost::asio::error::eof ||
        ec == boost::asio::error::connection_reset) {
      // user disconnects
    } else {
      LOG(ERROR) << "User connection error (" << ec << "): " << ec.message();
    }
    auto user_sess = std::dynamic_pointer_cast<UserSession>(conn);
    std::lock_guard<std::mutex> lock(user_mutex_);
    connection_pool_.erase(conn);
    uint32_t uid = user_sess->user_id();
    user_sessions_.erase(uid);
    LOG(INFO) << "Remove user session " << uid;
    conn->Stop();
  }
}

void Frontend::UpdateBackends(const BackendsUpdate& request,
                                    BackendsUpdateReply* reply) {
  backend_pool_.UpdateBackends(request, reply);
}

void Frontend::UpdateModelRoutes(const ModelRouteList& routes,
                                       RpcReply* reply) {
  std::lock_guard<std::mutex> lock(model_pool_mu_);
  for (auto model_route : routes.model_route()) {
    std::string model_sess_id = ModelSessionToString(
        model_route.model_session());
    auto itr = model_pool_.find(model_sess_id);
    if (itr == model_pool_.end()) {
      LOG(ERROR) << "Cannot find model handler for " << model_sess_id;
      continue;
    }
    auto model_handler = itr->second;
    model_handler->UpdateRoute(model_route);
  }
  reply->set_status(CTRL_OK);
}

std::shared_ptr<UserSession> Frontend::GetUserSession(uint32_t uid) {
  std::lock_guard<std::mutex> lock(user_mutex_);
  auto itr = user_sessions_.find(uid);
  if (itr == user_sessions_.end()) {
    return nullptr;
  }
  return itr->second;
}

std::shared_ptr<ModelHandler> Frontend::LoadModel(const LoadModelRequest& req) {
  LoadModelReply reply;
  CtrlStatus ret = sch_client_.LoadModelRpc(req, &reply);
  if (ret != CTRL_OK) {
    LOG(ERROR) << "Scheduler unreachable";
    return nullptr;
  }
  if (reply.status() != CTRL_OK) {
    LOG(ERROR) << "Load model error: " << CtrlStatus_Name(reply.status());
    return nullptr;
  }
  auto model_handler = std::make_shared<ModelHandler>(req.model_session(),
                                                      backend_pool_);
  model_handler->UpdateRoute(reply.model_route());
  {
    std::lock_guard<std::mutex> lock(model_pool_mu_);
    model_pool_.emplace(model_handler->model_session_id(), model_handler);
  }
  return model_handler;
}

void Frontend::RegisterUser(
    std::shared_ptr<UserSession> user_sess, const RequestProto& request,
    ReplyProto* reply) {
  uint32_t uid = request.user_id();
  user_sess->set_user_id(uid);
  std::lock_guard<std::mutex> lock(user_mutex_);
  auto itr = user_sessions_.find(uid);
  if (itr == user_sessions_.end()) {
    LOG(INFO) << "New user session: " << uid;
    user_sessions_.emplace(uid, user_sess);
  } else if (itr->second != user_sess) {
    LOG(INFO) << "Update user session: " << uid;
    user_sessions_[uid] = user_sess;
  }
  reply->set_user_id(uid);
  reply->set_status(CTRL_OK);
}

} // namespace app
} // namespace nexus
