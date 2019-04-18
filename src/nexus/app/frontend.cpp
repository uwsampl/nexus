#include <limits>
#include <gflags/gflags.h>

#include "nexus/app/frontend.h"
#include "nexus/common/config.h"

DECLARE_int32(load_balance);

namespace nexus {
namespace app {

Frontend::Frontend(std::string port, std::string rpc_port,
                   std::string sch_addr) :
    ServerBase(port),
    rpc_service_(this, rpc_port),
    rand_gen_(rd_()) {
  // Start RPC service
  rpc_service_.Start();
  // Init scheduler client
  if (sch_addr.find(':') == std::string::npos) {
    // Add default scheduler port if no port specified
    sch_addr += ":" + std::to_string(SCHEDULER_DEFAULT_PORT);
  }
  auto channel = grpc::CreateChannel(sch_addr,
                                     grpc::InsecureChannelCredentials());
  sch_stub_ = SchedulerCtrl::NewStub(channel);
  // Init Node ID and register frontend to scheduler
  Register();
}

Frontend::~Frontend() {
  if (running_) {
    Stop();
  }
}

void Frontend::Run(QueryProcessor* qp, size_t nthreads) {
  for (size_t i = 0; i < nthreads; ++i) {
    std::unique_ptr<Worker> worker(new Worker(qp, request_pool_));
    worker->Start();
    workers_.push_back(std::move(worker));
  }
  running_ = true;
  daemon_thread_ = std::thread(&Frontend::Daemon, this);
  LOG(INFO) << "Frontend server (id: " << node_id_ << ") is listening on " <<
      address();
  io_context_.run();
}

void Frontend::Stop() {
  running_ = false;
  // Unregister frontend
  Unregister();
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
  // Stop workers
  for (auto& worker : workers_) {
    worker->Stop();
  }
  for (auto& worker : workers_) {
    worker->Join();
  }
  daemon_thread_.join();
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
      request_pool_.AddNewRequest(std::make_shared<RequestContext>(
          user_sess, message, request_pool_));
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
      itr->second->HandleReply(result);
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
    backend_pool_.RemoveBackend(backend_conn);
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
    VLOG(1) << "Remove user session " << uid;
    conn->Stop();
  }
}

void Frontend::UpdateModelRoutes(const ModelRouteUpdates& request,
                                 RpcReply* reply) {
  int success = true;
  for (auto model_route : request.model_route()) {
    if (!UpdateBackendPoolAndModelRoute(model_route)) {
      success = false;
    }
  }
  if (success) {
    reply->set_status(CTRL_OK);
  } else {
    reply->set_status(MODEL_SESSION_NOT_LOADED);
  }
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
  return LoadModel(req, LoadBalancePolicy(FLAGS_load_balance));
}

std::shared_ptr<ModelHandler> Frontend::LoadModel(const LoadModelRequest& req,
                                                  LoadBalancePolicy lb_policy) {
  LoadModelReply reply;
  grpc::ClientContext context;
  grpc::Status status = sch_stub_->LoadModel(&context, req, &reply);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to connect to scheduler: " <<
        status.error_message() << "(" << status.error_code() << ")";
    return nullptr;
  }
  if (reply.status() != CTRL_OK) {
    LOG(ERROR) << "Load model error: " << CtrlStatus_Name(reply.status());
    return nullptr;
  }
  auto model_handler = std::make_shared<ModelHandler>(
      reply.model_route().model_session_id(), backend_pool_, lb_policy);
  // Only happens at Setup stage, so no concurrent modification to model_pool_
  model_pool_.emplace(model_handler->model_session_id(), model_handler);
  UpdateBackendPoolAndModelRoute(reply.model_route());

  return model_handler;
}

void Frontend::Register() {
  // Init node id
  std::uniform_int_distribution<uint32_t> dis(
      1, std::numeric_limits<uint32_t>::max());
  node_id_ = dis(rand_gen_);

  // Prepare request
  RegisterRequest request;
  request.set_node_type(FRONTEND_NODE);
  request.set_node_id(node_id_);
  request.set_server_port(port());
  request.set_rpc_port(rpc_service_.port());
  
  while (true) {
    grpc::ClientContext context;
    RegisterReply reply;
    grpc::Status status = sch_stub_->Register(&context, request, &reply);
    if (!status.ok()) {
      LOG(FATAL) << "Failed to connect to scheduler: " <<
          status.error_message() << "(" << status.error_code() << ")";
    }
    CtrlStatus ret = reply.status();
    if (ret == CTRL_OK) {
      beacon_interval_sec_ = reply.beacon_interval_sec();
      return;
    }
    if (ret != CTRL_FRONTEND_NODE_ID_CONFLICT) {
      LOG(FATAL) << "Failed to register frontend to scheduler: " <<
          CtrlStatus_Name(ret);
    }
    // Frontend ID conflict, need to generate a new one
    node_id_ = dis(rand_gen_);
    request.set_node_id(node_id_);
  }
}

void Frontend::Unregister() {
  UnregisterRequest request;
  request.set_node_type(FRONTEND_NODE);
  request.set_node_id(node_id_);

  grpc::ClientContext context;
  RpcReply reply;
  grpc::Status status = sch_stub_->Unregister(&context, request, &reply);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to connect to scheduler: " <<
        status.error_message() << "(" << status.error_code() << ")";
    return;
  }
  CtrlStatus ret = reply.status();
  if (ret != CTRL_OK) {
    LOG(ERROR) << "Failed to unregister frontend: " << CtrlStatus_Name(ret);
  }
}

void Frontend::KeepAlive() {
  grpc::ClientContext context;
  KeepAliveRequest request;
  request.set_node_type(FRONTEND_NODE);
  request.set_node_id(node_id_);
  RpcReply reply;
  grpc::Status status = sch_stub_->KeepAlive(&context, request, &reply);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to connect to scheduler: " <<
        status.error_message() << "(" << status.error_code() << ")";
    return;
  }
  CtrlStatus ret = reply.status();
  if (ret != CTRL_OK) {
    LOG(ERROR) << "KeepAlive error: " << CtrlStatus_Name(ret);
  }
}

bool Frontend::UpdateBackendPoolAndModelRoute(const ModelRouteProto& route) {
  auto& model_session_id = route.model_session_id();
  LOG(INFO) << "Update model route for " << model_session_id;
  // LOG(INFO) << route.DebugString();
  auto iter = model_pool_.find(model_session_id);
  if (iter == model_pool_.end()) {
    LOG(ERROR) << "Cannot find model handler for " << model_session_id;
    return false;
  }
  auto model_handler = iter->second;
  // Update backend pool first
  {
    std::lock_guard<std::mutex> lock(backend_sessions_mu_);
    auto old_backends = model_handler->BackendList();
    std::unordered_set<uint32_t> new_backends;
    // Add new backends
    for (auto backend : route.backend_rate()) {
      uint32_t backend_id = backend.info().node_id();
      if (backend_sessions_.count(backend_id) == 0) {
        backend_sessions_.emplace(
            backend_id, std::unordered_set<std::string>{model_session_id});
        backend_pool_.AddBackend(std::make_shared<BackendSession>(
            backend.info(), io_context_, this));
      } else {
        backend_sessions_.at(backend_id).insert(model_session_id);
      }
      new_backends.insert(backend_id);
    }
    // Remove unused backends
    for (auto backend_id : old_backends) {
      if (new_backends.count(backend_id) == 0) {
        backend_sessions_.at(backend_id).erase(model_session_id);
        if (backend_sessions_.at(backend_id).empty()) {
          LOG(INFO) << "Remove backend " << backend_id;
          backend_sessions_.erase(backend_id);
          backend_pool_.RemoveBackend(backend_id);
        }
      }
    }
  }
  // Update route to backends with throughput in model handler
  model_handler->UpdateRoute(route);
  return true;
}

void Frontend::RegisterUser(
    std::shared_ptr<UserSession> user_sess, const RequestProto& request,
    ReplyProto* reply) {
  uint32_t uid = request.user_id();
  user_sess->set_user_id(uid);
  std::lock_guard<std::mutex> lock(user_mutex_);
  auto itr = user_sessions_.find(uid);
  if (itr == user_sessions_.end()) {
    VLOG(1) << "New user session: " << uid;
    user_sessions_.emplace(uid, user_sess);
  } else if (itr->second != user_sess) {
    VLOG(1) << "Update user session: " << uid;
    user_sessions_[uid] = user_sess;
  }
  reply->set_user_id(uid);
  reply->set_status(CTRL_OK);
}

void Frontend::Daemon() {
  while (running_) {
    auto next_time = Clock::now() + std::chrono::seconds(beacon_interval_sec_);
    WorkloadStatsProto workload_stats;
    workload_stats.set_node_id(node_id_);
    for (auto const& iter : model_pool_) {
      auto model_session_id = iter.first;
      auto history = iter.second->counter()->GetHistory();
      auto model_stats = workload_stats.add_model_stats();
      model_stats->set_model_session_id(model_session_id);
      for (auto nreq : history) {
        model_stats->add_num_requests(nreq);
      }
    }
    ReportWorkload(workload_stats);
    std::this_thread::sleep_until(next_time);
  }
}

void Frontend::ReportWorkload(const WorkloadStatsProto& request) {
  grpc::ClientContext context;
  RpcReply reply;
  grpc::Status status = sch_stub_->ReportWorkload(&context, request,
                                                  &reply);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to connect to scheduler: " <<
        status.error_message() << "(" << status.error_code() << ")";
    return;
  }
  CtrlStatus ret = reply.status();
  if (ret != CTRL_OK) {
    LOG(ERROR) << "ReportWorkload error: " << CtrlStatus_Name(ret);
  }
}

} // namespace app
} // namespace nexus
