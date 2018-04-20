#include <glog/logging.h>
#include <unordered_set>

#include "nexus/common/config.h"
#include "nexus/common/model_db.h"
#include "nexus/backend/backend_server.h"

namespace nexus {
namespace backend {

BackendServer::BackendServer(std::string port, std::string rpc_port,
                             std::string sch_addr, size_t num_workers,
                             int gpu_id, std::string model_db_root) :
    ServerBase(port),
    gpu_id_(gpu_id),
    running_(false),
    rpc_service_(this, rpc_port),
    rand_gen_(rd_()) {
  // Init model information
  ModelDatabase::Singleton().Init(model_db_root);
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
  // Init node id and register backend to global scheduler
  Register();
  // Start workers
  for (size_t i = 0; i < num_workers; ++i) {
    std::unique_ptr<Worker> worker(new Worker(i, this, task_queue_));
    worker->Start();
    workers_.push_back(std::move(worker));
  }
  // Start GPU executor
  gpu_executor_.reset(new GpuExecutorMultiBatching(gpu_id, this));
  gpu_executor_->Start();
}

BackendServer::~BackendServer() {
  if (running_) {
    Stop();
  }
}

void BackendServer::Run() {
  running_ = true;
  // Start the daemon thread
  daemon_thread_ = std::thread(&BackendServer::Daemon, this);
  LOG(INFO) << "Backend server (id: " << node_id_ << ") is listening on " <<
      address();
  // Start the IO service
  io_service_.run();
}

void BackendServer::Stop() {
  running_ = false;
  // Unregister backend server
  Unregister();
  // Stop accept new connections
  ServerBase::Stop();
  // Stop all frontend connections
  for (auto conn: frontend_connections_) {
    conn->Stop();
  }
  frontend_connections_.clear();
  // Stop GPU executor
  gpu_executor_->Stop();
  // Stop workers
  for (auto& worker : workers_) {
    worker->Stop();
  }
  workers_.clear();
  // Stop daemon thread
  if (daemon_thread_.joinable()) {
    daemon_thread_.join();
  }
  // Stop RPC service
  rpc_service_.Stop();
  LOG(INFO) << "Backend server stopped";
}

void BackendServer::HandleAccept() {
  std::lock_guard<std::mutex> lock(frontend_mutex_);
  auto conn = std::make_shared<Connection>(std::move(socket_), this);
  frontend_connections_.insert(conn);
  conn->Start();
}

void BackendServer::HandleMessage(std::shared_ptr<Connection> conn,
                                  std::shared_ptr<Message> message) {
  if (message->type() != kBackendRequest) {
    LOG(INFO) << "Wrong message type: " << message->type();
    return;
  }
  auto task = std::make_shared<Task>(conn);
  task->DecodeQuery(message);
  task_queue_.push(std::move(task));
}

void BackendServer::HandleError(std::shared_ptr<Connection> conn,
                                boost::system::error_code ec) {
  if (ec == boost::asio::error::eof ||
      ec == boost::asio::error::connection_reset) {
    // frontend disconnects
  } else {
    LOG(ERROR) << "Frontend connection error (" << ec << "): " << ec.message();
  }
  std::lock_guard<std::mutex> lock(frontend_mutex_);
  frontend_connections_.erase(conn);
  conn->Stop();
}

void BackendServer::UpdateModelTable(const ModelTableConfig& request,
                                     RpcReply* reply) {
  SpinlockGuard lock(model_table_lock_);
  std::unordered_set<std::string> new_model_list;
  for (auto config : request.model_instance_config()) {
    auto model_sess = config.model_session();
    std::string session_id = ModelSessionToString(model_sess);
    new_model_list.insert(session_id);
    auto model_iter = model_table_.find(session_id);
    if (model_iter == model_table_.end()) {
      // Load new model instance
      auto model_id = ModelSessionToModelID(model_sess);
      auto info = ModelDatabase::Singleton().GetModelInfo(model_id);
      if (info == nullptr) {
        reply->set_status(MODEL_NOT_FOUND);
        return;
      }
      auto model = CreateModelInstance(gpu_id_, config, *info, task_queue_);
      model_table_.emplace(session_id, model);
      LOG(INFO) << "Load model instance " << session_id <<
          ", batch: " << config.batch();
    } else {
      auto model = model_iter->second;
      if (model->batch() != config.batch()) {
        // Update the batch size
        LOG(INFO) << "Update model instance " << session_id << ", batch: " <<
            model->batch() << " -> " << config.batch();
        model->set_batch(config.batch());
      }
    }
  }
  std::vector<std::string> to_remove;
  for (auto iter : model_table_) {
    if (new_model_list.find(iter.first) == new_model_list.end()) {
      to_remove.push_back(iter.first);
    }
  }
  for (auto session_id : to_remove) {
    LOG(INFO) << "Unload model instance " << session_id;
    model_table_.erase(session_id);
  }
  reply->set_status(CTRL_OK);
}

ModelInstancePtr BackendServer::GetModelInstance(
    const std::string& model_session_id) {
  SpinlockGuard lock(model_table_lock_);
  auto itr = model_table_.find(model_session_id);
  if (itr == model_table_.end()) {
    LOG(ERROR) << "Model session is not loaded: " << model_session_id;
    return nullptr;
  }
  return itr->second;
}

BackendServer::ModelTable BackendServer::GetModelTable() {
  SpinlockGuard lock(model_table_lock_);
  return model_table_;
}

void BackendServer::Daemon() {
  while (running_) {
    auto next_time = Clock::now() + std::chrono::seconds(beacon_interval_sec_);
    auto model_table = GetModelTable();
    BackendStatsProto backend_stats;
    backend_stats.set_node_id(node_id_);
    for (auto iter : model_table) {
      auto model_session_id = iter.first;
      auto model = iter.second;
      auto history = model->counter()->GetHistory();
      auto model_stats = backend_stats.add_model_stats();
      model_stats->set_model_session_id(model_session_id);
      for (auto nreq : history) {
        model_stats->add_num_requests(nreq);
      }
    }
    UpdateBackendStats(backend_stats);
    std::this_thread::sleep_until(next_time);
  }
}

void BackendServer::Register() {
  // Init node id
  std::uniform_int_distribution<uint32_t> dis(
      1, std::numeric_limits<uint32_t>::max());
  node_id_ = dis(rand_gen_);
  
  // Prepare request
  RegisterRequest request;
  request.set_node_type(BACKEND_NODE);
  request.set_node_id(node_id_);
  request.set_server_port(port());
  request.set_rpc_port(rpc_service_.port());
  GPUDevice* gpu_device = DeviceManager::Singleton().GetGPUDevice(
      gpu_id_);
  request.set_gpu_device_name(gpu_device->device_name());
  request.set_gpu_available_memory(gpu_device->FreeMemory());
  
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
    if (ret != CTRL_BACKEND_NODE_ID_CONFLICT) {
      LOG(FATAL) << "Failed to register backend to scheduler: " <<
          CtrlStatus_Name(ret);
    }
    // Backend ID conflict, need to generate a new one
    node_id_ = dis(rand_gen_);
    request.set_node_id(node_id_);
  }
}

void BackendServer::Unregister() {
  UnregisterRequest request;
  request.set_node_type(BACKEND_NODE);
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
    LOG(ERROR) << "Unregister error: " << CtrlStatus_Name(ret);
  }
}

void BackendServer::UpdateBackendStats(const BackendStatsProto& request) {
  grpc::ClientContext context;
  RpcReply reply;
  grpc::Status status = sch_stub_->UpdateBackendStats(&context, request,
                                                      &reply);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to connect to scheduler: " <<
        status.error_message() << "(" << status.error_code() << ")";
    return;
  }
  CtrlStatus ret = reply.status();
  if (ret != CTRL_OK) {
    LOG(ERROR) << "UpdateBackendStats error: " << CtrlStatus_Name(ret);
  }
}

} // namespace backend
} // namespace nexus
