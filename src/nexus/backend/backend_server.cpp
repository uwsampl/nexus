#include <gflags/gflags.h>
#include <glog/logging.h>
#include <pthread.h>
#include <unordered_set>

#include "nexus/common/config.h"
#include "nexus/common/model_db.h"
#include "nexus/backend/backend_server.h"
#include "nexus/backend/share_prefix_model.h"

DEFINE_bool(multi_batch, true, "Enable multi batching");
DEFINE_int32(occupancy_valid, 50, "Backup backend occupancy valid time in ms");

namespace nexus {
namespace backend {

BackendServer::BackendServer(std::string port, std::string rpc_port,
                             std::string sch_addr, int gpu_id,
                             size_t num_workers, std::vector<int> cores) :
    ServerBase(port),
    gpu_id_(gpu_id),
    running_(false),
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
  // Init GPU executor
  if (FLAGS_multi_batch) {
    LOG(INFO) << "Multi-batching is enabled";
    gpu_executor_.reset(new GpuExecutorMultiBatching(gpu_id));
  } else {
    LOG(INFO) << "Multi-batching is disabled";
    gpu_executor_.reset(new GpuExecutorNoMultiBatching(gpu_id));
  }
  if (cores.empty()) {
    gpu_executor_->Start();
  } else {
    gpu_executor_->Start(cores.back());
    cores.pop_back();
  }
  // Init workers
  if (num_workers == 0) {
    if (cores.empty()) {
      num_workers = 4;
    } else {
      num_workers = cores.size();
    }
  }
  for (size_t i = 0; i < num_workers; ++i) {
    std::unique_ptr<Worker> worker(new Worker(i, this, task_queue_));
    if (cores.empty()) {
      worker->Start();
    } else {
      worker->Start(cores[i % cores.size()]);
    }
    workers_.push_back(std::move(worker));
  }
  if (cores.size() > 0) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    int core = cores[num_workers % cores.size()];
    CPU_SET(core, &cpuset);
    int rc = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
    if (rc != 0) {
      LOG(ERROR) << "Error calling pthread_setaffinity_np: " << rc << "\n";
    }
    LOG(INFO) << "IO thread is pinned on CPU " << core;
  }
  // Init node id and register backend to global scheduler
  Register();
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
  switch (message->type()) {
    case kBackendRequest:
    case kBackendRelay: {
      auto task = std::make_shared<Task>(conn);
      task->DecodeQuery(message);
      task_queue_.push(std::move(task));
      break;
    }
    case kBackendRelayReply: {
      std::static_pointer_cast<BackupClient>(conn)->Reply(std::move(message));
      break;
    }
    default:
      LOG(INFO) << "Wrong message type: " << message->type();
  }
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

void BackendServer::UpdateModelTable(const ModelTableConfig& request) {
  // Update backend pool
  std::unordered_set<uint32_t> backend_list;
  std::unordered_map<uint32_t, BackendInfo> backend_infos;
  for (auto config : request.model_instance_config()) {
    for (auto const& info : config.backup_backend()) {
      backend_list.insert(info.node_id());
      backend_infos.emplace(info.node_id(), info);
    }
  }
  auto new_backends = backend_pool_.UpdateBackendList(backend_list);
  for (auto backend_id : new_backends) {
    backend_pool_.AddBackend(std::make_shared<BackupClient>(
        backend_infos.at(backend_id), io_service_, this));
  }
  
  // Update model table
  std::lock_guard<std::mutex> lock(model_table_mu_);
  std::unordered_set<std::string> all_sessions;
  for (auto config : request.model_instance_config()) {
    if (config.model_session_size() > 1) {
      std::shared_ptr<ModelExecutor> sp_model = nullptr;
      for (auto model_sess : config.model_session()) {
        std::string session_id = ModelSessionToString(model_sess);
        auto iter = model_table_.find(session_id);
        if (iter != model_table_.end()) {
          auto model = iter->second;
          if (model->IsSharePrefixModel()) {
            sp_model = model;
            break;
          } else {
            // Remove its original model
            gpu_executor_->RemoveModel(model);
            model_table_.erase(session_id);
          }
        }
      }
      if (sp_model == nullptr) {
        // Create a new prefix model
        LOG(INFO) << "Load prefix model instance " <<
            ModelSessionToString(config.model_session(0)) << ", batch: " <<
            config.batch() << ", backup: " << config.backup();
        auto model = std::make_shared<ModelExecutor>(gpu_id_, config,
                                                     task_queue_);
        gpu_executor_->AddModel(model);
        for (auto model_sess : config.model_session()) {
          std::string session_id = ModelSessionToString(model_sess);
          all_sessions.insert(session_id);
          model_table_.emplace(session_id, model);
        }
      } else {
        // Prefix model already exists
        // Need to update batch size, and add new model sessions sharing prefix
        auto sp_internal = dynamic_cast<SharePrefixModel*>(sp_model->model());
        if (sp_internal->batch() != config.batch()) {
          LOG(INFO) << "Update prefix model instance " <<
              sp_internal->model_session_id() << ", batch: " <<
              sp_internal->batch() << " -> " << config.batch();
          sp_model->SetBatch(config.batch());
        }
        for (auto model_sess : config.model_session()) {
          std::string session_id = ModelSessionToString(model_sess);
          all_sessions.insert(session_id);
          if (!sp_internal->HasModelSession(session_id)) {
            LOG(INFO) << "Add model session " << session_id <<
                " to prefix model " << sp_internal->model_session_id();
            sp_internal->AddModelSession(model_sess);
            model_table_.emplace(session_id, sp_model);
          }
        }
        sp_model->UpdateBackupBackends(config);
      }
    } else {
      // Regular model session
      auto model_sess = config.model_session(0);
      std::string session_id = ModelSessionToString(model_sess);
      all_sessions.insert(session_id);
      auto model_iter = model_table_.find(session_id);
      if (model_iter == model_table_.end()) {
        // Load new model instance
        auto model = std::make_shared<ModelExecutor>(gpu_id_, config,
                                                     task_queue_);
        model_table_.emplace(session_id, model);
        gpu_executor_->AddModel(model);
        LOG(INFO) << "Load model instance " << session_id <<
            ", batch: " << config.batch() << ", backup: " << config.backup();
      } else {
        auto model = model_iter->second;
        if (model->model()->batch() != config.batch()) {
          // Update the batch size
          LOG(INFO) << "Update model instance " << session_id << ", batch: " <<
              model->model()->batch() << " -> " << config.batch();
          model->SetBatch(config.batch());
        }
        model->UpdateBackupBackends(config);
      }
    }
  }
  std::vector<std::string> to_remove;
  for (auto iter : model_table_) {
    if (all_sessions.count(iter.first) == 0) {
      to_remove.push_back(iter.first);
    }
  }
  for (auto session_id : to_remove) {
    auto model = model_table_.at(session_id);
    model_table_.erase(session_id);
    if (!model->IsSharePrefixModel()) {
      LOG(INFO) << "Remove model instance " << session_id;
      gpu_executor_->RemoveModel(model);
    } else {
      auto sp_internal = dynamic_cast<SharePrefixModel*>(model->model());
      LOG(INFO) << "Remove model session " << session_id <<
          " from prefix model " << sp_internal->model_session_id();
      sp_internal->RemoveModelSession(session_id);
      if (sp_internal->num_model_sessions() == 0) {
        LOG(INFO) << "Remove prefix model instance " << session_id;
        gpu_executor_->RemoveModel(model);
      }
    }
  }
  
  // Update duty cycle
  gpu_executor_->SetDutyCycle(request.duty_cycle_us());
  LOG(INFO) << "Duty cycle: " << request.duty_cycle_us() << " us";
  //reply->set_status(CTRL_OK);
}

ModelExecutorPtr BackendServer::GetModel(const std::string& model_session_id) {
  std::lock_guard<std::mutex> lock(model_table_mu_);
  auto itr = model_table_.find(model_session_id);
  if (itr == model_table_.end()) {
    LOG(ERROR) << "Model session is not loaded: " << model_session_id;
    return nullptr;
  }
  return itr->second;
}

BackendServer::ModelTable BackendServer::GetModelTable() {
  std::lock_guard<std::mutex> lock(model_table_mu_);
  return model_table_;
}

std::shared_ptr<BackupClient> BackendServer::GetBackupClient(
    uint32_t backend_id) {
  auto backup = backend_pool_.GetBackend(backend_id);
  if (backup == nullptr) {
    return nullptr;
  }
  return std::static_pointer_cast<BackupClient>(backup);
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
    // LOG(INFO) << "Current utilization: " << gpu_executor_->CurrentUtilization();
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
