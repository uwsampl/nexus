#include <chrono>
#include <glog/logging.h>

#include "nexus/backend/backend_server.h"
#include "nexus/backend/model_ins.h"
#include "nexus/backend/worker.h"

namespace nexus {
namespace backend {

Worker::Worker(int index, BackendServer* server,
               BlockPriorityQueue<Task>& task_queue) :
    index_(index),
    server_(server),
    task_queue_(task_queue),
    running_(false) {}

void Worker::Start() {
  running_ = true;
  thread_ = std::thread(&Worker::Run, this);
}

void Worker::Stop() {
  running_ = false;
  if (thread_.joinable()) {
    thread_.join();
  }
}

void Worker::Run() {
  LOG(INFO) << "Worker " << index_ << " started";
  auto timeout = std::chrono::milliseconds(50);
  while (running_) {
    std::shared_ptr<Task> task = task_queue_.pop(timeout);
    if (task == nullptr) {
      continue;
    }
    Process(task);
  }
  LOG(INFO) << "Worker " << index_ << " stopped";
}

void Worker::Process(std::shared_ptr<Task> task) {
  switch (task->stage) {
    case kPreprocess: {
      task->model = server_->GetModel(task->query.model_session_id());
      if (task->model == nullptr) {
        std::stringstream ss;
        ss << "Model session is not loaded: " << task->query.model_session_id();
        task->result.set_status(MODEL_SESSION_NOT_LOADED);
        SendReply(std::move(task));
        break;
      }
      // Preprocess task
      if (!task->model->Preprocess(task)) {
        if (task->result.status() != CTRL_OK) {
          SendReply(std::move(task));
        } else {
          // Relay to the request to backup servers
          std::vector<uint32_t> backups = task->model->BackupBackends();
          double min_util = 1.;
          std::shared_ptr<BackupClient> best_backup = nullptr;
          for (auto backend_id : backups) {
            auto backup = server_->GetBackupClient(backend_id);
            double util = backup->GetUtilization();
            if (util < min_util) {
              min_util = util;
              best_backup = backup;
            }
          }
          if (best_backup != nullptr) {
            LOG(INFO) << "Forward request to backup " <<
                best_backup->node_id() << " with utilization " << min_util;
            best_backup->Forward(std::move(task));
          } else {
            task->model->Preprocess(task, true);
          }
        }
      }
      break;
    }
    case kPostprocess: {
      if (task->result.status() != CTRL_OK) {
        SendReply(std::move(task));
      } else {
        task->model->Postprocess(task);
        SendReply(std::move(task));
      }
      break;
    }
    default:
      LOG(ERROR) << "Wrong task stage: " << task->stage;
  }
}

void Worker::SendReply(std::shared_ptr<Task> task) {
  task->timer.Record("end");
  task->result.set_query_id(task->query.query_id());
  task->result.set_model_session_id(task->query.model_session_id());
  task->result.set_latency_us(task->timer.GetLatencyMicros("begin", "end"));
  task->result.set_queuing_us(task->timer.GetLatencyMicros("begin", "exec"));
  MessageType reply_type = kBackendReply;
  if (task->msg_type == kBackendRelay) {
    reply_type = kBackendRelayReply;
  }
  auto msg = std::make_shared<Message>(reply_type,
                                       task->result.ByteSizeLong());
  msg->EncodeBody(task->result);
  task->connection->Write(std::move(msg));
}

} // namespace backend
} // namespace nexus
