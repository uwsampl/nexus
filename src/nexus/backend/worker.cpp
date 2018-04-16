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
    running_(false) {
}

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
    // LOG(INFO) << "Worker " << index_ << " process query " <<
    //     task->query.query_id() << ", model session " <<
    //     task->query.model_session_id() << ", stage " << task->stage;
    Process(task);
  }
  LOG(INFO) << "Worker " << index_ << " stopped";
}

void Worker::Process(std::shared_ptr<Task> task) {
  switch (task->stage) {
    case kPreprocess: {
      task->model = server_->GetModelInstance(task->query.model_session_id());
      if (task->model == nullptr) {
        std::stringstream ss;
        ss << "Model session is not loaded: " << task->query.model_session_id();
        task->result.set_status(MODEL_SESSION_NOT_LOADED);
        SendReply(std::move(task));
        break;
      }
      if (!task->model->Preprocess(task)) {
        SendReply(std::move(task));
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
  if (task->query.debug()) {
    /*auto backend_lat = task->result.add_backend_latency();
    backend_lat->set_model_session_id(task->context.query->model_session_id());
    backend_lat->set_batch_time(task->timer.GetLatencyMillis(
        "begin", "batch"));
    backend_lat->set_process_time(task->timer.GetLatencyMillis(
        "batch", "end"));
    backend_lat->set_num_inputs(task->context.outputs.size());*/
  }
  auto msg = std::make_shared<Message>(kBackendReply,
                                       task->result.ByteSizeLong());
  msg->EncodeBody(task->result);
  task->connection->Write(std::move(msg));
}

} // namespace backend
} // namespace nexus
