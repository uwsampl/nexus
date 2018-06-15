#include <gflags/gflags.h>
#include <glog/logging.h>
#include <thread>

#include "nexus/backend/backend_server.h"
#include "nexus/backend/caffe_model.h"
#include "nexus/backend/gpu_executor.h"
#include "nexus/common/device.h"

namespace nexus {
namespace backend {

GpuExecutorMultiBatching::GpuExecutorMultiBatching(
    int gpu_id, BlockPriorityQueue<Task>& cpu_task_queue) : 
    gpu_id_(gpu_id),
    cpu_task_queue_(cpu_task_queue),
    running_(false) {
}

void GpuExecutorMultiBatching::Start() {
  running_ = true;
  thread_ = std::thread(&GpuExecutorMultiBatching::Run, this);
}

void GpuExecutorMultiBatching::Stop() {
  running_ = false;
  if (thread_.joinable()) {
    thread_.join();
  }
}

void GpuExecutorMultiBatching::AddModel(std::shared_ptr<ModelInstance> model) {
  std::lock_guard<std::mutex> lock(models_mu_);
  models_.emplace(model->model_session_id(), std::make_shared<ModelExecutor>(
      model, cpu_task_queue_));
}

void GpuExecutorMultiBatching::RemoveModel(
    std::shared_ptr<ModelInstance> model) {
  std::lock_guard<std::mutex> lock(models_mu_);
  models_.erase(model->model_session_id());
}

void GpuExecutorMultiBatching::AddTask(std::shared_ptr<Task> task) {
  models_.at(task->model->model_session_id())->AddTask(task);
}

void GpuExecutorMultiBatching::Run() {
#ifdef USE_CAFFE
  caffe::Caffe::set_mode(caffe::Caffe::GPU);
  caffe::Caffe::set_release_memory(false);
  caffe::Caffe::SetDevice(gpu_id_);
#endif

  NEXUS_CUDA_CHECK(cudaSetDevice(gpu_id_));
  auto min_cycle = std::chrono::microseconds(50);
  LOG(INFO) << "GpuExecutor started";
  while (running_) {
    auto cycle_start = std::chrono::high_resolution_clock::now();
    std::unordered_map<std::string, std::shared_ptr<ModelExecutor> > models;
    {
      // Take a snapshot
      std::lock_guard<std::mutex> lock(models_mu_);
      models = models_;
    }
    for (auto iter : models) {
      iter.second->Execute();
    }
    auto cycle_end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
          cycle_end - cycle_start);
    if (duration < min_cycle) {
      // ensure the cycle to be at least min_cycle to avoid acquiring lock
      // too frequently in the ModelInstance
      std::this_thread::sleep_for(min_cycle - duration);
    }
  }
  LOG(INFO) << "GpuExecutor stopped";
}

GpuExecutorNoMultiBatching::GpuExecutorNoMultiBatching(
    int gpu_id, BlockPriorityQueue<Task>& cpu_task_queue) :
    gpu_id_(gpu_id),
    cpu_task_queue_(cpu_task_queue) {}

void GpuExecutorNoMultiBatching::Start() {}

void GpuExecutorNoMultiBatching::Stop() {
  for (auto& iter : threads_) {
    iter.second->Stop();
  }
  threads_.clear();
}

void GpuExecutorNoMultiBatching::AddModel(
    std::shared_ptr<ModelInstance> model) {
  std::lock_guard<std::mutex> lock(mu_);
  std::unique_ptr<GpuExecutorMultiBatching> exec(
      new GpuExecutorMultiBatching(gpu_id_, cpu_task_queue_));
  exec->AddModel(model);
  exec->Start();
  threads_.emplace(model->model_session_id(), std::move(exec));
}

void GpuExecutorNoMultiBatching::RemoveModel(
    std::shared_ptr<ModelInstance> model) {
  std::lock_guard<std::mutex> lock(mu_);
  threads_.at(model->model_session_id())->Stop();
  threads_.erase(model->model_session_id());
}

void GpuExecutorNoMultiBatching::AddTask(std::shared_ptr<Task> task) {
  threads_.at(task->model->model_session_id())->AddTask(task);
}

} // namespace backend
} // namespace nexus
