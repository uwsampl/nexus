#include <gflags/gflags.h>
#include <glog/logging.h>
#include <thread>

#include "nexus/backend/backend_server.h"
#include "nexus/backend/gpu_executor.h"
#include "nexus/common/device.h"

namespace nexus {
namespace backend {

GpuExecutorMultiBatching::GpuExecutorMultiBatching(int gpu_id) :
    gpu_id_(gpu_id),
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


void GpuExecutorMultiBatching::AddModel(const std::string& model_sess_id,
                                        std::shared_ptr<ModelInstance> model) {
  std::lock_guard<std::mutex> lock(mu_);
  models_.emplace(model_sess_id, model);
}

void GpuExecutorMultiBatching::RemoveModel(const std::string& model_sess_id) {
  std::lock_guard<std::mutex> lock(mu_);
  models_.erase(model_sess_id);
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
    std::unordered_map<std::string, std::shared_ptr<ModelInstance> > models;
    {
      // Take a snapshot
      std::lock_guard<std::mutex> lock(mu_);
      models = models_;
    }
    for (auto iter : models) {
      iter.second->Forward();
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


GpuExecutorNoMultiBatching::GpuExecutorNoMultiBatching(int gpu_id) :
    gpu_id_(gpu_id) {}

void GpuExecutorNoMultiBatching::Start() {}

void GpuExecutorNoMultiBatching::Stop() {
  for (auto& iter : threads_) {
    iter.second->Stop();
  }
  threads_.clear();
}

void GpuExecutorNoMultiBatching::AddModel(
    const std::string& model_sess_id, std::shared_ptr<ModelInstance> model) {
  std::lock_guard<std::mutex> lock(mu_);
  threads_.emplace(model_sess_id,
                   std::make_shared<ModelThread>(gpu_id_, model_sess_id,
                                                 model));
}

void GpuExecutorNoMultiBatching::RemoveModel(const std::string& model_sess_id) {
  std::lock_guard<std::mutex> lock(mu_);
  threads_.at(model_sess_id)->Stop();
  threads_.erase(model_sess_id);
}

} // namespace backend
} // namespace nexus
