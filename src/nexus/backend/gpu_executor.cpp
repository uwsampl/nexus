#include <glog/logging.h>
#include <thread>

#include "nexus/backend/backend_server.h"
#include "nexus/backend/caffe_model.h"
#include "nexus/backend/gpu_executor.h"
#include "nexus/common/device.h"

namespace nexus {
namespace backend {

GpuExecutorMultiBatching::GpuExecutorMultiBatching(int gpu_id,
                                                   BackendServer* server) :
    gpu_id_(gpu_id),
    server_(server),
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

void GpuExecutorMultiBatching::Run() {
  auto min_cycle = std::chrono::microseconds(50);
  caffe::Caffe::set_mode(caffe::Caffe::GPU);
  caffe::Caffe::set_release_memory(false);
  caffe::Caffe::SetDevice(gpu_id_);
  NEXUS_CUDA_CHECK(cudaSetDevice(gpu_id_));
  LOG(INFO) << "GpuExecutor started";
  while (running_) {
    auto cycle_start = std::chrono::high_resolution_clock::now();
    auto models = server_->GetAllModelInstances();
    for (auto model : models) {
      model->Forward();
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


GpuExecutorNoMultiBatching::GpuExecutorNoMultiBatching(int gpu_id,
                                                       BackendServer* server) :
    gpu_id_(gpu_id),
    server_(server),
    running_(false) {
}

void GpuExecutorNoMultiBatching::Start() {
  running_ = true;
  auto models = server_->GetAllModelInstances();
  for (auto model : models) {
    threads_.push_back(std::thread(&GpuExecutorNoMultiBatching::Run, this,
                                   model));
  }
}

void GpuExecutorNoMultiBatching::Stop() {
  running_ = false;
  for (auto& thread : threads_) {
    thread.join();
  }
}

void GpuExecutorNoMultiBatching::Run(std::shared_ptr<ModelInstance> model) {
  auto min_cycle = std::chrono::microseconds(50);
  caffe::Caffe::set_mode(caffe::Caffe::GPU);
  caffe::Caffe::SetDevice(gpu_id_);
  NEXUS_CUDA_CHECK(cudaSetDevice(gpu_id_));
  LOG(INFO) << "GpuExecutor started for model " << model->model_id();
  while (running_) {
    auto cycle_start = std::chrono::high_resolution_clock::now();
    model->Forward();
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

} // namespace backend
} // namespace nexus
