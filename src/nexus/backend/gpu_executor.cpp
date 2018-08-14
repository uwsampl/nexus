#include <gflags/gflags.h>
#include <glog/logging.h>
#include <pthread.h>
#include <thread>

#include "nexus/backend/backend_server.h"
#include "nexus/backend/caffe_model.h"
#include "nexus/backend/gpu_executor.h"
#include "nexus/common/device.h"

namespace nexus {
namespace backend {

GpuExecutorMultiBatching::GpuExecutorMultiBatching(int gpu_id) : 
    gpu_id_(gpu_id),
    running_(false) {
}

void GpuExecutorMultiBatching::Start(int core) {
  running_ = true;
  thread_ = std::thread(&GpuExecutorMultiBatching::Run, this);
  if (core >= 0) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core, &cpuset);
    int rc = pthread_setaffinity_np(thread_.native_handle(),
                                    sizeof(cpu_set_t), &cpuset);
    if (rc != 0) {
      LOG(ERROR) << "Error calling pthread_setaffinity_np: " << rc << "\n";
    }
    LOG(INFO) << "GPU executor is pinned on CPU " << core;
  }
}

void GpuExecutorMultiBatching::Stop() {
  running_ = false;
  if (thread_.joinable()) {
    thread_.join();
  }
}

void GpuExecutorMultiBatching::AddModel(std::shared_ptr<ModelExecutor> model) {
  std::lock_guard<std::mutex> lock(models_mu_);
  if (model->backup()) {
    backup_models_.push_back(model);
  } else {
    models_.push_back(model);
  }
}

void GpuExecutorMultiBatching::RemoveModel(
    std::shared_ptr<ModelExecutor> model) {
  std::lock_guard<std::mutex> lock(models_mu_);
  for (auto iter = models_.begin(); iter != models_.end(); ++iter) {
    if (*iter == model) {
      models_.erase(iter);
      break;
    }
  }
}

double GpuExecutorMultiBatching::CurrentUtilization() {
  if (duty_cycle_us_ == 0) {
    // No model loaded so far
    return 0.;
  }
  std::lock_guard<std::mutex> lock(models_mu_);
  auto now = Clock::now();
  double exec_cycle = 0.;
  for (auto& model : models_) {
    int curr_queue_len = model->NumberOfOpenRequests();
    TimePoint last_exec_time = model->LastExecuteFinishTime();
    double elapse = std::chrono::duration_cast<std::chrono::microseconds>(
          now - last_exec_time).count();
    int est_queue_len = (int) std::min(elapse / duty_cycle_us_ * curr_queue_len,
                                       (double) model->model()->max_batch());
    LOG(INFO) << model->model()->model_session_id() <<
        " estimate batch size: " << est_queue_len;
    if (est_queue_len > 0) {
      exec_cycle += model->profile()->GetForwardLatency(est_queue_len);
    }
  }
  for (auto& model : backup_models_) {
    int queue_len = model->NumberOfOpenRequests();
    if (queue_len > 0) {
      exec_cycle += model->profile()->GetForwardLatency(queue_len);
    }
  }
  double util = exec_cycle / duty_cycle_us_;
  LOG(INFO) << "Utilization: " << util << " (exec/duty: " << exec_cycle <<
      " / " << duty_cycle_us_ << " us)";
  return exec_cycle / duty_cycle_us_;
}

void GpuExecutorMultiBatching::Run() {
#ifdef USE_CAFFE
  caffe::Caffe::set_mode(caffe::Caffe::GPU);
  caffe::Caffe::set_release_memory(false);
  caffe::Caffe::SetDevice(gpu_id_);
#endif

  NEXUS_CUDA_CHECK(cudaSetDevice(gpu_id_));
  //auto min_cycle = std::chrono::microseconds(50);
  double min_cycle_us = 50.; // us
  LOG(INFO) << "GpuExecutor started";
  while (running_) {
    std::vector<std::shared_ptr<ModelExecutor> > models;
    std::vector<std::shared_ptr<ModelExecutor> > backup_models;
    {
      // Take a snapshot
      std::lock_guard<std::mutex> lock(models_mu_);
      models = models_;
      backup_models = backup_models_;
    }
    double exec_cycle_us = 0.;
    for (auto model : models) {
      exec_cycle_us += model->Execute();
    }
    double budget = duty_cycle_us_ - exec_cycle_us;
    for (auto model : backup_models) {
      if (budget <= 0) {
        break;
      }
      uint32_t batch = 1;
      auto profile = model->profile();
      while (true) {
        double fwd_lat = profile->GetForwardLatency(batch);
        if (fwd_lat == 0 || fwd_lat >= budget) {
          --batch;
          break;
        }
        ++batch;
      }
      if (batch > 0) {
        auto lat = model->Execute(batch);
        budget -= lat;
        exec_cycle_us += lat;
      }
    }
    if (exec_cycle_us < min_cycle_us) {
      // ensure the cycle to be at least min_cycle to avoid acquiring lock
      // too frequently in the ModelInstance
      std::this_thread::sleep_for(std::chrono::microseconds(
          int(min_cycle_us - exec_cycle_us)));
    }
  }
  LOG(INFO) << "GpuExecutor stopped";
}

GpuExecutorNoMultiBatching::GpuExecutorNoMultiBatching(int gpu_id) :
    gpu_id_(gpu_id) {}

void GpuExecutorNoMultiBatching::Start(int core) {
  core_ = core;
}

void GpuExecutorNoMultiBatching::Stop() {
  for (auto& iter : threads_) {
    iter.second->Stop();
  }
  threads_.clear();
}

void GpuExecutorNoMultiBatching::AddModel(
    std::shared_ptr<ModelExecutor> model) {
  std::lock_guard<std::mutex> lock(mu_);
  std::unique_ptr<GpuExecutorMultiBatching> exec(
      new GpuExecutorMultiBatching(gpu_id_));
  exec->AddModel(model);
  exec->Start(core_);
  threads_.emplace(model->model()->model_session_id(), std::move(exec));
}

void GpuExecutorNoMultiBatching::RemoveModel(
    std::shared_ptr<ModelExecutor> model) {
  std::lock_guard<std::mutex> lock(mu_);
  auto sess_id = model->model()->model_session_id();
  threads_.at(sess_id)->Stop();
  threads_.erase(sess_id);
}

double GpuExecutorNoMultiBatching::CurrentUtilization() {
  // Doesn't support utilization
  return -1.;
}

} // namespace backend
} // namespace nexus
