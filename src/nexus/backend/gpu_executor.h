#ifndef NEXUS_BACKEND_BASE_GPU_EXECUTOR_H_
#define NEXUS_BACKEND_BASE_GPU_EXECUTOR_H_

#include <atomic>
#include <memory>
#include <thread>
#include <vector>
#include <unordered_map>

#include "nexus/backend/caffe_model.h"
#include "nexus/backend/model_ins.h"

namespace nexus {
namespace backend {

class BackendServer;

class GpuExecutor {
 public:
  virtual void Start() = 0;
  virtual void Stop() = 0;
  virtual void AddModel(const std::string& model_sess_id,
                        std::shared_ptr<ModelInstance> model) = 0;
  virtual void RemoveModel(const std::string& model_sess_id) = 0;
};

class GpuExecutorMultiBatching : public GpuExecutor {
 public:
  GpuExecutorMultiBatching(int gpu_id);

  inline int gpu_id() { return gpu_id_; }

  void Start() final;

  void Stop() final;

  void AddModel(const std::string& model_sess_id,
                std::shared_ptr<ModelInstance> model) final;

  void RemoveModel(const std::string& model_sess_id) final;

 private:
  void Run();

  int gpu_id_;
  std::atomic_bool running_;
  std::thread thread_;
  std::unordered_map<std::string, std::shared_ptr<ModelInstance> > models_;
  std::mutex mu_;
};

class GpuExecutorNoMultiBatching : public GpuExecutor {
 public:
  GpuExecutorNoMultiBatching(int gpu_id);

  inline int gpu_id() { return gpu_id_; }

  void Start();

  void Stop();

  void AddModel(const std::string& model_sess_id,
                std::shared_ptr<ModelInstance> model);

  void RemoveModel(const std::string& model_sess_id);

 private:
  class ModelThread {
   public:
    ModelThread(int gpu_id, const std::string& model_sess_id,
                std::shared_ptr<ModelInstance> model) :
        gpu_id_(gpu_id),
        model_session_id_(model_sess_id),
        model_(model),
        running_(true) {
      thread_ = std::thread(&ModelThread::Run, this);
    }

    void Stop() {
      running_ = false;
      thread_.join();
    }

   private:
    void Run() {
#ifdef USE_CAFFE
      caffe::Caffe::set_mode(caffe::Caffe::GPU);
      caffe::Caffe::set_release_memory(false);
      caffe::Caffe::SetDevice(gpu_id_);
#endif
      NEXUS_CUDA_CHECK(cudaSetDevice(gpu_id_));
      auto min_cycle = std::chrono::microseconds(50);
      LOG(INFO) << "ModelThread started for " << model_session_id_;
      while (running_) {
        auto cycle_start = std::chrono::high_resolution_clock::now();
        model_->Forward();
        auto cycle_end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
            cycle_end - cycle_start);
        if (duration < min_cycle) {
          // ensure the cycle to be at least min_cycle to avoid acquiring lock
          // too frequently in the ModelInstance
          std::this_thread::sleep_for(min_cycle - duration);
        }
      }
      LOG(INFO) << "ModelThread " << model_session_id_ << " stopped";
    }
    
    int gpu_id_;
    std::string model_session_id_;
    std::shared_ptr<ModelInstance> model_;
    std::atomic_bool running_;
    std::thread thread_;
  };

  int gpu_id_;
  BackendServer* server_;
  std::mutex mu_;
  std::unordered_map<std::string, std::shared_ptr<ModelThread> > threads_;
};

} // namespace backend
} // namespace nexus

#endif // NEXUS_BACKEND_BASE_GPU_EXECUTOR_H_
