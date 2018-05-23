#ifndef NEXUS_BACKEND_BASE_GPU_EXECUTOR_H_
#define NEXUS_BACKEND_BASE_GPU_EXECUTOR_H_

#include <atomic>
#include <memory>
#include <thread>
#include <vector>
#include <unordered_map>

#include "nexus/backend/model_exec.h"
#include "nexus/backend/model_ins.h"

namespace nexus {
namespace backend {

class GpuExecutor {
 public:
  virtual void Start() = 0;
  virtual void Stop() = 0;
  virtual void AddModel(std::shared_ptr<ModelInstance> model) = 0;
  virtual void RemoveModel(std::shared_ptr<ModelInstance> model) = 0;
  virtual void AddTask(std::shared_ptr<Task> task) = 0;
};

class GpuExecutorMultiBatching : public GpuExecutor {
 public:
  GpuExecutorMultiBatching(int gpu_id,
                           BlockPriorityQueue<Task>& cpu_task_queue);

  inline int gpu_id() { return gpu_id_; }

  void Start() final;

  void Stop() final;

  void AddModel(std::shared_ptr<ModelInstance> model) final;

  void RemoveModel(std::shared_ptr<ModelInstance> model) final;

  void AddTask(std::shared_ptr<Task> task) final;

 private:
  void Run();

  int gpu_id_;
  BlockPriorityQueue<Task>& cpu_task_queue_;
  std::atomic_bool running_;
  std::thread thread_;
  std::unordered_map<std::string, std::shared_ptr<ModelExecutor> > models_;
  std::mutex models_mu_;
};

class GpuExecutorNoMultiBatching : public GpuExecutor {
 public:
  GpuExecutorNoMultiBatching(int gpu_id,
                             BlockPriorityQueue<Task>& cpu_task_queue);

  inline int gpu_id() { return gpu_id_; }

  void Start();

  void Stop();

  void AddModel(std::shared_ptr<ModelInstance> model) final;

  void RemoveModel(std::shared_ptr<ModelInstance> model) final;

  void AddTask(std::shared_ptr<Task> task) final;

 private:
  int gpu_id_;
  BlockPriorityQueue<Task>& cpu_task_queue_;
  std::mutex mu_;
  std::unordered_map<std::string,
                     std::unique_ptr<GpuExecutorMultiBatching> > threads_;
};

} // namespace backend
} // namespace nexus

#endif // NEXUS_BACKEND_BASE_GPU_EXECUTOR_H_
