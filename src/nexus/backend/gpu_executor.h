#ifndef NEXUS_BACKEND_BASE_GPU_EXECUTOR_H_
#define NEXUS_BACKEND_BASE_GPU_EXECUTOR_H_

#include <memory>
#include <thread>
#include <vector>

#include "backend/model_ins.h"

namespace nexus {
namespace backend {

class BackendServer;

class GpuExecutor {
 public:
  virtual void Start() = 0;
  virtual void Stop() = 0;
};

class GpuExecutorMultiBatching : public GpuExecutor {
 public:
  GpuExecutorMultiBatching(int gpu_id, BackendServer* server);

  inline int gpu_id() { return gpu_id_; }

  void Start() final;

  void Stop() final;

 private:
  void Run();

  int gpu_id_;
  BackendServer* server_;
  volatile bool running_;
  std::thread thread_;
};

class GpuExecutorNoMultiBatching : public GpuExecutor {
 public:
  GpuExecutorNoMultiBatching(int gpu_id, BackendServer* server);

  inline int gpu_id() { return gpu_id_; }

  void Start();

  void Stop();

 private:
  void Run(std::shared_ptr<ModelInstance> model);

  int gpu_id_;
  BackendServer* server_;
  volatile bool running_;
  std::vector<std::thread> threads_;
};

} // namespace backend
} // namespace nexus

#endif // NEXUS_BACKEND_BASE_GPU_EXECUTOR_H_
