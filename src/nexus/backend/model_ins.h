#ifndef NEXUS_BACKEND_MODEL_INS_H_
#define NEXUS_BACKEND_MODEL_INS_H_

#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <yaml-cpp/yaml.h>

#include "nexus/backend/batch.h"
#include "nexus/backend/task.h"
#include "nexus/common/metric.h"
#include "nexus/common/model_def.h"
#include "nexus/proto/nnquery.pb.h"

namespace nexus {
namespace backend {

class ModelProfiler;

class ModelInstance {
 public:
  ModelInstance(int gpu_id, const ModelInstanceConfig& config,
                const YAML::Node& model_info);

  virtual ~ModelInstance();

  int gpu_id() const { return gpu_id_; }

  std::string framework() const { return model_session_.framework(); }

  std::string model_name() const { return model_session_.model_name(); }

  std::string model_session_id() const { return model_session_id_; }

  int version() const { return model_session_.version(); }

  std::string type() const { return type_; }
  
  uint32_t batch() const { return batch_.load(); }

  void set_batch(size_t batch);

  uint32_t max_batch() const { return max_batch_; }
      
  std::string profile_id() const {
    return ModelSessionToProfileID(model_session_);
  }

  std::shared_ptr<IntervalCounter> counter() const { return counter_; }

  virtual ArrayPtr CreateInputGpuArray() = 0;

  virtual std::unordered_map<std::string, size_t> OutputSizes() const = 0;

  virtual void Preprocess(std::shared_ptr<Task> task) = 0;

  virtual void Forward(BatchInput* batch_input, BatchOutput* batch_output) = 0;

  virtual void Postprocess(std::shared_ptr<Task> task) = 0;

 protected:
  int gpu_id_;
  ModelSession model_session_;
  std::string model_session_id_;
  std::atomic<uint32_t> batch_;
  uint32_t max_batch_;
  YAML::Node model_info_;
  std::string type_;
  std::shared_ptr<IntervalCounter> counter_;
  CPUDevice* cpu_device_;
  GPUDevice* gpu_device_;
};

using ModelInstancePtr = std::shared_ptr<ModelInstance>;

ModelInstancePtr CreateModelInstance(int gpu_id,
                                     const ModelInstanceConfig& config,
                                     const YAML::Node& info);

} // namespace backend
} // namespace nexus


#endif // NEXUS_BACKEND_MODEL_INS_H_
