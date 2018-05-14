#ifndef NEXUS_BACKEND_MODEL_INS_H_
#define NEXUS_BACKEND_MODEL_INS_H_

#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <yaml-cpp/yaml.h>

#include "nexus/backend/batch_task.h"
#include "nexus/backend/task.h"
#include "nexus/common/data_type.h"
#include "nexus/common/metric.h"
#include "nexus/common/model_def.h"
#include "nexus/proto/nnquery.pb.h"

namespace nexus {
namespace backend {

class ModelInstance {
 public:
  ModelInstance(int gpu_id, const ModelInstanceConfig& config);

  virtual ~ModelInstance();

  int gpu_id() const { return gpu_id_; }

  std::string framework() const { return model_session_.framework(); }

  std::string model_name() const { return model_session_.model_name(); }

  std::string model_session_id() const { return model_session_id_; }

  int version() const { return model_session_.version(); }

  std::string type() const { return model_info_["type"].as<std::string>(); }
  
  uint32_t batch() const { return batch_.load(); }

  void set_batch(size_t batch);

  uint32_t max_batch() const { return max_batch_; }
      
  std::string profile_id() const {
    return ModelSessionToProfileID(model_session_);
  }

  std::shared_ptr<IntervalCounter> counter() const { return counter_; }
  
  virtual Shape InputShape() const = 0;
  /*!
   * \brief
   * \return
   */
  virtual std::unordered_map<std::string, Shape> OutputShapes() const = 0;
  /*!
   * \brief Create input array in GPU memory that can hold input data up to
   * max batch size. This function can be called multiple times for double
   * buffering.
   * \return Array pointer with buffer allocated in GPU memory.
   */
  virtual ArrayPtr CreateInputGpuArray() = 0;
  /*!
   * \brief Get output array in GPU memory for storing output data up to
   * max batch size. This function should be only called once.
   * \return Map from output name to array pointer with buffer allocated
   * in GPU memory.
   * Empty map might be returned if a model doesn't support output in GPU memory.
   */
  virtual std::unordered_map<std::string, ArrayPtr> GetOutputGpuArrays() = 0;
  /*!
   * \brief
   * \return
   */
  virtual void Preprocess(std::shared_ptr<Task> task) = 0;
  /*!
   * \brief
   * \return
   */
  virtual void Forward(std::shared_ptr<BatchTask> batch_task) = 0;
  /*!
   * \brief
   * \return
   */
  virtual void Postprocess(std::shared_ptr<Task> task) = 0;

 protected:
  int gpu_id_;
  ModelSession model_session_;
  std::string model_session_id_;
  std::atomic<uint32_t> batch_;
  uint32_t max_batch_;
  YAML::Node model_info_;
  std::shared_ptr<IntervalCounter> counter_;
  CPUDevice* cpu_device_;
  GPUDevice* gpu_device_;
};

using ModelInstancePtr = std::shared_ptr<ModelInstance>;

ModelInstancePtr CreateModelInstance(int gpu_id,
                                     const ModelInstanceConfig& config);

} // namespace backend
} // namespace nexus

#endif // NEXUS_BACKEND_MODEL_INS_H_
