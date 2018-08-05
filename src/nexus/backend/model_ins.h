#ifndef NEXUS_BACKEND_MODEL_INS_H_
#define NEXUS_BACKEND_MODEL_INS_H_

#include <atomic>
#include <memory>
#include <mutex>
#include <string>

#include "nexus/backend/batch_task.h"
#include "nexus/backend/task.h"
#include "nexus/common/model_db.h"
#include "nexus/common/model_def.h"
#include "nexus/proto/nnquery.pb.h"

namespace nexus {
namespace backend {

/*!
 * \brief ModelInstance is an abstraction for a model instance developed in
 * different frameworks. It includes a set of APIs that is required for
 * pre- and post-process on inputs and outputs of the model, and forwarding the
 * model in a batch.
 */
class ModelInstance {
 public:
  /*!
   * \brief Construct a ModelInstance in given gpu and config.
   * \param gpu_id GPU index
   * \param config Configuration of model instance
   */
  ModelInstance(int gpu_id, const ModelInstanceConfig& config) :
      gpu_id_(gpu_id),
      model_session_(config.model_session(0)),
      batch_(config.batch()),
      max_batch_(config.max_batch()) {
    CHECK_GT(batch_, 0) << "batch must be greater than 0";
    CHECK_GT(max_batch_, 0) << "max_batch must be greater than 0";
    std::string model_id = ModelSessionToModelID(model_session_);
    auto info = ModelDatabase::Singleton().GetModelInfo(model_id);
    CHECK(info != nullptr) << "Model not found in the database";
    model_info_ = *info;
    model_session_id_ = ModelSessionToString(model_session_);
    cpu_device_ = DeviceManager::Singleton().GetCPUDevice();
    gpu_device_ = DeviceManager::Singleton().GetGPUDevice(gpu_id);
  }
  /*! \brief Deconstructs ModelInstance. */
  virtual ~ModelInstance() {}
  /*! \brief Get GPU ID that model is allocated on. */
  int gpu_id() const { return gpu_id_; }
  /*! \brief Get the framework name. */
  std::string framework() const { return model_session_.framework(); }
  /*! \brief Get the model name. */
  std::string model_name() const { return model_session_.model_name(); }
  /*! \brief Get the model version. */
  int version() const { return model_session_.version(); }
  /*! \brief Get the model session ID. */
  std::string model_session_id() const { return model_session_id_; }
  /*! \brief Get the model type. */
  std::string type() const { return model_info_["type"].as<std::string>(); }
  /*! \brief Get the suggested batch size. */
  uint32_t batch() const { return batch_.load(); }
  /*!
   * \brief Set the new batch size. This value should be no greater than
   * max_batch.
   * \param batch Batch size.
   */
  virtual void set_batch(size_t batch) {
    CHECK_LE(batch, max_batch_) << "Batch size must be less than max_batch";
    batch_.store(batch);
  }
  /*! \brief Get the max batch size allowed according to latency SLA. */
  uint32_t max_batch() const { return max_batch_; }
  /*! \brief Get the profile ID for this model instance. */
  std::string profile_id() const {
    return ModelSessionToProfileID(model_session_);
  }
  /*!
   * \brief Get input shape.
   * \return Input shape.
   */
  virtual Shape InputShape() = 0;
  /*!
   * \brief Get output shapes of the model.
   * \return Mapping from output blob name to its shape.
   */
  virtual std::unordered_map<std::string, Shape> OutputShapes() = 0;
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
   * \brief Preprocess the query in the task.
   * \param task Pointer to task.
   */
  virtual void Preprocess(std::shared_ptr<Task> task) = 0;
  /*!
   * \brief Forward batched task through the model on GPU.
   * \param task Pointer to batch task.
   */
  virtual void Forward(std::shared_ptr<BatchTask> batch_task) = 0;

  virtual void ForwardAsync(std::shared_ptr<BatchTask> batch_task) {
    LOG(WARNING) << "Don't support async forward";
    Forward(batch_task);
  }

  virtual void WaitOutput(std::shared_ptr<BatchTask> batch_task) {
    LOG(WARNING) << "Don't support async forward";
  }
  /*!
   * \brief Postprocess the query in the task.
   * \param task Pointer to task.
   */
  virtual void Postprocess(std::shared_ptr<Task> task) = 0;

 protected:
  /*! \brief GPU index */
  int gpu_id_;
  /*! \brief Model session information */
  ModelSession model_session_;
  /*! \brief Model session ID */
  std::string model_session_id_;
  /*! \brief Current batch size to use */
  std::atomic<uint32_t> batch_;
  /*! \brief Maximum batch size allowed given latency SLO */
  uint32_t max_batch_;
  /*! \brief Model metadata loaded from model database */
  YAML::Node model_info_;
  /*! \brief Pointer to CPU device */
  CPUDevice* cpu_device_;
  /*! \brief Pointer to GPU device */
  GPUDevice* gpu_device_;
};

/*!
 * \brief Create a model instance given GPU index and config.
 * \param gpu_id GPU index
 * \param config Model instance configuration
 * \param model Unique pointer to store the model instance
 */
void CreateModelInstance(int gpu_id, const ModelInstanceConfig& config,
                         std::unique_ptr<ModelInstance>* model);

} // namespace backend
} // namespace nexus

#endif // NEXUS_BACKEND_MODEL_INS_H_
