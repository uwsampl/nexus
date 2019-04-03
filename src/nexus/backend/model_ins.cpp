#include "nexus/backend/caffe_densecap_model.h"
#include "nexus/backend/caffe_model.h"
#include "nexus/backend/caffe2_model.h"
#include "nexus/backend/darknet_model.h"
#include "nexus/backend/model_ins.h"
#include "nexus/backend/share_prefix_model.h"
#include "nexus/backend/tensorflow_model.h"
#include "nexus/backend/tf_share_model.h"

#include <glog/logging.h>

namespace nexus {
namespace backend {

void CreateModelInstance(int gpu_id, const ModelInstanceConfig& config,
                         std::unique_ptr<ModelInstance>* model) {
  auto beg = Clock::now();
  if (config.model_session_size() > 1) {
    LOG(INFO) << "Create prefix model";
    model->reset(new SharePrefixModel(gpu_id, config));
  } else {
    std::string framework = config.model_session(0).framework();
    std::string model_name = config.model_session(0).model_name();
#ifdef USE_DARKNET
    if (framework == "darknet") {
      model->reset(new DarknetModel(gpu_id, config));
    } else
#endif
#ifdef USE_CAFFE
    if (framework == "caffe") {
      if (model_name == "densecap") {
        model->reset(new CaffeDenseCapModel(gpu_id, config));
      } else {
        model->reset(new CaffeModel(gpu_id, config));
      }
    } else
#endif
#ifdef USE_CAFFE2
    if (framework == "caffe2") {
      model->reset(new Caffe2Model(gpu_id, config));
    } else
#endif
#ifdef USE_TENSORFLOW
    if (framework == "tensorflow") {
      model->reset(new TensorflowModel(gpu_id, config));
    } else if (framework == "tf_share") {
      model->reset(new TFShareModel(gpu_id, config));
    } else
#endif
    {
      LOG(FATAL) << "Unknown framework " << framework;
    }
  }

  auto end = Clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
      end - beg);
  LOG(INFO) << "Loading model time: " << duration.count() << "ms";
}

ModelInstance::ModelInstance(int gpu_id, const ModelInstanceConfig &config) :
    gpu_id_(gpu_id),
    model_session_(config.model_session(0)),
    batch_(config.batch()),
    max_batch_(config.max_batch()) {
  CHECK_GT(batch_, 0) << "batch must be greater than 0";
  CHECK_GE(max_batch_, batch_) << "max_batch must be greater than batch";
  std::string model_id = ModelSessionToModelID(model_session_);
  auto info = ModelDatabase::Singleton().GetModelInfo(model_id);
  CHECK(info != nullptr) << "Model not found in the database";
  model_info_ = *info;
  model_session_id_ = ModelSessionToString(model_session_);
  cpu_device_ = DeviceManager::Singleton().GetCPUDevice();
#ifdef USE_GPU
  gpu_device_ = DeviceManager::Singleton().GetGPUDevice(gpu_id);
#endif
  LOG(INFO) << "Construct model " << model_session_id_ << ", batch " <<
            batch_ << ", max batch " << max_batch_;
}

ModelInstance::~ModelInstance() {
  LOG(INFO) << "Deconstruct model " << model_session_id_;
}
void ModelInstance::set_batch(size_t batch) {
  CHECK_LE(batch, max_batch_) << "Batch size must be less than max_batch";
  batch_.store(batch);
}
ArrayPtr ModelInstance::CreateInputGpuArrayWithRawPointer(float *ptr, size_t nfloats) {
  LOG(ERROR) << "Don't support create input gpu array with raw pointer";
  return nullptr;
}
void ModelInstance::RemoveInputGpuArray(ArrayPtr arr) {
  LOG(WARNING) << "Don't support remove input gpu array";
}
void ModelInstance::ForwardAsync(std::shared_ptr<BatchTask> batch_task) {
  LOG(WARNING) << "Don't support async forward";
  Forward(batch_task);
}
void ModelInstance::WaitOutput(std::shared_ptr<BatchTask> batch_task) {
  LOG(WARNING) << "Don't support async forward";
}
} // namespace backend
} // namespace nexus
