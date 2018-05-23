#include <chrono>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <thread>

#include "nexus/common/model_db.h"
#include "nexus/common/time_util.h"
#include "nexus/backend/caffe_densecap_model.h"
#include "nexus/backend/caffe_model.h"
#include "nexus/backend/caffe2_model.h"
#include "nexus/backend/darknet_model.h"
#include "nexus/backend/model_ins.h"
#include "nexus/backend/share_prefix_model.h"
#include "nexus/backend/tensorflow_model.h"

DEFINE_int32(count_interval, 1, "Interval to count number of requests in sec");

namespace nexus {
namespace backend {

ModelInstance::ModelInstance(int gpu_id, const ModelInstanceConfig& config) :
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
  counter_ = MetricRegistry::Singleton().CreateIntervalCounter(
      FLAGS_count_interval);
}

ModelInstance::~ModelInstance() {
  MetricRegistry::Singleton().RemoveMetric(counter_);
}

void ModelInstance::set_batch(size_t batch) {
  CHECK_LE(batch, max_batch_) << "Batch size must be less than max_batch";
  batch_.store(batch);
}

std::shared_ptr<ModelInstance> CreateModelInstance(
    int gpu_id, const ModelInstanceConfig& config) {
  std::shared_ptr<ModelInstance> model;
  if (config.model_session_size() > 1) {
    model = std::make_shared<SharePrefixModel>(gpu_id, config);
    return model;
  }

  std::string framework = config.model_session(0).framework();
  std::string model_name = config.model_session(0).model_name();
#ifdef USE_DARKNET
  if (framework == "darknet") {
    model = std::make_shared<DarknetModel>(gpu_id, config);
  } else
#endif
#ifdef USE_CAFFE
  if (framework == "caffe") {
    if (model_name == "densecap") {
      model = std::make_shared<CaffeDenseCapModel>(gpu_id, config);
    } else {
      model = std::make_shared<CaffeModel>(gpu_id, config);
    }
  } else
#endif
#ifdef USE_CAFFE2
  if (framework == "caffe2") {
    model = std::make_shared<Caffe2Model>(gpu_id, config);
  } else
#endif
#ifdef USE_TENSORFLOW
  if (framework == "tensorflow") {
      model = std::make_shared<TensorflowModel>(gpu_id, config);
  } else
#endif
  {
    LOG(FATAL) << "Unknown framework " << framework;
  }
  return model;
}

} // namespace backend
} // namespace nexus
