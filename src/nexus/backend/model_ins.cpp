#include "nexus/backend/caffe_densecap_model.h"
#include "nexus/backend/caffe_model.h"
#include "nexus/backend/caffe2_model.h"
#include "nexus/backend/darknet_model.h"
#include "nexus/backend/model_ins.h"
#include "nexus/backend/share_prefix_model.h"
#include "nexus/backend/tensorflow_model.h"

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

} // namespace backend
} // namespace nexus
