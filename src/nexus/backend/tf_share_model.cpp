#include "nexus/backend/tf_share_model.h"

namespace nexus {
namespace backend {

TFShareModel::TFShareModel(int gpu_id, const ModelInstanceConfig &config) :
    ModelInstance(gpu_id, config) {
  std::shared_ptr<TFShareInfo> tf_share_info;
  for (int i = 0; i < config.model_session_size(); ++i) {
    const auto &model_session = config.model_session(i);
    CHECK_EQ(model_session_.latency_sla(), model_session.latency_sla());
    const auto &model_name = model_session.model_name();
    auto info = ModelDatabase::Singleton().GetTFShareInfo(model_name);
    CHECK(info != nullptr) << "Cannot find TFShare model " << model_name;
    if (!tf_share_info)
      tf_share_info = info;
    CHECK(tf_share_info == info) << "Cannot run different TFShare models in a single instance. ("
        << tf_share_info->hack_internal_id << ") vs (" << info->hack_internal_id << ").";
  }

  ModelInstanceConfig model_config;
  {
    auto model_session = model_config.add_model_session();
    model_session->CopyFrom(model_session_);
    model_session->set_framework("tensorflow");
    model_session->set_model_name(tf_share_info->hack_internal_id);
    model_config.set_batch(batch_);
    model_config.set_max_batch(max_batch_);
  }
  CreateModelInstance(gpu_id, model_config, &tf_model_);
}

void TFShareModel::set_batch(size_t batch) {
  tf_model_->set_batch(batch);
}

Shape TFShareModel::InputShape() {
  return tf_model_->InputShape();
}

std::unordered_map<std::string, Shape> TFShareModel::OutputShapes() {
  return tf_model_->OutputShapes();
}

ArrayPtr TFShareModel::CreateInputGpuArray() {
  return tf_model_->CreateInputGpuArray();
}

std::unordered_map<std::string, ArrayPtr> TFShareModel::GetOutputGpuArrays() {
  CHECK(false) << "Doesn't support in-place output in GPU memory";
  return {};
}

void TFShareModel::Preprocess(std::shared_ptr<Task> task) {
  tf_model_->Preprocess(task);
}

void TFShareModel::Forward(std::shared_ptr<BatchTask> batch_task) {
  tf_model_->Forward(batch_task);
}

void TFShareModel::Postprocess(std::shared_ptr<Task> task) {
  tf_model_->Postprocess(task);
}

bool TFShareModel::AddModelSession(const ModelSession& model_sess) {
  std::lock_guard<std::mutex> lock(loaded_suffixes_mutex_);
  auto pair = loaded_suffixes_.emplace(model_sess.model_name());
  return pair.second;
}

bool TFShareModel::RemoveModelSession(const ModelSession& model_sess) {
  std::lock_guard<std::mutex> lock(loaded_suffixes_mutex_);
  size_t n = loaded_suffixes_.erase(model_sess.model_name());
  return n > 0;
}

size_t TFShareModel::num_model_sessions() {
  std::lock_guard<std::mutex> lock(loaded_suffixes_mutex_);
  return loaded_suffixes_.size();
}

}
}
