#include "nexus/backend/tf_share_model.h"

namespace nexus {
namespace backend {

TFShareModel::TFShareModel(int gpu_id, const ModelInstanceConfig &config) :
    ModelInstance(gpu_id, config),
    num_suffixes_(0) {
  for (int i = 0; i < config.model_session_size(); ++i) {
    const auto &model_session = config.model_session(i);
    CHECK_EQ(model_session_.latency_sla(), model_session.latency_sla());
    const auto &model_name = model_session.model_name();
    auto info = ModelDatabase::Singleton().GetTFShareInfo(model_name);
    CHECK(info != nullptr) << "Cannot find TFShare model " << model_name;
    if (!tf_share_info_)
      tf_share_info_ = info;
    CHECK(tf_share_info_ == info) << "Cannot run different TFShare models in a single instance. ("
                                  << tf_share_info_->hack_internal_id << ") vs (" << info->hack_internal_id << ").";
  }
  num_suffixes_ = tf_share_info_->suffix_models.size();

  ModelInstanceConfig model_config;
  {
    auto model_session = model_config.add_model_session();
    model_session->CopyFrom(model_session_);
    model_session->set_framework("tensorflow");
    model_session->set_model_name(tf_share_info_->hack_internal_id);
    model_config.set_batch(batch_);
    model_config.set_max_batch(max_batch_);
  }
  std::unique_ptr<ModelInstance> model;
  CreateModelInstance(gpu_id, model_config, &model);
  auto* ptr = dynamic_cast<TensorflowModel*>(model.release());
  CHECK_NE(ptr, nullptr);
  tf_model_.reset(ptr);
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
  std::vector<int32_t> slice_beg(num_suffixes_, -1);
  std::vector<int32_t> slice_end(num_suffixes_, -1);
  auto &tasks = batch_task->tasks();
  ModelSession model_sess;
  for (size_t i = 0; i < tasks.size();) {
    const auto &model_sess_id = tasks[i]->query.model_session_id();
    ParseModelID(model_sess_id, &model_sess);
    auto iter = tf_share_info_->suffix_models.find(model_sess.model_name());
    CHECK(iter != tf_share_info_->suffix_models.end())
      << "Cannot find model " << model_sess_id << " in " << tf_share_info_->hack_internal_id;
    const auto suffix_index = iter->second.suffix_index;
    CHECK_EQ(slice_beg[suffix_index], -1) << "Detected non-consecutive BatchTask";
    slice_beg[suffix_index] = i;
    while (i < tasks.size() && tasks[i]->query.model_session_id() == model_sess_id)
      ++i;
    slice_end[suffix_index] = i;
  }
  for (auto &v : slice_beg) if (v == -1) v = 0;
  for (auto &v : slice_end) if (v == -1) v = 0;

  auto &m = *tf_model_;
  size_t batch_size = batch_task->batch_size();
  auto in_tensor = m.input_tensors_[batch_task->GetInputArray()->tag()]->Slice(0, batch_size);
  m.set_slice_tensor(m.slice_beg_tensor_, slice_beg);
  m.set_slice_tensor(m.slice_end_tensor_, slice_end);

  std::vector<tf::Tensor> out_tensors;
  std::vector<std::pair<std::string, tf::Tensor>> inputs;
  inputs.emplace_back(tf_share_info_->input_layer, in_tensor);
  inputs.emplace_back(tf_share_info_->slice_beg_vector, *m.slice_beg_tensor_);
  inputs.emplace_back(tf_share_info_->slice_end_vector, *m.slice_end_tensor_);
  tf::Status status = m.session_->Run(inputs, m.output_layers_, {}, &out_tensors);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to run tensorflow: " << status.ToString();
    return;
  }
  std::unordered_map<std::string, Slice> slices;
  for (uint i = 0; i < m.output_layers_.size(); ++i) {
    const auto& name = m.output_layers_[i];
    const char* tensor_data = out_tensors[i].tensor_data().data();
    size_t nfloats = out_tensors[i].NumElements();
    auto out_arr = batch_task->GetOutputArray(name);
    float* out_data = out_arr->Data<float>();
    Memcpy(out_data, cpu_device_, tensor_data, cpu_device_, nfloats * sizeof(float));
    slices.emplace(name, Slice(batch_size, m.output_sizes_.at(name)));
  }
  batch_task->SliceOutputBatch(slices);
}

void TFShareModel::Postprocess(std::shared_ptr<Task> task) {
  tf_model_->Postprocess(task);
}

bool TFShareModel::AddModelSession(const ModelSession &model_sess) {
  std::lock_guard<std::mutex> lock(loaded_suffixes_mutex_);
  auto pair = loaded_suffixes_.emplace(model_sess.model_name());
  return pair.second;
}

bool TFShareModel::RemoveModelSession(const ModelSession &model_sess) {
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
