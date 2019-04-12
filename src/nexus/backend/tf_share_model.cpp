#include "nexus/backend/tf_share_model.h"
#include "nexus/backend/utils.h"
#include <boost/filesystem.hpp>

namespace fs = boost::filesystem;

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
  fs::path model_dir = fs::path(model_info_["model_dir"].as<std::string>());
  for (const auto& iter : tf_share_info_->suffix_models) {
    if (iter.second.class_names.empty())
      continue;
    fs::path cns_path = model_dir / iter.second.class_names;
    std::unordered_map<int, std::string> map;
    LoadClassnames(cns_path.string(), &map);
    classnames_.emplace(iter.first, map);
  }
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
  batch_.store(batch);
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
  std::vector<int32_t> slice_beg(num_suffixes_, 0);
  std::vector<int32_t> slice_len(num_suffixes_, 0);
  auto &tasks = batch_task->tasks();
  ModelSession model_sess;
  for (size_t i = 0; i < tasks.size();) {
    const auto &model_sess_id = tasks[i]->query.model_session_id();
    ParseModelID(model_sess_id, &model_sess);
    auto iter = tf_share_info_->suffix_models.find(model_sess.model_name());
    CHECK(iter != tf_share_info_->suffix_models.end())
      << "Cannot find model " << model_sess_id << " in " << tf_share_info_->hack_internal_id;
    const auto suffix_index = iter->second.suffix_index;
    CHECK_EQ(slice_len[suffix_index], 0) << "Detected non-consecutive BatchTask";
    slice_beg[suffix_index] = i;
    while (i < tasks.size() && tasks[i]->query.model_session_id() == model_sess_id)
      ++i;
    slice_len[suffix_index] = i - slice_beg[suffix_index];
  }

  auto &m = *tf_model_;
  size_t batch_size = batch_task->batch_size();
  auto in_tensor = m.input_tensors_[batch_task->GetInputArray()->tag()]->Slice(0, batch_size);
  m.set_slice_tensor(m.slice_beg_tensor_, slice_beg);
  m.set_slice_tensor(m.slice_end_tensor_, slice_len);

  std::vector<tf::Tensor> out_tensors;
  std::vector<std::pair<std::string, tf::Tensor>> inputs;
  inputs.emplace_back(tf_share_info_->input_layer, in_tensor);
  inputs.emplace_back(tf_share_info_->slice_beg_vector, *m.slice_beg_tensor_);
  inputs.emplace_back(tf_share_info_->slice_len_vector, *m.slice_end_tensor_);
  tf::Status status = m.session_->Run(inputs, m.output_layers_, {}, &out_tensors);
  if (!status.ok()) {
    LOG(FATAL) << "Failed to run tensorflow: " << status.ToString();
    return;
  }

  std::vector<std::shared_ptr<Output>> outputs(tasks.size());
  std::unordered_map<std::string, std::shared_ptr<Array>> arr;
  for (size_t i = 0; i < m.output_layers_.size(); ++i) {
    const auto& name = m.output_layers_[i];
    size_t num_elements = m.output_sizes_[name];
    auto out_array = batch_task->GetOutputArray(name);
    arr.clear();
    const int32_t beg = slice_beg[i], len = slice_len[i];
    const auto& out_tensor = out_tensors[i];
    CHECK_EQ(out_tensor.NumElements(), num_elements * len);
    for (int32_t j = 0; j < len; ++j) {
      auto out_buf = out_array->Slice(num_elements * j, num_elements);
      const char* tensor_data_base = out_tensor.tensor_data().data();
      const char* tensor_data = tensor_data_base + j * num_elements * sizeof(float);
      Memcpy(out_buf->Data<float>(), cpu_device_, tensor_data, cpu_device_, num_elements * sizeof(float));
      arr[name] = out_buf;
      const auto& input = batch_task->inputs()[beg + j];
      outputs[beg + j] = std::make_shared<Output>(input->task_id, input->index, arr);
    }
  }
  batch_task->set_outputs(outputs);
}

void TFShareModel::Postprocess(std::shared_ptr<Task> task) {
  ModelSession model_sess;
  ParseModelID(task->query.model_session_id(), &model_sess);
  auto suffix_info_iter = tf_share_info_->suffix_models.find(model_sess.model_name());
  CHECK(suffix_info_iter != tf_share_info_->suffix_models.end());
  const auto &suffix_info = suffix_info_iter->second;
  auto &m = *tf_model_;
  const QueryProto& query = task->query;
  QueryResultProto* result = &task->result;
  result->set_status(CTRL_OK);
  for (const auto& output : task->outputs) {
    if (suffix_info.type == "classification") {
      auto out_arr = output->arrays.at(suffix_info.output_layer);
      auto* out_data = out_arr->Data<float>();
      size_t output_size = m.output_sizes_.at(suffix_info.output_layer);
      auto iter = classnames_.find(suffix_info.model_name);
      if (iter == classnames_.end()) {
        PostprocessClassification(query, out_data, output_size, result);
      } else {
        PostprocessClassification(query, out_data, output_size, result, &iter->second);
      }
    } else if (suffix_info.type == "detection") {
      int im_height = task->attrs["im_height"].as<int>();
      int im_width = task->attrs["im_width"].as<int>();
      m.MarshalDetectionResult(query, output, im_height, im_width, result);
    } else {
      std::ostringstream oss;
      oss << "Unsupported model type " << type() << " for " << framework();
      result->set_status(MODEL_TYPE_NOT_SUPPORT);
      result->set_error_message(oss.str());
      break;
    }
  }
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
