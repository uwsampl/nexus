#include "nexus/backend/share_prefix_model.h"
#include "nexus/common/model_db.h"

namespace nexus {
namespace backend {

SharePrefixModel::SharePrefixModel(int gpu_id,
                                   const ModelInstanceConfig& config) : 
    ModelInstance(gpu_id, config) {
  prefix_length_ = -1;
  std::string model_id = ModelSessionToModelID(model_session_);
  for (int i = 1; i < config.model_session_size(); ++i) {
    std::string other_model_id = ModelSessionToModelID(config.model_session(i));
    int pref_len = ModelDatabase::Singleton().GetSharePrefixLength(
        model_id, other_model_id);
    if (prefix_length_ < 0 || pref_len < prefix_length_) {
      prefix_length_ = pref_len;
    }
  }
  CHECK_GT(prefix_length_, 0) << "No prefix layers shared among models";
  
  ModelInstanceConfig prefix_cfg;
  prefix_cfg.add_model_session()->CopyFrom(model_session_);
  prefix_cfg.set_batch(batch_);
  prefix_cfg.set_max_batch(max_batch_);
  prefix_cfg.set_end_index(prefix_length_);
  CreateModelInstance(gpu_id, prefix_cfg, &prefix_model_);
  for (auto iter : prefix_model_->OutputShapes()) {
    prefix_output_name_ = iter.first;
    prefix_output_shape_ = iter.second;
  }
  prefix_batch_output_arr_ = prefix_model_->GetOutputGpuArrays();

  max_suffix_output_size_ = 0; 
  for (int i = 0; i < config.model_session_size(); ++i) {
    ModelInstanceConfig suffix_cfg;
    suffix_cfg.add_model_session()->CopyFrom(config.model_session(i));
    suffix_cfg.set_batch(batch_);
    suffix_cfg.set_max_batch(max_batch_);
    suffix_cfg.set_start_index(prefix_length_);
    suffix_cfg.set_input_name(prefix_output_name_);
    // Don't include batch dim in the shape, so start from 1
    for (uint i = 1; i < prefix_output_shape_.ndims(); ++i) {
      suffix_cfg.add_input_shape(prefix_output_shape_.dim(i));
    }
    //LOG(INFO) << suffix_cfg.DebugString();

    std::unique_ptr<ModelInstance> suffix_model;
    CreateModelInstance(gpu_id, suffix_cfg, &suffix_model);
    auto model_sess_id = suffix_model->model_session_id();
    suffix_input_arrays_.emplace(model_sess_id,
                                 suffix_model->CreateInputGpuArray());
    auto suffix_output_shape = suffix_model->OutputShapes();
    CHECK_EQ(suffix_output_shape.size(), 1) << "All models must have only one "
        "output in the prefix batching";
    for (auto iter : suffix_output_shape) {
      size_t size = iter.second.NumElements(1);
      suffix_output_sizes_.emplace(model_sess_id, size);
      suffix_output_names_.emplace(model_sess_id, iter.first);
      if (size > max_suffix_output_size_) {
        max_suffix_output_size_ = size;
      }
    }
    suffix_models_.emplace(model_sess_id, std::move(suffix_model));
  }

  LOG(INFO) << "Prefix output shape: " << prefix_output_shape_ <<
      ", max suffix output size: " << max_suffix_output_size_;
}

void SharePrefixModel::set_batch(size_t batch) {
  CHECK_LE(batch, max_batch_) << "Batch size must be less than max_batch";
  batch_.store(batch);
  prefix_model_->set_batch(batch);
  for (auto& iter : suffix_models_) {
    iter.second->set_batch(batch);
  }
}

Shape SharePrefixModel::InputShape() {
  return prefix_model_->InputShape();
}

std::unordered_map<std::string, Shape> SharePrefixModel::OutputShapes() {
  std::lock_guard<std::mutex> lock(suffix_mu_);
  return {{"output", Shape({max_batch_, max_suffix_output_size_})}};
}

ArrayPtr SharePrefixModel::CreateInputGpuArray() {
  return prefix_model_->CreateInputGpuArray();
}

std::unordered_map<std::string, ArrayPtr> SharePrefixModel::GetOutputGpuArrays() {
  // Doesn't support in-place output in GPU memory
  return {};
}

void SharePrefixModel::Preprocess(std::shared_ptr<Task> task) {
  prefix_model_->Preprocess(task);
}

void SharePrefixModel::Forward(std::shared_ptr<BatchTask> batch_task) {
  // Do not allow to change the shared models during the forwarding
  // auto t1 = Clock::now();
  std::unordered_map<std::string, std::shared_ptr<ModelInstance> > suffix_models;
  {
    std::lock_guard<std::mutex> lock(suffix_mu_);
    suffix_models = suffix_models_;
  }
  auto suffix_output_arr = batch_task->GetOutputArray("output");

  // Replace the origin output arrays by prefix output GPU array and
  // Forward prefix model
  batch_task->SetOutputArrays(prefix_batch_output_arr_);
  VLOG(1) << "Forward prefix model " << prefix_model_->model_session_id() <<
      " with batch size " << batch_task->batch_size();

#if 1
  prefix_model_->ForwardAsync(batch_task);
  // auto t2 = Clock::now();

  uint32_t batch_size = batch_task->batch_size();
  auto tasks = batch_task->tasks();
  size_t offset = 0;
  float* prefix_batch_output_ptr = prefix_batch_output_arr_.
                                   at(prefix_output_name_)->Data<float>();
  size_t prefix_output_nfloats = prefix_output_shape_.NumElements(1);
  std::vector<ArrayPtr> suffix_input_arrs;
  std::vector<std::shared_ptr<BatchTask> > suffix_batch_tasks;
  
  // Prepare the suffix batch tasks
  for (uint32_t i = 0; i < batch_size; ++i) {
    auto task = tasks[i];
    auto model_sess_id = task->query.model_session_id();
    auto suffix_model = suffix_models.at(model_sess_id);
    task->suffix_model = suffix_model;
    auto suffix_batch_task = std::make_shared<BatchTask>(1);

    // Set input array in batch task
    auto suffix_input_arr = suffix_model->CreateInputGpuArrayWithRawPointer(
        prefix_batch_output_ptr, prefix_output_nfloats);
    prefix_batch_output_ptr += prefix_output_nfloats;
    suffix_batch_task->SetInputArray(suffix_input_arr);
    // Append input
    auto suffix_input = std::make_shared<Input>(
        task->deadline(), task->task_id, batch_task->inputs()[i]->index,
        suffix_input_arr);
    suffix_batch_task->AppendInput(suffix_input, task);

    // Set output array in batch task
    size_t suffix_output_nfloats = suffix_output_sizes_.at(model_sess_id);
    auto out_arr = suffix_output_arr->Slice(offset, suffix_output_nfloats);
    offset += suffix_output_nfloats;
    suffix_batch_task->SetOutputArrays({{
          suffix_output_names_.at(model_sess_id), out_arr }});

    suffix_input_arrs.push_back(suffix_input_arr);
    suffix_batch_tasks.push_back(suffix_batch_task);
  }
  // auto t3 = Clock::now();

  prefix_model_->WaitOutput(batch_task);
  // auto t4 = Clock::now();
  
  std::vector<std::shared_ptr<Output> > batch_outputs;
  for (int i = 0; i < batch_size; ++i) {
    auto task = tasks[i];
    auto suffix_batch_task = suffix_batch_tasks[i];
    task->suffix_model->Forward(suffix_batch_task);
    batch_outputs.push_back(suffix_batch_task->outputs()[0]);
    task->suffix_model->RemoveInputGpuArray(suffix_input_arrs[i]);
  }

  batch_task->set_outputs(batch_outputs);
  // auto t5 = Clock::now();
  // LOG(INFO) << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() <<
  //     " us, " << std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count() <<
  //     " us, "  << std::chrono::duration_cast<std::chrono::microseconds>(t4 - t3).count() <<
  //     " us, "  << std::chrono::duration_cast<std::chrono::microseconds>(t5 - t4).count() <<
  //     " us";

#else
  prefix_model_->Forward(batch_task);
  // auto t2 = Clock::now();
  
  // Append the outputs of prefix model to the input queue of corresponding
  // suffix model
  std::unordered_map<std::string, std::shared_ptr<BatchTask> > suffix_tasks;
  std::unordered_map<std::string, std::vector<int> > suffix_indices;
  auto prefix_outputs = batch_task->outputs();
  auto tasks = batch_task->tasks();
  for (int i = 0; i < prefix_outputs.size(); ++i) {
    auto prefix_output = prefix_outputs[i];
    auto task = tasks[i];
    auto model_sess_id = task->query.model_session_id();
    auto suffix_input = std::make_shared<Input>(
        task->deadline(), task->task_id, prefix_output->index,
        prefix_output->arrays.at(prefix_output_name_));
    if (suffix_tasks.find(model_sess_id) == suffix_tasks.end()) {
      auto suffix_task = std::make_shared<BatchTask>(
          suffix_models[model_sess_id]->max_batch());
      suffix_task->SetInputArray(suffix_input_arrays_[model_sess_id]);
      suffix_tasks.emplace(model_sess_id, suffix_task);
      suffix_indices.emplace(model_sess_id, std::vector<int>{});
    }
    suffix_tasks.at(model_sess_id)->AppendInput(suffix_input, task);
    suffix_indices.at(model_sess_id).push_back(i);
  }

  // Slice the output array for each suffix model and forward suffix model
  size_t offset = 0;
  std::vector<std::shared_ptr<Output> > batch_outputs(batch_task->batch_size());
  for (auto iter : suffix_tasks) {
    auto& model_sess_id = iter.first;
    auto suffix_task = iter.second;
    uint32_t batch = suffix_task->batch_size();
    size_t nfloats = batch * suffix_output_sizes_.at(model_sess_id);
    auto out_arr = suffix_output_arr->Slice(offset, nfloats);
    offset += nfloats;
    suffix_task->SetOutputArrays({{
          suffix_output_names_.at(model_sess_id), out_arr }});
    VLOG(1) << "Forward suffix model " << model_sess_id <<
        " with batch size " << suffix_task->batch_size();
    suffix_models.at(model_sess_id)->ForwardAsync(suffix_task);
  }
  for (auto iter : suffix_tasks) {
    auto& model_sess_id = iter.first;
    auto suffix_task = iter.second;
    auto origin_indices = suffix_indices.at(model_sess_id);
    suffix_models.at(model_sess_id)->WaitOutput(suffix_task);
    auto outputs = suffix_task->outputs();
    for (int i = 0; i < outputs.size(); ++i) {
      batch_outputs[origin_indices[i]] = outputs[i];
    }
  }
  // Set suffix outputs into the batch_task outputs
  batch_task->set_outputs(batch_outputs);
  // auto t3 = Clock::now();
  // LOG(INFO) << "Prefix: " << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() <<
  //     " us, suffix: " << std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count() <<
  //     " us";
#endif
}

void SharePrefixModel::Postprocess(std::shared_ptr<Task> task) {
  task->suffix_model->Postprocess(task);
}

int SharePrefixModel::num_model_sessions() {
  std::lock_guard<std::mutex> lock(suffix_mu_);
  return suffix_models_.size();
}

std::vector<std::string> SharePrefixModel::ModelSessions() {
  std::lock_guard<std::mutex> lock(suffix_mu_);
  std::vector<std::string> ret;
  for (auto& iter : suffix_models_) {
    ret.push_back(iter.first);
  }
  return ret;
}

bool SharePrefixModel::HasModelSession(const std::string& model_sess_id) {
  std::lock_guard<std::mutex> lock(suffix_mu_);
  return (suffix_models_.count(model_sess_id) > 0);
}

bool SharePrefixModel::AddModelSession(const ModelSession& model_sess) {
  std::string model_id = ModelSessionToModelID(model_sess);
  int pref_len = -1;
  for (auto& iter : suffix_models_) {
    ModelSession base_model_sess;
    ParseModelSession(iter.first, &base_model_sess);
    std::string base_model_id = ModelSessionToModelID(base_model_sess);
    pref_len = ModelDatabase::Singleton().GetSharePrefixLength(
        model_id, base_model_id);
    break;
  }
  if (pref_len != prefix_length_) {
    LOG(ERROR) << "New prefix length is not same as old (" << pref_len <<
        " vs " << prefix_length_ << ")";
    return false;
  }

  ModelInstanceConfig suffix_cfg;
  suffix_cfg.add_model_session()->CopyFrom(model_sess);
  suffix_cfg.set_batch(batch_);
  suffix_cfg.set_max_batch(max_batch_);
  suffix_cfg.set_start_index(prefix_length_);
  suffix_cfg.set_input_name(prefix_output_name_);
  // Don't include batch dim in the shape, so start from 1
  for (uint i = 1; i < prefix_output_shape_.ndims(); ++i) {
    suffix_cfg.add_input_shape(prefix_output_shape_.dim(i));
  }
  //LOG(INFO) << suffix_cfg.DebugString();

  std::unique_ptr<ModelInstance> suffix_model;
  CreateModelInstance(gpu_id_, suffix_cfg, &suffix_model);
  //auto suffix_model = CreateModelInstance(gpu_id_, suffix_cfg);
  auto model_sess_id = suffix_model->model_session_id();

  std::lock_guard<std::mutex> lock(suffix_mu_);
  suffix_input_arrays_.emplace(model_sess_id,
                               suffix_model->CreateInputGpuArray());
  auto suffix_output_shape = suffix_model->OutputShapes();
  CHECK_EQ(suffix_output_shape.size(), 1) << "All models must have only one "
      "output in the prefix batching";
  for (auto iter : suffix_output_shape) {
    size_t size = iter.second.NumElements(1);
    suffix_output_sizes_.emplace(model_sess_id, size);
    suffix_output_names_.emplace(model_sess_id, iter.first);
    if (size > max_suffix_output_size_) {
      max_suffix_output_size_ = size;
    }
  }
  suffix_models_.emplace(model_sess_id, std::move(suffix_model));
  return true;
}

void SharePrefixModel::RemoveModelSession(const std::string& model_sess_id) {
  std::lock_guard<std::mutex> lock(suffix_mu_);
  suffix_models_.erase(model_sess_id);
  suffix_input_arrays_.erase(model_sess_id);
  suffix_output_names_.erase(model_sess_id);
  suffix_output_sizes_.erase(model_sess_id);
}

} // namespace backend
} // namespace nexus
