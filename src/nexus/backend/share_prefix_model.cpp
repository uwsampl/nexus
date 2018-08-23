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

#if 0
  // sequentially execute each suffix task
  
  prefix_model_->ForwardAsync(batch_task);

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

    suffix_batch_tasks.push_back(suffix_batch_task);
    suffix_input_arrs.push_back(suffix_input_arr);
  }

  prefix_model_->WaitOutput(batch_task);
  
  std::vector<std::shared_ptr<Output> > batch_outputs;
  for (int i = 0; i < batch_size; ++i) {
    auto task = tasks[i];
    auto suffix_batch_task = suffix_batch_tasks[i];
    task->suffix_model->Forward(suffix_batch_task);
    batch_outputs.push_back(suffix_batch_task->outputs()[0]);
    task->suffix_model->RemoveInputGpuArray(suffix_input_arrs[i]);
  }

  batch_task->set_outputs(batch_outputs);

#else
  // Run suffix model in batch and in parallel
  
  prefix_model_->Forward(batch_task);

  auto tasks = batch_task->tasks();
  float* prefix_batch_output_ptr = prefix_batch_output_arr_.
                                   at(prefix_output_name_)->Data<float>();
  size_t prefix_output_nfloats = prefix_output_shape_.NumElements(1);
  size_t cpu_output_offset = 0;
  std::vector<std::tuple<std::shared_ptr<ModelInstance>,
                         std::shared_ptr<BatchTask>, ArrayPtr> > suffix_batches;

  for (int i = 0; i < tasks.size();) {
    int base = i;
    auto model_sess_id = tasks[i]->query.model_session_id();
    auto suffix_model = suffix_models.at(model_sess_id);
    std::vector<std::shared_ptr<Task> > suffix_tasks;
    suffix_tasks.push_back(tasks[i]);
    ++i;
    while (i < tasks.size() &&
           tasks[i]->query.model_session_id() == model_sess_id) {
      suffix_tasks.push_back(tasks[i]);
      ++i;
    }
    // Slice prefix GPU output array and use as suffix input array
    auto suffix_batch_size = suffix_tasks.size();
    auto suffix_batch_task = std::make_shared<BatchTask>(suffix_batch_size);
    auto suffix_batch_input_arr = suffix_model->CreateInputGpuArrayWithRawPointer(
        prefix_batch_output_ptr, prefix_output_nfloats * suffix_batch_size);
    prefix_batch_output_ptr += prefix_output_nfloats * suffix_batch_size;
    suffix_batch_task->SetInputArray(suffix_batch_input_arr);
    // Slice cpu output array and set as output array
    size_t suffix_output_nfloats = suffix_output_sizes_.at(model_sess_id);
    auto out_arr = suffix_output_arr->Slice(
        cpu_output_offset, suffix_output_nfloats * suffix_batch_size);
    cpu_output_offset += suffix_output_nfloats * suffix_batch_size;
    suffix_batch_task->SetOutputArrays({{
          suffix_output_names_.at(model_sess_id), out_arr }});
    // Append input into suffix batch task. Because each input array has the
    // same memory address as the batch input buffer, no memcpy occurs.
    size_t input_offset = 0;
    for (int j = 0; j < suffix_tasks.size(); ++j) {
      auto task = suffix_tasks[j];
      task->suffix_model = suffix_model;
      auto suffix_input_arr = suffix_batch_input_arr->Slice(
          input_offset, prefix_output_nfloats);
      input_offset += prefix_output_nfloats;
      auto suffix_input = std::make_shared<Input>(
          task->deadline(), task->task_id,
          batch_task->inputs()[base + j]->index, suffix_input_arr);
      suffix_batch_task->AppendInput(suffix_input, task);
    }

    suffix_batches.emplace_back(suffix_model, suffix_batch_task,
                                suffix_batch_input_arr);
  }
  // Wait prefix model finish
  //prefix_model_->WaitOutput(batch_task);
  // Start suffix batch task in parallel
  std::vector<std::shared_ptr<Output> > batch_outputs;
  for (int i = 0; i < suffix_batches.size(); ++i) {
    auto suffix_model = std::get<0>(suffix_batches[i]);
    auto suffix_batch_task = std::get<1>(suffix_batches[i]);
    auto suffix_batch_input_arr = std::get<2>(suffix_batches[i]);
    suffix_model->Forward(suffix_batch_task);
    for (auto output : suffix_batch_task->outputs()) {
      batch_outputs.push_back(output);
    }
    suffix_model->RemoveInputGpuArray(suffix_batch_input_arr);
  }
  batch_task->set_outputs(batch_outputs);
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
