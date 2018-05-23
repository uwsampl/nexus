#include "model_exec.h"

namespace nexus {
namespace backend {

ModelExecutor::ModelExecutor(std::shared_ptr<ModelInstance> model,
                             BlockPriorityQueue<Task>& task_queue) :
    model_(model),
    task_queue_(task_queue),
    batch_id_(0) {
  auto gpu_device = DeviceManager::Singleton().GetGPUDevice(model->gpu_id());
  profile_ = ModelDatabase::Singleton().GetModelProfile(
      gpu_device->device_name(), model->profile_id());
  input_array_ = model->CreateInputGpuArray();
}

void ModelExecutor::AddInput(std::shared_ptr<Task> task) {
  std::lock_guard<std::mutex> lock(mu_);
  for (auto input : task->inputs) {
    input_queue_.push(input);
  }
}

void ModelExecutor::Execute() {
  uint64_t batch_id = batch_id_.fetch_add(1);
  auto batch_task = std::make_shared<BatchTask>(batch_id, model_->max_batch());
  batch_task->SetInputArray(input_array_);
  
  auto t1 = std::chrono::high_resolution_clock::now();
  GetBatchInput(batch_task);
  auto t2 = std::chrono::high_resolution_clock::now();
  if (batch_task->batch_size() == 0) {
    return;
  }
  for (auto input : batch_task->inputs()) {
    input->task->timer.Record("batch");
  }
  
  auto t3 = std::chrono::high_resolution_clock::now();
  // Each time recompute output sizes because it might change for prefix model
  std::unordered_map<std::string, size_t> output_sizes;
  for (auto iter : model_->OutputShapes()) {
    output_sizes.emplace(iter.first, iter.second.NumElements(1));
  }
  batch_task->CreateOutputArrays(output_sizes,
                                 DeviceManager::Singleton().GetCPUDevice());
  model_->Forward(batch_task);
  auto t4 = std::chrono::high_resolution_clock::now();
  
  auto memcpy_lat = std::chrono::duration_cast<std::chrono::milliseconds>(
      t2 - t1).count();
  auto forward_lat = std::chrono::duration_cast<std::chrono::milliseconds>(
      t4 - t3).count();
  LOG(INFO) << model_->model_session_id() << " forwards batch " <<
      batch_task->batch_id() << ", size " << batch_task->batch_size() <<
      ", memcpy " << memcpy_lat << " ms, forward " << forward_lat << " ms";

  auto outputs = batch_task->outputs();
  for (auto output : batch_task->outputs()) {
    auto task = output->task;
    if (task->AddOutput(output)) {
      task->stage = kPostprocess;
      task_queue_.push(task);
    }
  }
}

void ModelExecutor::GetBatchInput(std::shared_ptr<BatchTask> batch_task) {
  std::lock_guard<std::mutex> lock(mu_);
  size_t batch_size = input_queue_.size();
  if (batch_size > model_->batch()) {
    batch_size = model_->batch();
  }
  TimePoint finish;
  if (profile_ != nullptr) {
    float latency = profile_->GetForwardLatency(batch_size);
    finish = Clock::now() + std::chrono::microseconds(int(latency));
  }
  while (batch_task->batch_size() < batch_size && !input_queue_.empty()) {
    auto input = std::move(input_queue_.top());
    input_queue_.pop();
    if (profile_ != nullptr && input->task->deadline() < finish) {
      if (input->task->AddVirtualOutput(input->index_in_task)) {
        input->task->stage = kPostprocess;
        task_queue_.push(input->task);
      }
    } else {
      batch_task->AppendInput(input);
    }
  }
}


} // namespace backend
} // namespace nexus

