#include "model_exec.h"

namespace nexus {
namespace backend {

ModelExecutor::ModelExecutor(std::shared_ptr<ModelInstance> model,
                             BlockPriorityQueue<Task>& task_queue) :
    model_(model),
    task_queue_(task_queue),
    batch_id_(0) {
  auto gpu_device = DeviceManager::Singleton().GetGPUDevice(model->gpu_id());
  profile_ = ModelDatabase::Singleton().GetModelProfile(gpu_device->name(),
                                                        model->profile_id());
  input_array_ = model->CreateInputGpuArray();
  output_sizes_ = model->OutputSizes();
}

void ModelExecutor::AddInput(std::shared_ptr<Task> task) {
  std::lock_guard<std::mutex> lock(mu_);
  for (auto input : task->inputs) {
    input_queue_.push(input);
  }
}

void ModelExecutor::Execute() {
  auto t1 = std::chrono::high_resolution_clock::now();
  auto input_batch = GetBatchInput();
  auto t2 = std::chrono::high_resolution_clock::now();
  if (input_batch == nullptr) {
    return;
  }
  auto inputs = input_batch->inputs();
  for (size_t i = 0; i < inputs.size(); ++i) {
    inputs[i]->task->timer.Record("batch");
  }
  
  auto t3 = std::chrono::high_resolution_clock::now();
  auto output_batch = std::make_shared<BatchOutput>(input_batch->batch_id(),
                                                    input_batch->batch_size());
  output_batch->CreateArrays(output_sizes_,
                             DeviceManager::Singleton().GetCPUDevice());
  model_->Forward(input_batch.get(), output_batch.get());
  auto t4 = std::chrono::high_resolution_clock::now();
  
  auto memcpy_lat = std::chrono::duration_cast<std::chrono::milliseconds>(
      t2 - t1).count();
  auto forward_lat = std::chrono::duration_cast<std::chrono::milliseconds>(
      t4 - t3).count();
  LOG(INFO) << model_->model_session_id() << " forwards batch " <<
      input_batch->batch_id() << ", size " << input_batch->batch_size() <<
      ", memcpy " << memcpy_lat << " ms, forward " << forward_lat << " ms";

  auto outputs = output_batch->GetOutputs();
  for (size_t i = 0; i < inputs.size(); ++i) {
    auto task = inputs[i]->task;
    if (task->AddOutput(inputs[i]->index, std::move(outputs[i]))) {
      task->stage = kPostprocess;
      task_queue_.push(task);
    }
  }
}

std::unique_ptr<BatchInput> ModelExecutor::GetBatchInput() {
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
  uint64_t bid = batch_id_.fetch_add(1);
  std::unique_ptr<BatchInput> batch_input(new BatchInput(bid,
                                                         input_array_));
  while (batch_input->batch_size() < batch_size && !input_queue_.empty()) {
    auto input = std::move(input_queue_.top());
    input_queue_.pop();
    if (profile_ != nullptr && input->task->deadline() < finish) {
      if (input->task->AddVirtualOutput(input->index)) {
        input->task->stage = kPostprocess;
        task_queue_.push(input->task);
      }
    } else {
      batch_input->Append(input);
    }
  }
  if (batch_input->batch_size() == 0) {
    return nullptr;
  }
  return std::move(batch_input);
}


} // namespace backend
} // namespace nexus

