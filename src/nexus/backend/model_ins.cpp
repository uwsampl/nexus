#include <chrono>
#include <glog/logging.h>
#include <thread>

#include "nexus/common/model_db.h"
#include "nexus/common/time_util.h"
#include "nexus/backend/caffe_densecap_model.h"
#include "nexus/backend/caffe_model.h"
#include "nexus/backend/darknet_model.h"
#include "nexus/backend/model_ins.h"
#include "nexus/backend/tensorflow_model.h"

namespace nexus {
namespace backend {

ModelInstance::ModelInstance(int gpu_id, const std::string& model_name,
                             uint32_t version, const std::string& type,
                             uint32_t batch, uint32_t max_batch,
                             BlockPriorityQueue<Task>& task_queue):
    gpu_id_(gpu_id),
    model_name_(model_name),
    version_(version),
    type_(type),
    batch_(batch),
    max_batch_(max_batch),
    need_update_max_batch_(false),
    task_queue_(task_queue),
    batch_id_(0) {
  cpu_device_ = DeviceManager::Singleton().GetCPUDevice();
  gpu_device_ = DeviceManager::Singleton().GetGPUDevice(gpu_id);
  meter_ = MetricRegistry::Singleton().CreateMovingAverage(model_name, 1, 5);
}

void ModelInstance::Init() {
  InitBatchInputArray();
}
/*
void ModelInstance::UpdateMaxBatch(size_t max_batch) {
  std::lock_guard<std::mutex> lock(input_mutex_);
  if (max_batch_ != max_batch) {
    LOG(INFO) << "Update max batch size to " << max_batch;
    max_batch_ = max_batch;
    need_update_max_batch_ = true;
  }
}
*/
bool ModelInstance::Preprocess(std::shared_ptr<Task> task) {
  std::vector<ArrayPtr> input_arrays;
  PreprocessImpl(task, &input_arrays);
  if (task->result.status() != CTRL_OK) {
    return false;
  }
  task->stage = kForward;
  AppendInputs(task, input_arrays);
  return true;
}

void ModelInstance::Forward(size_t min_batch) {
  auto t1 = std::chrono::high_resolution_clock::now();
  auto input_batch = GetBatchInput(min_batch);
  auto t2 = std::chrono::high_resolution_clock::now();
  if (input_batch == nullptr) {
    return;
  }
  auto inputs = input_batch->inputs();
  for (size_t i = 0; i < inputs.size(); ++i) {
    inputs[i]->task->timer.Record("batch");
  }
  
  auto t3 = std::chrono::high_resolution_clock::now();
  auto output_batch = std::make_shared<BatchOutput>(
      input_batch->batch_id(), input_batch->batch_size());
  {
    std::lock_guard<std::mutex> lock(output_mutex_);
    output_pool_.emplace(output_batch->batch_id(), output_batch);
  }
  ForwardImpl(input_batch.get(), output_batch.get());
  auto t4 = std::chrono::high_resolution_clock::now();
  
  auto memcpy_lat = std::chrono::duration_cast<std::chrono::milliseconds>(
      t2 - t1).count();
  auto forward_lat = std::chrono::duration_cast<std::chrono::milliseconds>(
      t4 - t3).count();
  LOG(INFO) << model_name_ << " batch " << input_batch->batch_id() <<
      ": size " << input_batch->batch_size() << ", memcpy " << memcpy_lat <<
      " ms, forward " << forward_lat << " ms";

  auto outputs = output_batch->GetOutputs();
  for (size_t i = 0; i < inputs.size(); ++i) {
    auto task = inputs[i]->task;
    if (task->AddOutput(inputs[i]->index, std::move(outputs[i]))) {
      task->stage = kPostprocess;
      task_queue_.push(task);
    }
  }
}

void ModelInstance::Postprocess(std::shared_ptr<Task> task) {
  for (auto& out : task->outputs) {
    PostprocessImpl(task, out.get());
    if (out->Finish()) {
      RemoveOutput(out->batch_id());
    }
  }
}

uint64_t ModelInstance::Tick() {
  return meter_->Tick();
}

double ModelInstance::GetRate() {
  return meter_->rate();
}

std::unique_ptr<BatchInput> ModelInstance::GetBatchInput(size_t min_batch) {
  std::lock_guard<std::mutex> lock(input_mutex_);
  /*if (need_update_max_batch_) {
    UpdateMaxBatchImpl();
    need_update_max_batch_ = false;
    }*/
  if (input_queue_.size() < min_batch) {
    return nullptr;
  }
  size_t batch_size = batch_;
  if (input_queue_.size() < batch_size) {
    batch_size = input_queue_.size();
  }
  float latency = static_cast<uint32_t>(
      ModelDatabase::Singleton().GetModelForwardLatency(
          gpu_device_->device_name(), profile_id(), batch_size));
  bool has_estimate;
  TimePoint finish;
  if (latency <= 0) {
    bool has_estimate = false;
  } else {
    finish = Clock::now() + std::chrono::microseconds(int(latency));
  }
  uint64_t bid = batch_id_.fetch_add(1);
  std::unique_ptr<BatchInput> batch_input(
      new BatchInput(bid, batch_input_array_));
  std::shared_ptr<Task> last_remove_task = nullptr;
  while (batch_input->batch_size() < batch_) {
    if (input_queue_.empty()) {
      break;
    }
    auto input = std::move(input_queue_.top());
    input_queue_.pop();
    if (has_estimate && input->task->deadline() < finish) {
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

void ModelInstance::AppendInputs(std::shared_ptr<Task> task,
                                 const std::vector<ArrayPtr>& input_arrays) {
  std::lock_guard<std::mutex> lock(input_mutex_);
  for (uint i = 0; i < input_arrays.size(); ++i) {
    // insert a placeholder in the outputs
    task->outputs.push_back(nullptr);
    auto input = std::make_shared<Input>(task, input_arrays[i], i);
    input_queue_.push(input);
  }
  meter_->Increase(input_arrays.size());
}

void ModelInstance::RemoveOutput(uint64_t batch_id) {
  std::lock_guard<std::mutex> lock(output_mutex_);
  output_pool_.erase(batch_id);
}

std::shared_ptr<ModelInstance> CreateModelInstance(
    int gpu_id, const ModelInstanceDesc& model_inst_desc, YAML::Node info,
    BlockPriorityQueue<Task>& task_queue) {
  const auto& model_sess = model_inst_desc.model_session();
  auto framework = model_sess.framework();
  auto model_name = model_sess.model_name();
  auto model_type = info["type"].as<std::string>();
  uint32_t version = info["version"].as<uint32_t>();
  uint32_t batch = model_inst_desc.batch();
  uint32_t max_batch = model_inst_desc.max_batch();
  if (model_inst_desc.memory_usage() > 0) {
    info["memory_usage"] = model_inst_desc.memory_usage();
  }
  if (model_sess.image_height() > 0) {
    CHECK_GT(model_sess.image_width(), 0) << "image_height and image_width " <<
        "must be both set";
    info["image_height"] = model_sess.image_height();
    info["image_width"] = model_sess.image_width();
  }
  std::shared_ptr<ModelInstance> model;
  if (framework == "darknet") {
    model = std::make_shared<DarknetModel>(
        gpu_id, model_name, version, model_type, batch, max_batch, task_queue,
        info);
  } else if (framework == "caffe") {
    if (model_name == "densecap") {
      model = std::make_shared<CaffeDenseCapModel>(
          gpu_id, model_name, version, model_type, batch, max_batch, task_queue,
          info);
    } else {
      model = std::make_shared<CaffeModel>(
          gpu_id, model_name, version, model_type, batch, max_batch, task_queue,
          info);
    }
  } else if (framework == "tensorflow") {
    model = std::make_shared<TensorflowModel>(
        gpu_id, model_name, version, model_type, batch, max_batch, task_queue,
        info);
  } else {
    LOG(FATAL) << "Unknown framework " << framework;
  }
  model->Init();
  return model;
}

} // namespace backend
} // namespace nexus
