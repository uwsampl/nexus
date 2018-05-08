#include <glog/logging.h>

#include "nexus/backend/batch.h"
#include "nexus/backend/task.h"
#include "nexus/common/util.h"

namespace nexus {
namespace backend {

Input::Input(std::shared_ptr<Task> task, std::shared_ptr<Array> arr, int idx) :
      DeadlineItem(task->deadline()),
      task(task),
      array(arr),
      index(idx) {
}

BatchInput::BatchInput(uint64_t batch_id, std::shared_ptr<Array> arr) :
    batch_id_(batch_id),
    array_(arr),
    num_elements_(0) {
  write_pt_ = array_->Data<float>();
}

void BatchInput::Append(std::shared_ptr<Input> input) {
  CHECK_EQ(array_->data_type(), input->array->data_type()) <<
      "Input data type is not float";
  CHECK_LE(num_elements_ + input->array->num_elements(),
           array_->num_elements()) << "Exceeds BatchInput buffer capacity";
  inputs_.push_back(input);
  auto src_array = input->array;
  const float* src_data = src_array->Data<float>();
  size_t nbytes = src_array->num_elements() * sizeof(float);
  Memcpy(write_pt_, array_->device(), src_data, src_array->device(), nbytes);
  write_pt_ += src_array->num_elements();
}

Output::Output(std::shared_ptr<BatchOutput> batch_output,
               const std::unordered_map<std::string, ArrayPtr>& arrays) :
    batch_output_(batch_output),
    arrays_(arrays) {}

uint64_t Output::batch_id() const {
  return batch_output_->batch_id();
}

ArrayPtr Output::GetArray(const std::string& name) const {
  CHECK_GT(arrays_.count(name), 0) << "Array " << name << " is not present";
  return arrays_.at(name);
}

ArrayPtr Output::operator[](const std::string& name) const {
  return GetArray(name);
}

BatchOutput::BatchOutput(uint64_t batch_id, size_t batch_size) :
    batch_id_(batch_id),
    batch_size_(batch_size) {}

void BatchOutput::SetArrays(
    const std::unordered_map<std::string, ArrayPtr>& arrays) {
  arrays_ = arrays;
}

void BatchOutput::CreateArrays(
    const std::unordered_map<std::string, size_t>& sizes, Device* device) {
  for (auto iter : sizes) {
    auto arr = std::make_shared<Array>(DT_FLOAT, iter.second * batch_size_,
                                       device);
    arrays_.emplace(iter.first, arr);
  }
}

ArrayPtr BatchOutput::GetArray(const std::string& name) {
  CHECK_GT(arrays_.count(name), 0) << "Batch array " << name <<
      " is not created";
  return arrays_.at(name);
}

void BatchOutput::SliceBatch(
    const std::unordered_map<std::string, Slice>& slices) {
  CHECK_EQ(arrays_.size(), slices.size()) << "Number of outputs must " <<
      "match with number of slices";
  slices_ = slices;
}

std::vector<std::unique_ptr<Output> > BatchOutput::GetOutputs() {
  std::vector<std::unique_ptr<Output> > rets;
  for (size_t i = 0; i < batch_size_; ++i) {
    std::unordered_map<std::string, ArrayPtr> slice_arrays;
    for (auto iter : arrays_) {
      auto const& slice = slices_.at(iter.first);
      slice_arrays.emplace(iter.first, iter.second->Slice(
          slice.offset(i), slice.num_elements(i)));
    }
    rets.push_back(std::unique_ptr<Output>(new Output(
        shared_from_this(), slice_arrays)));
  }
  return rets;
}

} // namespace backend
} // namespace nexus
