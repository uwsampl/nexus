#include "nexus/backend/batch_task.h"
#include "nexus/common/util.h"

namespace nexus {
namespace backend {

BatchTask::BatchTask(uint64_t batch_id, uint32_t max_batch) :
    batch_id_(batch_id),
    max_batch_(max_batch),
    input_write_pt_(nullptr),
    input_nfloats_(0) {}

void BatchTask::SetInputArray(ArrayPtr arr) {
  input_array_ = arr;
  input_write_pt_ = input_array_->Data<float>();
  input_nfloats_ = 0;
}

void BatchTask::CreateInputArray(size_t input_size, Device* device) {
  input_array_ = std::make_shared<Array>(DT_FLOAT, max_batch_ * input_size,
                                         device);
  input_write_pt_ = input_array_->Data<float>();
  input_nfloats_ = 0;
}

void BatchTask::SetOutputArrays(
    const std::unordered_map<std::string, ArrayPtr>& arrays) {
  output_arrays_ = arrays;
}

void BatchTask::CreateOutputArrays(
    const std::unordered_map<std::string, size_t>& sizes, Device* device) {
  uint32_t batch = max_batch_;
  if (inputs_.size() > 0) {
    batch = inputs_.size();
  }
  for (auto iter : sizes) {
    auto arr = std::make_shared<Array>(DT_FLOAT, batch * iter.second,
                                       device);
    output_arrays_.emplace(iter.first, arr);
  }
}

ArrayPtr BatchTask::GetOutputArray(const std::string& name) const {
  CHECK_GT(output_arrays_.count(name), 0) << "Output array " << name <<
      " doesn't exist";
  return output_arrays_.at(name);
}

void BatchTask::AppendInput(std::shared_ptr<Input> input) {
  CHECK_EQ(input_array_->data_type(), input->array->data_type()) <<
      "Input data type is not float";
  CHECK_LT(inputs_.size(), max_batch_) << "Exceed max batch size";
  CHECK_LE(input_nfloats_ + input->array->num_elements(),
           input_array_->num_elements()) << "Exceeds input_array capacity";
  inputs_.push_back(input);
  auto in_arr = input->array;
  const float* src_data = in_arr->Data<float>();
  Memcpy(input_write_pt_, input_array_->device(), src_data, in_arr->device(),
         in_arr->num_elements() * sizeof(float));
  input_write_pt_ += in_arr->num_elements();
}

void BatchTask::SliceOutputBatch(
    const std::unordered_map<std::string, Slice>& slices) {
  CHECK(outputs_.empty()) << "Batch output is already sliced";
  CHECK_EQ(output_arrays_.size(), slices.size()) << "Number of outputs must "
      "match the number of slices";
  for (uint i = 0; i < inputs_.size(); ++i) {
    auto input = inputs_[i];
    std::unordered_map<std::string, ArrayPtr> slice_arrays;
    for (auto iter : output_arrays_) {
      auto const& slice = slices.at(iter.first);
      slice_arrays.emplace(iter.first, iter.second->Slice(
          slice.offset(i), slice.num_elements(i)));
    }
    outputs_.push_back(std::make_shared<Output>(slice_arrays, input->task,
                                                input->index_in_task));
  }
}

void BatchTask::set_outputs(
    const std::vector<std::shared_ptr<Output>>& outputs) {
  CHECK_EQ(outputs.size(), inputs_.size()) << "Number of outputs must match "
      "number of inputs";
  outputs_.clear();
  outputs_ = outputs;
}

} // namespace backend
} // namespace nexus
