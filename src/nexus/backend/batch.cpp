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

uint64_t Output::batch_id() const {
  return batch_output_->batch_id();
}

ArrayPtr Output::GetOutput(int i) const {
  CHECK_LT(i, outputs_.size()) << "Output index " << i << " is larger than " <<
      " number of outputs " << outputs_.size();
  return outputs_[i];
}

ArrayPtr Output::GetOutput(const std::string& name) const {
  CHECK_GT(output_names_.size(), 0) << "Output name is not set";
  for (size_t i = 0; i < output_names_.size(); ++i) {
    if (output_names_[i] == name) {
      return outputs_[i];
    }
  }
  return nullptr;
}

bool Output::Finish() const {
  return batch_output_->Finish();
}

BatchOutput::BatchOutput(uint64_t batch_id, size_t batch_size) :
    batch_id_(batch_id),
    batch_size_(batch_size),
    finished_(0) {
}

void BatchOutput::SetOutputBatch(const std::vector<float*>& outputs,
                                 const std::vector<Slice>& slices,
                                 Device* device) {
  CHECK_EQ(outputs.size(), slices.size()) << "Number of outputs must " <<
      "match with number of slices (" << outputs.size() << " vs " <<
      slices.size() << ")";
  for (uint i = 0; i < outputs.size(); ++i) {
    size_t nfloats = slices[i].total_elements();
    auto buf = std::make_shared<Buffer>(
        outputs[i], nfloats * sizeof(float), device, true);
    auto arr = std::make_shared<Array>(DT_FLOAT, nfloats, buf);
    outputs_.push_back(arr);
  }
  slices_ = slices;
}

void BatchOutput::SetOutputBatch(const std::vector<ArrayPtr>& outputs,
                                 const std::vector<Slice>& slices) {
  CHECK_EQ(outputs.size(), slices.size()) << "Number of outputs must " <<
      "match with number of slices (" << outputs.size() << " vs " <<
      slices.size() << ")";
  outputs_ = outputs;
  slices_ = slices;
}

void BatchOutput::SetOutputBatch(const std::vector<ArrayPtr>& outputs,
                                 const std::vector<Slice>& slices,
                                 const std::vector<std::string>& names) {
  CHECK_EQ(outputs.size(), slices.size()) << "Number of outputs must" <<
      " match with number of slices (" << outputs.size() << " vs " <<
      slices.size() << ")";
  CHECK_EQ(outputs.size(), names.size()) << "Number of outputs must" <<
      " match with number of names (" << outputs.size() << " vs " <<
      names.size() << ")";
  outputs_ = outputs;
  slices_ = slices;
  output_names_ = names;
}

std::vector<std::unique_ptr<Output> > BatchOutput::GetOutputs() {
  std::vector<std::unique_ptr<Output> > rets;
  for (size_t i = 0; i < batch_size_; ++i) {
    std::vector<ArrayPtr> slice_outputs(outputs_.size());
    for (size_t j = 0; j < outputs_.size(); ++j) {
      slice_outputs[j] = outputs_[j]->Slice(slices_[j].offset(i),
                                            slices_[j].num_elements(i));
    }
    rets.push_back(std::unique_ptr<Output>(new Output(
        this, slice_outputs, output_names_)));
  }
  return rets;
}

bool BatchOutput::Finish() {
  uint result = ++finished_;
  if (result == batch_size_) {
    return true;
  }
  return false;
}

} // namespace backend
} // namespace nexus
