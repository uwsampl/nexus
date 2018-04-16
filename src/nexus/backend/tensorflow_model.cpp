#include "tensorflow_model.h"

#include <boost/filesystem.hpp>
#include <fstream>
#include <opencv2/opencv.hpp>

#include "nexus/backend/slice.h"
#include "nexus/common/image.h"
#include "nexus/common/util.h"
// Tensorflow headers
#include "tensorflow/core/common_runtime/direct_session.h"
#include "tensorflow/core/common_runtime/gpu/process_state.h"

namespace fs = boost::filesystem;

namespace nexus {
namespace backend {

TensorflowModel::TensorflowModel(int gpu_id, const std::string& model_name,
                                 uint32_t version, const std::string& type,
                                 uint32_t batch, uint32_t max_batch,
                                 BlockPriorityQueue<Task>& task_queue,
                                 const YAML::Node& info) :
    ModelInstance(gpu_id, model_name, version, type, batch, max_batch,
                  task_queue) {
  CHECK(info["model_file"]) << "Missing model_file in the model info";
  CHECK(info["input_mean"]) << "Missing input_mean in the model info";
  CHECK_EQ(info["input_mean"].size(), 3) << "input_mean must have 3 values";
  CHECK(info["input_std"]) << "Missing input_std in the model info";
  CHECK_EQ(info["input_std"].size(), 3) << "input_std must have 3 values";
  // Init session options
  auto gpu_opt = gpu_option_.config.mutable_gpu_options();
  gpu_opt->set_visible_device_list(std::to_string(gpu_id));
  gpu_opt->set_allocator_type("BFC");
  if (info["memory_usage"]) {
    size_t memory_usage = info["memory_usage"].as<size_t>();
    LOG(INFO) << "model memory usage: " << memory_usage << " B";
    gpu_opt->set_per_process_gpu_memory_fraction(
        double(memory_usage) / gpu_device_->TotalMemory());
    gpu_opt->set_allow_growth(false);
  } else {
    gpu_opt->set_allow_growth(true);
  }
  (*cpu_option_.config.mutable_device_count())["GPU"] = 0;
  // Init session and load model
  session_.reset(tf::NewSession(gpu_option_));
  fs::path model_dir = fs::path(info["model_dir"].as<std::string>());
  fs::path model_file = model_dir / info["model_file"].as<std::string>();
  CHECK(fs::exists(model_file)) << "model file " << model_file <<
      " doesn't exist";
  tf::GraphDef graph_def;
  tf::Status status;
  status = tf::ReadBinaryProto(gpu_option_.env, model_file.string(), &graph_def);
  // for (auto node : graph_def.node()) {
  //   LOG(INFO) << node.name() << ": " << node.device();
  // }
  if (!status.ok()) {
    LOG(FATAL) << "Failed to load model " << model_file << " : " <<
        status.ToString();
  }
  status = session_->Create(graph_def);
  if (!status.ok()) {
    LOG(FATAL) << "Failed to add graph to session: " << status.ToString();
  }
  // Get the input and output shape
  image_height_ = info["image_height"].as<int>();
  image_width_ = info["image_width"].as<int>();
  input_size_ = image_height_ * image_width_ * 3;
  input_layer_ = info["input_layer"].as<std::string>();
  output_layer_ = info["output_layer"].as<std::string>();
  // Load configs
  for (uint i = 0; i < info["input_mean"].size(); ++i) {
    input_mean_.push_back(info["input_mean"][i].as<float>());
  }
  for (uint i = 0; i < info["input_std"].size(); ++i) {
    input_std_.push_back(info["input_std"][i].as<float>());
  }
  // Load class names
  if (info["class_names"]) {
    fs::path cns_path = model_dir / info["class_names"].as<std::string>();
    LoadClassnames(cns_path.string());
  }
  // Get the GPU allocator for creating input buffer
  tf::ProcessState* process_state = tf::ProcessState::singleton();
  gpu_allocator_ = process_state->GetGPUAllocator(
      gpu_option_.config.gpu_options(), tf::TfGpuId(0), 0);
}

TensorflowModel::~TensorflowModel() {
  session_->Close();
}

std::string TensorflowModel::profile_id() const {
  std::stringstream ss;
  ss << "tensorflow:" << model_name_ << ":" << version_;
  return ss.str();
}

void TensorflowModel::InitBatchInputArray() {
  input_tensor_.reset(new tf::Tensor(
      gpu_allocator_, tf::DT_FLOAT,
      {max_batch_, image_height_, image_width_, 3}));
  char* gpu_data = const_cast<char*>(input_tensor_->tensor_data().data());
  auto buf = std::make_shared<Buffer>(
      gpu_data, input_tensor_->NumElements() * sizeof(float), gpu_device_);
  batch_input_array_ = std::make_shared<Array>(
      DT_FLOAT, input_tensor_->NumElements(), buf);
}
/*
void TensorflowModel::UpdateMaxBatchImpl() {
  InitBatchInputArray();
}
*/
void TensorflowModel::PreprocessImpl(std::shared_ptr<Task> task,
                                     std::vector<ArrayPtr>* input_arrays) {
  auto prepare_image = [&](cv::Mat& image) {
    auto in_arr = std::make_shared<Array>(DT_FLOAT, input_size_, cpu_device_);
    // create a cv::Mat using buffer allocated in the in_arr
    cv::Mat resized(image_width_, image_height_, CV_32FC3, in_arr->Data<void>());
    cv::resize(image, resized, cv::Size(image_width_, image_height_));
    for (cv::Point3_<float>& p : cv::Mat_<cv::Point3_<float> >(resized)) {
      p.x = (p.x - input_mean_[0]) / input_std_[0];
      p.y = (p.y - input_mean_[1]) / input_std_[1];
      p.z = (p.z - input_mean_[2]) / input_std_[2];
    }
    input_arrays->push_back(in_arr);
  };

  const auto& query = task->query;
  const auto& input_data = query.input();
  switch (input_data.data_type()) {
    case DT_IMAGE: {
      cv::Mat img_rgb = DecodeImage(input_data.image(), CO_RGB);
      cv::Mat img;
      img_rgb.convertTo(img, CV_32FC3);
      if (query.window_size() > 0) {
        for (int i = 0; i < query.window_size(); ++i) {
          const auto& rect = query.window(i);
          cv::Mat crop_img = img(cv::Rect(
              rect.left(), rect.top(), rect.right() - rect.left(),
              rect.bottom() - rect.top()));
          prepare_image(crop_img);
        }
      } else {
        prepare_image(img);
      }
      break;
    }
    default:
      task->result.set_status(INPUT_TYPE_INCORRECT);
      task->result.set_error_message("Input type incorrect: " +
                                     DataType_Name(input_data.data_type()));
      break;
  }
}

void TensorflowModel::ForwardImpl(BatchInput* batch_input,
                                  BatchOutput* batch_output) {
  size_t batch_size = batch_input->batch_size();
  //auto in_tensor = input_tensor_->Slice(0, batch_size);
  auto in_tensor = input_tensor_->Slice(0, batch_);
  //tf::Tensor& in_tensor = *(input_tensor_.get());
  std::vector<tf::Tensor> out_tensors;
  tf::Status status = session_->Run(
      {{input_layer_, in_tensor}}, {output_layer_}, {}, &out_tensors);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to run tensorflow: " << status.ToString();
    return;
  }
  const char* tensor_data = out_tensors[0].tensor_data().data();
  size_t nfloats = out_tensors[0].NumElements();
  auto out_arr = std::make_shared<Array>(DT_FLOAT, nfloats, cpu_device_);
  float* out_data = out_arr->Data<float>();
  Memcpy(out_data, cpu_device_, tensor_data, gpu_device_,
         nfloats * sizeof(float));
  // size_t output_size = nfloats / batch_size;
  size_t output_size = nfloats / batch_;
  batch_output->SetOutputBatch({out_arr}, {Slice(batch_size, output_size)});
}

void TensorflowModel::PostprocessImpl(std::shared_ptr<Task> task,
                                      Output* output) {
  const QueryProto& query = task->query;
  QueryResultProto* result = &task->result;
  auto out_arr = output->GetOutput(0);
  float* out_data = out_arr->Data<float>();
  size_t count = out_arr->num_elements();
  if (type_ == "classification") {
    result->set_status(CTRL_OK);
    float threshold = 0.;
    MarshalClassificationResult(query, out_data, count, threshold, 
                                result);
  } else {
    result->set_status(MODEL_TYPE_NOT_SUPPORT);
    std::ostringstream oss;
    oss << "Unsupported model type " << type() << " for " << framework();
    result->set_error_message(oss.str());
  }
}

void TensorflowModel::LoadClassnames(const std::string& filepath) {
  std::ifstream infile(filepath);
  CHECK(infile.good()) << "Classname file " << filepath << " doesn't exist";
  std::string line;
  while (std::getline(infile, line)) {
    classnames_.push_back(line);
  }
  infile.close();
}

void TensorflowModel::MarshalClassificationResult(
    const QueryProto& query, const float* prob, size_t nprobs, float threshold,
    QueryResultProto* result) {
  std::vector<std::string> output_fields(query.output_field().begin(),
                                         query.output_field().end());
  if (output_fields.size() == 0) {
    output_fields.push_back("class_name");
  }
  float max_prob = 0.;
  int max_idx = -1;
  for (int i = 0; i < (int) nprobs; ++i) {
    float p = prob[i];
    if (p >= threshold) {
      if (p > max_prob) {
        max_prob = p;
        max_idx = i;
      }
    }
  }
  if (max_idx > -1) {
    auto record = result->add_output();
    for (auto field : output_fields) {
      if (field == "class_id") {
        auto value = record->add_named_value();
        value->set_name("class_id");
        value->set_data_type(DT_INT);
        value->set_i(max_idx);
      } else if (field == "class_prob") {
        auto value = record->add_named_value();
        value->set_name("class_prob");
        value->set_data_type(DT_FLOAT);
        value->set_f(max_prob);
      } else if (field == "class_name") {
        auto value = record->add_named_value();
        value->set_name("class_name");
        value->set_data_type(DT_STRING);
        if (classnames_.size() > max_idx) {
          value->set_s(classnames_.at(max_idx));
        }
      }
    }
  }
}

} // namespace backend
} // namespace nexus
