#ifdef USE_TENSORFLOW

#include <boost/filesystem.hpp>
#include <fstream>
#include <opencv2/opencv.hpp>

#include "nexus/backend/postprocess.h"
#include "nexus/backend/slice.h"
#include "nexus/backend/tensorflow_model.h"
#include "nexus/common/image.h"
#include "nexus/common/util.h"
// Tensorflow headers
#include "tensorflow/core/common_runtime/direct_session.h"
#include "tensorflow/core/common_runtime/gpu/process_state.h"

namespace fs = boost::filesystem;

namespace nexus {
namespace backend {

TensorflowModel::TensorflowModel(int gpu_id, const ModelInstanceConfig& config):
    ModelInstance(gpu_id, config),
    first_input_array_(true) {
  CHECK(model_info_["model_file"]) << "Missing model_file in the model info";
  CHECK(model_info_["input_mean"]) << "Missing input_mean in the model info";
  CHECK_EQ(model_info_["input_mean"].size(), 3) << "input_mean must have " <<
      "3 values";
  CHECK(model_info_["input_std"]) << "Missing input_std in the model info";
  CHECK_EQ(model_info_["input_std"].size(), 3) << "input_std must have " <<
      "3 values";

  // Init session options
  auto gpu_opt = gpu_option_.config.mutable_gpu_options();
  gpu_opt->set_visible_device_list(std::to_string(gpu_id));
  gpu_opt->set_allocator_type("BFC");
  if (config.memory_usage() > 0) {
    double memory_usage = config.memory_usage();
    LOG(INFO) << "model memory usage: " << memory_usage << " B";
    gpu_opt->set_per_process_gpu_memory_fraction(
        memory_usage / gpu_device_->TotalMemory());
    gpu_opt->set_allow_growth(false);
  } else {
    gpu_opt->set_allow_growth(true);
  }
  (*cpu_option_.config.mutable_device_count())["GPU"] = 0;
  
  // Init session and load model
  session_.reset(tf::NewSession(gpu_option_));
  fs::path model_dir = fs::path(model_info_["model_dir"].as<std::string>());
  fs::path model_file = model_dir / model_info_["model_file"].as<std::string>();
  CHECK(fs::exists(model_file)) << "model file " << model_file <<
      " doesn't exist";
  tf::GraphDef graph_def;
  tf::Status status;
  status = tf::ReadBinaryProto(gpu_option_.env, model_file.string(),
                               &graph_def);
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
  if (model_session_.image_height() > 0) {
    image_height_ = model_session_.image_height();
    image_width_ = model_session_.image_width();
  } else {
    image_height_ = model_info_["image_height"].as<int>();
    image_width_ = model_info_["image_width"].as<int>();
  }
  // Tensorflow uses NHWC by default. More details see
  // https://www.tensorflow.org/versions/master/performance/performance_guide
  input_shape_.set_dims({max_batch_, image_height_, image_width_, 3});
  input_size_ = input_shape_.NumElements(1);
  input_layer_ = model_info_["input_layer"].as<std::string>();
  output_layer_ = model_info_["output_layer"].as<std::string>();

  // Get the GPU allocator for creating input buffer
  tf::ProcessState* process_state = tf::ProcessState::singleton();
  gpu_allocator_ = process_state->GetGPUAllocator(
      gpu_option_.config.gpu_options(), tf::TfGpuId(0), 0);

  // Dry run the model to get the outpue size
  tf::Tensor* in_tensor = NewInputTensor();
  std::vector<tf::Tensor> out_tensors;
  status = session_->Run({{input_layer_, *in_tensor}}, {output_layer_}, {},
                         &out_tensors);
  if (!status.ok()) {
    LOG(FATAL) << "Failed to run " << model_session_id_ << ": " <<
        status.ToString();
  }
  tf::TensorShape tf_shape = out_tensors[0].shape();
  std::vector<int> shape;
  for (int i = 0; i < tf_shape.dims(); ++i) {
    shape.push_back(tf_shape.dim_size(i));
  }
  output_shape_.set_dims(shape);
  output_size_ = output_shape_.NumElements(1);

  LOG(INFO) << "Model " << model_session_id_ << ", input " <<
      input_layer_ << ": " << input_shape_ << " (" << input_size_ <<
      "), output " << output_layer_ << ": " << output_shape_ <<
      " (" << output_size_ << ")";

  // Load preprocessing configs
  for (uint i = 0; i < model_info_["input_mean"].size(); ++i) {
    input_mean_.push_back(model_info_["input_mean"][i].as<float>());
  }
  for (uint i = 0; i < model_info_["input_std"].size(); ++i) {
    input_std_.push_back(model_info_["input_std"][i].as<float>());
  }

  // Load class names
  if (model_info_["class_names"]) {
    fs::path cns_path = model_dir / model_info_["class_names"].
                        as<std::string>();
    LoadClassnames(cns_path.string());
  }
}

TensorflowModel::~TensorflowModel() {
  session_->Close();
}

Shape TensorflowModel::InputShape() {
  return input_shape_;
}

std::unordered_map<std::string, Shape> TensorflowModel::OutputShapes() {
  return {{output_layer_, output_shape_}};
}

ArrayPtr TensorflowModel::CreateInputGpuArray() {
  tf::Tensor* tensor;
  if (first_input_array_) {
    tensor = input_tensors_[0].get();
    first_input_array_ = false;
  } else {
    tensor = NewInputTensor();
  }
  char* gpu_data = const_cast<char*>(tensor->tensor_data().data());
  auto buf = std::make_shared<Buffer>(
      gpu_data, tensor->NumElements() * sizeof(float), gpu_device_);
  auto arr = std::make_shared<Array>(DT_FLOAT, tensor->NumElements(), buf);
  arr->set_tag(input_tensors_.size() - 1);
  return arr;
}

std::unordered_map<std::string, ArrayPtr> TensorflowModel::GetOutputGpuArrays(){
  // Because TF always returns output in CPU memory, doesn't support in-place
  // output in GPU memory
  return {};
}

void TensorflowModel::Preprocess(std::shared_ptr<Task> task) {
  auto prepare_image = [&](cv::Mat& image) {
    // Tensorflow uses NHWC by default. More details see
    // https://www.tensorflow.org/versions/master/performance/performance_guide
    auto in_arr = std::make_shared<Array>(DT_FLOAT, input_size_, cpu_device_);
    // create a cv::Mat using buffer allocated in the in_arr
    cv::Mat resized(image_width_, image_height_, CV_32FC3,
                    in_arr->Data<void>());
    cv::resize(image, resized, cv::Size(image_width_, image_height_));
    for (cv::Point3_<float>& p : cv::Mat_<cv::Point3_<float> >(resized)) {
      p.x = (p.x - input_mean_[0]) / input_std_[0];
      p.y = (p.y - input_mean_[1]) / input_std_[1];
      p.z = (p.z - input_mean_[2]) / input_std_[2];
    }
    task->AppendInput(in_arr);
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

void TensorflowModel::Forward(std::shared_ptr<BatchTask> batch_task) {
  size_t batch_size = batch_task->batch_size();
  auto in_tensor = input_tensors_[batch_task->GetInputArray()->tag()]->Slice(
      0, batch_size);
  // auto in_tensor = input_tensors_[batch_input->array()->tag()]->Slice(
  //     0, batch_);
  std::vector<tf::Tensor> out_tensors;
  tf::Status status = session_->Run({{input_layer_, in_tensor}},
                                    {output_layer_}, {}, &out_tensors);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to run tensorflow: " << status.ToString();
    return;
  }
  const char* tensor_data = out_tensors[0].tensor_data().data();
  size_t nfloats = out_tensors[0].NumElements();
  auto out_arr = batch_task->GetOutputArray(output_layer_);
  float* out_data = out_arr->Data<float>();
  Memcpy(out_data, cpu_device_, tensor_data, gpu_device_,
         nfloats * sizeof(float));
  batch_task->SliceOutputBatch({{
        output_layer_, Slice(batch_size, output_size_) }});
}

void TensorflowModel::Postprocess(std::shared_ptr<Task> task) {
  const QueryProto& query = task->query;
  QueryResultProto* result = &task->result;
  result->set_status(CTRL_OK);
  for (auto output : task->outputs) {
    auto out_arr = output->arrays.at(output_layer_);
    float* out_data = out_arr->Data<float>();
    size_t count = out_arr->num_elements();
    if (type() == "classification") {
      if (classnames_.empty()) {
        PostprocessClassification(query, out_data, output_size_, result);
      } else {
        PostprocessClassification(query, out_data, output_size_, result,
                                  &classnames_);
      }
    } else {
      std::ostringstream oss;
      oss << "Unsupported model type " << type() << " for " << framework();
      result->set_status(MODEL_TYPE_NOT_SUPPORT);
      result->set_error_message(oss.str());
      break;
    }
  }
}

tf::Tensor* TensorflowModel::NewInputTensor() {
  tf::TensorShape shape;
  for (auto dim : input_shape_.dims()) {
    shape.AddDim(dim);
  }
  tf::Tensor* tensor = new tf::Tensor(gpu_allocator_, tf::DT_FLOAT, shape);
  input_tensors_.emplace_back(tensor);
  return tensor;
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

} // namespace backend
} // namespace nexus

#endif // USE_TENSORFLOW
