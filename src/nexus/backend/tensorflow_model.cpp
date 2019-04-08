#ifdef USE_TENSORFLOW

#include <boost/filesystem.hpp>
#include <opencv2/opencv.hpp>
#include <glog/logging.h>
#include "tensorflow/core/common_runtime/gpu/gpu_process_state.h"

#include "nexus/backend/slice.h"
#include "nexus/backend/tensorflow_model.h"
#include "nexus/backend/utils.h"
#include "nexus/common/image.h"
//#undef LOG
//#undef VLOG
//#undef CHECK_EQ
//#undef CHECK_LE
//#undef CHECK_LT
//#undef CHECK_GE
//#undef CHECK_GT
//#undef _LOGGING_H_

namespace fs = boost::filesystem;

namespace nexus {
namespace backend {

TensorflowModel::TensorflowModel(int gpu_id, const ModelInstanceConfig& config):
    ModelInstance(gpu_id, config),
    first_input_array_(true), num_suffixes_(0) {
  CHECK(model_info_["model_file"]) << "Missing model_file in the model info";

  // Init session options
  auto gpu_opt = gpu_option_.config.mutable_gpu_options();
  gpu_opt->set_visible_device_list(std::to_string(gpu_id));
  gpu_opt->set_allocator_type("BFC");
  LOG(INFO) << "model memory usage: " << config.memory_usage() << " B";
  gpu_opt->set_allow_growth(true);
//  if (config.memory_usage() > 0) {
//    double memory_usage = config.memory_usage();
//    LOG(INFO) << "model memory usage: " << memory_usage << " B";
//    gpu_opt->set_per_process_gpu_memory_fraction(
//        memory_usage / gpu_device_->TotalMemory());
//    gpu_opt->set_allow_growth(false);
//  } else {
//    gpu_opt->set_allow_growth(true);
//  }
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
  input_shape_.set_dims({static_cast<int>(max_batch_), image_height_, image_width_, 3});
  input_size_ = input_shape_.NumElements(1);
  input_layer_ = model_info_["input_layer"].as<std::string>();

  if (model_info_["output_layer"].IsSequence()) {
    for (size_t i = 0; i < model_info_["output_layer"].size(); ++i) {
      output_layers_.push_back(model_info_["output_layer"][i].as<std::string>());
    }
  } else {
    output_layers_.push_back(model_info_["output_layer"].as<std::string>());
  }
  LOG(INFO) << "Model " << model_session_id_ << ", input: " <<
      input_layer_ << ", shape: " << input_shape_ << " (" << input_size_ <<
      ")";

  if (model_name() == "ssd_mobilenet" || model_name() == "ssd_mobilenet_0.75") {
    input_data_type_ = DT_UINT8;
  } else {
    input_data_type_ = DT_FLOAT;
  }

  // Get the GPU allocator for creating input buffer
  tf::GPUProcessState* process_state = tf::GPUProcessState::singleton();
  gpu_allocator_ = process_state->GetGPUAllocator(
      gpu_option_.config.gpu_options(), tf::TfGpuId(0), 0);

  // Dry run the model to get the outpue size
  auto in_tensor = NewInputTensor()->Slice(0, 1);
  std::vector<std::pair<std::string, tf::Tensor>> inputs;
  std::vector<tf::Tensor> out_tensors;
  inputs.emplace_back(input_layer_, in_tensor);
  if (model_info_["slice_beg_vector"]) {
    // TFShareModel
    auto slice_beg_vector = model_info_["slice_beg_vector"].as<std::string>();
    auto slice_end_vector = model_info_["slice_end_vector"].as<std::string>();
    num_suffixes_ = model_info_["suffix_models"].size();
    tf::TensorShape shape;
    shape.AddDim(num_suffixes_);
    slice_beg_tensor_.reset(new tf::Tensor(/* gpu_allocator_, */tf::DT_INT32, shape));
    slice_end_tensor_.reset(new tf::Tensor(/* gpu_allocator_, */tf::DT_INT32, shape));
    set_slice_tensor(slice_beg_tensor_, std::vector<int32_t>(num_suffixes_, 0));
    set_slice_tensor(slice_end_tensor_, std::vector<int32_t>(num_suffixes_, 1));
    inputs.emplace_back(slice_beg_vector, slice_beg_tensor_->Slice(0, num_suffixes_));
    inputs.emplace_back(slice_end_vector, slice_end_tensor_->Slice(0, num_suffixes_));
  }
  status = session_->Run(inputs, output_layers_, {}, &out_tensors);
  if (!status.ok()) {
    LOG(FATAL) << "Failed to run " << model_session_id_ << ": " <<
        status.ToString();
  }
  for (uint i = 0; i < out_tensors.size(); ++i) {
    tf::TensorShape tf_shape = out_tensors[i].shape();
    std::vector<int> dims;
    for (int i = 0; i < tf_shape.dims(); ++i) {
      dims.push_back(tf_shape.dim_size(i));
    }
    if (dims.size() == 1) {
      dims.push_back(1);
    }
    Shape shape(dims);
    size_t out_size = shape.NumElements(1);
    output_shapes_.emplace(output_layers_[i], shape);
    output_sizes_.emplace(output_layers_[i], out_size);
    LOG(INFO) << "Output " << output_layers_[i] << ", shape: " << shape <<
        " (" << out_size << ")";
  }

  // Load preprocessing configs
  if (model_info_["input_mean"]) {
    CHECK_EQ(model_info_["input_mean"].size(), 3) << "input_mean must have " <<
        "3 values";
    for (uint i = 0; i < model_info_["input_mean"].size(); ++i) {
      input_mean_.push_back(model_info_["input_mean"][i].as<float>());
    }
  }
  if (model_info_["input_std"]) {
    CHECK_EQ(model_info_["input_std"].size(), 3) << "input_std must have " <<
        "3 values";
    for (uint i = 0; i < model_info_["input_std"].size(); ++i) {
      input_std_.push_back(model_info_["input_std"][i].as<float>());
    }
  }

  // Load class names
  if (model_info_["class_names"]) {
    fs::path cns_path = model_dir / model_info_["class_names"].
                        as<std::string>();
    LoadClassnames(cns_path.string(), &classnames_);
  }
}

TensorflowModel::~TensorflowModel() {
  session_->Close();
}

Shape TensorflowModel::InputShape() {
  return input_shape_;
}

std::unordered_map<std::string, Shape> TensorflowModel::OutputShapes() {
  return output_shapes_;
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
  size_t nbytes = tensor->NumElements() * type_size(input_data_type_);
  auto buf = std::make_shared<Buffer>(gpu_data, nbytes, gpu_device_);
  auto arr = std::make_shared<Array>(input_data_type_, tensor->NumElements(),
                                     buf);
  arr->set_tag(input_tensors_.size() - 1);
  return arr;
}

std::unordered_map<std::string, ArrayPtr> TensorflowModel::GetOutputGpuArrays(){
  // Because TF always returns output in CPU memory, doesn't support in-place
  // output in GPU memory
  return {};
}

void TensorflowModel::Preprocess(std::shared_ptr<Task> task) {
  // Tensorflow uses NHWC by default. More details see
  // https://www.tensorflow.org/versions/master/performance/performance_guide

  auto prepare_image_default = [&](cv::Mat& image) {
    // Convert to image in float
    cv::Mat fimg;
    image.convertTo(fimg, CV_32FC3);
    // create a cv::Mat using buffer allocated in the in_arr
    auto in_arr = std::make_shared<Array>(DT_FLOAT, input_size_, cpu_device_);
    cv::Mat resized(image_height_, image_width_, CV_32FC3,
                    in_arr->Data<void>());
    cv::resize(fimg, resized, cv::Size(image_width_, image_height_));
    task->AppendInput(in_arr);
  };

  auto prepare_image_ssd = [&](cv::Mat& image) {
    auto in_arr = std::make_shared<Array>(DT_UINT8, input_size_, cpu_device_);
    // create a cv::Mat using buffer allocated in the in_arr
    cv::Mat resized(image_width_, image_height_, CV_8UC3,
                    in_arr->Data<void>());
    cv::resize(image, resized, cv::Size(image_width_, image_height_));
    task->AppendInput(in_arr);
  };

  std::function<void(cv::Mat&)> prepare_image;
  if (model_name() == "ssd_mobilenet" || model_name() == "ssd_mobilenet_0.75") {
    prepare_image = prepare_image_ssd;
  } else {
    prepare_image = prepare_image_default;
  }

  const auto& query = task->query;
  const auto& input_data = query.input();
  switch (input_data.data_type()) {
    case DT_IMAGE: {
      cv::Mat img = DecodeImage(input_data.image(), CO_RGB);
      task->attrs["im_height"] = img.rows;
      task->attrs["im_width"] = img.cols;
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
  std::vector<tf::Tensor> out_tensors;
  tf::Status status = session_->Run({{input_layer_, in_tensor}},
                                    output_layers_, {}, &out_tensors);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to run tensorflow: " << status.ToString();
    return;
  }
  std::unordered_map<std::string, Slice> slices;
  for (uint i = 0; i < output_layers_.size(); ++i) {
    const auto& name = output_layers_[i];
    const char* tensor_data = out_tensors[i].tensor_data().data();
    size_t nfloats = out_tensors[i].NumElements();
    auto out_arr = batch_task->GetOutputArray(name);
    float* out_data = out_arr->Data<float>();
    Memcpy(out_data, cpu_device_, tensor_data, cpu_device_,
           nfloats * sizeof(float));
    slices.emplace(name, Slice(batch_size, output_sizes_.at(name)));
  }
  batch_task->SliceOutputBatch(slices);
}

void TensorflowModel::Postprocess(std::shared_ptr<Task> task) {
  const QueryProto& query = task->query;
  QueryResultProto* result = &task->result;
  result->set_status(CTRL_OK);
  for (auto output : task->outputs) {
    if (type() == "classification") {
      auto out_arr = output->arrays.at(output_layers_[0]);
      float* out_data = out_arr->Data<float>();
      size_t count = out_arr->num_elements();
      size_t output_size = output_sizes_.at(output_layers_[0]);
      if (classnames_.empty()) {
        PostprocessClassification(query, out_data, output_size, result);
      } else {
        PostprocessClassification(query, out_data, output_size, result,
                                  &classnames_);
      }
    } else if (type() == "detection") {
      int im_height = task->attrs["im_height"].as<int>();
      int im_width = task->attrs["im_width"].as<int>();
      MarshalDetectionResult(query, output, im_height, im_width, result);
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
  tf::Tensor* tensor;
  if (input_data_type_ == DT_UINT8) {
    tensor = new tf::Tensor(gpu_allocator_, tf::DT_UINT8, shape);
  } else {
    tensor = new tf::Tensor(gpu_allocator_, tf::DT_FLOAT, shape);
  }
  input_tensors_.emplace_back(tensor);
  return tensor;
}

void TensorflowModel::MarshalDetectionResult(
    const QueryProto& query, std::shared_ptr<Output> output,
    int im_height, int im_width, QueryResultProto* result) {
  int num_boxes = static_cast<int>(output->arrays.at("num_detections")->
                                   Data<float>()[0]);
  float* boxes = output->arrays.at("detection_boxes")->Data<float>();
  float* scores = output->arrays.at("detection_scores")->Data<float>();
  float* classes = output->arrays.at("detection_classes")->Data<float>();

  std::vector<std::string> output_fields(query.output_field().begin(),
                                         query.output_field().end());
  if (output_fields.size() == 0) {
    output_fields.push_back("rect");
    output_fields.push_back("class_name");
  }
  for (int i = 0; i < num_boxes; ++i) {
    auto record = result->add_output();
    int class_id = static_cast<int>(classes[i]);
    for (auto field : output_fields) {
      if (field == "rect") {
        auto value = record->add_named_value();
        value->set_name("rect");
        value->set_data_type(DT_RECT);
        auto rect = value->mutable_rect();
        rect->set_top(int(im_height * boxes[i * 4]));
        rect->set_left(int(im_width * boxes[i * 4 + 1]));
        rect->set_bottom(int(im_height * boxes[i * 4 + 2]));
        rect->set_right(int(im_width * boxes[i * 4 + 3]));
      } else if (field == "score") {
        auto value = record->add_named_value();
        value->set_name("score");
        value->set_data_type(DT_FLOAT);
        value->set_f(scores[i]);
      } else if (field == "class_id") {
        auto value = record->add_named_value();
        value->set_name("class_id");
        value->set_data_type(DT_INT32);
        value->set_i(class_id);
      } else if (field == "class_name") {
        auto value = record->add_named_value();
        value->set_name("class_name");
        value->set_data_type(DT_STRING);
        auto iter = classnames_.find(class_id);
        if (iter == classnames_.end()) {
          LOG(ERROR) << "Cannot find class name for class id " << class_id;
        } else {
          value->set_s(iter->second);
        }
      }
    }
  }
}

void TensorflowModel::set_slice_tensor(const std::unique_ptr<tf::Tensor>& dst, const std::vector<int32_t> &src) {
  CHECK_EQ(dst->dtype(), tf::DT_INT32);
  CHECK_EQ(dst->NumElements(), src.size());
  Memcpy(const_cast<char*>(dst->tensor_data().data()), cpu_device_,
      src.data(), cpu_device_, sizeof(int32_t) * src.size());
}

} // namespace backend
} // namespace nexus

#endif // USE_TENSORFLOW
