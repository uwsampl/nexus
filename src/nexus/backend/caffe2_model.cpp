#ifdef USE_CAFFE2

#include <boost/filesystem.hpp>
#include <fstream>
#include <glog/logging.h>
#include <opencv2/opencv.hpp>
#include <sstream>

#include "nexus/backend/caffe2_model.h"
#include "nexus/backend/postprocess.h"
#include "nexus/backend/slice.h"
#include "nexus/common/image.h"
#include "nexus/common/util.h"
#include "nexus/proto/control.pb.h"
// Caffe2 headers
#include "caffe2/utils/proto_utils.h"
#include "caffe/proto/caffe.pb.h"

namespace fs = boost::filesystem;

namespace nexus {
namespace backend {

Caffe2Model::Caffe2Model(int gpu_id, const ModelInstanceConfig& config) :
    ModelInstance(gpu_id, config),
    first_input_array_(true) {
  CHECK(model_info_["init_net"]) << "Missing cfg_file in the model info";
  CHECK(model_info_["predict_net"]) << "Missing weight_file in the model info";
  CHECK(model_info_["mean_file"] || model_info_["mean_value"])
      << "Missing mean_file or mean_value in the model info";
  // load caffe model
  fs::path model_dir = fs::path(model_info_["model_dir"].as<std::string>());
  fs::path init_net_path = model_dir / model_info_["init_net"].
                           as<std::string>();
  fs::path predict_net_path = model_dir / model_info_["predict_net"].
                              as<std::string>();
  CHECK(fs::exists(init_net_path)) << "init_net file " << init_net_path <<
      " doesn't exist";
  CHECK(fs::exists(predict_net_path)) << "predict_net file " <<
      predict_net_path << " doesn't exist";

  // Init GPU context
  caffe2::DeviceOption option;
  option.set_cuda_gpu_id(gpu_id);
  option.set_device_type(caffe2::CUDA);
  gpu_ctx_.reset(new caffe2::CUDAContext(option));

  // Load network from protobuf
  caffe2::NetDef init_net, predict_net;
  LoadModel(init_net_path.string(), predict_net_path.string(), config,
            &init_net, &predict_net);
  init_net.mutable_device_option()->CopyFrom(option);
  predict_net.mutable_device_option()->CopyFrom(option);
  // Use caffe2 async dag net, for now use 2 workers by default
  // TODO: probably allow to tune number of workers in the future
  predict_net.set_type("async_dag");
  predict_net.set_num_workers(1);
  net_name_ = predict_net.name();

  // New workspace
  workspace_.reset(new caffe2::Workspace());
  // Load weights from init_net
  CAFFE_ENFORCE(workspace_->RunNetOnce(init_net));
  // Create predict network
  CAFFE_ENFORCE(workspace_->CreateNet(predict_net));
  net_ = workspace_->GetNet(net_name_);
  LOG(INFO) << "Caffe2 model support async run: " << net_->SupportsAsync();
  
  // Get input size of a single input
  input_size_ = input_shape_.NumElements(1);
  // Dry run network to get output tensor, shape, and size
  uint32_t blob_idx;
  caffe2::Blob* input_blob;
  std::tie(blob_idx, input_blob) = NewInputBlob();
  std::string blob_name = input_blobs_[blob_idx].first;
  workspace_->RenameBlob(blob_name, input_blob_name_);
  CAFFE_ENFORCE(workspace_->RunNet(net_name_));
  workspace_->RenameBlob(input_blob_name_, blob_name);
  output_tensor_ = workspace_->GetBlob(output_blob_name_)->
                   GetMutable<caffe2::TensorCUDA>();
  output_shape_.set_dims(output_tensor_->dims());
  output_size_ = output_shape_.NumElements(1);

  LOG(INFO) << "Model " << model_session_id_ << ", input " <<
      input_blob_name_ << ": " << input_shape_ << " (" << input_size_ <<
      "), output " << output_blob_name_ << ": " << output_shape_ <<
      " (" << output_size_ << ")";
  
  // Get preprocessing parameters
  if (model_info_["scale"]) {
    scale_ = model_info_["scale"].as<float>();
  } else {
    scale_ = 1.;
  }
  if (model_info_["mean_file"]) {
    has_mean_file_ = true;
    fs::path mean_file = model_dir / model_info_["mean_file"].as<std::string>();
    caffe::BlobProto mean_proto;
    caffe2::ReadProtoFromBinaryFile(mean_file.string().c_str(), &mean_proto);
    size_t mean_size = 1;
    if (mean_proto.has_height()) {
      mean_size = mean_proto.num() * mean_proto.channels() *
                  mean_proto.height() * mean_proto.width();
    } else {
      for (int i = 0; i < mean_proto.shape().dim_size(); ++i) {
        mean_size *= mean_proto.shape().dim(i);
      }
    }
    CHECK_EQ(mean_size, input_size_) << "Mean blob size must be equal to " <<
        "input size";
    mean_blob_.resize(mean_size);
    for (uint i = 0; i < mean_size; ++i) {
      mean_blob_[i] = mean_proto.data(i);
    }
  } else {
    has_mean_file_ = false;
    const YAML::Node& mean_values = model_info_["mean_value"];
    CHECK(mean_values.IsSequence()) << "mean_value in the config is " <<
        "not sequence";
    CHECK_EQ(mean_values.size(), 3) << "mean_value must have 3 values";
    for (uint i = 0; i < mean_values.size(); ++i) {
      mean_value_.push_back(mean_values[i].as<float>());
    }
  }
  
  // Load classnames
  if (model_info_["class_names"]) {
    fs::path cns_path = model_dir / model_info_["class_names"].
                        as<std::string>();
    LoadClassnames(cns_path.string());
  }
}

Shape Caffe2Model::InputShape() {
  return input_shape_;
}

std::unordered_map<std::string, Shape> Caffe2Model::OutputShapes() {
  return {{ output_blob_name_, output_shape_ }};
}

ArrayPtr Caffe2Model::CreateInputGpuArray() {
  uint32_t blob_idx;
  caffe2::Blob* blob;
  if (first_input_array_) {
    blob_idx = 0;
    blob = input_blobs_[blob_idx].second;
    first_input_array_ = false;
  } else {
    std::tie(blob_idx, blob) = NewInputBlob();
  }
  size_t nfloats = max_batch_ * input_size_;
  auto tensor = blob->Get<caffe2::TensorCUDA>();
  auto buf = std::make_shared<Buffer>(tensor.mutable_data<float>(),
                                      nfloats * sizeof(float), gpu_device_);
  auto arr = std::make_shared<Array>(DT_FLOAT, nfloats, buf);
  arr->set_tag(blob_idx);
  return arr;
}

ArrayPtr Caffe2Model::CreateInputGpuArrayWithRawPointer(float* ptr,
                                                        size_t nfloats) {
  uint32_t blob_idx;
  caffe2::Blob* blob;
  std::tie(blob_idx, blob) = NewInputBlob(ptr, nfloats);
  auto tensor = blob->Get<caffe2::TensorCUDA>();
  auto buf = std::make_shared<Buffer>(tensor.mutable_data<float>(),
                                      nfloats * sizeof(float), gpu_device_);
  auto arr = std::make_shared<Array>(DT_FLOAT, nfloats, buf);
  arr->set_tag(blob_idx);
  return arr;
}

void Caffe2Model::RemoveInputGpuArray(ArrayPtr arr) {
  auto blob_name = input_blobs_.at(arr->tag()).first;
  workspace_->RemoveBlob(blob_name);
  input_blobs_.erase(arr->tag());
}

std::unordered_map<std::string, ArrayPtr> Caffe2Model::GetOutputGpuArrays() {
  size_t nfloats = max_batch_ * output_size_;
  auto buf = std::make_shared<Buffer>(output_tensor_->mutable_data<float>(),
                                      nfloats * sizeof(float), gpu_device_);
  auto arr = std::make_shared<Array>(DT_FLOAT, nfloats, buf);
  return {{ output_blob_name_, arr }};
}

void Caffe2Model::Preprocess(std::shared_ptr<Task> task) {
  auto prepare_image = [&](cv::Mat& image) {
    auto in_arr = std::make_shared<Array>(DT_FLOAT, input_size_, cpu_device_);
    cv::Mat resized_img;
    cv::resize(image, resized_img, cv::Size(image_width_, image_height_));
    float* out_ptr = in_arr->Data<float>();
    int out_index;
    for (int h = 0; h < image_height_; ++h) {
      const uchar* ptr = resized_img.ptr<uchar>(h);
      int in_index = 0;
      for (int w = 0; w < image_width_; ++w) {
        for (int c = 0; c < 3; ++c) {
          out_index = (c * image_height_ + h) * image_width_ + w;
          float pixel = static_cast<float>(ptr[in_index++]);
          if (has_mean_file_) {
            out_ptr[out_index] = (pixel - mean_blob_[out_index]) * scale_;
          } else {
            out_ptr[out_index] = (pixel - mean_value_[c]) * scale_;
          }
        }
      }
    }
    task->AppendInput(in_arr);
  };

  const auto& query = task->query;
  const auto& input_data = query.input();
  switch (input_data.data_type()) {
    case DT_IMAGE: {
      cv::Mat cv_img_bgr = DecodeImage(input_data.image(), CO_BGR);
      if (query.window_size() > 0) {
        for (int i = 0; i < query.window_size(); ++i) {
          const auto& rect = query.window(i);
          cv::Mat crop_img = cv_img_bgr(cv::Rect(
              rect.left(), rect.top(), rect.right() - rect.left(),
              rect.bottom() - rect.top()));
          prepare_image(crop_img);
        }
      } else {
        prepare_image(cv_img_bgr);
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

void Caffe2Model::Forward(std::shared_ptr<BatchTask> batch_task) {
  // Get corresponding input blob
  std::string blob_name;
  caffe2::Blob* blob;
  std::tie(blob_name, blob) = input_blobs_[batch_task->GetInputArray()->tag()];
  
  // Reshape input blob to current batch size
  size_t batch = batch_task->batch_size();
  std::vector<int> input_shape = input_shape_.dims();
  input_shape[0] = batch;
  blob->GetMutable<caffe2::TensorCUDA>()->Resize(input_shape);
  
  // Run the net
  workspace_->RenameBlob(blob_name, input_blob_name_);
  CAFFE_ENFORCE(net_->Run());
  workspace_->RenameBlob(input_blob_name_, blob_name);

  // Copy to output
  auto out_arr = batch_task->GetOutputArray(output_blob_name_);
  Memcpy(out_arr->Data<void>(), out_arr->device(),
         output_tensor_->data<float>(), gpu_device_,
         batch * output_size_ * sizeof(float));
  batch_task->SliceOutputBatch({{
        output_blob_name_, Slice(batch, output_size_)}});
}

void Caffe2Model::ForwardAsync(std::shared_ptr<BatchTask> batch_task) {
  // Get corresponding input blob
  std::string blob_name;
  caffe2::Blob* blob;
  std::tie(blob_name, blob) = input_blobs_[batch_task->GetInputArray()->tag()];
  
  // Reshape input blob to current batch size
  uint32_t batch = batch_task->batch_size();
  std::vector<int> input_shape = input_shape_.dims();
  input_shape[0] = batch;
  blob->GetMutable<caffe2::TensorCUDA>()->Resize(input_shape);
  
  // Run the net
  workspace_->RenameBlob(blob_name, input_blob_name_);
  CAFFE_ENFORCE(net_->RunAsync());
  workspace_->RenameBlob(input_blob_name_, blob_name);
}

void Caffe2Model::WaitOutput(std::shared_ptr<BatchTask> batch_task) {
  uint32_t batch = batch_task->batch_size();
  net_->Wait();
  auto out_arr = batch_task->GetOutputArray(output_blob_name_);
  Memcpy(out_arr->Data<void>(), out_arr->device(),
         output_tensor_->data<float>(), gpu_device_,
         batch * output_size_ * sizeof(float));
  batch_task->SliceOutputBatch({{
        output_blob_name_, Slice(batch, output_size_)}});
}

void Caffe2Model::Postprocess(std::shared_ptr<Task> task) {
  const QueryProto& query = task->query;
  QueryResultProto* result = &task->result;
  result->set_status(CTRL_OK);
  for (auto& output : task->outputs) {
    auto out_arr = output->arrays.at(output_blob_name_);
    float* out_data = out_arr->Data<float>();
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

void Caffe2Model::LoadModel(const std::string& init_path,
                            const std::string& predict_path,
                            const ModelInstanceConfig& config,
                            caffe2::NetDef* init_net,
                            caffe2::NetDef* predict_net) {
  std::unordered_set<std::string> external_inputs;
  std::unordered_set<std::string> external_outputs;
  if (config.start_index() == 0 && config.end_index() == 0) {
    CAFFE_ENFORCE(caffe2::ReadProtoFromFile(init_path, init_net));
    CAFFE_ENFORCE(caffe2::ReadProtoFromFile(predict_path, predict_net));
  } else {
    caffe2::NetDef full_init_net, full_predict_net;
    CAFFE_ENFORCE(caffe2::ReadProtoFromFile(init_path, &full_init_net));
    CAFFE_ENFORCE(caffe2::ReadProtoFromFile(predict_path, &full_predict_net));
    int num_ops = full_predict_net.op_size();
    int start_index = config.start_index();
    int end_index = config.end_index() == 0 ? num_ops : config.end_index();
    int op_idx = 0;
    predict_net->set_name(full_predict_net.name());
    for (int i = start_index; i < end_index; ++i) {
      auto const& op = full_predict_net.op(i);
      predict_net->add_op()->CopyFrom(op);
      for (auto& input : op.input()) {
        auto iter = external_outputs.find(input);
        if (iter == external_outputs.end()) {
          external_inputs.insert(input);
        } else {
          external_outputs.erase(iter);
        }
      }
      for (auto& output : op.output()) {
        if (output[0] != '_') {
          external_outputs.insert(output);
        }
      }
    }
    for (auto input : external_inputs) {
      predict_net->add_external_input(input);
    }
    for (auto output : external_outputs) {
      predict_net->add_external_output(output);
    }
    //LOG(INFO) << predict_net->DebugString();
    for (int i = 0; i < full_init_net.op_size(); ++i) {
      auto const& op = full_init_net.op(i);
      if (external_inputs.find(op.output(0)) != external_inputs.end()) {
        init_net->add_op()->CopyFrom(op);
      }
    }
    CHECK_EQ(external_outputs.size(), 1) << "Number of outputs must be 1";
  }
  if (config.start_index() == 0) {
    input_blob_name_ = model_info_["input_blob"].as<std::string>();
    if (model_session_.image_height() > 0) {
      image_height_ = model_session_.image_height();
      image_width_ = model_session_.image_width();
    } else {
      image_height_ = model_info_["image_height"].as<int>();
      image_width_ = model_info_["image_width"].as<int>();
    }
    input_shape_.set_dims({max_batch_, 3, image_height_, image_width_});
  } else {
    // Add input placeholder
    input_blob_name_ = config.input_name();
    auto placeholder = init_net->add_op();
    placeholder->add_output(input_blob_name_);
    placeholder->set_type("ConstantFill");
    auto arg = placeholder->add_arg();
    arg->set_name("shape");
    arg->add_ints(1);

    std::vector<int> shape;
    shape.push_back(max_batch_);
    for (auto dim : config.input_shape()) {
      shape.push_back(dim);
    }
    input_shape_.set_dims(shape);
  }
  if (config.end_index() == 0) {
    output_blob_name_ = model_info_["output_blob"].as<std::string>();
  } else {
    for (auto iter : external_outputs) {
      output_blob_name_ = iter;
    }
  }
  // Set context of all operators to be CUDA
  for (int i = 0; i < predict_net->op_size(); ++i) {
    auto device_option = predict_net->mutable_op(i)->mutable_device_option();
    device_option->set_cuda_gpu_id(gpu_id_);
    device_option->set_device_type(caffe2::CUDA);
  }
}

std::pair<uint32_t, caffe2::Blob*> Caffe2Model::NewInputBlob() {
  uint32_t idx;
  std::string blob_name;
  caffe2::Blob* blob;
  if (input_blobs_.empty()) {
    idx = 0;
    blob_name = input_blob_name_ + "-" + std::to_string(idx);
    blob = workspace_->RenameBlob(input_blob_name_, blob_name);
  } else {
    for (idx = 1; ; ++idx) {
      if (input_blobs_.count(idx) == 0) {
        break;
      }
    }
    blob_name = input_blob_name_ + "-" + std::to_string(idx);
    blob = workspace_->CreateBlob(blob_name);
  }
  auto tensor = blob->GetMutable<caffe2::TensorCUDA>();
  tensor->Resize(input_shape_.dims());
  tensor->mutable_data<float>();
  tensor->Reserve(input_shape_.dims(), gpu_ctx_.get());
  input_blobs_.emplace(idx, std::make_pair(blob_name, blob));
  return std::make_pair(idx, blob);
}

std::pair<uint32_t, caffe2::Blob*> Caffe2Model::NewInputBlob(float* ptr,
                                                             size_t nfloats) {
  uint32_t batch = nfloats / input_size_;
  CHECK_GT(batch, 0) << "Capacity is too small to fit in batch size 1";
  //LOG(INFO) << "Batch size: " << batch;
  uint32_t idx;
  for (idx = 1; ; ++idx) {
    if (input_blobs_.count(idx) == 0) {
      break;
    }
  }
  std::string blob_name = input_blob_name_ + "-" + std::to_string(idx);
  caffe2::Blob* blob;
  blob = workspace_->CreateBlob(blob_name);
  auto tensor = blob->GetMutable<caffe2::TensorCUDA>();
  auto dims = input_shape_.dims();
  dims[0] = batch;
  tensor->Resize(dims);
  tensor->ShareExternalPointer<float>(ptr, nfloats * sizeof(float));
  input_blobs_.emplace(idx, std::make_pair(blob_name, blob));
  return std::make_pair(idx, blob);
}

void Caffe2Model::LoadClassnames(const std::string& filepath) {
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

#endif // USE_CAFFE2
