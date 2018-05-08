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

Caffe2Model::Caffe2Model(int gpu_id, const ModelInstanceConfig& config,
                         const YAML::Node& info) :
    ModelInstance(gpu_id, config, info),
    first_input_array_(true) {
  CHECK(info["init_net"]) << "Missing cfg_file in the model info";
  CHECK(info["predict_net"]) << "Missing weight_file in the model info";
  CHECK(info["mean_file"] || info["mean_value"])
      << "Missing mean_file or mean_value in the model info";
  // load caffe model
  fs::path model_dir = fs::path(info["model_dir"].as<std::string>());
  fs::path init_net_path = model_dir / info["init_net"].as<std::string>();
  fs::path predict_net_path = model_dir / info["predict_net"].as<std::string>();
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
  CAFFE_ENFORCE(caffe2::ReadProtoFromFile(init_net_path.string(), &init_net));
  CAFFE_ENFORCE(caffe2::ReadProtoFromFile(predict_net_path.string(),
                                          &predict_net));
  init_net.mutable_device_option()->CopyFrom(option);
  predict_net.mutable_device_option()->CopyFrom(option);
  net_name_ = predict_net.name();

  // New workspace
  workspace_.reset(new caffe2::Workspace());
  // Load weights from init_net
  CAFFE_ENFORCE(workspace_->RunNetOnce(init_net));
  // Create predict network
  CAFFE_ENFORCE(workspace_->CreateNet(predict_net));
  
  input_blob_name_ = info["input_blob"].as<std::string>();
  output_blob_name_ = info["output_blob"].as<std::string>();
  if (model_session_.image_height() > 0) {
    image_height_ = model_session_.image_height();
    image_width_ = model_session_.image_width();
  } else {
    image_height_ = info["image_height"].as<int>();
    image_width_ = info["image_width"].as<int>();
  }

  input_shape_.resize(4);
  input_shape_[0] = max_batch_;
  input_shape_[1] = 3;
  input_shape_[2] = image_height_;
  input_shape_[3] = image_width_;

  std::string blob_name;
  caffe2::Blob* input_blob;
  std::tie(blob_name, input_blob) = NewInputBlob();
  auto input_tensor = input_blob->Get<caffe2::TensorCUDA>();
  // input size of a single input
  input_size_ = input_tensor.size_from_dim(1);

  // Dry run network to get output tensor and size
  workspace_->RenameBlob(blob_name, input_blob_name_);
  CAFFE_ENFORCE(workspace_->RunNet(net_name_));
  workspace_->RenameBlob(input_blob_name_, blob_name);
  output_tensor_ = &workspace_->GetBlob(output_blob_name_)->
                   Get<caffe2::TensorCUDA>();
  output_size_ = output_tensor_->size_from_dim(1);

  LOG(INFO) << model_session_id_ << " input size: " << input_size_ <<
      ", output size: " << output_size_;
  
  // Load scale factor and mean value
  if (info["scale"]) {
    scale_ = info["scale"].as<float>();
  } else {
    scale_ = 1.;
  }
  if (info["mean_file"]) {
    has_mean_file_ = true;
    fs::path mean_file = model_dir / info["mean_file"].as<std::string>();
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
    const YAML::Node& mean_values = info["mean_value"];
    CHECK(mean_values.IsSequence()) << "mean_value in the config is " <<
        "not sequence";
    CHECK_EQ(mean_values.size(), 3) << "mean_value must have 3 values";
    for (uint i = 0; i < mean_values.size(); ++i) {
      mean_value_.push_back(mean_values[i].as<float>());
    }
  }
  
  // Load classnames
  if (info["class_names"]) {
    fs::path cns_path = model_dir / info["class_names"].as<std::string>();
    LoadClassnames(cns_path.string());
  }
}

ArrayPtr Caffe2Model::CreateInputGpuArray() {
  caffe2::Blob* blob;
  if (first_input_array_) {
    blob = input_blobs_[0].second;
    first_input_array_ = false;
  } else {
    std::string name;
    std::tie(name, blob) = NewInputBlob();
  }
  auto tensor = blob->Get<caffe2::TensorCUDA>();
  auto buf = std::make_shared<Buffer>(tensor.mutable_data<float>(),
                                      tensor.nbytes(), gpu_device_);
  auto arr = std::make_shared<Array>(DT_FLOAT, tensor.size(), buf);
  arr->set_tag(input_blobs_.size() - 1);
  return arr;
}

std::unordered_map<std::string, size_t> Caffe2Model::OutputSizes() const {
  return {{output_blob_name_, output_size_}};
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

void Caffe2Model::Forward(BatchInput* batch_input, BatchOutput* batch_output) {
  // Get corresponding input blob
  std::string blob_name;
  caffe2::Blob* blob;
  std::tie(blob_name, blob) = input_blobs_[batch_input->array()->tag()];
  
  // Reshape input blob to current batch size
  size_t batch = batch_input->batch_size();
  std::vector<int> input_shape(input_shape_);
  input_shape[0] = batch;
  blob->GetMutable<caffe2::TensorCUDA>()->Resize(input_shape);
  
  // Run the net
  workspace_->RenameBlob(blob_name, input_blob_name_);
  CAFFE_ENFORCE(workspace_->RunNet(net_name_));
  workspace_->RenameBlob(input_blob_name_, blob_name);

  // Copy to output
  auto out_arr = batch_output->GetArray(output_blob_name_);
  Memcpy(out_arr->Data<void>(), out_arr->device(),
         output_tensor_->data<float>(), gpu_device_,
         batch * output_size_ * sizeof(float));
  batch_output->SliceBatch({{output_blob_name_, Slice(batch, output_size_)}});
}

void Caffe2Model::Postprocess(std::shared_ptr<Task> task) {
  const QueryProto& query = task->query;
  QueryResultProto* result = &task->result;
  result->set_status(CTRL_OK);
  for (auto& output : task->outputs) {
    auto out_arr = output->GetArray(output_blob_name_);
    float* out_data = out_arr->Data<float>();
    if (type_ == "classification") {
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

std::pair<std::string, caffe2::Blob*> Caffe2Model::NewInputBlob() {
  std::string blob_name = input_blob_name_ + "-" +
                          std::to_string(input_blobs_.size());
  caffe2::Blob* blob;
  if (input_blobs_.empty()) {
    blob = workspace_->RenameBlob(input_blob_name_, blob_name);
  } else {
    blob = workspace_->CreateBlob(blob_name);
  }
  auto tensor = blob->GetMutable<caffe2::TensorCUDA>();
  tensor->Resize(input_shape_);
  tensor->mutable_data<float>();
  tensor->Reserve(input_shape_, gpu_ctx_.get());
  input_blobs_.emplace_back(blob_name, blob);
  return std::make_pair(blob_name, blob);
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
