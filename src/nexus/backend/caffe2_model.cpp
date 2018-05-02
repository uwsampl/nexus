#if USE_CAFFE2 == 1

#include <boost/filesystem.hpp>
#include <fstream>
#include <glog/logging.h>
#include <opencv2/opencv.hpp>
#include <sstream>

#include "nexus/backend/caffe2_model.h"
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

Caffe2Model::Caffe2Model(int gpu_id, const std::string& model_name,
                         uint32_t version, const std::string& type,
                         uint32_t batch, uint32_t max_batch,
                         BlockPriorityQueue<Task>& task_queue,
                         const YAML::Node& info) :
    ModelInstance(gpu_id, model_name, version, type, batch, max_batch,
                  task_queue) {
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
  image_height_ = info["image_height"].as<int>();
  image_width_ = info["image_width"].as<int>();

  input_shape_.resize(4);
  input_shape_[0] = max_batch;
  input_shape_[1] = 3;
  input_shape_[2] = image_height_;
  input_shape_[3] = image_width_;

  input_tensor_ = workspace_->CreateBlob(input_blob_name_)->
                  GetMutable<caffe2::TensorCUDA>();
  input_tensor_->Resize(input_shape_);
  input_tensor_->mutable_data<float>();
  input_tensor_->Reserve(input_shape_, gpu_ctx_.get());
  // input size of a single input
  input_size_ = input_tensor_->size_from_dim(1);

  // Dry run to get output tensor and size
  CAFFE_ENFORCE(workspace_->RunNet(net_name_));
  output_tensor_ = &workspace_->GetBlob(output_blob_name_)->
                   Get<caffe2::TensorCUDA>();
  output_size_ = output_tensor_->size_from_dim(1);

  LOG(INFO) << "caffe2 model " << net_name_ << " input tensor: " <<
      input_tensor_->DebugString() << ", output tensor: " <<
      output_tensor_->DebugString();
  
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

std::string Caffe2Model::profile_id() const {
  std::stringstream ss;
  ss << "caffe2:" << model_name_ << ":" << version_;
  return ss.str();
}

void Caffe2Model::InitBatchInputArray() {
  auto buf = std::make_shared<Buffer>(input_tensor_->mutable_data<float>(),
                                      input_tensor_->nbytes(), gpu_device_);
  batch_input_array_ = std::make_shared<Array>(DT_FLOAT, input_tensor_->size(),
                                               buf);
}

void Caffe2Model::PreprocessImpl(std::shared_ptr<Task> task,
                                 std::vector<ArrayPtr>* input_arrays) {
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
    input_arrays->push_back(in_arr);
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

void Caffe2Model::ForwardImpl(BatchInput* batch_input,
                              BatchOutput* batch_output) {
  // reshape input blob to current batch size
  size_t batch = batch_input->batch_size();
  std::vector<int> input_shape(input_shape_);
  input_shape[0] = batch;
  input_tensor_->Resize(input_shape);
  CAFFE_ENFORCE(workspace_->RunNet(net_name_));
  auto out_arr = std::make_shared<Array>(DT_FLOAT, batch * output_size_,
                                         cpu_device_);
  Memcpy(out_arr->Data<void>(), cpu_device_, output_tensor_->data<float>(),
         gpu_device_, batch * output_size_ * sizeof(float));
  batch_output->SetOutputBatch({out_arr}, {Slice(batch, output_size_)});
}

void Caffe2Model::PostprocessImpl(std::shared_ptr<Task> task, Output* output) {
  const QueryProto& query = task->query;
  QueryResultProto* result = &task->result;
  auto out_arr = output->GetOutputs()[0];
  float* out_data = out_arr->Data<float>();
  if (type_ == "classification") {
    result->set_status(CTRL_OK);
    float threshold = 0.;
    MarshalClassificationResult(query, out_data, output_size_, threshold,
                                &task->result);
  } else {
    result->set_status(MODEL_TYPE_NOT_SUPPORT);
    std::ostringstream oss;
    oss << "Unsupported model type " << type() << " for " << framework();
    result->set_error_message(oss.str());
  }
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

void Caffe2Model::MarshalClassificationResult(
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

#endif // USE_CAFFE2 == 1
