#include <boost/filesystem.hpp>
#include <fstream>
#include <glog/logging.h>
#include <opencv2/opencv.hpp>
#include <sstream>

#include "nexus/backend/caffe_model.h"
#include "nexus/backend/slice.h"
#include "nexus/common/image.h"
#include "nexus/common/util.h"
#include "nexus/proto/control.pb.h"

namespace fs = boost::filesystem;

namespace nexus {
namespace backend {

CaffeModel::CaffeModel(int gpu_id, const std::string& model_name,
                       uint32_t version, const std::string& type,
                       uint32_t batch, uint32_t max_batch,
                       BlockPriorityQueue<Task>& task_queue,
                       const YAML::Node& info) :
    ModelInstance(gpu_id, model_name, version, type, batch, max_batch,
                  task_queue) {
  CHECK(info["cfg_file"]) << "Missing cfg_file in the model info";
  CHECK(info["weight_file"]) << "Missing weight_file in the model info";
  CHECK(info["mean_file"] || info["mean_value"])
      << "Missing mean_file or mean_value in the model info";
  // load caffe model
  fs::path model_dir = fs::path(info["model_dir"].as<std::string>());
  fs::path cfg_path = model_dir / info["cfg_file"].as<std::string>();
  fs::path weight_path = model_dir / info["weight_file"].as<std::string>();
  CHECK(fs::exists(cfg_path)) << "cfg file " << cfg_path <<
      " doesn't exist";
  CHECK(fs::exists(weight_path)) << "weight file " << weight_path <<
      " doesn't exist";
  // init gpu device
  caffe::Caffe::SetDevice(gpu_id);
  caffe::Caffe::set_mode(caffe::Caffe::GPU);
  // load network
  net_.reset(new caffe::ServeNet<float>(cfg_path.string(), max_batch));
  net_->CopyTrainedLayersFrom(weight_path.string());
  // get input and output shape
  // NOTE: currently we only consider single input and single output
  CHECK_EQ(net_->num_inputs(), 1)
      << "CaffeModel only support caffe model that has single input";
  CHECK_EQ(net_->num_outputs(), 1)
      << "CaffeModel only support caffe model that has single output";
  input_shape_ = std::vector<int>(net_->input_blobs()[0]->shape());
  output_shape_ = std::vector<int>(net_->output_blobs()[0]->shape());
  input_blob_indices_ = net_->input_blob_indices();
  // get the single input and output size
  input_size_ = 1;
  output_size_ = 1;
  for (size_t i = 1; i < input_shape_.size(); ++i) {
    input_size_ *= input_shape_[i];
  }
  for (size_t i = 1; i < output_shape_.size(); ++i) {
    output_size_ *= output_shape_[i];
  }
  LOG(INFO) << "model " << model_name_ << ": input size " << input_size_ <<
      ", output size " << output_size_;
  // set up data transformer
  caffe::TransformationParameter transform_param;
  if (info["scale"]) {
    transform_param.set_scale(info["scale"].as<float>());
  }
  transform_param.set_crop_size(input_shape_[2]); // height of input_shape_
  if (info["mean_file"]) {
    fs::path mean_file = model_dir / info["mean_file"].as<std::string>();
    transform_param.set_mean_file(mean_file.string());
  } else {
    const YAML::Node& mean_values = info["mean_value"];
    CHECK(mean_values.IsSequence()) <<
        "mean_value in the config is not sequence";
    for (uint i = 0; i < mean_values.size(); ++i) {
      transform_param.add_mean_value(mean_values[i].as<float>());
    }
  }
  transformer_.reset(new caffe::DataTransformer<float>(
      transform_param, net_->phase()));
  // resize image shape
  if (info["image_dim"]) {
    image_dim_ = info["image_dim"].as<int>();
  } else {
    image_dim_ = input_shape_[2];
  }
  // whether enbable prefix batching
  if (info["prefix_layer"]) {
    prefix_layer_ = info["prefix_layer"].as<std::string>();
    prefix_index_ = net_->layer_index_by_name(prefix_layer_);
    LOG(INFO) << "Prefix layer up to " << prefix_layer_ << "(" <<
        prefix_index_ << ")";
  } else {
    prefix_layer_ = "";
    prefix_index_ = -1;
  }
  // load classnames
  if (info["class_names"]) {
    fs::path cns_path = model_dir / info["class_names"].as<std::string>();
    LoadClassnames(cns_path.string());
  }
}

std::string CaffeModel::profile_id() const {
  std::stringstream ss;
  ss << "caffe:" << model_name_ << ":" << version_;
  return ss.str();
}

void CaffeModel::InitBatchInputArray() {
  input_blob_ = net_->input_blobs()[input_blob_indices_[0]];
  auto buf = std::make_shared<Buffer>(
      input_blob_->mutable_gpu_data(), input_blob_->count() * sizeof(float),
      gpu_device_);
  batch_input_array_ = std::make_shared<Array>(
      DT_FLOAT, input_blob_->count(), buf);
}
/*
void CaffeModel::UpdateMaxBatchImpl() {
  input_shape_[0] = max_batch_;
  // Set the release_memory flag to free GPU memory in case max batch decreases
  caffe::Caffe::set_release_memory(true);
  net_->input_blobs()[input_blob_indices_[0]]->Reshape(input_shape_);
  net_->Reshape();
  caffe::Caffe::set_release_memory(false);
  InitBatchInputArray();
}
*/
void CaffeModel::PreprocessImpl(std::shared_ptr<Task> task,
                                std::vector<ArrayPtr>* input_arrays) {
  auto prepare_image = [&](cv::Mat& image) {
    auto in_arr = std::make_shared<Array>(DT_FLOAT, input_size_, cpu_device_);
    cv::Mat resized_image;
    cv::resize(image, resized_image, cv::Size(image_dim_, image_dim_));
    std::vector<int> blob_shape = input_shape_;
    blob_shape[0] = 1;
    caffe::Blob<float> blob(blob_shape);
    blob.data()->set_cpu_data(in_arr->Data<void>());
    transformer_->Transform(resized_image, &blob);
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

void CaffeModel::ForwardImpl(BatchInput* batch_input,
                             BatchOutput* batch_output) {
  size_t batch = batch_input->batch_size();
  // reshape input blob to current batch size
  std::vector<int> input_shape = input_shape_;
  input_shape[0] = batch;
  input_blob_->Reshape(input_shape);
  auto out_arr = std::make_shared<Array>(DT_FLOAT, batch * output_size_,
                                         cpu_device_);
  // We don't need to reshape the network, because during the forwarding
  // Caffe will reshape every layers based on the input batch size
  if (prefix_index_ >= 0) {
#if 1
    // Forward the shared prefix part
    net_->ForwardTo(prefix_index_);
    auto prefix_top_blob_id = net_->top_ids(prefix_index_)[0];
    auto prefix_top_blob = net_->blobs()[prefix_top_blob_id];
    float* prefix_top_gpu_data = prefix_top_blob->mutable_gpu_data();
    std::vector<int> bottom_shape = prefix_top_blob->shape();
    bottom_shape[0] = 1; // change to single batch
    boost::shared_ptr<caffe::Blob<float> > split_bottom_blob(
        new caffe::Blob<float>(bottom_shape));
    int split_count = split_bottom_blob->count();
    std::vector<int> blob_shape = input_shape_;
    // Forward the split part
    for (size_t i = 0; i < batch; ++i) {
      split_bottom_blob->set_gpu_data(prefix_top_gpu_data + i * split_count);
      net_->set_blob(prefix_top_blob_id, split_bottom_blob);
      net_->ForwardFrom(prefix_index_ + 1);
      const caffe::Blob<float>* output_blob = net_->output_blobs()[0];
      Memcpy(out_arr->Data<void>() + i * output_size_, cpu_device_,
             output_blob->gpu_data(), gpu_device_, output_size_ * sizeof(float));
    }
    // restore prefix top blob
    net_->set_blob(prefix_top_blob_id, prefix_top_blob);
#else
    // No prefix batching
    auto input_blob = net_->blobs()[0];
    float* gpu_data = input_blob->mutable_gpu_data();
    input_shape[0] = 1;
    boost::shared_ptr<caffe::Blob<float> > single_input_blob(
        new caffe::Blob<float>(input_shape));
    for (size_t i = 0; i < batch; ++i) {
      single_input_blob->set_gpu_data(gpu_data + i * input_size_);
      net_->set_blob(0, single_input_blob);
      auto output_blob = net_->Forward()[0];
      Memcpy(out_arr->Data<void>() + i * output_size_, cpu_device_,
             output_blob->gpu_data(), gpu_device_, output_size_ * sizeof(float));
    }
    net_->set_blob(0, input_blob);
#endif
  } else {
    const caffe::Blob<float>* output_blob = net_->Forward()[0];
    Memcpy(out_arr->Data<void>(), cpu_device_, output_blob->gpu_data(),
           gpu_device_, output_blob->count() * sizeof(float));
  }
  batch_output->SetOutputBatch({out_arr}, {Slice(batch, output_size_)});
}

void CaffeModel::PostprocessImpl(std::shared_ptr<Task> task, Output* output) {
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

void CaffeModel::LoadClassnames(const std::string& filepath) {
  std::ifstream infile(filepath);
  CHECK(infile.good()) << "Classname file " << filepath << " doesn't exist";
  std::string line;
  while (std::getline(infile, line)) {
    classnames_.push_back(line);
  }
  infile.close();
}

void CaffeModel::MarshalClassificationResult(
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
