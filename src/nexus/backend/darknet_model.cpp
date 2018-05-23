#ifdef USE_DARKNET

#include <boost/filesystem.hpp>
#include <fstream>
#include <glog/logging.h>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <unordered_set>

#include "nexus/backend/darknet_model.h"
#include "nexus/backend/postprocess.h"
#include "nexus/backend/slice.h"
#include "nexus/common/image.h"
#include "nexus/proto/control.pb.h"

namespace fs = boost::filesystem;

namespace nexus {
namespace backend {

namespace {

image cvmat_to_image(const cv::Mat& cv_img_rgb) {
  int c = cv_img_rgb.channels();
  int h = cv_img_rgb.rows;
  int w = cv_img_rgb.cols;
  image img = make_image(w, h, c);
  int count = 0;
  for (int k = 0; k < c; ++k) {
    for (int i = 0; i < h; ++i) {
      for (int j = 0; j < w; ++j) {
        cv::Vec3b pixel = cv_img_rgb.at<cv::Vec3b>(i, j);
        img.data[count++] = pixel.val[k] / 255.;
      }
    }
  }
  return img;
}

}

DarknetModel::DarknetModel(int gpu_id, const ModelInstanceConfig& config) :
    ModelInstance(gpu_id, config),
    first_input_array_(true) {
  // load darknet model
  CHECK(model_info_["cfg_file"]) << "Missing cfg_file in the model info";
  CHECK(model_info_["weight_file"]) << "Missing weight_file in the model info";
  fs::path model_dir = fs::path(model_info_["model_dir"].as<std::string>());
  fs::path cfg_path = model_dir / model_info_["cfg_file"].as<std::string>();
  fs::path weight_path = model_dir / model_info_["weight_file"].
                         as<std::string>();
  CHECK(fs::exists(cfg_path)) << "Config file " << cfg_path <<
      " doesn't exist";
  CHECK(fs::exists(weight_path)) << "Weight file " << weight_path <<
      " doesn't exist";
  image_height_ = 0;
  image_width_ = 0;
  if (model_session_.image_height() > 0) {
    CHECK_GT(model_session_.image_width(), 0) << "image_height and " <<
        "image_width must be set at same time";
    image_height_ = model_session_.image_height();
    image_width_ = model_session_.image_width();
  }

  // switch the current directory to the model directory as required
  // for loading a model in the darknet
  fs::path curr_dir = fs::current_path();
  fs::current_path(weight_path.parent_path());
  net_ = parse_network_cfg_spec(const_cast<char*>(cfg_path.string().c_str()),
                                gpu_id, max_batch_, image_width_,
                                image_height_);
  load_weights(net_, const_cast<char*>(weight_path.string().c_str()));
  fs::current_path(curr_dir);

  // Get input and output's shape and size
  auto input_layer = net_->layers[0];
  input_shape_.set_dims({max_batch_, input_layer.c, input_layer.h,
          input_layer.w});
  input_size_ = input_shape_.NumElements(1);
  auto output_layer = get_network_output_layer(net_);
  output_shape_.set_dims({max_batch_, output_layer.out_c, output_layer.out_h,
          output_layer.out_w});
  output_size_ = output_shape_.NumElements(1);
  LOG(INFO) << "Model " << model_session_id_ << ": input shape " <<
      input_shape_ << " (" << input_size_ << "), output shape " <<
      output_shape_ << " (" << output_size_ << ")";
  // find the output layer id
  for (int i = net_->n - 1; i > 0; --i) {
    if (net_->layers[i].type != COST) {
      output_layer_id_ = i;
      break;
    }
  }
  // Darknet doesn't have name for layers, so hardcode name as "output"
  output_name_ = "output"; 
  // load classnames
  if (model_info_["class_names"]) {
    fs::path cns_path = model_dir / model_info_["class_names"].
                        as<std::string>();
    LoadClassnames(cns_path.string());
  }
}

DarknetModel::~DarknetModel() {
  free_network(net_);
}

Shape DarknetModel::InputShape() {
  return input_shape_;
}

std::unordered_map<std::string, Shape> DarknetModel::OutputShapes() {
  return {{ output_name_, output_shape_ }};
}

ArrayPtr DarknetModel::CreateInputGpuArray() {
  std::shared_ptr<Array> arr;
  if (first_input_array_) {
    auto buf = std::make_shared<Buffer>(
        net_->input_gpu, max_batch_ * input_size_ * sizeof(float),
        gpu_device_);
    arr = std::make_shared<Array>(DT_FLOAT, max_batch_ * input_size_, buf);
    first_input_array_ = false;
  } else {
    arr = std::make_shared<Array>(DT_FLOAT, max_batch_ * input_size_,
                                  gpu_device_);
  }
  return arr;
}

std::unordered_map<std::string, ArrayPtr> DarknetModel::GetOutputGpuArrays() {
  auto buf = std::make_shared<Buffer>(net_->layers[output_layer_id_].output_gpu,
                                      max_batch_ * output_size_ * sizeof(float),
                                      gpu_device_);
  auto arr = std::make_shared<Array>(DT_FLOAT, max_batch_ * output_size_, buf);
  return {{ output_name_, arr }};
}

void DarknetModel::Preprocess(std::shared_ptr<Task> task) {
  auto prepare_image = [&](cv::Mat& cv_img) {
    image img = cvmat_to_image(cv_img);
    //image resized = resize_image(img, net_->w, net_->h);
    image resized = letterbox_image(img, net_->w, net_->h);
    size_t nfloats = net_->w * net_->h * 3;
    auto buf = std::make_shared<Buffer>(
        resized.data, nfloats * sizeof(float), cpu_device_, true);
    free_image(img);
    /*
    cv::Mat resized_image;
    cv::resize(cv_img, resized_image, cv::Size(net_->w, net_->h));
    image input = cvmat_to_image(resized_image);
    size_t nfloats = net_->w * net_->h * 3;
    auto buf = std::make_shared<Buffer>(
        input.data, nfloats * sizeof(float), cpu_device_, true);
    */
    auto in_arr = std::make_shared<Array>(DT_FLOAT, nfloats, buf);
    task->AppendInput(in_arr);
  };

  const auto& query = task->query;
  const auto& input_data = query.input();
  switch (input_data.data_type()) {
    case DT_IMAGE: {
      cv::Mat cv_img_rgb = DecodeImage(input_data.image(), CO_RGB);
      task->attrs["im_height"] = cv_img_rgb.rows;
      task->attrs["im_width"] = cv_img_rgb.cols;
      if (query.window_size() > 0) {
        for (int i = 0; i < query.window_size(); ++i) {
          auto rect = query.window(i);
          cv::Mat crop_img = cv_img_rgb(cv::Rect(
              rect.left(), rect.top(), rect.right() - rect.left(),
              rect.bottom() - rect.top()));
          prepare_image(crop_img);
        }
      } else {
        prepare_image(cv_img_rgb);
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

void DarknetModel::Forward(std::shared_ptr<BatchTask> batch_task) {
  size_t batch_size = batch_task->batch_size();
  net_->input_gpu = batch_task->GetInputArray()->Data<float>();
  set_batch_network_lightweight(net_, batch_size);
  network_predict_gpu_nocopy(net_);
  layer l = net_->layers[output_layer_id_];
  auto out_arr = batch_task->GetOutputArray(output_name_);
  Memcpy(out_arr->Data<void>(), cpu_device_, l.output_gpu, gpu_device_,
         batch_size * output_size_ * sizeof(float));
  batch_task->SliceOutputBatch({{
        output_name_, Slice(batch_size, output_size_) }});
}

void DarknetModel::Postprocess(std::shared_ptr<Task> task) {
  const auto& query = task->query;
  auto* result = &task->result;
  result->set_status(CTRL_OK);
  for (auto& output : task->outputs) {
    auto out_arr = output->arrays.at(output_name_);
    float* out_data = out_arr->Data<float>();
    // TODO: check predicates in the query
    if (type() == "detection") {
      layer l = net_->layers[net_->n - 1];
      size_t nboxes = l.w * l.h * l.n;
      size_t nprobs = nboxes * (l.classes + 1);
      int* boxes = new int[nboxes * 4];
      float* probs = new float[nprobs];
      int only_objectness = 0;
      float tree_threshold = 0.5;
      int relative = 1;
      float nms = 0.3;
      float threshold = 0.24;
      int im_height = task->attrs["im_height"].as<int>();
      int im_width = task->attrs["im_width"].as<int>();
      output_detection_results(
          out_data, l, im_width, im_height, net_->w, net_->h, threshold, probs,
          nprobs, boxes, nboxes * 4, only_objectness, nullptr, tree_threshold,
          relative, nms);
      MarshalDetectionResult(query, probs, nprobs, boxes, nboxes, result);
      delete[] boxes;
      delete[] probs;
    } else if (type() == "classification") {
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

void DarknetModel::LoadClassnames(const std::string& filepath) {
  std::ifstream infile(filepath);
  CHECK(infile.good()) << "Classname file " << filepath << " doesn't exist";
  std::string line;
  while (std::getline(infile, line)) {
    classnames_.push_back(line);
  }
  LOG(INFO) << "Load " << classnames_.size() << " class names from " <<
      filepath;
}

void DarknetModel::MarshalDetectionResult(
    const QueryProto& query, const float* probs, size_t nprobs,
    const int* boxes, size_t nboxes, QueryResultProto* result) {
  std::vector<std::string> output_fields(query.output_field().begin(),
                                         query.output_field().end());
  if (output_fields.size() == 0) {
    output_fields.push_back("rect");
    output_fields.push_back("class_name");
  }
  size_t nclasses_plus_1 = nprobs / nboxes;
  for (size_t i = 0; i < nboxes; ++i) {
    const float* ps = &probs[i * nclasses_plus_1];
    const int* bs = &boxes[i * 4];
    bool allzero = true;
    for (size_t j = 0; j < 4; ++j) {
      if (bs[j] != 0) {
        allzero = false;
        break;
      }
    }
    if (allzero) {
      continue;
    }
    float max_prob = 0.;
    int max_idx = -1;
    for (size_t j = 0; j < nclasses_plus_1 - 1; ++j) {
      float p = float(ps[j]);
      if (p > 0) {
        if (p > max_prob) {
          max_prob = p;
          max_idx = j;
        }
      }
    }
    // fill in detection result
    auto record = result->add_output();
    for (auto field : output_fields) {
      if (field == "rect") {
        auto value = record->add_named_value();
        value->set_name("rect");
        value->set_data_type(DT_RECT);
        auto rect = value->mutable_rect();
        rect->set_left(bs[0]);
        rect->set_right(bs[1]);
        rect->set_top(bs[2]);
        rect->set_bottom(bs[3]);
      } else if (field == "objectness") {
        auto value = record->add_named_value();
        value->set_name("objectness");
        value->set_data_type(DT_FLOAT);
        value->set_f(ps[nclasses_plus_1 - 1]);
      } else if (field == "class_id") {
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

#endif // USE_DARKNET
