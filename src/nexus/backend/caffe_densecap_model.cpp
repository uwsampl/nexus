#ifdef USE_CAFFE

#include <boost/filesystem.hpp>
#include <boost/make_shared.hpp>
#include <fstream>
#include <glog/logging.h>
#include <opencv2/opencv.hpp>

#include "nexus/backend/caffe_densecap_model.h"
#include "nexus/common/image.h"
#include "nexus/common/util.h"
// Caffe headers
#include "caffe/layers/slice_layer.hpp"
#include "caffe/util/nms.hpp"

namespace nexus {
namespace backend {

namespace fs = boost::filesystem;

CaffeDenseCapModel::CaffeDenseCapModel(int gpu_id,
                                       const ModelInstanceConfig& config,
                                       const YAML::Node& info) :
    ModelInstance(gpu_id, config, info) {
  CHECK(info["feature_prototxt"]) << "Missing feature_prototxt in the config";
  CHECK(info["rnn_prototxt"]) << "Missing rnn_prototxt in the config";
  CHECK(info["embed_prototxt"]) << "Missing embed_prototxt in the config";
  CHECK(info["model_file"]) << "Missing model_file in the config";
  CHECK(info["vocab_file"]) << "Missing vocab_file in the config";
  CHECK(info["mean_value"]) << "Missing mean_value in the config";
  CHECK_EQ(info["mean_value"].size(), 3);
  // load config
  max_boxes_ = info["max_boxes"].as<int>();
  max_timestep_ = info["max_timestep"].as<int>();
  nms_threshold_ = info["nms_threshold"].as<float>();
  score_threshold_ = info["score_threshold"].as<float>();
  for (uint i = 0; i < info["mean_value"].size(); ++i) {
    mean_values_.push_back(info["mean_value"][i].as<float>());
  }
  for (uint i = 0; i < info["bbox_mean"].size(); ++i) {
    bbox_mean_.push_back(info["bbox_mean"][i].as<float>());
  }
  for (uint i = 0; i < info["bbox_stds"].size(); ++i) {
    bbox_stds_.push_back(info["bbox_stds"][i].as<float>());
  }
  // init gpu device
  caffe::Caffe::SetDevice(gpu_id);
  caffe::Caffe::set_mode(caffe::Caffe::GPU);
  // load caffe model
  fs::path model_dir = fs::path(info["model_dir"].as<std::string>());
  fs::path feature_prototxt = model_dir / info["feature_prototxt"].
                              as<std::string>();
  fs::path rnn_prototxt = model_dir / info["rnn_prototxt"].
                          as<std::string>();
  fs::path embed_prototxt = model_dir / info["embed_prototxt"].
                            as<std::string>();
  fs::path model_file = model_dir / info["model_file"].as<std::string>();
  feature_net_.reset(new caffe::ServeNet<float>(
      feature_prototxt.string(), max_batch_));
  rnn_net_.reset(new caffe::ServeNet<float>(
      rnn_prototxt.string(), max_batch_ * max_boxes_, 1));
  embed_net_.reset(new caffe::ServeNet<float>(
      embed_prototxt.string(), max_batch_ * max_boxes_, 1));
  caffe::NetParameter caffemodel;
  caffe::ReadNetParamsFromBinaryFileOrDie(model_file.string(), &caffemodel);
  feature_net_->CopyTrainedLayersFrom(caffemodel);
  rnn_net_->CopyTrainedLayersFrom(caffemodel);
  embed_net_->CopyTrainedLayersFrom(caffemodel);
  // set up input and output size
  image_height_ = model_session_.image_height();
  image_width_ = model_session_.image_width();
  /*
  int target_size = info["target_size"].as<int>();
  int max_size = info["max_size"].as<int>();
  float aspect_ratio = info["aspect_ratio"].as<float>();
  if (aspect_ratio > 1) {
    input_width_ = (int) round(
        std::min((float) max_size, target_size * aspect_ratio));
    input_height_ = (int) round(input_width_ / aspect_ratio);
  } else {
    input_height_ = (int) round(
        std::min((float) max_size, target_size / aspect_ratio));
    input_width_ = (int) round(input_height_ * aspect_ratio);
  }*/
  LOG(INFO) << "input shape: " << image_height_ << " x " << image_width_;
  input_size_ = 3 * image_height_ * image_width_;
  input_shape_.push_back(max_batch_);
  input_shape_.push_back(3);
  input_shape_.push_back(image_height_);
  input_shape_.push_back(image_width_);
  feature_net_input_idx_ = feature_net_->input_blob_indices()[0];
  // Reshape the input blob and feature_net according to our input size
  feature_net_->input_blobs()[0]->Reshape(input_shape_);
  feature_net_->Reshape();
  // load vocabulary
  fs::path vocab_file = model_dir / info["vocab_file"].as<std::string>();
  LoadVocabulary(vocab_file.string());
  // set helper buffer
  multiplier_.reset(new caffe::Blob<float>({max_boxes_}));
  caffe::caffe_gpu_set(max_boxes_, (float) 1., multiplier_->mutable_gpu_data());
  best_words_.resize(max_batch_ * max_boxes_);

  output_sizes_.emplace("rois", max_boxes_ * 5);
  output_sizes_.emplace("bbox_offsets", max_boxes_ * 4);
  output_sizes_.emplace("captions", max_boxes_ * max_timestep_);
  output_sizes_.emplace("scores", max_boxes_ * 2);
}

ArrayPtr CaffeDenseCapModel::CreateInputGpuArray() {
  boost::shared_ptr<caffe::Blob<float> > blob;
  if (input_blobs_.empty()) {
    blob = feature_net_->blobs()[feature_net_input_idx_];
  } else {
    blob = boost::make_shared<caffe::Blob<float> >(input_shape_);
  }
  auto buf = std::make_shared<Buffer>(blob->mutable_gpu_data(),
                                      blob->count() * sizeof(float),
                                      gpu_device_);
  auto arr = std::make_shared<Array>(DT_FLOAT, blob->count(), buf);
  arr->set_tag(input_blobs_.size());
  input_blobs_.push_back(blob);
  return arr;
}

std::unordered_map<std::string, size_t> CaffeDenseCapModel::OutputSizes()
    const {
  return output_sizes_;
}

void CaffeDenseCapModel::Preprocess(std::shared_ptr<Task> task) {
  const auto& query = task->query;
  const auto& input_data = query.input();
  if (input_data.data_type() != DT_IMAGE) {
    task->result.set_status(INPUT_TYPE_INCORRECT);
    task->result.set_error_message("Input type incorrect: " +
                                   DataType_Name(input_data.data_type()));
    return;
  }
  cv::Mat cv_img_bgr = DecodeImage(input_data.image(), CO_BGR);
  cv::Mat img;
  cv_img_bgr.convertTo(img, CV_32FC3);
  for (cv::Point3_<float>& p : cv::Mat_<cv::Point3_<float> >(img)) {
    p.x -= mean_values_[0];
    p.y -= mean_values_[1];
    p.z -= mean_values_[2];
  }
  int im_height = cv_img_bgr.rows;
  int im_width = cv_img_bgr.cols;
  int height = input_shape_[2];
  int width = input_shape_[3];
  cv::Mat resized;
  cv::resize(img, resized, cv::Size(width, height));
  float scale_h = float(height) / im_height;
  float scale_w = float(width) / im_width;
  // set the attributes
  task->attrs["im_height"] = im_height;
  task->attrs["im_width"] = im_width;
  task->attrs["scale_h"] = scale_h;
  task->attrs["scale_w"] = scale_w;
  // transpose the image
  const float* im_data = (const float*) resized.data;
  auto in_arr = std::make_shared<Array>(DT_FLOAT, input_size_, cpu_device_);
  float* input = in_arr->Data<float>();
  for (int h = 0; h < height; ++h) {
    for (int w = 0; w < width; ++w) {
      for (int c = 0; c < 3; ++c) {
        input[(c * height + h) * width + w] = im_data[(h * width + w) * 3 + c];
      }
    }
  }
  task->AppendInput(in_arr);
}

void CaffeDenseCapModel::Forward(BatchInput* batch_input,
                                 BatchOutput* batch_output) {
  // Get all output arrays
  auto rois_arr = batch_output->GetArray("rois");
  auto bbox_offsets_arr = batch_output->GetArray("bbox_offsets");
  auto captions_arr = batch_output->GetArray("captions_arr");
  auto scores_arr = batch_output->GetArray("scores");
  
  auto t1 = std::chrono::high_resolution_clock::now();
  // Prepare image blob
  auto blob = input_blobs_[batch_input->array()->tag()];
  int batch = batch_input->batch_size();
  std::vector<int> input_shape(input_shape_);
  input_shape[0] = batch;
  blob->Reshape(input_shape);
  feature_net_->set_blob(feature_net_input_idx_, blob);
  
  // Prepare im_info blob
  auto im_info_blob = feature_net_->input_blobs()[1];
  float* im_info = im_info_blob->mutable_cpu_data();
  im_info[0] = input_shape_[2];  // input image height
  im_info[1] = input_shape_[3];  // input image width
  im_info[2] = batch_input->inputs()[0]->task->attrs["scale_h"].as<float>();
  
  // set the slice points
  auto split_fc7_layer = dynamic_cast<caffe::SliceLayer<float>*>(
      feature_net_->layer_by_name("split_fc7").get());
  auto split_fc8_layer = dynamic_cast<caffe::SliceLayer<float>*>(
      feature_net_->layer_by_name("split_fc8").get());
  split_fc7_layer->SetSlicePoints({batch});
  split_fc8_layer->SetSlicePoints({batch});
  // forward feature_net_
  feature_net_->Forward();
  
  // get outputs from feature_net_
  auto num_proposals_blob = feature_net_->blob_by_name("num_proposals");
  auto rois_blob = feature_net_->blob_by_name("rois");
  auto scores_blob = feature_net_->blob_by_name("cls_probs");
  int total_proposals = scores_blob->shape(0);

  // Copy outputs of feature_net from GPU to CPU
  float* num_proposals = new float[batch];
  float* rois = rois_arr->Data<float>();
  float* scores = scores_arr->Data<float>();
  Memcpy(num_proposals, cpu_device_, num_proposals_blob->gpu_data(),
         gpu_device_, batch * sizeof(float));
  // add offset batch is because [0 - batch-1] rois are global rois
  Memcpy(rois, rois_arr->device(),
         rois_blob->gpu_data() + rois_blob->offset(batch), gpu_device_,
         total_proposals * 5 * sizeof(float));
  Memcpy(scores, scores_arr->device(), scores_blob->gpu_data(), gpu_device_,
         total_proposals * 2 * sizeof(float));
  auto t2 = std::chrono::high_resolution_clock::now();
  //LOG(INFO) << *num_proposals;
  //LOG(INFO) << total_proposals;

  // Get all blobs needed for RNN
  // outputs from feature_net_
  auto _global_features_blob = feature_net_->blob_by_name("global_features");
  auto region_features_blob = feature_net_->blob_by_name("region_features");
  // inputs to rnn_net_
  auto global_features_blob = rnn_net_->blob_by_name("global_features");
  auto cont_sentence_blob = rnn_net_->blob_by_name("cont_sentence");
  // outputs from rnn_net_
  auto word_probs_blob = rnn_net_->blob_by_name("probs");
  auto bbox_pred_blob = rnn_net_->blob_by_name("bbox_pred");
  // input to embed_net_
  auto input_sentence_blob = embed_net_->input_blobs()[0];
  // output from embed_net_
  auto embedded_sentence_blob = embed_net_->blob_by_name("embedded_sentence");
  int nfeat = _global_features_blob->shape(2);
  int nwords = word_probs_blob->shape(2);
  
  // prepare inputs to rnn_net_
  global_features_blob->Reshape({global_features_blob->shape(0),
          total_proposals, global_features_blob->shape(2)});
  int offset = 0;
  // broadcase global features to same dimension as region features
  for (int i = 0; i < batch; ++i) {
    caffe::caffe_gpu_gemm(
        CblasNoTrans, CblasNoTrans, num_proposals[i], nfeat, 1,
        (float)1., multiplier_->gpu_data(), _global_features_blob->gpu_data(),
        (float)0., global_features_blob->mutable_gpu_data() + offset);
    offset += int(num_proposals[i]) * nfeat;
  }
  rnn_net_->set_blob("input_features", region_features_blob);
  cont_sentence_blob->Reshape({1, total_proposals});
  // init cont_sentence with 0 first
  caffe::caffe_gpu_set(total_proposals, (float) 0.,
                       cont_sentence_blob->mutable_gpu_data());
  
  // pass the image features throught rnn_net_ first
  rnn_net_->Forward();
  
  // Prepare the inputs to embed_net_ and rnn_net_
  // reset cont_sentence to 1
  caffe::caffe_gpu_set(total_proposals, (float) 1.,
                       cont_sentence_blob->mutable_gpu_data());
  input_sentence_blob->Reshape({1, total_proposals});
  // start with EOS for all sentences
  caffe::caffe_gpu_set(total_proposals, (float) 0.,
                       input_sentence_blob->mutable_gpu_data());
  // hook the output of embed_net_ to rnn_net_
  rnn_net_->set_blob("input_features", embedded_sentence_blob);
  rnn_net_->set_blob("global_features", embedded_sentence_blob);
  
  // Get the output buffers
  float* bbox_offsets = bbox_offsets_arr->Data<float>();
  // TODO: current assume captions array allocated in CPU only
  float* captions = captions_arr->Data<float>();

  // forward rnn_net_
  for (int step = 0; step < max_timestep_; ++step) {
    embed_net_->Forward();
    rnn_net_->Forward();
    const float* word_prob = word_probs_blob->cpu_data();
    int finished = 0;
    for (int i = 0; i < total_proposals; ++i) {
      int max_idx = -1;
      float max_prob = 0.;
      for (int j = 0; j < nwords; ++j) {
        if (word_prob[j] > max_prob) {
          max_prob = word_prob[j];
          max_idx = j;
        }
      }
      // LOG(INFO) << "proposal " << i << ": " << max_idx << " " << max_prob;
      if (step == 0 || captions[i * max_timestep_ + step - 1] != 0) {
        if (max_idx == 0) {
          ++finished;
        }
        best_words_[i] = max_idx;
        captions[i * max_timestep_ + step] = max_idx;
        Memcpy(bbox_offsets + i * 4, bbox_offsets_arr->device(),
               bbox_pred_blob->gpu_data() + i * 4, gpu_device_,
               4 * sizeof(float));
      } else {
        ++finished;
        best_words_[i] = 0.;
        captions[i * max_timestep_ + step] = 0;
      }
      word_prob += nwords;
    }
    if (finished == total_proposals) {
      break;
    }
    // copy the best words to input_sentence gpu data
    Memcpy(input_sentence_blob->mutable_gpu_data(), gpu_device_,
           best_words_.data(), cpu_device_, total_proposals * sizeof(float));
  }

  auto t3 = std::chrono::high_resolution_clock::now();
  auto feature_lat = std::chrono::duration_cast<std::chrono::milliseconds>(
      t2 - t1);
  auto caption_lat = std::chrono::duration_cast<std::chrono::milliseconds>(
      t3 - t2);
  LOG(INFO) << "feature latency: " << feature_lat.count() << " ms, caption " <<
      " latency: " << caption_lat.count() << " ms";

  // Slice output
  std::unordered_map<std::string, Slice> slices{
    {"rois", Slice(batch, num_proposals, 5)},
    {"bbox_offsets", Slice(batch, num_proposals, 4)},
    {"captions", Slice(batch, num_proposals, max_timestep_)},
    {"scores", Slice(batch, num_proposals, 2)},
  };
  batch_output->SliceBatch(slices);
  delete[] num_proposals;
}

void CaffeDenseCapModel::Postprocess(std::shared_ptr<Task> task) {
  const QueryProto& query = task->query;
  QueryResultProto* result = &task->result;
  std::vector<std::string> output_fields(query.output_field().begin(),
                                         query.output_field().end());
  if (output_fields.size() == 0) {
    output_fields.push_back("rect");
    output_fields.push_back("caption");
  }
  
  auto& output = task->outputs[0];
  int nboxes = output->GetArray("rois")->num_elements() / 5;
  float* rois = output->GetArray("rois")->Data<float>();
  float* boxes = rois;
  float* bbox_offsets = output->GetArray("bbox_offsets")->Data<float>();
  float* captions = output->GetArray("captions")->Data<float>();
  float* scores = output->GetArray("scores")->Data<float>();
  // get attributes
  int im_height = task->attrs["im_height"].as<int>();
  int im_width = task->attrs["im_width"].as<int>();
  float scale = task->attrs["scale_h"].as<float>();
  int* order = new int[nboxes];
  for (int i = 0; i < nboxes; ++i) {
    order[i] = i;
    // transform bbox offsets
    for (int j = 0; j < 4; ++j) {
      bbox_offsets[i * 4 + j] = bbox_offsets[i * 4 + j] * bbox_stds_[j] +
                                bbox_mean_[j];
    }
  }
  TransformBbox(im_height, im_width, scale, nboxes, rois, bbox_offsets, boxes);
  std::sort(order, order + nboxes, [&](int a, int b) {
      return scores[b * 2 + 1] < scores[a * 2 + 1];
    });
  int num_out;
  int* keep_out = new int[nboxes];
  caffe::nms_cpu(boxes, order, nboxes, nms_threshold_, keep_out, &num_out);

  int total_boxes = 0;
  for (int i = 0; i < num_out; ++i) {
    int idx = keep_out[i];
    if (scores[idx * 2 + 1] <= score_threshold_) {
      continue;
    }
    std::string sentence;
    for (int step = 0; step < max_timestep_; ++step) {
      int word = (int) captions[idx * max_timestep_ + step];
      if (word == 0) {
        break;
      }
      sentence += vocabulary_[word] + " ";
    }
    float* box = boxes + idx * 4;
    // Add caption result
    auto record = result->add_output();
    for (auto field : output_fields) {
      if (field == "rect") {
        auto value = record->add_named_value();
        value->set_name("rect");
        value->set_data_type(DT_RECT);
        auto rect = value->mutable_rect();
        rect->set_left(int(round(box[0])));
        rect->set_top(int(round(box[1])));
        rect->set_right(int(round(box[2])));
        rect->set_bottom(int(round(box[3])));
      } else if (field == "caption") {
        auto value = record->add_named_value();
        value->set_name("caption");
        value->set_data_type(DT_STRING);
        value->set_s(sentence);
      } else if (field == "score") {
        auto value = record->add_named_value();
        value->set_name("score");
        value->set_data_type(DT_FLOAT);
        value->set_f(scores[idx * 2 + 1]);
      }
    }
    ++total_boxes;
  }
  //LOG(INFO) << "total boxes: " << total_boxes;
  delete[] order;
  delete[] keep_out;
}

void CaffeDenseCapModel::LoadVocabulary(const std::string& filename) {
  std::ifstream fin(filename);
  vocabulary_.push_back("<EOS>");
  std::string line;
  while (std::getline(fin, line)) {
    vocabulary_.push_back(line);
  }
  fin.close();
  LOG(INFO) << "Load " << vocabulary_.size() << " vocabs from " << filename;
}

void CaffeDenseCapModel::TransformBbox(
    int im_height, int im_width, float scale, int nboxes, const float* rois,
    const float* bbox_deltas, float* out) {
  for (int i = 0; i < nboxes; ++i) {
    float x1 = rois[i * 5 + 1] / scale;
    float y1 = rois[i * 5 + 2] / scale;
    float x2 = rois[i * 5 + 3] / scale;
    float y2 = rois[i * 5 + 4] / scale;
    float width = x2 - x1 + 1;
    float height = y2 - y1 + 1;
    float ctr_x = x1 + 0.5 * width;
    float ctr_y = y1 + 0.5 * height;
    float dx = bbox_deltas[i * 4];
    float dy = bbox_deltas[i * 4 + 1];
    float dw = bbox_deltas[i * 4 + 2];
    float dh = bbox_deltas[i * 4 + 3];
    ctr_x += dx * width;
    ctr_y += dy * height;
    width *= exp(dw);
    height *= exp(dh);
    out[i * 4] = std::max(std::min(ctr_x - 0.5 * width, im_width - 1.), 0.);
    out[i * 4 + 1] = std::max(std::min(ctr_y - 0.5 * height, im_height - 1.),
                              0.);
    out[i * 4 + 2] = std::max(std::min(ctr_x + 0.5 * width, im_width - 1.), 0.);
    out[i * 4 + 3] = std::max(std::min(ctr_y + 0.5 * height, im_height - 1.),
                              0.);
  }
}

} // namespace backend
} // namespace nexus

#endif // USE_CAFFE
