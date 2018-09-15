#include <boost/filesystem.hpp>
#include <cmath>
#include <fstream>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <iostream>
#include <cstdlib>
#include <string>
#include <time.h>
#include <thread>
#include <vector>
#include <unordered_map>
#include <yaml-cpp/yaml.h>

#include "nexus/common/device.h"
#include "nexus/common/block_queue.h"
#include "nexus/common/model_db.h"
#include "nexus/backend/model_exec.h"
#include "nexus/backend/model_ins.h"
#include "nexus/proto/nnquery.pb.h"

DEFINE_int32(gpu, 0, "GPU device id");
DEFINE_string(framework, "", "Framework");
DEFINE_string(model, "", "Model name");
DEFINE_int32(model_version, 1, "Version");
DEFINE_string(image_dir, "", "Image directory");
DEFINE_int32(batch, 0, "Batch size");
DEFINE_int32(num_models, 0, "Number of models");
DEFINE_int32(repeat, 100, "Repeat times for profiling");

namespace nexus {
namespace backend {

using duration = std::chrono::microseconds;
namespace fs = boost::filesystem;

class ModelProfiler {
 public:
  ModelProfiler(int gpu, const std::string& framework,
                const std::string& model_name, int model_version,
                const std::string& image_dir, int height=0, int width=0) :
      gpu_(gpu) {
    model_info_ = ModelDatabase::Singleton().GetModelInfo(
        framework, model_name, model_version);
    CHECK(model_info_ != nullptr) << "Cannot find model info for " <<
        framework << ":" << model_name << ":" << model_version;
    // Init model session
    model_sess_.set_framework(framework);
    model_sess_.set_model_name(model_name);
    model_sess_.set_version(model_version);
    model_sess_.set_latency_sla(50000);
    if (height > 0) {
      CHECK_GT(width, 0) << "Height and width must be set together";
      model_sess_.set_image_height(height);
      model_sess_.set_image_width(width);
    } else {
      if ((*model_info_)["resizable"] &&
          (*model_info_)["resizable"].as<bool>()) {
        // Set default image size for resizable CNN
        model_sess_.set_image_height(
            (*model_info_)["image_height"].as<uint32_t>());
        model_sess_.set_image_width(
            (*model_info_)["image_width"].as<uint32_t>());
      }
    }
    LOG(INFO) << model_sess_.DebugString();
    model_sessions_.push_back(ModelSessionToString(model_sess_));
    LOG(INFO) << "Profile model " << ModelSessionToProfileID(model_sess_);
    // Get test dataset
    ListImages(image_dir);
    // Init GPU device
    NEXUS_CUDA_CHECK(cudaSetDevice(gpu_));
    gpu_device_ = DeviceManager::Singleton().GetGPUDevice(gpu_);
  }

  void LoadModel(int batch) {
    BlockPriorityQueue<Task> task_queue;
    std::vector<std::shared_ptr<ModelExecutor> > models;
    size_t origin_freemem = gpu_device_->FreeMemory();
    ModelInstanceConfig config;
    config.add_model_session()->CopyFrom(model_sess_);
    config.set_batch(batch);
    config.set_max_batch(batch);
    for (int i = 0; i < FLAGS_num_models; ++i) {
      models.push_back(
          std::make_shared<ModelExecutor>(gpu_, config, task_queue));
    }
    size_t memory_use = origin_freemem - gpu_device_->FreeMemory();
    LOG(INFO) << "Number of models: " << FLAGS_num_models;
    LOG(INFO) << "Memory use: " << memory_use << " B";
  }

  void Profile(int batch, int repeat=100) {
    size_t origin_freemem = gpu_device_->FreeMemory();
    LOG(INFO) << "Origin free memory: " << origin_freemem;

    ModelInstanceConfig config;
    config.add_model_session()->CopyFrom(model_sess_);
    std::vector<std::string> share_models =
        ModelDatabase::Singleton().GetPrefixShareModels(
            ModelSessionToModelID(model_sess_));
    CHECK_GE(share_models.size(), FLAGS_num_models - 1) << "Number of models is too large";
    for (int i = 0; i < FLAGS_num_models - 1; ++i) {
      auto model_id = share_models[i];
      LOG(INFO) << model_id;
      auto share_sess = config.add_model_session();
      ParseModelID(model_id, share_sess);
      share_sess->set_latency_sla(50000);
      model_sessions_.push_back(ModelSessionToString(*share_sess));
    }
    LOG(INFO) << config.DebugString();
    BlockPriorityQueue<Task> task_queue;

    // preprocess
    std::vector<std::shared_ptr<Task> > preproc_tasks;
    {
      config.set_batch(1);
      config.set_max_batch(1);
      //auto model = CreateModelInstance(gpu_, config);
      std::unique_ptr<ModelInstance> model;
      CreateModelInstance(gpu_, config, &model);
      // prepare the input
      int num_inputs = batch * (repeat + 1);
      if (num_inputs > 1000) {
        num_inputs = 1000;
      }
      for (int i = 0; i < num_inputs; ++i) {
        // Load input image
        int idx = rand() % test_images_.size();
        std::string im;
        ReadImage(test_images_[idx], &im);
        auto task = std::make_shared<Task>();
        auto input = task->query.mutable_input();
        input->set_data_type(DT_IMAGE);
        auto image = input->mutable_image();
        image->set_data(im);
        image->set_format(ImageProto::JPEG);
        image->set_color(true);
        std::vector<ArrayPtr> input_arrays;
        auto beg = std::chrono::high_resolution_clock::now();
        model->Preprocess(task);
        auto end = std::chrono::high_resolution_clock::now();
        preproc_tasks.push_back(task);
      }
    }
    std::this_thread::sleep_for(std::chrono::microseconds(200));
    LOG(INFO) << "Preprocess finished";

    // forward and postprocess
    int dryrun = 4;
    config.set_batch(batch);
    config.set_max_batch(batch);
    auto model = std::make_shared<ModelExecutor>(gpu_, config, task_queue);
    std::vector<uint64_t> forward_lats;
    for (int i = 0; i < batch * (repeat + dryrun); ++i) {
      int idx = i % preproc_tasks.size();
      auto task = std::make_shared<Task>();
      task->SetDeadline(std::chrono::milliseconds(1000000));
      task->query.set_query_id(i);
      task->query.set_model_session_id(
          model_sessions_[i % model_sessions_.size()]);
      task->attrs = preproc_tasks[idx]->attrs;
      task->AppendInput(preproc_tasks[idx]->inputs[0]->array);
      model->AddPreprocessedTask(task);
    }
    // dry run
    for (int i = 0; i < dryrun; ++i) {
      model->Execute();
    }
    // start meansuring forward latency
    for (int i = 0; i < repeat; ++i) {
      auto beg = std::chrono::high_resolution_clock::now();
      model->Execute();
      auto end = std::chrono::high_resolution_clock::now();
      forward_lats.push_back(
          std::chrono::duration_cast<duration>(end - beg).count());
    }
    size_t curr_freemem = gpu_device_->FreeMemory();
    size_t memory_usage = origin_freemem - curr_freemem;
    for (int i = 0; i < batch * (repeat + dryrun); ++i) {
      auto task = task_queue.pop();
      CHECK_EQ(task->result.status(), CTRL_OK) << "Error detected: " <<
          task->result.status();
      auto beg = std::chrono::high_resolution_clock::now();
      model->Postprocess(task);
      auto end = std::chrono::high_resolution_clock::now();
    }
    auto stats = GetStats<uint64_t>(forward_lats);
    float min = std::get<0>(stats);
    float mean = std::get<1>(stats);
    float std = std::get<2>(stats);
    CHECK_EQ(task_queue.size(), 0) << "Task queue is not empty";
    std::this_thread::sleep_for(std::chrono::microseconds(200));

    LOG(INFO) << "Final free memory: " << gpu_device_->FreeMemory();
    preproc_tasks.clear();
    
    // output to file
    std::cout << gpu_device_->device_name() << "\n";
    std::cout << "Forward latency\n";
    std::cout << "batch size: " << batch << "\n";
    std::cout << "Forwared latency (min/mean/std): " << min << " us, " <<
        mean << " us, " << std << " us\n";
    std::cout << "Memory usage: " << memory_usage << " B\n";
  }

 private:
  template<class T>
  std::tuple<float, float, float> GetStats(const std::vector<T>& lats) {
    float min = lats[0];
    float mean = 0.;
    float std = 0.;
    for (uint i = 0; i < lats.size(); ++i) {
      if (lats[i] < min) {
        min = lats[i];
      }
      mean += lats[i];
    }
    mean /= lats.size();
    for (uint i = 0; i < lats.size(); ++i) {
      std += (lats[i] - mean) * (lats[i] - mean);
    }
    std = sqrt(std / (lats.size() - 1));
    return std::make_tuple(min, mean, std);
  }

  void ListImages(const std::string& root_dir) {
    fs::directory_iterator end_iter;
    for (fs::directory_iterator it(root_dir); it != end_iter; ++it) {
      test_images_.push_back(it->path().string());
    }
    LOG(INFO) << "Number of test images: " << test_images_.size();
  }

  void ReadImage(const std::string& file_path, std::string* content) {
    std::ifstream fin(file_path, std::ios::binary | std::ios::ate);
    size_t fsize = fin.tellg();
    fin.seekg(0);
    content->resize(fsize);
    fin.read(&((*content)[0]), fsize);
    fin.close();
  }

 private:
  int gpu_;
  ModelSession model_sess_;
  const YAML::Node* model_info_;
  std::string framework_;
  std::string model_name_;
  int version_;
  int height_;
  int width_;
  std::vector<std::string> test_images_;
  std::vector<std::string> model_sessions_;
  GPUDevice* gpu_device_;
};

} // namespace backend
} // namespace nexus


int main(int argc, char** argv) {
  using namespace nexus;
  using namespace nexus::backend;
  
  // log to stderr
  FLAGS_logtostderr = 1;
  // Init glog
  google::InitGoogleLogging(argv[0]);
  // Parse command line flags
  google::ParseCommandLineFlags(&argc, &argv, true);
  // Setup backtrace on segfault
  google::InstallFailureSignalHandler();
  // Check flags
  CHECK_GT(FLAGS_framework.length(), 0) << "Missing framework";
  CHECK_GT(FLAGS_model.length(), 0) << "Missing model";
  CHECK_GT(FLAGS_image_dir.length(), 0) << "Missing image_dir";
  CHECK_GT(FLAGS_batch, 0) << "Missing batch";
  CHECK_GT(FLAGS_num_models, 0) << "Missing num_models";
  srand(time(NULL));
  ModelProfiler profiler(FLAGS_gpu, FLAGS_framework, FLAGS_model,
                         FLAGS_model_version, FLAGS_image_dir);
  profiler.Profile(FLAGS_batch, FLAGS_repeat);
  //profiler.LoadModel(FLAGS_batch);
}
