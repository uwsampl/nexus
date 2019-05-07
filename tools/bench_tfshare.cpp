#include <utility>

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
DEFINE_string(model, "", "Model name");
DEFINE_string(image_dir, "", "Image directory");
DEFINE_string(batch_sizes, "", "list of batch sizes for each suffix model, separated by comma");
DEFINE_int32(height, 0, "Image height");
DEFINE_int32(width, 0, "Image width");
DEFINE_int32(dryrun, 5, "Warmup times for profiling");
DEFINE_int32(repeat, 30, "Repeat times for profiling");

namespace nexus {
namespace backend {

using duration = std::chrono::microseconds;
namespace fs = boost::filesystem;

class BenchTFShare {
 public:
  BenchTFShare(int gpu, const std::string &model_name,
               const std::string &image_dir, int height, int width,
               std::vector<int> batch_sizes) :
      gpu_(gpu), batch_sizes_(std::move(batch_sizes)) {
    // Init model session
    const std::string framework = "tf_share";
    const int model_version = 1;
    for (size_t i = 0; i < batch_sizes_.size(); ++i) {
      const auto name = model_name + "_" + std::to_string(i);
      auto model_info_ = ModelDatabase::Singleton().GetModelInfo(framework, name, model_version);
      CHECK(model_info_ != nullptr) << "Cannot find model info for "
                                    << framework << ":" << model_name << ":" << model_version;
      ModelSession model_sess;
      model_sess.set_framework(framework);
      model_sess.set_model_name(name);
      model_sess.set_version(model_version);
      model_sess.set_latency_sla(50000);
      if (height > 0) {
        CHECK_GT(width, 0) << "Height and width must be set together";
        model_sess.set_image_height(height);
        model_sess.set_image_width(width);
      } else {
        if ((*model_info_)["resizable"] && (*model_info_)["resizable"].as<bool>()) {
          // Set default image size for resizable CNN
          model_sess.set_image_height((*model_info_)["image_height"].as<uint32_t>());
          model_sess.set_image_width((*model_info_)["image_width"].as<uint32_t>());
        }
      }
      model_sessions_.push_back(model_sess);
    }
    // Get test dataset
    ListImages(image_dir);
    // Init GPU device
    NEXUS_CUDA_CHECK(cudaSetDevice(gpu_));
    gpu_device_ = DeviceManager::Singleton().GetGPUDevice(gpu_);
  }

  void Bench(int dryrun, int repeat) {
    ModelInstanceConfig config;
    for (const auto &model_sess : model_sessions_)
      config.add_model_session()->CopyFrom(model_sess);
    LOG(INFO) << config.DebugString();
    BlockPriorityQueue<Task> task_queue;

    // preprocess
    std::vector<std::shared_ptr<Task>> preproc_tasks;
    {
      config.set_batch(1);
      config.set_max_batch(1);
      std::unique_ptr<ModelInstance> model;
      CreateModelInstance(gpu_, config, &model);
      // prepare the input
      size_t num_inputs = test_images_.size();
      if (num_inputs > 1000) {
        num_inputs = 1000;
      }
      for (size_t idx = 0; idx < num_inputs; ++idx) {
        // Load input image
        std::string im;
        ReadImage(test_images_[idx], &im);
        auto task = std::make_shared<Task>();
        auto input = task->query.mutable_input();
        input->set_data_type(DT_IMAGE);
        auto image = input->mutable_image();
        image->set_data(im);
        image->set_format(ImageProto::JPEG);
        image->set_color(true);
        model->Preprocess(task);
        preproc_tasks.push_back(task);
      }
    }
    LOG(INFO) << "Preprocess finished";

    // forward and postprocess
    uint32_t total_batch = 0;
    for (auto batch_size : batch_sizes_)
      total_batch += batch_size;
    config.set_batch(total_batch);
    config.set_max_batch(total_batch);
    auto model = std::make_shared<ModelExecutor>(gpu_, config, task_queue);
    int total_tasks = 0;
    for (int num_try = -dryrun; num_try < repeat; ++num_try) {
      for (size_t i = 0; i < batch_sizes_.size(); ++i) {
        auto model_sess_id = ModelSessionToString(model_sessions_[i]);
        for (size_t j = 0; j < batch_sizes_[i]; ++j) {
          auto idx = total_tasks % preproc_tasks.size();
          auto task = std::make_shared<Task>();
          task->SetDeadline(std::chrono::milliseconds(1000000));
          task->query.set_query_id(total_tasks);
          task->query.set_model_session_id(model_sess_id);
          task->attrs = preproc_tasks[idx]->attrs;
          task->AppendInput(preproc_tasks[idx]->inputs[0]->array);
          model->AddPreprocessedTask(task);
          ++total_tasks;
        }
      }
    }
    printf("total tasks: %d\n", total_tasks);

    // start measuring forward latency
    std::vector<uint64_t> forward_lats;
    for (int i = -dryrun; i < repeat; ++i) {
      auto beg = std::chrono::high_resolution_clock::now();
      model->Execute();
      auto end = std::chrono::high_resolution_clock::now();
      auto time_us = std::chrono::duration_cast<duration>(end - beg).count();
      if (i >= 0)
        forward_lats.push_back(time_us);
      fprintf(stdout, "\x1b[2Krepeat %d/%d: %.3fms\r", i + 1, repeat, time_us / 1000.);
      fflush(stdout);
    }
    for (int i = 0; i < total_tasks; ++i) {
      auto task = task_queue.pop();
      CHECK_EQ(task->result.status(), CTRL_OK) << "Error detected: " << task->result.status();
    }
    CHECK_EQ(task_queue.size(), 0) << "Task queue is not empty";
    preproc_tasks.clear();

    // output
    double mean, std;
    std::tie(mean, std) = GetStats<uint64_t>(forward_lats);
    printf("\nbatch sizes =");
    for (auto batch_size : batch_sizes_)
      printf(" %d", batch_size);
    printf("\n");
    printf("mean %11.6f\n", mean / 1e3);
    printf("std  %11.6f\n", std / 1e3);
  }

 private:
  template<class T>
  std::pair<double, double> GetStats(const std::vector<T> &lats) {
    double mean = 0.;
    double std = 0.;
    for (uint i = 0; i < lats.size(); ++i) {
      mean += lats[i];
    }
    mean /= lats.size();
    for (uint i = 0; i < lats.size(); ++i) {
      std += (lats[i] - mean) * (lats[i] - mean);
    }
    std = sqrt(std / (lats.size() - 1));
    return {mean, std};
  }

  void ListImages(const std::string &root_dir) {
    fs::directory_iterator end_iter;
    for (fs::directory_iterator it(root_dir); it != end_iter; ++it) {
      test_images_.push_back(it->path().string());
    }
    LOG(INFO) << "Number of test images: " << test_images_.size();
  }

  void ReadImage(const std::string &file_path, std::string *content) {
    std::ifstream fin(file_path, std::ios::binary | std::ios::ate);
    size_t fsize = fin.tellg();
    fin.seekg(0);
    content->resize(fsize);
    fin.read(&((*content)[0]), fsize);
    fin.close();
  }

 private:
  int gpu_;
  std::vector<std::string> test_images_;
  std::vector<ModelSession> model_sessions_;
  std::vector<int> batch_sizes_;
  GPUDevice *gpu_device_;
};

} // namespace backend
} // namespace nexus


std::vector<int> parse_int_list(const std::string &str) {
  std::vector<int> list;
  int number = 0;
  for (size_t i = 0; i < str.size(); ++i) {
    if ('0' <= str[i] && str[i] <= '9') {
      number = number * 10 + str[i] - '0';
    } else if (str[i] != ',') {
      LOG(FATAL) << "unexpected character " << str[i];
    }
    if (str[i] == ',' || i + 1 == str.size()) {
      list.push_back(number);
      number = 0;
    }
  }
  return list;
}

int main(int argc, char **argv) {
  using namespace nexus;
  using namespace nexus::backend;

  // log to stderr
  FLAGS_logtostderr = true;
  // Init glog
  google::InitGoogleLogging(argv[0]);
  // Parse command line flags
  google::ParseCommandLineFlags(&argc, &argv, true);
  // Setup backtrace on segfault
  google::InstallFailureSignalHandler();
  // Check flags
  CHECK_GT(FLAGS_model.length(), 0) << "Missing model";
  CHECK_GT(FLAGS_image_dir.length(), 0) << "Missing image_dir";

  auto batch_sizes = parse_int_list(FLAGS_batch_sizes);

  BenchTFShare bench(FLAGS_gpu, FLAGS_model, FLAGS_image_dir,
                     FLAGS_height, FLAGS_width, batch_sizes);
  bench.Bench(FLAGS_dryrun, FLAGS_repeat);
}
