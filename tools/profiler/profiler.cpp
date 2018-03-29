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

#include "common/device.h"
#include "common/block_queue.h"
#include "common/model_def.h"
#include "backend/model_ins.h"
#include "proto/nnquery.pb.h"

DEFINE_int32(gpu, 0, "GPU device id");
DEFINE_int32(repeat, 10, "repeat times");
DEFINE_string(config, "config/backend_example.yml", "config file");
DEFINE_string(model_db, "config/model_db.yml", "model meta information database");
DEFINE_string(image_dir, "", "Image directory");
DEFINE_int32(min_batch, 1, "Minimum batch size");
DEFINE_int32(max_batch, 256, "Maximum batch size");
DEFINE_string(framework, "", "Framework");
DEFINE_string(model, "", "Model name");
DEFINE_string(output, "", "Output file");
DEFINE_int32(height, 0, "Image height");
DEFINE_int32(width, 0, "Image width");

using namespace nexus;
using namespace nexus::backend;
using duration = std::chrono::milliseconds;
namespace fs = boost::filesystem;

namespace {
std::unordered_map<Framework, std::string> framework_rootdir;
std::unordered_map<ModelId, YAML::Node> model_info_table;
std::vector<std::string> test_images;
}

void LoadModelInfo(const std::string& config_file,
                   const std::string& db_file) {
  YAML::Node config = YAML::LoadFile(config_file);
  const YAML::Node& model_dir = config["model_dir"];
  for (auto it = model_dir.begin(); it != model_dir.end(); ++it) {
    Framework framework = get_Framework(it->first.as<std::string>());
    const std::string& path = it->second.as<std::string>();
    framework_rootdir.emplace(framework, path);
  }

  YAML::Node db = YAML::LoadFile(db_file);
  const YAML::Node& models = db["models"];
  for (uint i = 0; i < models.size(); ++i) {
    YAML::Node model_info = models[i];
    if (!model_info["framework"]) {
      LOG(FATAL) << "Missing framework in the model config";
    }
    if (!model_info["model_name"]) {
      LOG(FATAL) << "Missing model_name in the model config";
    }
    if (!model_info["type"]) {
      LOG(FATAL) << "Missing type in the model config";
    }
    Framework framework = get_Framework(
        model_info["framework"].as<std::string>());
    std::string model_name = model_info["model_name"].as<std::string>();
    if (!model_info["model_dir"]) {
      if (framework_rootdir.find(framework) == framework_rootdir.end()) {
        LOG(FATAL) << "Cannot find model root directory for framework " <<
            Framework_name(framework);
      }
      model_info["model_dir"] = framework_rootdir.at(framework);
    }
    ModelId model_id(framework, model_name);
    model_info_table[model_id] = model_info;
  }
}

void ListImages(const std::string& root_dir) {
  fs::directory_iterator end_iter;
  for (fs::directory_iterator it(root_dir); it != end_iter; ++it) {
    test_images.push_back(it->path().string());
  }
  LOG(INFO) << "Number of test images: " << test_images.size();
}

void ReadImage(const std::string& file_path, std::string* content) {
  std::ifstream fin(file_path, std::ios::binary | std::ios::ate);
  size_t fsize = fin.tellg();
  //LOG(INFO) << "Image file size: " << fsize << " B";
  fin.seekg(0);
  content->resize(fsize);
  fin.read(&((*content)[0]), fsize);
  fin.close();
}

template<class T>
std::pair<float, float> GetStats(const std::vector<T>& lats) {
  float mean = 0.;
  float std = 0.;
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

void Profile(ModelId model_id, int repeat) {
  LOG(INFO) << "Profile model " << model_id;
  // Init GPU and get GPU info
  NEXUS_CUDA_CHECK(cudaSetDevice(FLAGS_gpu));
  auto gpu_device = DeviceManager::Singleton().GetGPUDevice(FLAGS_gpu);
  cudaDeviceProp prop;
  NEXUS_CUDA_CHECK(cudaGetDeviceProperties(&prop, FLAGS_gpu));
  // Get model info and type
  BlockPriorityQueue<Task> task_queue;
  if (model_info_table.find(model_id) == model_info_table.end()) {
    LOG(FATAL) << "Model " << model_id << " is not contained in the model DB";
  }
  auto info = model_info_table.at(model_id);
  ModelType type = get_ModelType(info["type"].as<std::string>());
  // Prepare query
  ModelSession model_sess;
  model_sess.set_framework(model_id.first);
  model_sess.set_model_name(model_id.second);
  model_sess.set_version(1);
  model_sess.set_latency_sla(500);
  if (FLAGS_height > 0) {
    CHECK_GT(FLAGS_width, 0) << "Height and width must be set together";
    model_sess.set_image_height(FLAGS_height);
    model_sess.set_image_width(FLAGS_width);
  }
  std::string model_session_id = ModelSessionToString(model_sess);
  size_t origin_freemem = gpu_device->FreeMemory();
  std::vector<uint64_t> preprocess_lats;
  std::vector<uint64_t> postprocess_lats;
  std::unordered_map<int, std::tuple<float, float, size_t> > forward_stats;
  //std::shared_ptr<ModelInstance> model = nullptr;
  ModelInstanceDesc desc;
  desc.mutable_model_session()->CopyFrom(model_sess);
  
  for (int batch = FLAGS_min_batch; batch <= FLAGS_max_batch; ++batch) {
    desc.set_batch(batch);
    desc.set_max_batch(batch);
    auto model = CreateModelInstance(FLAGS_gpu, desc, info, task_queue);
    // latencies
    std::vector<uint64_t> forward_lats;
    for (int i = 0; i < batch * (repeat + 1); ++i) {
      // Load input image
      int idx = rand() % test_images.size();
      std::string im;
      ReadImage(test_images[idx], &im);
      auto task = std::make_shared<Task>();
      task->SetDeadline(std::chrono::milliseconds(100000));
      task->query.set_query_id(i);
      task->query.set_model_session_id(model_session_id);
      auto input = task->query.mutable_input();
      input->set_data_type(DT_IMAGE);
      auto image = input->mutable_image();
      image->set_data(im);
      image->set_format(ImageProto::JPEG);
      image->set_color(true);
      auto beg = std::chrono::high_resolution_clock::now();
      model->Preprocess(task);
      auto end = std::chrono::high_resolution_clock::now();
      preprocess_lats.push_back(
          std::chrono::duration_cast<duration>(end - beg).count());
    }
    // dry run
    model->Forward();
    // start meansuring forward latency
    for (int i = 0; i < repeat; ++i) {
      auto beg = std::chrono::high_resolution_clock::now();
      model->Forward();
      auto end = std::chrono::high_resolution_clock::now();
      forward_lats.push_back(
          std::chrono::duration_cast<duration>(end - beg).count());
    }
    size_t curr_freemem = gpu_device->FreeMemory();
    size_t memory_usage = origin_freemem - curr_freemem;
    LOG(INFO) << "memory usage: " << memory_usage;
    for (int i = 0; i < batch * (repeat + 1); ++i) {
      auto task = task_queue.pop();
      auto beg = std::chrono::high_resolution_clock::now();
      model->Postprocess(task);
      auto end = std::chrono::high_resolution_clock::now();
      postprocess_lats.push_back(
          std::chrono::duration_cast<duration>(end - beg).count());
    }
    float mean, std;
    std::tie(mean, std) = GetStats<uint64_t>(forward_lats);
    forward_stats.emplace(batch, std::make_tuple(mean, std, memory_usage));
    if (batch < FLAGS_min_batch) {
      return;
    }
  }
  // output to file
  std::ofstream fout(FLAGS_output, std::ofstream::out);
  fout << ModelSessionToString(model_sess, false) << "\n";
  fout << gpu_device->device_name() << "\n";
  fout << "Forward latency\n";
  fout << "batch,latency(ms),std(ms),memory(B)\n";
  for (int batch = FLAGS_min_batch; batch <= FLAGS_max_batch; ++batch) {
    float mean, std;
    size_t memory_usage;
    std::tie(mean, std, memory_usage) = forward_stats.at(batch);
    fout << batch << "," << mean << "," << std << "," << memory_usage << "\n";
  }
  float mean, std;
  std::tie(mean, std) = GetStats<uint64_t>(preprocess_lats);
  fout << "Preprocess latency\nmean(ms),std(ms)\n";
  fout << mean << "," << std << "\n";
  std::tie(mean, std) = GetStats<uint64_t>(postprocess_lats);
  fout << "Postprocess latency\nmean(ms),std(ms)\n";
  fout << mean << "," << std << "\n";
  fout.close();
}

int main(int argc, char** argv) {
  // log to stderr
  FLAGS_logtostderr = 1;
  // Init glog
  google::InitGoogleLogging(argv[0]);
  // Parse command line flags
  google::ParseCommandLineFlags(&argc, &argv, true);
  // Setup backtrace on segfault
  google::InstallFailureSignalHandler();

  if (FLAGS_framework.length() == 0) {
    LOG(FATAL) << "Missing framework";
  }
  if (FLAGS_model.length() == 0) {
    LOG(FATAL) << "Missing model";
  }
  if (FLAGS_image_dir.length() == 0) {
    LOG(FATAL) << "Missing image_dir";
  }
  if (FLAGS_output.length() == 0) {
    LOG(FATAL) << "Missing output";
  }
  srand(time(NULL));
  Framework framework = get_Framework(FLAGS_framework);
  ModelId model_id(framework, FLAGS_model);
  LoadModelInfo(FLAGS_config, FLAGS_model_db);
  ListImages(FLAGS_image_dir);
  Profile(model_id, FLAGS_repeat);
}
