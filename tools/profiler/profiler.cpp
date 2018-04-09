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
//#include "nexus/common/model_def.h"
#include "nexus/backend/model_ins.h"
#include "nexus/proto/nnquery.pb.h"

DEFINE_int32(gpu, 0, "GPU device id");
DEFINE_string(framework, "", "Framework");
DEFINE_string(model, "", "Model name");
DEFINE_int32(model_version, 1, "Version");
DEFINE_string(model_root, "", "Model root directory");
DEFINE_string(image_dir, "", "Image directory");
DEFINE_int32(min_batch, 1, "Minimum batch size");
DEFINE_int32(max_batch, 256, "Maximum batch size");
DEFINE_string(output, "", "Output file");
DEFINE_int32(height, 0, "Image height");
DEFINE_int32(width, 0, "Image width");
DEFINE_int32(repeat, 10, "Repeat times for profiling");

using namespace nexus;
using namespace nexus::backend;
using duration = std::chrono::microseconds;
namespace fs = boost::filesystem;

namespace {
std::vector<std::string> test_images;
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

void Profile() {
  // Init GPU and get GPU info
  NEXUS_CUDA_CHECK(cudaSetDevice(FLAGS_gpu));
  auto gpu_device = DeviceManager::Singleton().GetGPUDevice(FLAGS_gpu);
  cudaDeviceProp prop;
  NEXUS_CUDA_CHECK(cudaGetDeviceProperties(&prop, FLAGS_gpu));
  // Get model info
  auto info = ModelDatabase::Singleton().GetModelInfo(
      FLAGS_framework, FLAGS_model, FLAGS_model_version);
  // Prepare query
  ModelSession model_sess;
  model_sess.set_framework(FLAGS_framework);
  model_sess.set_model_name(FLAGS_model);
  model_sess.set_version(FLAGS_model_version);
  model_sess.set_latency_sla(50000);
  if (FLAGS_height > 0) {
    CHECK_GT(FLAGS_width, 0) << "Height and width must be set together";
    model_sess.set_image_height(FLAGS_height);
    model_sess.set_image_width(FLAGS_width);
  } else {
    if (info["resizable"] && info["resizable"].as<bool>()) {
      // Set default image size for resizable CNN
      model_sess.set_image_height(info["image_height"].as<uint32_t>());
      model_sess.set_image_width(info["image_width"].as<uint32_t>());
    }
  }
  LOG(INFO) << "Profile model " << ModelSessionToProfileID(model_sess);
  
  std::string model_session_id = ModelSessionToString(model_sess);
  size_t origin_freemem = gpu_device->FreeMemory();
  std::vector<uint64_t> preprocess_lats;
  std::vector<uint64_t> postprocess_lats;
  std::unordered_map<int, std::tuple<float, float, size_t> > forward_stats;
  ModelInstanceDesc desc;
  desc.mutable_model_session()->CopyFrom(model_sess);
  BlockPriorityQueue<Task> task_queue;
  
  for (int batch = FLAGS_min_batch; batch <= FLAGS_max_batch; ++batch) {
    desc.set_batch(batch);
    desc.set_max_batch(batch);
    auto model = CreateModelInstance(FLAGS_gpu, desc, info, task_queue);
    // latencies
    std::vector<uint64_t> forward_lats;
    for (int i = 0; i < batch * (FLAGS_repeat + 1); ++i) {
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
      if (i > 0) {
        preprocess_lats.push_back(
            std::chrono::duration_cast<duration>(end - beg).count());
      }
    }
    // dry run
    model->Forward();
    // start meansuring forward latency
    for (int i = 0; i < FLAGS_repeat; ++i) {
      auto beg = std::chrono::high_resolution_clock::now();
      model->Forward();
      auto end = std::chrono::high_resolution_clock::now();
      forward_lats.push_back(
          std::chrono::duration_cast<duration>(end - beg).count());
    }
    size_t curr_freemem = gpu_device->FreeMemory();
    size_t memory_usage = origin_freemem - curr_freemem;
    LOG(INFO) << "memory usage: " << memory_usage;
    for (int i = 0; i < batch * (FLAGS_repeat + 1); ++i) {
      auto task = task_queue.pop();
      auto beg = std::chrono::high_resolution_clock::now();
      model->Postprocess(task);
      auto end = std::chrono::high_resolution_clock::now();
      if (i > 0) {
        postprocess_lats.push_back(
            std::chrono::duration_cast<duration>(end - beg).count());
      }
    }
    float mean, std;
    std::tie(mean, std) = GetStats<uint64_t>(forward_lats);
    forward_stats.emplace(batch, std::make_tuple(mean, std, memory_usage));
    if (batch < FLAGS_min_batch) {
      return;
    }
  }
  // output to file
  std::ostream* fout;
  if (FLAGS_output.length() == 0) {
    fout = &std::cout;
  } else {
    fout = new std::ofstream(FLAGS_output, std::ofstream::out);
  }
  
  *fout << ModelSessionToProfileID(model_sess) << "\n";
  *fout << gpu_device->device_name() << "\n";
  *fout << "Forward latency\n";
  *fout << "batch,latency(us),std(us),memory(B)\n";
  for (int batch = FLAGS_min_batch; batch <= FLAGS_max_batch; ++batch) {
    float mean, std;
    size_t memory_usage;
    std::tie(mean, std, memory_usage) = forward_stats.at(batch);
    *fout << batch << "," << mean << "," << std << "," << memory_usage << "\n";
  }
  float mean, std;
  std::tie(mean, std) = GetStats<uint64_t>(preprocess_lats);
  *fout << "Preprocess latency\nmean(us),std(us)\n";
  *fout << mean << "," << std << "\n";
  std::tie(mean, std) = GetStats<uint64_t>(postprocess_lats);
  *fout << "Postprocess latency\nmean(us),std(us)\n";
  *fout << mean << "," << std << "\n";
  if (fout != &std::cout) {
    delete fout;
  }
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
  if (FLAGS_model_root.length() == 0) {
    LOG(FATAL) << "Missing model_root";
  }
  if (FLAGS_framework.length() == 0) {
    LOG(FATAL) << "Missing framework";
  }
  if (FLAGS_model.length() == 0) {
    LOG(FATAL) << "Missing model";
  }
  if (FLAGS_image_dir.length() == 0) {
    LOG(FATAL) << "Missing image_dir";
  }
  srand(time(NULL));
  ModelDatabase::Singleton().Init(FLAGS_model_root);
  ListImages(FLAGS_image_dir);
  Profile();
}
