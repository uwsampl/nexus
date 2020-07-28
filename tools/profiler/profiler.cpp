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
DEFINE_int32(min_batch, 1, "Minimum batch size");
DEFINE_int32(max_batch, 256, "Maximum batch size");
DEFINE_string(output, "", "Output file");
DEFINE_int32(height, 0, "Image height");
DEFINE_int32(width, 0, "Image width");
DEFINE_bool(share_prefix, false, "Enable share prefix");
DEFINE_int32(repeat, 10, "Repeat times for profiling");

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

  void Profile(int min_batch, int max_batch, const std::string output="",
               int repeat=10) {
    std::vector<uint64_t> preprocess_lats;
    std::vector<uint64_t> postprocess_lats;
    std::unordered_map<int, std::tuple<float, float, uint64_t> > forward_stats;
    ModelInstanceConfig config;
    config.add_model_session()->CopyFrom(model_sess_);
    if (FLAGS_share_prefix) {
      std::vector<std::string> share_models =
          ModelDatabase::Singleton().GetPrefixShareModels(
              ModelSessionToModelID(model_sess_));
      for (auto model_id : share_models) {
        LOG(INFO) << model_id;
        auto share_sess = config.add_model_session();
        ParseModelID(model_id, share_sess);
        share_sess->set_latency_sla(50000);
        if (model_sess_.image_height() > 0) {
          share_sess->set_image_height(model_sess_.image_height());
          share_sess->set_image_width(model_sess_.image_width());
        }
        model_sessions_.push_back(ModelSessionToString(*share_sess));
      }
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
      int num_inputs = max_batch * (repeat + 1);
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
        if (i > 0) {
          preprocess_lats.push_back(
              std::chrono::duration_cast<duration>(end - beg).count());
        }
      }
    }
    std::this_thread::sleep_for(std::chrono::microseconds(200));
    LOG(INFO) << "Preprocess finished";

    // output to file
    std::ostream* fout;
    if (output.length() == 0) {
      fout = &std::cout;
    } else {
      fout = new std::ofstream(output, std::ofstream::out);
    }

    if (FLAGS_share_prefix) {
      *fout << ModelSessionToProfileID(model_sess_) << "-prefix\n";
    } else {
      *fout << ModelSessionToProfileID(model_sess_) << "\n";
    }
    *fout << gpu_device_->device_name() << "\n";
    *fout << gpu_device_->uuid() << "\n";
    *fout << "Forward latency\n";
    *fout << "batch,latency(us),std(us),memory(B),repeat\n";
    *fout << std::flush;

    // forward and postprocess
    int dryrun = 4;
    for (int batch = min_batch; batch <= max_batch; ++batch) {
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
      auto memory_usage = model->GetPeakMemoryUsage();
      CHECK_GE(memory_usage, 0) << "get_peak_memory_usage not implemented";
      LOG(INFO) << "memory usage: " << memory_usage;
      for (int i = 0; i < batch * (repeat + dryrun); ++i) {
        auto task = task_queue.pop();
        CHECK_EQ(task->result.status(), CTRL_OK) << "Error detected: " <<
            task->result.status();
        auto beg = std::chrono::high_resolution_clock::now();
        model->Postprocess(task);
        auto end = std::chrono::high_resolution_clock::now();
        if (i > 0 && postprocess_lats.size() < 2000) {
          postprocess_lats.push_back(
              std::chrono::duration_cast<duration>(end - beg).count());
        }
      }
      float mean, std;
      std::tie(mean, std) = GetStats<uint64_t>(forward_lats);
      forward_stats.emplace(batch, std::make_tuple(mean, std, memory_usage));
      CHECK_EQ(task_queue.size(), 0) << "Task queue is not empty";

      // output to file
      *fout << batch << "," << mean << "," << std << "," << memory_usage << "," << repeat << std::endl;

      std::this_thread::sleep_for(std::chrono::microseconds(200));
    }

    LOG(INFO) << "Final free memory: " << gpu_device_->FreeMemory();
    preproc_tasks.clear();

    // output to file
    float mean, std;
    *fout << "Preprocess latency (mean,std,repeat)\n";
    std::tie(mean, std) = GetStats<uint64_t>(preprocess_lats);
    *fout << mean << "," << std << "," << preprocess_lats.size() << "\n";
    *fout << "Postprocess latency (mean,std,repeat)\n";
    std::tie(mean, std) = GetStats<uint64_t>(postprocess_lats);
    *fout << mean << "," << std << "," << postprocess_lats.size() << "\n";
    *fout << std::flush;
    if (fout != &std::cout) {
      delete fout;
    }
  }

 private:
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
  if (FLAGS_framework == "tf_share" && FLAGS_share_prefix)
    LOG(FATAL) << "Cannot use --share_prefix on TFShare models";
  srand(time(NULL));
  ModelProfiler profiler(FLAGS_gpu, FLAGS_framework, FLAGS_model,
                         FLAGS_model_version, FLAGS_image_dir, FLAGS_height,
                         FLAGS_width);
  profiler.Profile(FLAGS_min_batch, FLAGS_max_batch, FLAGS_output,
                   FLAGS_repeat);
}
