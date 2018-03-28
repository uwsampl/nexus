#include <boost/filesystem.hpp>
#include <fstream>

#include "common/model_profile.h"
#include "common/util.h"

namespace fs = boost::filesystem;

namespace nexus {

ModelProfile::ModelProfile(const std::string& filepath) {
  LoadProfile(filepath);
}

void ModelProfile::LoadProfile(const std::string& filepath) {
  std::ifstream fin(filepath);
  CHECK(fin.good()) << "Profile file " << filepath << " doesn't exist";
  std::getline(fin, model_id_);
  // TODO: add version 1 for now, should udpate it
  model_id_ += ":1";
  std::getline(fin, gpu_device_name_);
  std::string line;
  std::vector<std::string> tokens;
  std::getline(fin, line);
  std::getline(fin, line);
  while (true) {
    std::getline(fin, line);
    if (line.find(",") == std::string::npos) {
      break;
    }
    SplitString(line, ',', &tokens);
    ProfileEntry entry;
    uint32_t batch = stoi(tokens[0]);
    entry.latency_mean = stof(tokens[1]);
    entry.latency_std = stof(tokens[2]);
    entry.memory_usage = stoll(tokens[3]);
    forward_lats_.emplace(batch, entry);
  }
  std::getline(fin, line);
  std::getline(fin, line);
  SplitString(line, ',', &tokens);
  preprocess_.latency_mean = stof(tokens[0]);
  preprocess_.latency_std = stof(tokens[1]);
  std::getline(fin, line);
  std::getline(fin, line);
  std::getline(fin, line);
  SplitString(line, ',', &tokens);
  postprocess_.latency_mean = stof(tokens[0]);
  postprocess_.latency_std = stof(tokens[1]);
}

float ModelProfile::GetForwardLatency(uint32_t batch) const {
  if (forward_lats_.find(batch) == forward_lats_.end()) {
    return -1;
  }
  auto entry = forward_lats_.at(batch);
  return entry.latency_mean + entry.latency_std;
}

float ModelProfile::GetPreprocessLatency() const {
  return preprocess_.latency_mean + preprocess_.latency_std;
}

float ModelProfile::GetPostprocessLatency() const {
  return postprocess_.latency_mean + postprocess_.latency_std;
}

size_t ModelProfile::GetMemoryUsage(uint32_t batch) const {
  if (forward_lats_.find(batch) == forward_lats_.end()) {
    return 0;
  }
  return forward_lats_.at(batch).memory_usage;
}

uint32_t ModelProfile::GetMaxBatch(float latency_sla) const {
  float latency_budget = latency_sla - network_latency_;
  latency_budget -= GetPreprocessLatency();
  latency_budget -= GetPostprocessLatency();
  // divide by 2 is because half of time will spend in batching
  latency_budget /= 2;
  uint32_t batch = 1;
  while (true) {
    if (forward_lats_.find(batch) == forward_lats_.end()) {
      break;
    }
    auto entry = forward_lats_.at(batch);
    if (entry.latency_mean + entry.latency_std > latency_budget) {
      break;
    }
    ++batch;
  }
  --batch;
  batch = (batch == 0) ? 1 : batch;
  return batch;
}

std::pair<uint32_t, float> ModelProfile::GetMaxThroughput(float latency_sla)
    const {
  float max_throughput = 0;
  uint32_t best_batch = 0;
  // divide by 2 is becuase half of time will spend in batching
  float exec_budget = (latency_sla - network_latency_ - GetPreprocessLatency() -
                       GetPostprocessLatency()) * 0.5;
  for (uint32_t batch = 1; ; ++batch) {
    float forward_lat = GetForwardLatency(batch);
    if (forward_lat < 0 || forward_lat > exec_budget) {
      break;
    }
    float tp = batch * 1000 / forward_lat;
    if (tp > max_throughput) {
      max_throughput = tp;
      best_batch = batch;
    }
  }
  return {best_batch, max_throughput};
}

void ModelProfileTable::Init(const std::string& profile_dir) {
  fs::path root_dir(profile_dir);
  CHECK(fs::exists(root_dir)) << "Model profile dir " << profile_dir <<
      "doesn't exist";
  CHECK(fs::is_directory(root_dir)) << profile_dir << " is not a directory";
  fs::directory_iterator end_iter;
  for (fs::directory_iterator it(root_dir); it != end_iter; ++it) {
    auto path = it->path();
    if (!fs::is_directory(path)) {
      continue;
    }
    LOG(INFO) << "Load model profiles for GPU " << path.filename().string();
    for (fs::directory_iterator it2(path); it2 != end_iter; ++it2) {
      LOG(INFO) << "- Load model profile " << it2->path().string();
      ModelProfile profile(it2->path().string());
      model_profiles_.emplace(profile.key(), profile);
    }
  }
}

const ModelProfile* ModelProfileTable::GetModelProfile(
    const std::string& gpu_device, const std::string& model_id) const {
  auto profile_itr = model_profiles_.find({gpu_device, model_id});
  if (profile_itr == model_profiles_.end()) {
    LOG(ERROR) << "Cannot find model profile for " << model_id;
    return nullptr;
  }
  return &profile_itr->second;
}

float ModelProfileTable::GetModelForwardLatency(const std::string& gpu_device,
                                                const std::string& model_id,
                                                uint32_t batch) const {
  auto profile_itr = model_profiles_.find({gpu_device, model_id});
  if (profile_itr == model_profiles_.end()) {
    LOG(ERROR) << "Cannot find model profile for " << model_id;
    return 1.;
  }
  return profile_itr->second.GetForwardLatency(batch);
}

size_t ModelProfileTable::GetModelMemoryUsage(const std::string& gpu_device,
                                              const std::string& model_id,
                                              uint32_t batch) const {
  auto profile_itr = model_profiles_.find({gpu_device, model_id});
  if (profile_itr == model_profiles_.end()) {
    LOG(ERROR) << "Cannot find model profile for " << model_id;
    return 0;
  }
  return profile_itr->second.GetMemoryUsage(batch);
}


} // namespace nexus
