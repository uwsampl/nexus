#include <boost/filesystem.hpp>
#include <fstream>

#include "nexus/common/model_db.h"
#include "nexus/common/model_def.h"
#include "nexus/common/util.h"

namespace fs = boost::filesystem;

namespace nexus {

ModelProfile::ModelProfile(const std::string& filepath) {
  LoadProfile(filepath);
}

void ModelProfile::LoadProfile(const std::string& filepath) {
  std::ifstream fin(filepath);
  CHECK(fin.good()) << "Profile file " << filepath << " doesn't exist";
  std::getline(fin, profile_id_);
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
    return 0;
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

uint32_t ModelProfile::GetMaxBatch(float latency_sla_ms) const {
  float latency_budget = latency_sla_ms * 1000 - network_latency_;
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

std::pair<uint32_t, float> ModelProfile::GetMaxThroughput(float latency_sla_ms)
    const {
  float max_throughput = 0;
  uint32_t best_batch = 0;
  // divide by 2 is becuase half of time will spend in batching
  float exec_budget = (latency_sla_ms * 1000 - network_latency_ -
                       GetPreprocessLatency() - GetPostprocessLatency()) * 0.5;
  for (uint32_t batch = 1; ; ++batch) {
    float forward_lat = GetForwardLatency(batch);
    if (forward_lat < 0 || forward_lat > exec_budget) {
      break;
    }
    float tp = batch * 1e6 / forward_lat;
    if (tp > max_throughput) {
      max_throughput = tp;
      best_batch = batch;
    }
  }
  return {best_batch, max_throughput};
}

ModelDatabase& ModelDatabase::Singleton() {
  static ModelDatabase model_db_;
  return model_db_;
}

void ModelDatabase::Init(const std::string& db_root_dir) {
  db_root_dir_ = db_root_dir;
  fs::path db_dir(db_root_dir);
  CHECK(fs::is_directory(db_dir)) << "Database root directory " <<
      db_dir << " doesn't exist";
  // Check model store directory exists
  fs::path model_store_dir = db_dir / "store";
  CHECK(fs::is_directory(model_store_dir)) << "Model store directory " <<
      model_store_dir << " doesn't exist";
  model_store_dir_ = model_store_dir.string();
  // Load model DB file
  fs::path db_file = db_dir / "db" / "model_db.yml";
  CHECK(fs::exists(db_file)) << "Model DB file " << db_file << " doesn't exist";
  LoadModelInfo(db_file.string());
  // Load model profiles
  fs::path profile_dir = db_dir / "profiles";
  CHECK(fs::is_directory(profile_dir)) << "Model profile directory " <<
      profile_dir << " doesn't exist";
  LoadModelProfiles(profile_dir.string());
}

const YAML::Node& ModelDatabase::GetModelInfo(const std::string& model_id)
    const {
  auto itr = model_info_table_.find(model_id);
  if (itr == model_info_table_.end()) {
    LOG(FATAL) << "Cannot find model info for " << model_id;
  }
  return itr->second;
}

const YAML::Node& ModelDatabase::GetModelInfo(
    const std::string& framework, const std::string& model_name,
    uint32_t version) const {
  auto model_id = ModelID(framework, model_name, version);
  auto itr = model_info_table_.find(model_id);
  if (itr == model_info_table_.end()) {
    LOG(FATAL) << "Cannot find model info for " << model_id;
  }
  return itr->second;
}

const ModelProfile* ModelDatabase::GetModelProfile(
    const std::string& gpu_device, const std::string& profile_id) const {
  auto itr = device_profile_table_.find(gpu_device);
  if (itr == device_profile_table_.end()) {
    LOG(ERROR) << "Cannot find model profile for GPU " << gpu_device;
    return nullptr;
  }
  auto& profile_table = itr->second;
  auto itr2 = profile_table.find(profile_id);
  if (itr2 == profile_table.end()) {
    LOG(ERROR) << "Cannot find model profile " << profile_id << " on " <<
        gpu_device;
    return nullptr;
  }
  return &itr2->second;
}

float ModelDatabase::GetModelForwardLatency(const std::string& gpu_device,
                                            const std::string& profile_id,
                                            uint32_t batch) const {
  auto profile = GetModelProfile(gpu_device, profile_id);
  if (profile == nullptr) {
    return 1.;
  }
  return profile->GetForwardLatency(batch);
}

size_t ModelDatabase::GetModelMemoryUsage(const std::string& gpu_device,
                                          const std::string& profile_id,
                                          uint32_t batch) const {
  auto profile = GetModelProfile(gpu_device, profile_id);
  if (profile == nullptr) {
    return 0;
  }
  return profile->GetMemoryUsage(batch);
}

void ModelDatabase::LoadModelInfo(const std::string& db_file) {
  LOG(INFO) << "Load model DB from " << db_file;
  YAML::Node db = YAML::LoadFile(db_file);
  const YAML::Node& models = db["models"];
  for (uint i = 0; i < models.size(); ++i) {
    YAML::Node model_info = models[i];
    if (!model_info["framework"]) {
      LOG(FATAL) << "Missing framework in the model DB";
    }
    if (!model_info["model_name"]) {
      LOG(FATAL) << "Missing model_name in the model DB";
    }
    if (!model_info["version"]) {
      LOG(FATAL) << "Missing version in the model DB";
    }
    if (!model_info["type"]) {
      LOG(FATAL) << "Missing type in the model DB";
    }
    model_info["model_dir"] = model_store_dir_;
    std::string framework = model_info["framework"].as<std::string>();
    std::string model_name = model_info["model_name"].as<std::string>();
    uint32_t version = model_info["version"].as<uint32_t>();
    std::string model_id = ModelID(framework, model_name, version);
    model_info_table_[model_id] = model_info;
  }
}

void ModelDatabase::LoadModelProfiles(const std::string& profile_dir) { 
  fs::path root_dir(profile_dir);
  fs::directory_iterator end_iter;
  for (fs::directory_iterator dir_itr(root_dir); dir_itr != end_iter;
       ++dir_itr) {
    auto path = dir_itr->path();
    if (!fs::is_directory(path)) {
      continue;
    }
    LOG(INFO) << "Load model profiles for GPU " << path.filename().string();
    for (fs::directory_iterator file_itr(path); file_itr != end_iter;
         ++file_itr) {
      LOG(INFO) << "- Load model profile " << file_itr->path().string();
      ModelProfile profile(file_itr->path().string());
      auto device = profile.gpu_device_name();
      if (device_profile_table_.find(device) == device_profile_table_.end()) {
        device_profile_table_.emplace(device, ProfileTable());
      }
      device_profile_table_.at(device).emplace(profile.profile_id(), profile);
    }
  }
}

} // namespace nexus
