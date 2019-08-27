#include <cmath>
#include <fstream>
#include <queue>
#include <boost/filesystem.hpp>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include "nexus/common/model_db.h"
#include "nexus/common/model_def.h"
#include "nexus/common/util.h"

std::pair<double, double> MergeMeanStd(double mean1, double std1, int n1,
                                       double mean2, double std2, int n2) {
  double mean = ((n1 - 1) * mean1 + (n2 - 1) * mean2) / (n1 + n2 - 1);
  double var = ((n1 - 1) * std1 * std1 + (n2 - 1) * std2 * std2
      + n1 * (mean1 - mean) * (mean1 - mean)
      + n2 * (mean2 - mean) * (mean2 - mean)) / (n1 + n2 - 1);
  return {mean, std::sqrt(var)};
}

namespace fs = boost::filesystem;

DEFINE_string(model_root, "", "Model root dicrectory");
DEFINE_double(profile_multiplier, 1.15, "Multiplier to forward latency in profile.");

namespace nexus {

void MergeMeanStd(ProfileEntry& dst, const ProfileEntry& src) {
  double mean, std;
  std::tie(mean, std) = ::MergeMeanStd(
      dst.latency_mean, dst.latency_std, dst.repeat,
      src.latency_mean, src.latency_std, src.repeat);
  dst.latency_mean = mean;
  dst.latency_std = std;
  dst.repeat += src.repeat;
}

ModelProfile::ModelProfile(const std::string& filepath) {
  LoadProfile(filepath);
}

void ModelProfile::MergeProfile(const ModelProfile& rhs) {
  uint32_t batch = 1;
  while (batch <= forward_lats_.size() && batch <= rhs.forward_lats_.size()) {
    auto &rec1 = forward_lats_.at(batch);
    const auto &rec2 = rhs.forward_lats_.at(batch);
    MergeMeanStd(rec1, rec2);
    ++batch;
  }
  while (batch <= rhs.forward_lats_.size()) {
    forward_lats_[batch] = rhs.forward_lats_.at(batch);
    ++batch;
  }
  MergeMeanStd(preprocess_, rhs.preprocess_);
  MergeMeanStd(postprocess_, rhs.postprocess_);
}

void ModelProfile::LoadProfile(const std::string& filepath) {
  std::ifstream fin(filepath);
  CHECK(fin.good()) << "Profile file " << filepath << " doesn't exist";
  std::getline(fin, profile_id_);
  std::getline(fin, gpu_device_name_);
  std::string line;
  std::vector<std::string> tokens;
  std::getline(fin, gpu_uuid_);
  std::getline(fin, line); // Forward latency
  std::getline(fin, line); // batch,latency(us),std(us),memory(B),repeat
  while (true) {
    std::getline(fin, line);
    if (line.find("Preprocess latency (mean,std,repeat)") == 0)
      break;
    SplitString(line, ',', &tokens);
    ProfileEntry entry;
    uint32_t batch = stoi(tokens[0]);
    entry.latency_mean = stof(tokens[1]) * FLAGS_profile_multiplier;
    entry.latency_std = stof(tokens[2]) * FLAGS_profile_multiplier;
    entry.memory_usage = stoll(tokens[3]);
    entry.repeat = std::stoi(tokens[4]);
    forward_lats_.emplace(batch, entry);
  }
  std::getline(fin, line);
  SplitString(line, ',', &tokens);
  preprocess_.latency_mean = stof(tokens[0]) * FLAGS_profile_multiplier;
  preprocess_.latency_std = stof(tokens[1]) * FLAGS_profile_multiplier;
  preprocess_.repeat = std::stoi(tokens[2]);
  std::getline(fin, line); // Postprocess latency (mean,std,repeat)
  std::getline(fin, line);
  SplitString(line, ',', &tokens);
  postprocess_.latency_mean = stof(tokens[0]) * FLAGS_profile_multiplier;
  postprocess_.latency_std = stof(tokens[1]) * FLAGS_profile_multiplier;
  postprocess_.repeat = std::stoi(tokens[2]);
}

float ModelProfile::GetForwardLatency(uint32_t batch) const {
  if (forward_lats_.find(batch) == forward_lats_.end()) {
    LOG(FATAL) << "Cannot find forward latency: model=" << profile_id() << " batch=" << batch;
    return 0.;
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
  float latency_budget = latency_sla_ms * 1000 - network_latency_us_;
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
  float exec_budget = (latency_sla_ms * 1000 - network_latency_us_ -
                       GetPreprocessLatency() - GetPostprocessLatency()) * 0.5;
  for (uint32_t batch = 1; batch <= forward_lats_.size() ; ++batch) {
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
  CHECK_GT(FLAGS_model_root.length(), 0) << "Missing model_root";
  static ModelDatabase model_db_(FLAGS_model_root);
  return model_db_;
}

const YAML::Node* ModelDatabase::GetModelInfo(const std::string& model_id)
    const {
  auto itr = model_info_table_.find(model_id);
  if (itr == model_info_table_.end()) {
    LOG(ERROR) << "Cannot find model info for " << model_id;
    return nullptr;
  }
  return &itr->second;
}

const YAML::Node* ModelDatabase::GetModelInfo(
    const std::string& framework, const std::string& model_name,
    uint32_t version) const {
  auto model_id = ModelID(framework, model_name, version);
  auto itr = model_info_table_.find(model_id);
  if (itr == model_info_table_.end()) {
    LOG(ERROR) << "Cannot find model info for " << model_id;
    return nullptr;
  }
  return &itr->second;
}

const ModelProfile* ModelDatabase::GetModelProfile(
    const std::string& gpu_device,
    const std::string& gpu_uuid,
    const std::string& profile_id) const {
  auto itr = device_profile_table_.find(gpu_device);
  if (itr == device_profile_table_.end()) {
    LOG(ERROR) << "Cannot find model profile for GPU " << gpu_device;
    return nullptr;
  }
  auto& profile_table = itr->second;
  auto key = profile_id + ":" + gpu_uuid;
  auto itr2 = profile_table.find(key);
  if (itr2 != profile_table.end())
    return &itr2->second;

  std::vector<std::string> tokens;
  SplitString(profile_id, ':', &tokens);
  const auto &model = tokens[1];
  auto pos = model.rfind('_');
  if (pos == std::string::npos) {
    LOG(ERROR) << "Cannot find model profile " << key << " on " << gpu_device;
    return nullptr;
  }
  auto mirror_model = model.substr(0, pos) + "_0";
  auto mirror_profile_id = tokens[0] + ":" + mirror_model;
  for (size_t i = 2; i < tokens.size(); ++i) {
    mirror_profile_id += ':';
    mirror_profile_id += tokens[i];
  }
  auto key3 = mirror_profile_id + ':' + gpu_uuid;
  auto itr3 = profile_table.find(key3);
  if (itr3 != profile_table.end())
    return &itr3->second;
  LOG(ERROR) << "Cannot find model profile " << key
             << " or " << key3 << " on " << gpu_device;
  return nullptr;
}

std::shared_ptr<TFShareInfo> ModelDatabase::GetTFShareInfo(const std::string& model_name) const {
  auto iter = tf_share_models_.find(model_name);
  if (iter != tf_share_models_.end())
    return iter->second;
  return nullptr;
}

int ModelDatabase::GetSharePrefixLength(const std::string& model_id1,
                                        const std::string& model_id2) const {
  auto iter = share_prefix_models_.find(model_id1);
  if (iter == share_prefix_models_.end()) {
    return 0;
  }
  auto const& shares = iter->second;
  if (shares.find(model_id2) == shares.end()) {
    return 0;
  }
  return shares.at(model_id2);
}

std::vector<std::string> ModelDatabase::GetPrefixShareModels(
    const std::string& model_id) const {
  auto find = share_prefix_models_.find(model_id);
  if (find == share_prefix_models_.end()) {
    return {};
  }
  auto const& shares = find->second;
  std::vector<std::string> models;
  for (auto iter : shares) {
    models.push_back(iter.first);
  }
  return models;
}

ModelDatabase::ModelDatabase(const std::string& db_root_dir) {
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

void ModelDatabase::LoadModelInfo(const std::string& db_file) {
  VLOG(1) << "Load model DB from " << db_file;
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

  const YAML::Node& shares = db["share_prefix"];
  for (uint i = 0; i < shares.size(); ++i) {
    auto const& share = shares[i];
    int prefix_length = share["prefix_length"].as<int>();
    VLOG(1) << "prefix length: " << prefix_length;
    std::vector<std::string> share_models;
    for (uint j = 0; j < share["models"].size(); ++j) {
      auto const& model = share["models"][j];
      std::string model_id = ModelID(model["framework"].as<std::string>(),
                                     model["model_name"].as<std::string>(),
                                     model["version"].as<int>());
      share_models.push_back(model_id);
      if (share_prefix_models_.find(model_id) == share_prefix_models_.end()) {
        share_prefix_models_.emplace(model_id, PrefixMap());
      }
      VLOG(1) << " - " << model_id;
    }
    for (uint j = 0; j < share_models.size(); ++j) {
      for (uint k = j + 1; k < share_models.size(); ++k) {
        share_prefix_models_[share_models[j]].emplace(share_models[k],
                                                      prefix_length);
        share_prefix_models_[share_models[k]].emplace(share_models[j],
                                                      prefix_length);
      }
    }
  }

  const YAML::Node& tf_share = db["tf_share"];
  for (uint i = 0; i < tf_share.size(); ++i) {
    const auto& node = tf_share[i];
    auto info = std::make_shared<TFShareInfo>(node);
    std::vector<std::string> output_layers(info->suffix_models.size());
    for (const auto &suffix_model : info->suffix_models) {
      const auto &name = suffix_model.first;
      CHECK(tf_share_models_.count(name) == 0) << "Duplicated model " << name;
      tf_share_models_[name] = info;
      CHECK(model_info_table_.count(name) == 0) << "Duplicated model " << name;
      output_layers[suffix_model.second.suffix_index] = suffix_model.second.output_layer;

      // FIXME: hack for the ModelInstance constructor
      YAML::Node model_info;
      model_info["model_dir"] = model_store_dir_;
      std::string model_id = ModelID("tf_share", name, 1);
      model_info_table_[model_id] = model_info;
    }

    // TODO refactor ModelInstance constructor so that it doesn't look up the ModelDB Singleton
    YAML::Node model_info = node;
    model_info["framework"] = "tensorflow";
    model_info["model_name"] = info->hack_internal_id;
    model_info["version"] = 1;
    model_info["type"] = "classification";  // FIXME
    model_info["model_dir"] = model_store_dir_;
    model_info["output_layer"] = output_layers;
    std::string model_id = ModelID("tensorflow", info->hack_internal_id, 1);
    model_info_table_[model_id] = model_info;
  }
}

void ModelDatabase::LoadModelProfiles(const std::string& profile_dir) {
  std::vector<fs::path> files;
  fs::directory_iterator end_iter;

  std::queue<fs::path> dirs;
  dirs.emplace(profile_dir);
  while (!dirs.empty()) {
    auto dir = dirs.front();
    dirs.pop();
    for (fs::directory_iterator dir_itr(dir); dir_itr != end_iter; ++dir_itr) {
      auto path = dir_itr->path();
      if (dir_itr->path().filename().string()[0] == '.') {
        continue;
      }
      if (fs::is_directory(path)) {
        dirs.emplace(path);
      } else {
        files.emplace_back(path);
      }
    }
  }

  for (const auto &filepath : files) {
    ModelProfile profile(filepath.string());
    auto device = profile.gpu_device_name();
    if (device_profile_table_.find(device) == device_profile_table_.end()) {
      device_profile_table_.emplace(device, ProfileTable());
    }
    auto &table = device_profile_table_[device];

    auto key_generic = profile.profile_id() + ":generic";
    auto it = table.find(key_generic);
    if (it == table.end()) {
      table[key_generic] = profile;
    } else {
      it->second.MergeProfile(profile);
    }

    if (profile.gpu_uuid() != "generic") {
      auto key = profile.profile_id() + ":" + profile.gpu_uuid();
      table[key] = profile;
    }
  }
}

TFShareSuffixInfo::TFShareSuffixInfo(size_t suffix_index_, const YAML::Node &node) :
    suffix_index(suffix_index_),
    model_name(node["model_name"].as<std::string>()),
    output_layer(node["output_layer"].as<std::string>()),
    type(node["type"].as<std::string>()),
    class_names(node["class_names"].as<std::string>()) {
}

TFShareInfo::TFShareInfo(const YAML::Node &node) :
    model_file(node["model_file"].as<std::string>()),
    input_layer(node["input_layer"].as<std::string>()),
    slice_beg_vector(node["slice_beg_vector"].as<std::string>()),
    slice_len_vector(node["slice_len_vector"].as<std::string>()),
    image_height(node["image_height"].as<int>()),
    image_width(node["image_width"].as<int>()) {
  hack_internal_id = "tf_share";
  const auto& models = node["suffix_models"];
  for (size_t i = 0; i < models.size(); ++i) {
    TFShareSuffixInfo suffix(i, models[i]);
    CHECK(suffix_models.count(suffix.model_name) == 0) << "Duplicated model_name " << suffix.model_name;
    suffix_models.emplace(suffix.model_name, suffix);
    hack_internal_id += '|';
    hack_internal_id += suffix.model_name;
  }
}


} // namespace nexus
