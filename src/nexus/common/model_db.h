#ifndef NEXUS_COMMON_MODEL_DB_H_
#define NEXUS_COMMON_MODEL_DB_H_

#include <memory>
#include <unordered_map>
#include <yaml-cpp/yaml.h>

namespace nexus {

struct ProfileEntry {
  // latency in us unit
  float latency_mean;
  float latency_std;
  size_t memory_usage;
};

class ModelProfile {
 public:
  ModelProfile() {}

  ModelProfile(const std::string& file_path);

  void LoadProfile(const std::string& file_path);

  std::string profile_id() const { return profile_id_; }

  std::string gpu_device_name() const { return gpu_device_name_; }

  float GetForwardLatency(uint32_t batch) const;

  float GetPreprocessLatency() const;

  float GetPostprocessLatency() const;

  size_t GetMemoryUsage(uint32_t batch) const;
  /*!
   * \brief Computes the maximum batch size to use within latency_sla
   * \param latency_sla Latency SLA in ms
   * \return max batch size
   */
  uint32_t GetMaxBatch(float latency_sla_ms) const;
  /*!
   * \brief Computes the maximum throughput can be achieved within latency_sla
   * \param latency_sla Latency SLA in ms
   * \return pair of best batch size and max throughput
   */
  std::pair<uint32_t, float> GetMaxThroughput(float latency_sla_ms) const;

 private:
  std::string profile_id_;
  std::string gpu_device_name_;
  std::unordered_map<uint32_t, ProfileEntry> forward_lats_;
  ProfileEntry preprocess_;
  ProfileEntry postprocess_;
  const float network_latency_us_ = 2000; // us
};

struct TFShareSuffixInfo {
  size_t suffix_index;
  std::string model_name;
  std::string output_layer;
  std::string type;
  std::string class_names;

  TFShareSuffixInfo(size_t suffix_index_, const YAML::Node &node);
};

struct TFShareInfo {
  std::string model_file;
  std::string input_layer;
  std::string slice_beg_vector;
  std::string slice_end_vector;
  int image_height;
  int image_width;
  std::unordered_map<std::string, TFShareSuffixInfo> suffix_models;

  std::string hack_internal_id;
  explicit TFShareInfo(const YAML::Node &node);
};

class ModelDatabase {
 public:
  static ModelDatabase& Singleton();

  const YAML::Node* GetModelInfo(const std::string& model_id) const;

  const YAML::Node* GetModelInfo(const std::string& framework,
                                 const std::string& model_name,
                                 uint32_t version) const;

  const ModelProfile* GetModelProfile(const std::string& gpu_device,
                                      const std::string& profile_id) const;

  float GetModelForwardLatency(const std::string& gpu_device,
                               const std::string& profile_id,
                               uint32_t batch) const;
  
  size_t GetModelMemoryUsage(const std::string& gpu_device,
                             const std::string& profile_id,
                             uint32_t batch) const;

  int GetSharePrefixLength(const std::string& model_id1,
                           const std::string& model_id2) const;

  std::vector<std::string> GetPrefixShareModels(const std::string& model_id)
      const;

  std::shared_ptr<TFShareInfo> GetTFShareInfo(const std::string& model_name) const;

 private:
  ModelDatabase(const std::string& model_root);

  void LoadModelInfo(const std::string& db_file);

  void LoadModelProfiles(const std::string& profile_dir);

 private:
  using ProfileTable = std::unordered_map<std::string, ModelProfile>;
  using PrefixMap = std::unordered_map<std::string, uint32_t>;

  /*! \brief Model database root directory */
  std::string db_root_dir_;
  /*! \brief Model store directory */
  std::string model_store_dir_;
  /*! \brief Map from model ID to model information */
  std::unordered_map<std::string, YAML::Node> model_info_table_;
  /*! \brief Map from device name to profile table */
  std::unordered_map<std::string, ProfileTable> device_profile_table_;

  std::unordered_map<std::string, PrefixMap> share_prefix_models_;
  /*! \brief Map from model name to TFShareInfo */
  std::unordered_map<std::string, std::shared_ptr<TFShareInfo>> tf_share_models_;
};

} // namespace nexus

#endif // NEXUS_COMMON_MODEL_DB_H_
