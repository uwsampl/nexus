#ifndef NEXUS_COMMON_MODEL_PROFILE_H_
#define NEXUS_COMMON_MODEL_PROFILE_H_

#include <unordered_map>

#include "common/model_def.h"

namespace nexus {

// model profile key is a pair of <GPU device name, model session id>
using ProfileKey = std::pair<std::string, std::string>;

struct ProfileEntry {
  float latency_mean;
  float latency_std;
  size_t memory_usage;
};

class ModelProfile {
 public:
  ModelProfile() {}

  ModelProfile(const std::string& file_path);

  void LoadProfile(const std::string& file_path);

  std::string model_id() const { return model_id_; }

  std::string gpu_device_name() const { return gpu_device_name_; }

  ProfileKey key() const { return {gpu_device_name_, model_id_}; }

  float GetForwardLatency(uint32_t batch) const;

  float GetPreprocessLatency() const;

  float GetPostprocessLatency() const;

  size_t GetMemoryUsage(uint32_t batch) const;
  /*!
   * \brief Computes the maximum batch size to use within latency_sla
   * \param latency_sla Latency SLA in ms
   * \return max batch size
   */
  uint32_t GetMaxBatch(float latency_sla) const;
  /*!
   * \brief Computes the maximum throughput can be achieved within latency_sla
   * \param latency_sla Latency SLA in ms
   * \return pair of best batch size and max throughput
   */
  std::pair<uint32_t, float> GetMaxThroughput(float latency_sla) const;

 private:
  std::string model_id_;
  std::string gpu_device_name_;
  std::unordered_map<uint32_t, ProfileEntry> forward_lats_;
  ProfileEntry preprocess_;
  ProfileEntry postprocess_;
  const float network_latency_ = 2; // 2 ms
};

} // namespace nexus

namespace std {
template <>
struct hash<nexus::ProfileKey> {
  size_t operator()(const nexus::ProfileKey& k) const {
    return hash<string>()(k.first) ^ hash<string>()(k.second);
  }
};
} // namespace std

namespace nexus {

class ModelProfileTable {
 public:
  static ModelProfileTable& Singleton() {
    static ModelProfileTable model_profile_table_;
    return model_profile_table_;
  }

  void Init(const std::string& profile_dir);

  const ModelProfile* GetModelProfile(const std::string& gpu_device,
                                      const std::string& model_id) const;

  float GetModelForwardLatency(
      const std::string& gpu_device, const std::string& model_id,
      uint32_t batch) const;
  
  size_t GetModelMemoryUsage(
      const std::string& gpu_device, const std::string& model_id,
      uint32_t batch) const;

 private:
  ModelProfileTable() {}

  std::unordered_map<ProfileKey, ModelProfile> model_profiles_;
};

} // namespace nexus

#endif // NEXUS_COMMON_MODEL_PROFILE_H_
