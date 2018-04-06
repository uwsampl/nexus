#include "nexus/common/model_def.h"

namespace nexus {

std::unordered_map<std::string, Framework> NameToFramework = {
  {"darknet", DARKNET},
  {"caffe", CAFFE},
  {"tensorflow", TENSORFLOW},
};

std::unordered_map<std::string, ModelType> NameToModelType = {
  {"classification", kClassification},
  {"detection", kDetection},
  {"caption", kCaption},
};

} // namespace nexus

namespace std {

ostream& operator<<(ostream& stream, const nexus::ModelId& model_id) {
  stream << nexus::Framework_name(model_id.first) << ":" << model_id.second;
  return stream;
}

} // namespace nexus
