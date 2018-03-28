#ifndef NEXUS_COMMON_MODEL_DEF_H_
#define NEXUS_COMMON_MODEL_DEF_H_

#include <glog/logging.h>
#include <ostream>
#include <sstream>
#include <unordered_map>
#include <utility>

#include "common/util.h"
#include "proto/nnquery.pb.h"

namespace nexus {

inline Framework get_Framework(std::string name) {
  extern std::unordered_map<std::string, Framework> NameToFramework;
  if (NameToFramework.find(name) == NameToFramework.end()) {
    LOG(FATAL) << "Unknown framework: " << name;
  }
  return NameToFramework.at(name);
}

inline std::string Framework_name(Framework f) {
  switch (f) {
    case DARKNET:
      return "darknet";
    case CAFFE:
      return "caffe";
    case TENSORFLOW:
      return "tensorflow";
    default:
      return "unknown";
  }
}

// ModelId is pair <Framework, model name>
using ModelId = std::pair<Framework, std::string>;
// ModelType defines the task of a model
enum ModelType {
  kClassification = 0,
  kDetection,
  kCaption,
};

inline std::string ModelType_Name(ModelType type) {
  switch (type) {
    case kClassification:
      return "classification";
    case kDetection:
      return "detection";
    case kCaption:
      return "caption";
    default:
      return "";
  }
}

inline ModelType get_ModelType(std::string s) {
  extern std::unordered_map<std::string, ModelType> NameToModelType;
  return NameToModelType.at(s);
}

inline std::string ModelSessionToString(const ModelSession& model_session,
                                        bool has_latency_sla = true) {
  std::stringstream ss;
  ss << Framework_name(model_session.framework()) << ":" <<
      model_session.model_name() << ":" << model_session.version();
  if (model_session.image_height() > 0) {
    ss << ":" << model_session.image_height() << "x" <<
        model_session.image_width();
  }
  if (has_latency_sla) {
    ss << ":" << model_session.latency_sla();
  }
  return ss.str();
}

inline bool ParseModelSession(const std::string& str, ModelSession* sess,
                              bool has_latency_sla = true) {
  std::vector<std::string> tokens;
  SplitString(str, ':', &tokens);
  if (has_latency_sla) {
    if (tokens.size() < 4) {
      return false;
    }
  } else if (tokens.size() < 3) {
      return false;
  }
  
  sess->set_framework(get_Framework(tokens[0]));
  sess->set_model_name(tokens[1]);
  sess->set_version(std::stoi(tokens[2]));
  
  auto decode_dims = [&](const std::string& s) {
    std::vector<std::string> image_dims;
    SplitString(s, 'x', &image_dims);
    if (image_dims.size() != 2) {
      return false;
    }
    sess->set_image_height(std::stoi(image_dims[0]));
    sess->set_image_width(std::stoi(image_dims[1]));
    return true;
  };
  
  if (has_latency_sla) {
    if (tokens.size() == 4) {
      sess->set_latency_sla(std::stof(tokens[3]));
    } else {
      sess->set_latency_sla(std::stof(tokens[4]));
      if (!decode_dims(tokens[3])) {
        return false;
      }
    }
  } else {
    if (tokens.size() > 2) {
      if (!decode_dims(tokens[3])) {
        return false;
      }
    }
  }
  return true;
}

} // namespace nexus

namespace std {

template <>
struct hash<nexus::Framework> {
  size_t operator()(const nexus::Framework& k) const {
    return k;
  }
};

template <>
struct hash<nexus::ModelId> {
  size_t operator()(const nexus::ModelId& k) const {
    return (k.first) ^ hash<string>()(k.second);
  }
};

ostream& operator<<(ostream& stream, const nexus::ModelId& model_id);

} // namespace std

#endif // NEXUS_COMMON_MODEL_DEF_H_
