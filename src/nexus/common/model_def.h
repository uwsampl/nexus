#ifndef NEXUS_COMMON_MODEL_DEF_H_
#define NEXUS_COMMON_MODEL_DEF_H_

#include <ostream>
#include <sstream>
#include <unordered_map>
#include <utility>

#include "nexus/common/util.h"
#include "nexus/proto/nnquery.pb.h"

namespace nexus {

inline std::string ModelID(const std::string& framework,
                           const std::string& model_name,
                           uint32_t version) {
  std::stringstream ss;
  ss << framework << ":" << model_name << ":" << version;
  return ss.str();
}

inline void ParseModelID(const std::string model_id,
                         ModelSession* model_session) {
  std::vector<std::string> tokens;
  SplitString(model_id, ':', &tokens);
  model_session->set_framework(tokens[0]);
  model_session->set_model_name(tokens[1]);
  model_session->set_version(std::stoi(tokens[2]));
}

inline std::string ModelSessionToModelID(const ModelSession& model_session) {
  std::stringstream ss;
  ss << model_session.framework() << ":" << model_session.model_name() << ":"
     << model_session.version();
  return ss.str();
}

inline std::string ModelSessionToProfileID(const ModelSession& model_session) {
  std::stringstream ss;
  ss << model_session.framework() << ":" << model_session.model_name() << ":"
     << model_session.version();
  if (model_session.image_height() > 0) {
    ss << ":" << model_session.image_height() << "x" <<
        model_session.image_width();
  }
  return ss.str();
}

inline std::string ModelSessionToString(const ModelSession& model_session) {
  std::stringstream ss;
  ss << model_session.framework() << ":" <<
      model_session.model_name() << ":" << model_session.version();
  if (model_session.image_height() > 0) {
    ss << ":" << model_session.image_height() << "x" <<
        model_session.image_width();
  }
  ss << ":" << model_session.latency_sla();
  return ss.str();
}

inline bool ParseModelSession(const std::string& str, ModelSession* sess) {
  std::vector<std::string> tokens;
  SplitString(str, ':', &tokens);
  if (tokens.size() < 4) {
    return false;
  }
  sess->set_framework(tokens[0]);
  sess->set_model_name(tokens[1]);
  sess->set_version(std::stoi(tokens[2]));
  if (tokens.size() == 4) {
    sess->set_latency_sla(std::stoi(tokens[3]));
  } else {
    sess->set_latency_sla(std::stoi(tokens[4]));
    // decode image size
    std::vector<std::string> image_dims;
    SplitString(tokens[3], 'x', &image_dims);
    if (image_dims.size() != 2) {
      return false;
    }
    sess->set_image_height(std::stoi(image_dims[0]));
    sess->set_image_width(std::stoi(image_dims[1]));
  }
  return true;
}

} // namespace nexus

#endif // NEXUS_COMMON_MODEL_DEF_H_
