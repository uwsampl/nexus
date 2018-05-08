#include <glog/logging.h>
#include <unordered_set>

#include "nexus/backend/postprocess.h"

namespace nexus {
namespace backend {

void PostprocessClassification(const QueryProto& query, const float* prob,
                               size_t nprobs, QueryResultProto* result,
                               const std::vector<std::string>* classnames) {
  // TODO: handle top k and threshold in the query
  if (classnames != nullptr) {
    CHECK_EQ(classnames->size(), nprobs) << "Mismatch between number of " <<
        "class names and number of outputs";
  }
  std::unordered_set<std::string> output_fields(query.output_field().begin(),
                                                query.output_field().end());
  if (output_fields.empty()) {
    output_fields.insert("class_id");
    output_fields.insert("class_name");
  }
  float max_prob = 0.;
  int max_idx = -1;
  for (int i = 0; i < (int) nprobs; ++i) {
    float p = prob[i];
    if (p > max_prob) {
      max_prob = p;
      max_idx = i;
    }
  }
  if (max_idx > -1) {
    auto record = result->add_output();
    for (auto field : output_fields) {
      if (field == "class_id") {
        auto value = record->add_named_value();
        value->set_name("class_id");
        value->set_data_type(DT_INT);
        value->set_i(max_idx);
      } else if (field == "class_prob") {
        auto value = record->add_named_value();
        value->set_name("class_prob");
        value->set_data_type(DT_FLOAT);
        value->set_f(max_prob);
      } else if (field == "class_name") {
        auto value = record->add_named_value();
        value->set_name("class_name");
        value->set_data_type(DT_STRING);
        if (classnames != nullptr) {
          value->set_s(classnames->at(max_idx));
        }
      }
    }
  }
}

} // namespace backend
} // namespace nexus
