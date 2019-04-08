#include <fstream>
#include <glog/logging.h>
#include <unordered_set>

#include "nexus/common/util.h"
#include "nexus/backend/utils.h"

namespace nexus {
namespace backend {

void LoadClassnames(const std::string& filepath,
                    std::unordered_map<int, std::string>* classnames) {
  std::ifstream infile(filepath);
  CHECK(infile.good()) << "Classname file " << filepath << " doesn't exist";
  std::string line;
  int class_id = 0;
  while (std::getline(infile, line)) {
    std::vector<std::string> items;
    SplitString(line, ',', &items);
    if (items.size() == 1) {
      classnames->emplace(class_id++, line);
    } else {
      int idx = std::stoi(items[0]);
      classnames->emplace(idx, items[1]);
    }
  }
  infile.close();
}

void PostprocessClassification(
    const QueryProto& query, const float* prob, size_t nprobs,
    QueryResultProto* result,
    const std::unordered_map<int, std::string>* classnames) {
  // TODO: handle top k and threshold in the query
  if (classnames != nullptr) {
    CHECK_EQ(classnames->size(), nprobs) << "Mismatch between number of " <<
        "class names and number of outputs";
  }
  std::unordered_set<std::string> output_fields(query.output_field().begin(),
                                                query.output_field().end());
  if (output_fields.empty()) {
    output_fields.insert("class_id");
    output_fields.insert("class_prob");
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
        value->set_data_type(DT_INT32);
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
          auto iter = classnames->find(max_idx);
          if (iter == classnames->end()) {
            LOG(ERROR) << "Cannot find class name for class id " << max_idx;
          } else {
            value->set_s(iter->second);
          }
        }
      }
    }
  }
}

} // namespace backend
} // namespace nexus
