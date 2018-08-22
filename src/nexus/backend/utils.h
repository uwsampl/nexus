#ifndef NEXUS_BACKEND_UTILS_H_
#define NEXUS_BACKEND_UTILS_H_

#include <glog/logging.h>
#include <unordered_map>

#include "nexus/proto/nnquery.pb.h"

namespace nexus {
namespace backend {

void LoadClassnames(const std::string& filepath,
                    std::unordered_map<int, std::string>* classnames);

void PostprocessClassification(
    const QueryProto& query, const float* prob, size_t nprobs,
    QueryResultProto* result,
    const std::unordered_map<int, std::string>* classnames = nullptr);
                               

} // namespace backend
} // namespace nexus

#endif // NEXUS_BACKEND_UTILS_H_
