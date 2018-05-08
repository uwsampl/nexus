#ifndef NEXUS_BACKEND_POSTPROCESS_H_
#define NEXUS_BACKEND_POSTPROCESS_H_

#include "nexus/proto/nnquery.pb.h"

namespace nexus {
namespace backend {

void PostprocessClassification(
    const QueryProto& query, const float* prob, size_t nprobs,
    QueryResultProto* result,
    const std::vector<std::string>* classnames = nullptr);
                               

} // namespace backend
} // namespace nexus

#endif // NEXUS_BACKEND_POSTPROCESS_H_
