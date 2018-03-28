#include "backend/slice.h"
#include <glog/logging.h>

namespace nexus {
namespace backend {

Slice::Slice(size_t nsplits, size_t nfloats) :
    equal_split_(true) {
  size_t offset = 0;
  for (size_t i = 0; i < nsplits; ++i) {
    offsets_.push_back(offset);
    offset += nfloats;
  }
  total_elements_ = offset;
  equal_slice_size_ = nfloats;
}

Slice::Slice(std::vector<size_t> nfloats, size_t multiplier) :
    equal_split_(false) {
  size_t offset = 0;
  for (auto size : nfloats) {
    offsets_.push_back(offset);
    size_t slice_size = size * multiplier;
    slice_sizes_.push_back(slice_size);
    offset += slice_size;
  }
  total_elements_ = offset;
}

Slice::Slice(std::vector<float> nfloats, size_t multiplier) :
    equal_split_(false) {
  size_t offset = 0;
  for (auto size : nfloats) {
    offsets_.push_back(offset);
    size_t slice_size = size_t(size) * multiplier;
    slice_sizes_.push_back(slice_size);
    offset += slice_size;
  }
  total_elements_ = offset;
}

Slice::Slice(size_t nsplits, float* nfloats, size_t multiplier) :
    equal_split_(false) {
  size_t offset = 0;
  for (size_t i = 0; i < nsplits; ++i) {
    offsets_.push_back(offset);
    size_t slice_size = size_t(nfloats[i]) * multiplier;
    slice_sizes_.push_back(slice_size);
    offset += slice_size;
  }
  total_elements_ = offset;
}

size_t Slice::offset(int idx) const {
  CHECK_LT(idx, offsets_.size()) << "Index " << idx << " exceeds the boundary "
                                 << offsets_.size();
  return offsets_[idx];
}

size_t Slice::num_elements(int idx) const {
  CHECK_LT(idx, offsets_.size()) << "Index " << idx << " exceeds the boundary "
                                 << offsets_.size();
  if (equal_split_) {
    return equal_slice_size_;
  }
  return slice_sizes_[idx];
}

} // namespace backend
} // namespace nexus
