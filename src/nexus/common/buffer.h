#ifndef NEXUS_COMMON_BUFFER_H_
#define NEXUS_COMMON_BUFFER_H_

#include <memory>

#include "common/device.h"

namespace nexus {

class Buffer {
 public:
  // disable copy
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;

  Buffer() :
      data_(nullptr),
      nbytes_(0),
      own_data_(false) {}

  explicit Buffer(size_t nbytes, Device* device) :
      nbytes_(nbytes),
      device_(device),
      own_data_(true) {
    data_ = device->Allocate(nbytes_);
    //LOG(INFO) << "Allocate " << nbytes_ << " on " << device->name();
  }

  explicit Buffer(void* data, size_t nbytes, Device* device,
                  bool own_data = false) :
      data_(data),
      nbytes_(nbytes),
      device_(device),
      own_data_(own_data) {}

  ~Buffer() {
    if (own_data_) {
      device_->Free(data_);
      //LOG(INFO) << "Free " << nbytes_ << " on " << device_->name();
    }
  }

  size_t nbytes() { return nbytes_; }

  void* data() { return data_; }

  const void* data() const { return data_; }

  Device* device() const { return device_; }
  
 private:
  void* data_;
  size_t nbytes_;
  Device* device_;
  bool own_data_;
};

} // namespace nexus

#endif // NEXUS_COMMON_BUFFER_H_
