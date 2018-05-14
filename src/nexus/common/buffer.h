#ifndef NEXUS_COMMON_BUFFER_H_
#define NEXUS_COMMON_BUFFER_H_

#include <memory>

#include "nexus/common/device.h"

namespace nexus {

class Buffer : public std::enable_shared_from_this<Buffer> {
 public:
  // disable copy
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;

  Buffer() :
      data_(nullptr),
      nbytes_(0),
      own_data_(false),
      shared_from_(nullptr) {}

  explicit Buffer(size_t nbytes, Device* device) :
      nbytes_(nbytes),
      device_(device),
      own_data_(true),
      shared_from_(nullptr) {
    data_ = device->Allocate(nbytes_);
    //LOG(INFO) << "Allocate " << nbytes_ << " on " << device->name();
  }

  explicit Buffer(void* data, size_t nbytes, Device* device,
                  bool own_data = false) :
      data_(data),
      nbytes_(nbytes),
      device_(device),
      own_data_(own_data),
      shared_from_(nullptr) {}

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

  std::shared_ptr<Buffer> Slice(size_t offset, size_t nbytes) {
    CHECK_LE(offset + nbytes, nbytes_) << "Slice exceeds buffer boundary";
    return std::shared_ptr<Buffer>(new Buffer(
        shared_from_this(), offset, nbytes));
  }
  
 private:
  Buffer(std::shared_ptr<Buffer> origin, size_t offset, size_t nbytes) :
      data_((char*) origin->data_ + offset),
      nbytes_(nbytes),
      device_(origin->device_),
      own_data_(false),
      shared_from_(origin) {}
  
  void* data_;
  size_t nbytes_;
  Device* device_;
  bool own_data_;
  std::shared_ptr<Buffer> shared_from_;
};

} // namespace nexus

#endif // NEXUS_COMMON_BUFFER_H_
