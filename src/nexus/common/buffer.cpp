#include "nexus/common/buffer.h"
#include <glog/logging.h>

namespace nexus {
std::shared_ptr<Buffer> Buffer::Slice(size_t offset, size_t nbytes) {
    CHECK_LE(offset + nbytes, nbytes_) << "Slice exceeds buffer boundary";
    return std::shared_ptr<Buffer>(new Buffer(
            shared_from_this(), offset, nbytes));
}
}
