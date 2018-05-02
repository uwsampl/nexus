#ifndef NEXUS_COMMON_IMAGE_H_
#define NEXUS_COMMON_IMAGE_H_

#include <opencv2/core/core.hpp>

#include "nexus/proto/nnquery.pb.h"

namespace nexus {

enum ChannelOrder {
  CO_RGB = 0,
  CO_BGR = 1,
};

cv::Mat DecodeImage(const ImageProto& image, ChannelOrder order);

} // namespace nexus

#endif // NEXUS_COMMON_IMAGE_H_
