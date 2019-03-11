#include <glog/logging.h>
#include <opencv2/opencv.hpp>
#include <string>

#include "nexus/common/image.h"

namespace nexus {

cv::Mat DecodeImage(const ImageProto& image, ChannelOrder order) {
  cv::Mat img_bgr;
  const std::string& data = image.data();
  std::vector<char> vec_data(data.c_str(), data.c_str() + data.size());
  int cv_read_flag = image.color() ? cv::IMREAD_COLOR : cv::IMREAD_GRAYSCALE;
  img_bgr = cv::imdecode(vec_data, cv_read_flag);
  if (!img_bgr.data) {
    LOG(ERROR) << "Could not decode image";
  }
  if (order == CO_BGR) {
    return img_bgr;
  }
  cv::Mat img_rgb;
  cv::cvtColor(img_bgr, img_rgb, cv::COLOR_BGR2RGB);
  return img_rgb;
}

} // namespace nexus
