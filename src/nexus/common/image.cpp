#include <glog/logging.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <string>

#include "nexus/common/image.h"

namespace nexus {

cv::Mat DecodeImage(const ImageProto& image, ChannelOrder order) {
  cv::Mat img_bgr;
  const std::string& data = image.data();
  std::vector<char> vec_data(data.c_str(), data.c_str() + data.size());
  int cv_read_flag = (image.color() ? CV_LOAD_IMAGE_COLOR :
                      CV_LOAD_IMAGE_GRAYSCALE);
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
/*
cv::Mat decode_image_rgb(const NNImage& nn_image) {
  cv::Mat img_bgr, img_rgb;
  const std::string& data = nn_image.data();
  std::vector<char> vec_data(data.c_str(), data.c_str() + data.size());
  int cv_read_flag = (nn_image.color() ? CV_LOAD_IMAGE_COLOR :
                      CV_LOAD_IMAGE_GRAYSCALE);
  img_bgr = cv::imdecode(vec_data, cv_read_flag);
  if (!img_bgr.data) {
    LOG(ERROR) << "Could not decode nn_image";
  }
  cv::cvtColor(img_bgr, img_rgb, cv::COLOR_BGR2RGB);
  return img_rgb;
}
*/

} // namespace nexus
