#include <boost/filesystem.hpp>
#include <boost/filesystem/operations.hpp>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <opencv2/opencv.hpp>

#include <fstream>
#include <iterator>
#include <string>
#include <unordered_map>
#include <vector>

#include "nexus/common/image.h"

DEFINE_string(hack_image_root, "", "HACK: path to directory of images");

class _Hack_Images {
public:
  _Hack_Images(const std::string &root) {
    if (root.empty())
      return;
    LOG(INFO) << "Initializing _Hack_Images, root: " << root;
    auto root_path = boost::filesystem::absolute(root);
    for (auto it = boost::filesystem::recursive_directory_iterator(root_path),
              end = boost::filesystem::recursive_directory_iterator();
         it != end; ++it) {
      if (!boost::filesystem::is_regular_file(it->path()))
        continue;
      if (it->path().extension().string() != ".jpg")
        continue;

      std::ifstream fin(it->path().string(), std::ios::binary);
      std::istreambuf_iterator<char> fin_beg(fin), fin_end;
      std::vector<char> data(fin_beg, fin_end);

      auto rel_path = boost::filesystem::relative(it->path(), root_path);
      data_.emplace(rel_path.string(), std::move(data));
    }
    LOG(INFO) << "Found " << data_.size() << " images in " << root;

    LOG(INFO) << "Keys of some random images:";
    auto iter = data_.begin();
    for (int i = 0; i < 10 && iter != data_.end(); ++i, ++iter) {
      LOG(INFO) << "  " << iter->first;
    }

    LOG(INFO) << "_Hack_Images initilization finished";
  }

  const std::vector<char> &get(const std::string &filename) const {
    auto iter = data_.find(filename);
    return iter != data_.end() ? iter->second : empty_;
  }

private:
  std::unordered_map<std::string, std::vector<char>> data_;
  std::vector<char> empty_;
};

namespace nexus {

cv::Mat DecodeImageImpl(const std::vector<char> &vec_data, bool color,
                        ChannelOrder order) {
  cv::Mat img_bgr;
  int cv_read_flag = color ? cv::IMREAD_COLOR : cv::IMREAD_GRAYSCALE;
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

cv::Mat _Hack_DecodeImageByFilename(const ImageProto &image,
                                    ChannelOrder order) {
  static _Hack_Images *_images = new _Hack_Images(FLAGS_hack_image_root);
  const auto &vec_data = _images->get(image.hack_filename());
  if (vec_data.empty()) {
    if (image.hack_filename() != "__init_Hack_Images")
      LOG(ERROR) << "Cannot find image by filename: " << image.hack_filename();
    return {};
  }
  return DecodeImageImpl(vec_data, image.color(), order);
}

cv::Mat DecodeImage(const ImageProto &image, ChannelOrder order) {
  if (image.hack_filename().empty()) {
    const std::string &data = image.data();
    std::vector<char> vec_data(data.c_str(), data.c_str() + data.size());
    return DecodeImageImpl(vec_data, image.color(), order);
  } else {
    return _Hack_DecodeImageByFilename(image, order);
  }
}

} // namespace nexus
