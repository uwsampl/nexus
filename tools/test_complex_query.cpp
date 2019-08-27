#include <gflags/gflags.h>
#include <glog/logging.h>
#include <iostream>
#include "nexus/common/model_db.h"
#include "nexus/scheduler/complex_query.h"

using namespace nexus;
using namespace nexus::scheduler;

DEFINE_int32(avg_interval, 10, "Moving average interval for backend rate");  // for the sch_info.cpp linking error

ComplexQuery::NodeID add_node(ComplexQuery &cq, const std::string &gpu,
                              const std::string &framework, const std::string &model_name,
                              int image_width, int image_height) {
  ComplexQuery::NodeID node(framework, model_name);
  auto model_sess_id = framework + ':' + model_name + ":0";
  auto profile_id = framework + ':' + model_name + ":1";
  if (image_height > 0) {
    profile_id += ":" + std::to_string(image_height) + "x" + std::to_string(image_width);
  }
  auto *profile = ModelDatabase::Singleton().GetModelProfile(gpu, "generic", profile_id);
  CHECK(profile != nullptr);
  cq.AddNode(node, model_sess_id, *profile);
  return node;
}

void add_node(ComplexQuery &cq, ComplexQuery::NodeID &node, const std::string &profile_id, const std::string &gpu) {
  auto model_sess_id = node.framework + ':' + node.model_name + ":0";
  auto *profile = ModelDatabase::Singleton().GetModelProfile(gpu, "generic", profile_id);
  cq.AddNode(node, model_sess_id, *profile);
}

int main(int argc, char** argv) {
  FLAGS_logtostderr = 1;
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InstallFailureSignalHandler();

  const int SLO_MS = 400;
  const int SEGMENTS = 500;
  const std::string gpu = "GeForce_GTX_1080_Ti";

  ComplexQuery cq("cq_id", SLO_MS * 1000, SEGMENTS);
  auto node_ssd = add_node(cq, gpu, "tensorflow", "ssd_mobilenet", 300, 300);
  auto node_inception = add_node(cq, gpu, "tensorflow", "inception_0", 0, 0);
  auto node_vgg = add_node(cq, gpu, "tensorflow", "vgg16_0", 0, 0);
  cq.AddChild(node_ssd, node_inception);
  cq.AddChild(node_ssd, node_vgg);
  cq.Finalize();

  cq.SetRequestRate(node_ssd, 200);
  cq.SetRequestRate(node_inception, 50);
  cq.SetRequestRate(node_vgg, 100);
  cq.DynamicProgramming();
  std::cout << "minimal number of GPUs: " << cq.GetMinimalGPUs() << std::endl;
  auto split = cq.GetSLOms();
  for (auto &node : split)
    std::cout << "  " <<  node.first.ToString() << ": " << node.second << "ms" << std::endl;
}
