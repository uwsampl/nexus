#include <gflags/gflags.h>

#include "nexus/app/app_base.h"

using namespace nexus;
using namespace nexus::app;

class TrafficApp : public AppBase {
 public:
  TrafficApp(std::string port, std::string rpc_port, std::string sch_addr,
             size_t nthreads, int latency_slo, int ssd_latency_ms) :
      AppBase(port, rpc_port, sch_addr, nthreads),
      latency_slo_(latency_slo),
      ssd_latency_ms_(ssd_latency_ms),
      rec_latency_ms_(latency_slo - ssd_latency_ms) {}

  void Setup() final {
    ssd_model_ = GetModelHandler("tensorflow", "ssd_mobilenet", 1,
                                 ssd_latency_ms_, 0, {}, LB_DeficitRR);
    car_model_ = GetModelHandler("caffe2", "googlenet_cars", 1, rec_latency_ms_);
    face_model_ = GetModelHandler("caffe2", "vgg_face_0", 1, rec_latency_ms_);
    auto func1 = [&](std::shared_ptr<RequestContext> ctx) {
      auto ssd_output = ssd_model_->Execute(ctx, ctx->const_request().input());
      return std::vector<VariablePtr>{
          std::make_shared<Variable>("ssd_output", ssd_output)};
    };
    auto func2 = [&](std::shared_ptr<RequestContext> ctx) {
      auto ssd_output = ctx->GetVariable("ssd_output")->result();
      std::vector<std::shared_ptr<QueryResult> > results;
      std::vector<RectProto> car_boxes;
      std::vector<RectProto> face_boxes;
      for (int i = 0; i < ssd_output->num_records(); ++i) {
        auto& rec = (*ssd_output)[i];
        auto name = rec["class_name"].as<std::string>();
        if (name == "car" || name == "truck") {
          car_boxes.push_back(rec["rect"].as<RectProto>());
        } else if (name == "person") {
          face_boxes.push_back(rec["rect"].as<RectProto>());
        }
      }
      if (!car_boxes.empty()) {
        results.push_back(
            car_model_->Execute(ctx, ctx->const_request().input(), {}, 1,
                                car_boxes));
      }
      if (!face_boxes.empty()) {
        results.push_back(
            face_model_->Execute(ctx, ctx->const_request().input(), {}, 1,
                                 face_boxes));
      }
      return std::vector<VariablePtr>{
          std::make_shared<Variable>("rec_output", results)};
    };
    auto func3 = [&](std::shared_ptr<RequestContext> ctx) {
      auto rec_output = ctx->GetVariable("rec_output");
      if (rec_output->count() > 0) {
        rec_output->result()->ToProto(ctx->reply());
      }
      return std::vector<VariablePtr>{};
    };
    ExecBlock* b1 = new ExecBlock(0, func1, {});
    ExecBlock* b2 = new ExecBlock(1, func2, {"ssd_output"});
    ExecBlock* b3 = new ExecBlock(2, func3, {"rec_output"});
    qp_ = new QueryProcessor({b1, b2, b3});
  }

 private:
  RectProto GetRect(int left, int right, int top, int bottom) {
    RectProto rect;
    rect.set_left(left);
    rect.set_right(right);
    rect.set_top(top);
    rect.set_bottom(bottom);
    return rect;
  }

  int latency_slo_;
  int ssd_latency_ms_;
  int rec_latency_ms_;
  std::shared_ptr<ModelHandler> ssd_model_;
  std::shared_ptr<ModelHandler> car_model_;
  std::shared_ptr<ModelHandler> face_model_;
};

DEFINE_string(port, "9001", "Server port");
DEFINE_string(rpc_port, "9002", "RPC port");
DEFINE_string(sch_addr, "127.0.0.1", "Scheduler address");
DEFINE_int32(nthread, 4, "Number of threads processing requests");
DEFINE_int32(latency, 0, "Latency SLO for query in ms");
DEFINE_int32(ssd_latency, 0, "Latency SLO for SSD model in ms");

int main(int argc, char** argv) {
  // log to stderr
  FLAGS_logtostderr = 1;
  // Init glog
  google::InitGoogleLogging(argv[0]);
  // Parse command line flags
  google::ParseCommandLineFlags(&argc, &argv, true);
  // Setup backtrace on segfault
  google::InstallFailureSignalHandler();

  LOG(INFO) << "App port " << FLAGS_port << ", rpc port " << FLAGS_rpc_port;
  // Create the frontend server
  TrafficApp app(FLAGS_port, FLAGS_rpc_port, FLAGS_sch_addr, FLAGS_nthread,
                 FLAGS_latency, FLAGS_ssd_latency);
  LaunchApp(&app);

  return 0;
}
