#include <gflags/gflags.h>

#include "nexus/app/app_base.h"

using namespace nexus;
using namespace nexus::app;

class TrafficApp : public AppBase {
 public:
  TrafficApp(std::string port, std::string rpc_port, std::string sch_addr,
             size_t nthreads, int latency_slo) :
      AppBase(port, rpc_port, sch_addr, nthreads),
      latency_slo_(latency_slo){}
  void buildDependency() {
    
  }
  void add_model(ModelSession* blank, ModelSession model) {
    blank->set_framework(model.framework());
    blank->set_model_name(model.model_name());
    blank->set_version(model.version());
    blank->set_latency_sla(model.latency_sla());
    blank->set_image_height(model.image_height());
    blank->set_image_width(model.image_width());
    blank->set_estimate_latency(200);
  }
  void Setup() final {
   //200 is estimate_latency
    SetComplexQuery();
    ssd_model_ = GetModelHandler(true, "tensorflow", "ssd_mobilenet", 1,
                                 200, 1, {});
    car_model_ = GetModelHandler(true, "caffe2", "googlenet_cars", 1, 200, 1);
    face_model_ = GetModelHandler(true, "caffe2", "vgg_face_0", 1, 200, 1, {});
    //build relationship graph
    LoadDependency(request);
    LoadDependencyRequest request;
    LoadDependencyProto *proto = request.mutable_dependency();
    proto->set_n(3);
    proto->set_m(2);
    proto->set_latency(latency_slo_);
    
    auto ssd = ssd_model_->GetModelSession();
    auto car = car_model_->GetModelSession();
    auto face = face_model_->GetModelSession();
    
    auto* model = proto->add_models();
    add_model(model, ssd);
    model = proto->add_models();
    add_model(model, car);
    model = proto->add_models();
    add_model(model, face);
    
    auto* edge = proto->add_edges();
    auto* v = edge->mutable_v1();
    add_model(v, ssd);
    v = edge->mutable_v2();
    add_model(v, car);
    
    edge = proto->add_edges();
    v = edge->mutable_v1();
    add_model(v, ssd);
    v = edge->mutable_v2();
    add_model(v, face);
    
    LoadDependency(request);
    
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
DEFINE_int32(latency, 400, "Latency SLO for query in ms");

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
                 FLAGS_latency);
  LaunchApp(&app);

  return 0;
}
