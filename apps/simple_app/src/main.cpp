#include <gflags/gflags.h>

#include "nexus/app/app_base.h"

using namespace nexus;
using namespace nexus::app;

class SimpleApp : public AppBase {
 public:
  SimpleApp(std::string port, std::string rpc_port, std::string sch_addr,
            size_t nthreads, const std::string& framework,
            const std::string& model_name, int version, int latency_sla_ms,
            float estimate_workload) :
      AppBase(port, rpc_port, sch_addr, nthreads),
      framework_(framework),
      model_name_(model_name),
      version_(version),
      latency_sla_ms_(latency_sla_ms),
      estimate_workload_(estimate_workload) {}

  void Setup() final {
    model_ = GetModelHandler(framework_, model_name_, version_,
                             latency_sla_ms_, estimate_workload_);
  }

  void Process(const RequestProto& request, ReplyProto* reply) final {
    auto output = model_->Execute(request.input());
    output->FillReply(reply);
  }
  
 private:
  std::string framework_;
  std::string model_name_;
  int version_;
  int latency_sla_ms_;
  float estimate_workload_;
  std::shared_ptr<ModelHandler> model_;
};

DEFINE_string(port, "9001", "Server port");
DEFINE_string(rpc_port, "9002", "RPC port");
DEFINE_string(sch_addr, "127.0.0.1", "Scheduler address");
DEFINE_int32(nthread, 1000, "Number of threads processing requests "
             "(default: 1000)");
DEFINE_string(framework, "", "Framework (caffe, darknet, tensorflow)");
DEFINE_string(model, "", "Model name");
DEFINE_int32(model_version, 1, "Model version (default: 1)");
DEFINE_int32(latency, 500, "Latency SLA in ms (default: 500)");
DEFINE_double(workload, 0, "Estimated request rate (default: 0)");

int main(int argc, char** argv) {
  // log to stderr
  FLAGS_logtostderr = 1;
  // Init glog
  google::InitGoogleLogging(argv[0]);
  // Parse command line flags
  google::ParseCommandLineFlags(&argc, &argv, true);
  // Setup backtrace on segfault
  google::InstallFailureSignalHandler();

  CHECK_GT(FLAGS_framework.length(), 0) << "Missing framework";
  CHECK_GT(FLAGS_model.length(), 0) << "Missing model";
  LOG(INFO) << "App port " << FLAGS_port << ", rpc port " << FLAGS_rpc_port;
  // Create the frontend server
  SimpleApp app(FLAGS_port, FLAGS_rpc_port, FLAGS_sch_addr, FLAGS_nthread,
                FLAGS_framework, FLAGS_model, FLAGS_model_version,
                FLAGS_latency, FLAGS_workload);
  LaunchApp(&app);

  return 0;
}
