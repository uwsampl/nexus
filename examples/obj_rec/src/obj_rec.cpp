#include <gflags/gflags.h>

#include "nexus/app/app_base.h"

using namespace nexus;
using namespace nexus::app;

class ObjRecApp : public AppBase {
 public:
  ObjRecApp(std::string port, std::string rpc_port, std::string sch_addr,
            size_t nthreads) :
      AppBase(port, rpc_port, sch_addr, nthreads) {
  }

  void Setup() final {
    model_ = GetModelHandler("caffe2", "vgg16", 1, 1000);
  }

  void Process(const RequestProto& request, ReplyProto* reply) final {
    auto output = model_->Execute(request.input(),
                                  {"class_id", "class_prob", "class_name"});
    output->FillReply(reply);
  }
  
 private:
  std::shared_ptr<ModelHandler> model_;
};

DEFINE_string(port, "9001", "Server port");
DEFINE_string(rpc_port, "9002", "RPC port");
DEFINE_string(sch_addr, "127.0.0.1", "Scheduler IP address");
DEFINE_int32(nthread, 1000, "Number of threads processing requests "
             "(default: 1000)");

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
  ObjRecApp app(FLAGS_port, FLAGS_rpc_port, FLAGS_sch_addr, FLAGS_nthread);
  LaunchApp(&app);

  return 0;
}
