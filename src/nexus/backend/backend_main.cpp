#include <gflags/gflags.h>
#include <glog/logging.h>
#include <iostream>
#include <string>
#include <vector>

#include "nexus/common/config.h"
#include "nexus/common/util.h"
#include "nexus/backend/backend_server.h"

using namespace nexus::backend;

DEFINE_string(port, std::to_string(BACKEND_DEFAULT_PORT), "server port");
DEFINE_string(rpc_port, std::to_string(BACKEND_DEFAULT_RPC_PORT), "RPC port");
DEFINE_string(sch_addr, "127.0.0.1", "scheduler IP address "
              "(use default port 10001 if no port specified)");
DEFINE_string(model_root, "", "model root directory");
DEFINE_uint64(num_workers, 4, "number of workers (default: 4)");
DEFINE_int32(gpu, 0, "gpu device ID (default: 0)");

int main(int argc, char** argv) {
  // log to stderr
  FLAGS_logtostderr = 1;
  // Init glog
  google::InitGoogleLogging(argv[0]);
  // Parse command line flags
  google::ParseCommandLineFlags(&argc, &argv, true);
  // Setup backtrace on segfault
  google::InstallFailureSignalHandler();
  // Check whether the config file is specified
  if (FLAGS_model_root.length() == 0) {
    LOG(FATAL) << "Missing model_root";
  }
  // Decide server IP address
  LOG(INFO) << "Backend server: port " << FLAGS_port << ", rpc port " <<
      FLAGS_rpc_port << ", workers " << FLAGS_num_workers << ", gpu " <<
      FLAGS_gpu;
  // Create the backend server
  BackendServer server(FLAGS_port, FLAGS_rpc_port, FLAGS_sch_addr,
                       FLAGS_num_workers, FLAGS_gpu, FLAGS_model_root);
  server.Run();
  return 0;
}
