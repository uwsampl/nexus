#include <gflags/gflags.h>
#include <glog/logging.h>
#include <iostream>
#include <string>
#include <vector>

#include "common/util.h"
#include "backend/backend_server.h"

using namespace nexus::backend;

DEFINE_string(ip, "127.0.0.1", "server IP address");
DEFINE_string(port, "8001", "server port");
DEFINE_string(rpc_ip, "", "RPC IP address or prefix (default: same as ip)");
DEFINE_string(rpc_port, "8002", "RPC port");
DEFINE_string(sch_addr, "127.0.0.1:10001", "scheduler address");
//DEFINE_string(model_db, "", "model meta information database");
DEFINE_string(config, "", "config file");
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
  if (FLAGS_config.length() == 0) {
    LOG(FATAL) << "Missing config_file";
  }
  // Decide server IP address
  std::string ip = nexus::GetIpAddress(FLAGS_ip);
  if (ip.length() == 0) {
    LOG(FATAL) << "Cannot find any network interface for prefix " << FLAGS_ip;
  }
  std::string rpc_ip;
  if (FLAGS_rpc_ip.length() == 0) {
    rpc_ip = ip;
  } else {
    rpc_ip = nexus::GetIpAddress(FLAGS_rpc_ip);
    if (rpc_ip.length() == 0) {
      LOG(FATAL) << "Cannot find the network interface for prefix " <<
          FLAGS_rpc_ip;
    }
  }
  LOG(INFO) << "Backend address " << ip << ":" << FLAGS_port <<
      ", rpc address " << rpc_ip << ":" << FLAGS_rpc_port << ", workers " <<
      FLAGS_num_workers << ", gpu " << FLAGS_gpu;
  // Create the backend server
  BackendServer server(FLAGS_port, FLAGS_rpc_port, FLAGS_sch_addr,
                       FLAGS_num_workers, FLAGS_gpu);
  server.LoadConfigFromFile(FLAGS_config);
  server.Run();
  return 0;
}
