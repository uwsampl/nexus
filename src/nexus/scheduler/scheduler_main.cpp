#include <gflags/gflags.h>

#include "nexus/common/util.h"
#include "nexus/scheduler/scheduler.h"

using namespace nexus::scheduler;

DEFINE_string(ip, "127.0.0.1", "server IP address");
DEFINE_string(port, "10001", "RPC port");
DEFINE_string(config, "", "config file");
DEFINE_double(epoch, 5., "epoch time in seconds (default: 5s)");

int main(int argc, char** argv) {
  // log to stderr
  FLAGS_logtostderr = 1;
  // Init glog
  google::InitGoogleLogging(argv[0]);
  // Parse command line flags
  google::ParseCommandLineFlags(&argc, &argv, true);
  // Setup backtrace on segfault
  google::InstallFailureSignalHandler();
  // Decide server IP address
  std::string ip = nexus::GetIpAddress(FLAGS_ip);
  if (ip.length() == 0) {
    LOG(FATAL) << "Cannot find any network interface for prefix " << FLAGS_ip;
  }
  //std::string address = ip + ":" + FLAGS_port;
  //LOG(INFO) << "Scheduler address " << address;
  // Create scheduler
  Scheduler scheduler(FLAGS_port, 4, FLAGS_epoch);
  if (FLAGS_config.length() > 0) {
    scheduler.LoadConfigFile(FLAGS_config);
  }
  scheduler.Run();
  while (true) {
    ;
  }
}
