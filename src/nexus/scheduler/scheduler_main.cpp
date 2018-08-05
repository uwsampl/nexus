#include <gflags/gflags.h>

#include "nexus/common/config.h"
#include "nexus/common/util.h"
#include "nexus/scheduler/scheduler.h"

using namespace nexus::scheduler;

DEFINE_string(port, "10001", "RPC port");
DEFINE_string(workload, "", "Static workload config file");

int main(int argc, char** argv) {
  // log to stderr
  FLAGS_logtostderr = 1;
  // Init glog
  google::InitGoogleLogging(argv[0]);
  // Parse command line flags
  google::ParseCommandLineFlags(&argc, &argv, true);
  // Setup backtrace on segfault
  google::InstallFailureSignalHandler();
  // Create scheduler
  Scheduler scheduler(FLAGS_port, 4);
  if (FLAGS_workload.length() > 0) {
    scheduler.LoadWorkloadFile(FLAGS_workload);
  }
  scheduler.Run();
  while (true) {
    ;
  }
}
