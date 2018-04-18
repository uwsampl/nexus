#include <gflags/gflags.h>

#include "nexus/common/config.h"
#include "nexus/common/util.h"
#include "nexus/scheduler/scheduler.h"

using namespace nexus::scheduler;

DEFINE_string(port, "10001", "RPC port");
DEFINE_string(model_root, "", "model root directory");
DEFINE_string(workload_file, "", "workload file");
DEFINE_int32(beacon, BEACON_INTERVAL_SEC, "beacon interval in seconds "
             "(default: 5 sec)");
DEFINE_int32(epoch, EPOCH_INTERVAL_SEC, "beacon interval in seconds "
             "(default: 30 sec)");

int main(int argc, char** argv) {
  // log to stderr
  FLAGS_logtostderr = 1;
  // Init glog
  google::InitGoogleLogging(argv[0]);
  // Parse command line flags
  google::ParseCommandLineFlags(&argc, &argv, true);
  // Setup backtrace on segfault
  google::InstallFailureSignalHandler();
  if (FLAGS_model_root.length() == 0) {
    LOG(FATAL) << "Missing model_root";
  }
  // Create scheduler
  Scheduler scheduler(FLAGS_port, 4, FLAGS_model_root, FLAGS_beacon,
                      FLAGS_epoch);
  if (FLAGS_workload_file.length() > 0) {
    scheduler.LoadWorkloadFile(FLAGS_workload_file);
  }
  scheduler.Run();
  while (true) {
    ;
  }
}
