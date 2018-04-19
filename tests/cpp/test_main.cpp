#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

DEFINE_string(test_data, "", "Test data directory");

int main(int argc, char ** argv) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  testing::InitGoogleTest(&argc, argv);
  testing::FLAGS_gtest_death_test_style = "threadsafe";
  if (FLAGS_test_data.length() == 0) {
    LOG(FATAL) << "Missing test_data";
  }
  return RUN_ALL_TESTS();
}
