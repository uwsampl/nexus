#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

DEFINE_string(model_db, "", "Test model db directory");

int main(int argc, char ** argv) {
  testing::InitGoogleTest(&argc, argv);
  testing::FLAGS_gtest_death_test_style = "threadsafe";
  google::ParseCommandLineFlags(&argc, &argv, true);
  if (FLAGS_model_db.length() == 0) {
    LOG(FATAL) << "Missing model_db";
  }
  return RUN_ALL_TESTS();
}
