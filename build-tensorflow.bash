#!/bin/bash
set -e
set -x

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
SRC_DIR="$SCRIPT_DIR/build-dep-src"
INSTALL_DIR="$SCRIPT_DIR/build-dep-install"

mkdir -p "$SRC_DIR"
mkdir -p "$INSTALL_DIR"

cd "$SRC_DIR"

if [ ! -d "$SRC_DIR/tensorflow" ]; then
    # TensorFlow 2.0 master
    git clone https://github.com/tensorflow/tensorflow.git
fi
cd tensorflow
git checkout 906f537c0be010929a0bda3c7d061de9d3d8d5b0

# Tensorflow build config
export PYTHON_BIN_PATH="${PYTHON_BIN_PATH:-/usr/bin/python}"
export PYTHON_LIB_PATH="$($PYTHON_BIN_PATH -c 'import site; print(site.getsitepackages()[0])')"
export TF_ENABLE_XLA=0
export TF_NEED_OPENCL_SYCL=0
export TF_NEED_ROCM=0
export TF_NEED_CUDA=1
export TF_NEED_TENSORRT=0
export TF_CUDA_COMPUTE_CAPABILITIES="${TF_CUDA_COMPUTE_CAPABILITIES:-3.5,3.7,5.2,6.0,6.1,7.0,7.5}"
export TF_CUDA_PATHS="${TF_CUDA_PATHS:-/usr/local/cuda-10.0,/usr}"
export TF_CUDA_CLANG=0
export GCC_HOST_COMPILER_PATH="${GCC_HOST_COMPILER_PATH:-/usr/bin/gcc}"
export TF_NEED_MPI=0
export TF_SET_ANDROID_WORKSPACE=0
export CC_OPT_FLAGS="-march=native -Wno-sign-compare"
export PATH="$PATH:$INSTALL_DIR/bazel"
./configure

# Fix grpc (https://github.com/tensorflow/tensorflow/issues/33758#issuecomment-547867642)
git checkout -- tensorflow/workspace.bzl
rm -f third_party/grpc/backport-pr-18950.patch
git apply << EOM
From 73640aaec2ab0234d9fff138e3c9833695570c0a Mon Sep 17 00:00:00 2001
From: Hiroshi Ogawa <hi.ogawa.zz@gmail.com>
Date: Wed, 30 Oct 2019 15:06:40 +0900
Subject: [PATCH] Backport grpc PR 18950

---
 tensorflow/workspace.bzl                 |  1 +
 third_party/grpc/backport-pr-18950.patch | 66 ++++++++++++++++++++++++
 2 files changed, 67 insertions(+)
 create mode 100644 third_party/grpc/backport-pr-18950.patch

diff --git a/tensorflow/workspace.bzl b/tensorflow/workspace.bzl
index d361130b358c3..5c120709bc652 100755
--- a/tensorflow/workspace.bzl
+++ b/tensorflow/workspace.bzl
@@ -517,6 +517,7 @@ def tf_repositories(path_prefix = "", tf_repo_name = ""):
         sha256 = "67a6c26db56f345f7cee846e681db2c23f919eba46dd639b09462d1b6203d28c",
         strip_prefix = "grpc-4566c2a29ebec0835643b972eb99f4306c4234a3",
         system_build_file = clean_dep("//third_party/systemlibs:grpc.BUILD"),
+        patch_file = clean_dep("//third_party/grpc:backport-pr-18950.patch"),
         urls = [
             "https://storage.googleapis.com/mirror.tensorflow.org/github.com/grpc/grpc/archive/4566c2a29ebec0835643b972eb99f4306c4234a3.tar.gz",
             "https://github.com/grpc/grpc/archive/4566c2a29ebec0835643b972eb99f4306c4234a3.tar.gz",
diff --git a/third_party/grpc/backport-pr-18950.patch b/third_party/grpc/backport-pr-18950.patch
new file mode 100644
index 0000000000000..55290dfb40567
--- /dev/null
+++ b/third_party/grpc/backport-pr-18950.patch
@@ -0,0 +1,66 @@
+diff --git a/src/core/lib/gpr/log_linux.cc b/src/core/lib/gpr/log_linux.cc
+index 561276f0c2..8b597b4cf2 100644
+--- a/src/core/lib/gpr/log_linux.cc
++++ b/src/core/lib/gpr/log_linux.cc
+@@ -40,7 +40,7 @@
+ #include <time.h>
+ #include <unistd.h>
+
+-static long gettid(void) { return syscall(__NR_gettid); }
++static long sys_gettid(void) { return syscall(__NR_gettid); }
+
+ void gpr_log(const char* file, int line, gpr_log_severity severity,
+              const char* format, ...) {
+@@ -70,7 +70,7 @@ void gpr_default_log(gpr_log_func_args* args) {
+   gpr_timespec now = gpr_now(GPR_CLOCK_REALTIME);
+   struct tm tm;
+   static __thread long tid = 0;
+-  if (tid == 0) tid = gettid();
++  if (tid == 0) tid = sys_gettid();
+
+   timer = static_cast<time_t>(now.tv_sec);
+   final_slash = strrchr(args->file, '/');
+diff --git a/src/core/lib/gpr/log_posix.cc b/src/core/lib/gpr/log_posix.cc
+index b6edc14ab6..2f7c6ce376 100644
+--- a/src/core/lib/gpr/log_posix.cc
++++ b/src/core/lib/gpr/log_posix.cc
+@@ -31,7 +31,7 @@
+ #include <string.h>
+ #include <time.h>
+
+-static intptr_t gettid(void) { return (intptr_t)pthread_self(); }
++static intptr_t sys_gettid(void) { return (intptr_t)pthread_self(); }
+
+ void gpr_log(const char* file, int line, gpr_log_severity severity,
+              const char* format, ...) {
+@@ -86,7 +86,7 @@ void gpr_default_log(gpr_log_func_args* args) {
+   char* prefix;
+   gpr_asprintf(&prefix, "%s%s.%09d %7" PRIdPTR " %s:%d]",
+                gpr_log_severity_string(args->severity), time_buffer,
+-               (int)(now.tv_nsec), gettid(), display_file, args->line);
++               (int)(now.tv_nsec), sys_gettid(), display_file, args->line);
+
+   fprintf(stderr, "%-70s %s\n", prefix, args->message);
+   gpr_free(prefix);
+diff --git a/src/core/lib/iomgr/ev_epollex_linux.cc b/src/core/lib/iomgr/ev_epollex_linux.cc
+index b6d13b44d1..e1cda21b3e 100644
+--- a/src/core/lib/iomgr/ev_epollex_linux.cc
++++ b/src/core/lib/iomgr/ev_epollex_linux.cc
+@@ -1103,7 +1103,7 @@ static void end_worker(grpc_pollset* pollset, grpc_pollset_worker* worker,
+ }
+
+ #ifndef NDEBUG
+-static long gettid(void) { return syscall(__NR_gettid); }
++static long sys_gettid(void) { return syscall(__NR_gettid); }
+ #endif
+
+ /* pollset->mu lock must be held by the caller before calling this.
+@@ -1123,7 +1123,7 @@ static grpc_error* pollset_work(grpc_pollset* pollset,
+ #define WORKER_PTR (&worker)
+ #endif
+ #ifndef NDEBUG
+-  WORKER_PTR->originator = gettid();
++  WORKER_PTR->originator = sys_gettid();
+ #endif
+   if (grpc_polling_trace.enabled()) {
+     gpr_log(GPR_INFO,
EOM

# Fix install_headers (https://github.com/tensorflow/tensorflow/pull/36013)
git checkout -- tensorflow/core/BUILD
git apply << EOM
diff --git a/tensorflow/core/BUILD b/tensorflow/core/BUILD
index 422df45c79..d2dac95add 100644
--- a/tensorflow/core/BUILD
+++ b/tensorflow/core/BUILD
@@ -4481,7 +4481,7 @@ transitive_hdrs(
     name = "headers",
     visibility = ["//tensorflow:__subpackages__"],
     deps = [
-        ":core_cpu",
+        ":core",
         ":framework",
         ":lib",
         ":protos_all_cc",
EOM

# Build
bazel build --config=opt --config=noaws --config=nogcp --config=nohdfs --config=nonccl \
    --noincompatible_do_not_split_linking_cmdline \
    //tensorflow:libtensorflow_cc.so \
    //tensorflow:libtensorflow_framework.so \
    //tensorflow:install_headers

# Copy files
rm -rf "$INSTALL_DIR/tensorflow"
cp -av bazel-out/k8-opt/bin/tensorflow "$INSTALL_DIR/tensorflow"
