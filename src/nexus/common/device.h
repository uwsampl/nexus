#ifndef NEXUS_COMMON_DEVICE_H_
#define NEXUS_COMMON_DEVICE_H_

#include <algorithm>
#include <glog/logging.h>
#include <sstream>
#include <string>
#include <vector>

#ifdef USE_GPU
#include <cuda_runtime.h>
#endif

namespace nexus {

enum DeviceType {
  kCPU = 0,
  kGPU = 1,
};

class DeviceManager; // forward declare

class Device {
 public:
  virtual void* Allocate(size_t nbytes) = 0;

  virtual void Free(void* buf) = 0;

  virtual std::string name() const = 0;
  
  DeviceType type() const { return type_; }

  bool operator==(const Device& other) const {
    return name() == other.name();
  }

 protected:
  Device(DeviceType type) : type_(type) {}
  // disable copy
  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;

 private:
  DeviceType type_;
};

class CPUDevice : public Device {
 public:
  void* Allocate(size_t nbytes) final {
    void* buf = malloc(nbytes);
    return buf;
  }

  void Free(void* buf) final {
    free(buf);
  }

  std::string name() const final { return "cpu"; }

 private:
  CPUDevice() : Device(kCPU) {}
  friend class DeviceManager;
};

#ifdef USE_GPU

#define NEXUS_CUDA_CHECK(condition)                              \
  do {                                                          \
    cudaError_t err = (condition);                              \
    CHECK_EQ(err, cudaSuccess) << cudaGetErrorString(err);      \
  } while (0)

class GPUDevice : public Device {
 public:
  int gpu_id() const { return gpu_id_; }

  void* Allocate(size_t nbytes) final {
    void* buf;
    NEXUS_CUDA_CHECK(cudaSetDevice(gpu_id_));
    cudaError_t err = cudaMalloc(&buf, nbytes);
    if (err != cudaSuccess) {
      throw cudaGetErrorString(err);
    }
    return buf;
  }

  void Free(void* buf) final {
    NEXUS_CUDA_CHECK(cudaFree(buf));
  }

  std::string name() const final { return name_; }

  std::string device_name() { return device_name_; }

  size_t FreeMemory() const {
    size_t free_mem, total_mem;
    NEXUS_CUDA_CHECK(cudaSetDevice(gpu_id_));
    NEXUS_CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    return free_mem;
  }

  size_t TotalMemory() const { return total_memory_; }

private:
  GPUDevice(int gpu_id) :
      Device(kGPU), gpu_id_(gpu_id) {
    std::stringstream ss;
    ss << "gpu:" << gpu_id;
    name_ = ss.str();
    cudaDeviceProp prop;
    NEXUS_CUDA_CHECK(cudaSetDevice(gpu_id_));
    NEXUS_CUDA_CHECK(cudaGetDeviceProperties(&prop, gpu_id_));
    device_name_.assign(prop.name, strlen(prop.name));
    std::replace(device_name_.begin(), device_name_.end(), ' ', '_');
    total_memory_ = prop.totalGlobalMem;
    LOG(INFO) << "GPU " << gpu_id << " " << device_name_ << ": total memory " <<
        total_memory_ / 1024. / 1024. / 1024. << "GB";
  }
  friend class DeviceManager;

 private:
  int gpu_id_;
  std::string name_;
  std::string device_name_;
  size_t total_memory_;
};

#endif

class DeviceManager {
 public:
  static DeviceManager& Singleton() {
    static DeviceManager device_manager;
    return device_manager;
  }

  CPUDevice* GetCPUDevice() const {
    return cpu_device_;
  }

#ifdef USE_GPU
  GPUDevice* GetGPUDevice(int gpu_id) const {
    CHECK_LT(gpu_id, gpu_devices_.size()) << "GPU id " << gpu_id <<
        " exceeds number of GPU devices (" << gpu_devices_.size() << ")";
    return gpu_devices_[gpu_id];
  }
#endif

 private:
  DeviceManager() {
    cpu_device_ = new CPUDevice();
    int gpu_count;
#ifdef USE_GPU
    NEXUS_CUDA_CHECK(cudaGetDeviceCount(&gpu_count));
    for (int i = 0; i < gpu_count; ++i) {
      gpu_devices_.push_back(new GPUDevice(i));
    }
#endif
  }

  CPUDevice* cpu_device_;
#ifdef USE_GPU
  std::vector<GPUDevice*> gpu_devices_;
#endif
};

} // namespec nexus

#endif // NEXUS_COMMON_DEVICE_H_
