#ifndef NEXUS_COMMON_DEVICE_H_
#define NEXUS_COMMON_DEVICE_H_

#include <algorithm>
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

  void* Allocate(size_t nbytes) final;

  void Free(void* buf) final;

  std::string name() const final { return name_; }

  std::string device_name() { return device_name_; }

  size_t FreeMemory() const;

  size_t TotalMemory() const { return total_memory_; }

private:
  explicit GPUDevice(int gpu_id);
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
  GPUDevice* GetGPUDevice(int gpu_id) const;
#endif

 private:
  DeviceManager();

  CPUDevice* cpu_device_;
#ifdef USE_GPU
  std::vector<GPUDevice*> gpu_devices_;
#endif
};

} // namespec nexus

#endif // NEXUS_COMMON_DEVICE_H_
