#include "nexus/common/device.h"
#include <glog/logging.h>

namespace nexus {

#ifdef USE_GPU
GPUDevice::GPUDevice(int gpu_id) :
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

void *GPUDevice::Allocate(size_t nbytes) {
    void* buf;
    NEXUS_CUDA_CHECK(cudaSetDevice(gpu_id_));
    cudaError_t err = cudaMalloc(&buf, nbytes);
    if (err != cudaSuccess) {
        throw cudaGetErrorString(err);
    }
    return buf;
}

size_t GPUDevice::FreeMemory() const {
    size_t free_mem, total_mem;
    NEXUS_CUDA_CHECK(cudaSetDevice(gpu_id_));
    NEXUS_CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    return free_mem;
}

void GPUDevice::Free(void *buf) {
    NEXUS_CUDA_CHECK(cudaFree(buf));
}


GPUDevice *DeviceManager::GetGPUDevice(int gpu_id) const {
    CHECK_LT(gpu_id, gpu_devices_.size()) << "GPU id " << gpu_id <<
                                          " exceeds number of GPU devices (" << gpu_devices_.size() << ")";
    return gpu_devices_[gpu_id];
}
#endif

DeviceManager::DeviceManager() {
    cpu_device_ = new CPUDevice();
    int gpu_count;
#ifdef USE_GPU
    NEXUS_CUDA_CHECK(cudaGetDeviceCount(&gpu_count));
#endif
    for (int i = 0; i < gpu_count; ++i) {
        gpu_devices_.push_back(new GPUDevice(i));
    }
}
}
