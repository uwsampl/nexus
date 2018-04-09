#include <arpa/inet.h>
#include <cuda_runtime.h>
#include <glog/logging.h>
#include <ifaddrs.h>
#include <netinet/in.h>
#include <sstream>

#include "nexus/common/util.h"

namespace nexus {

void SplitString(const std::string& str, char delim,
                 std::vector<std::string>* tokens) {
  std::stringstream ss;
  ss.str(str);
  std::string token;
  tokens->clear();
  while (std::getline(ss, token, delim)) {
    tokens->push_back(token);
  }
}

void Memcpy(void* dst, const Device* dst_device, const void* src,
            const Device* src_device, size_t nbytes) {
  if (dst == src && *dst_device == *src_device) {
    return;
  }
  DeviceType dst_type = dst_device->type();
  DeviceType src_type = src_device->type();
  if (dst_type == kCPU) {
    if (src_type == kCPU) {
      memcpy(dst, src, nbytes);
    } else { // src_type == kGPU
      NEXUS_CUDA_CHECK(cudaMemcpy(dst, src, nbytes, cudaMemcpyDeviceToHost));
    }
  } else { // dst_type == kGPU
    if (src_type == kCPU) {
      NEXUS_CUDA_CHECK(cudaMemcpy(dst, src, nbytes, cudaMemcpyHostToDevice));
    } else { // src_type == kGPU
      NEXUS_CUDA_CHECK(cudaMemcpy(dst, src, nbytes, cudaMemcpyDeviceToDevice));
    }
  }
}

namespace {
/*! \brief the list of all IPv4 addresses */
std::vector<in_addr> Ipv4Interfaces;
} // namespace

void ListIpv4Address() {
  if (Ipv4Interfaces.size() > 0) {
    Ipv4Interfaces.clear();
  }
  struct ifaddrs* ifAddrStruct = nullptr;
  struct ifaddrs* ifa = nullptr;
  // get network interface addresses
  getifaddrs(&ifAddrStruct);
  // iterate over all addresses
  for (ifa = ifAddrStruct; ifa != nullptr; ifa = ifa->ifa_next) {
    if (!ifa->ifa_addr) {
      continue;
    }
    if (ifa->ifa_addr->sa_family == AF_INET) {
      // IPv4 Address
      in_addr* addr = &((sockaddr_in*) ifa->ifa_addr)->sin_addr;
      Ipv4Interfaces.push_back(*addr);
    } else if (ifa->ifa_addr->sa_family == AF_INET6) {
      continue;
      // IPv6 Address
      /*in6_addr* addr = &((sockaddr_in6*) ifa->ifa_addr)->sin6_addr;
      char ipv6[INET6_ADDRSTRLEN];
      inet_ntop(AF_INET6, addr, ipv6, INET6_ADDRSTRLEN);
      //printf("%s IP Address %s\n", ifa->ifa_name, ipv6);
      ret = std::string(ipv6);*/
    }
  }
  if (ifAddrStruct != nullptr) {
    freeifaddrs(ifAddrStruct);
  }
}

void ConvertPrefix(const std::string& prefix, uint32_t* addr, uint32_t* mask) {
  char *pref = new char[prefix.length() + 1];
  strcpy(pref, prefix.c_str());
  char *pch = strchr(pref, '/');
  if (pch == nullptr) {
    *mask = 0xffffffff;
  } else {
    *pch = 0;
    ++pch;
    int prefix_len = atoi(pch);
    if (prefix_len > 32 || prefix_len < 0) {
      LOG(FATAL) << "Wrong prefix length: " << prefix_len;
    }
    *mask = ~(uint32_t)((1 << (32 - prefix_len)) - 1);
  }
  uint32_t prefix_addr = 0;
  pch = strtok(pref, ".");
  while (pch != nullptr) {
    prefix_addr = (prefix_addr << 8) | (uint8_t) atoi(pch);
    pch = strtok(NULL, ".");
  }
  *addr = prefix_addr & *mask;
  delete[] pref;
}

std::string GetIpAddress(const std::string& prefix) {
  if (Ipv4Interfaces.empty()) {
    ListIpv4Address();
  }
  uint32_t prefix_addr;
  uint32_t prefix_mask;
  ConvertPrefix(prefix, &prefix_addr, &prefix_mask);
  for (size_t i = 0; i < Ipv4Interfaces.size(); ++i) {
    const in_addr* addr = &Ipv4Interfaces[i];
    if ((ntohl(addr->s_addr) & prefix_mask) == prefix_addr) {
      char addr_str[INET_ADDRSTRLEN];
      inet_ntop(AF_INET, addr, addr_str, INET_ADDRSTRLEN);
      return std::string(addr_str);
    }
  }
  return "";
}

} // namespace nexus
