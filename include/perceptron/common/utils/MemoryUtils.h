#ifndef PERCEPTRON_COMMON_UTILS_MEMORYUTILS_H
#define PERCEPTRON_COMMON_UTILS_MEMORYUTILS_H

#include "perceptron/common/Common.h"

#include <cuda_runtime.h>

namespace perceptron {
namespace utils {

struct cu_memory_deleter {
  void
  operator()(void *ptr) const {
    CUDA_CHECK(cudaFree(ptr));
  }
};

struct cu_pinned_deleter {
  void
  operator()(void *ptr) const {
    CUDA_CHECK(cudaHostUnregister(ptr));
  }
};

struct cu_host_deleter {
  void
  operator()(void *ptr) const {
    CUDA_CHECK(cudaFreeHost(ptr));
  }
};

template<typename T>
using CudaHostOwner = std::unique_ptr<T, cu_host_deleter>;

template<typename T>
using CudaDeviceOwner = std::unique_ptr<T, cu_memory_deleter>;

template<typename T>
using CudaPinnedOwner = std::unique_ptr<T, cu_pinned_deleter>;

template<typename T>
CudaDeviceOwner<T>
cu_make_memory_unique(std::size_t size) {
  T *ptr = nullptr;
  CUDA_CHECK(cudaMalloc(&ptr, size * sizeof(T)));

  return CudaDeviceOwner<T>(ptr, cu_memory_deleter{});
}

template<typename T>
CudaDeviceOwner<T>
cu_make_memory_unique() {
  return cu_make_memory_unique<T>(1);
}

template<typename T>
CudaDeviceOwner<T>
cu_make_pitched_memory_unique(std::size_t nrows,
                              std::size_t ncols,
                              std::size_t &pitch) {
  T *ptr = nullptr;
  CUDA_CHECK(cudaMallocPitch(&ptr, &pitch, sizeof(T) * ncols, nrows));
  pitch /= sizeof(T);

  return CudaDeviceOwner<T>(ptr, cu_memory_deleter{});
}

template<typename T>
CudaPinnedOwner<T>
cu_make_pinned_memory_unique(T *ptr, const std::size_t size) {
  CUDA_CHECK(cudaHostRegister(ptr, sizeof(T) * size, cudaHostRegisterDefault));

  return CudaPinnedOwner<T>{ptr, cu_pinned_deleter{}};
}

template<typename T>
CudaPinnedOwner<T>
cu_make_pinned_memory_unique(T *ptr) {
  return cu_make_pinned_memory_unique(ptr, 1);
}

template<typename T>
CudaHostOwner<T>
cu_make_host_memory_unique(const std::size_t size) {
  T *ptr = nullptr;
  CUDA_CHECK(cudaHostAlloc(&ptr, sizeof(T) * size, cudaHostAllocDefault));

  return CudaHostOwner<T>{ptr, cu_host_deleter{}};
}

template<typename T>
CudaHostOwner<T>
cu_make_host_memory_unique() {
  return cu_make_host_memory_unique<T>(1);
}

template<typename T>
void
cu_memcpy_async(T *dst,
                const T *src,
                size_t count,
                enum cudaMemcpyKind kind,
                cudaStream_t stream = (cudaStream_t) nullptr) {
  CUDA_CHECK(cudaMemcpyAsync(dst, src, sizeof(T) * count, kind, stream));
}

template<typename T>
void
cu_memcpy2D_async(T *dst,
                  size_t dpitch,
                  const T *src,
                  size_t spitch,
                  size_t width,
                  size_t height,
                  enum cudaMemcpyKind kind,
                  cudaStream_t stream = (cudaStream_t) nullptr) {
  CUDA_CHECK(cudaMemcpy2DAsync(dst, sizeof(T) * dpitch, src, sizeof(T) * spitch, sizeof(T) * width, height, kind, stream));
}

template<typename T>
void
cu_memset_async(T *devPtr, int value, size_t count, cudaStream_t stream = (cudaStream_t) nullptr) {
  CUDA_CHECK(cudaMemsetAsync(devPtr, value, sizeof(T) * count, stream));
}

template<typename T>
void
cu_memset2D_async(T *devPtr,
                  size_t pitch,
                  int value,
                  size_t width,
                  size_t height,
                  cudaStream_t stream = (cudaStream_t) nullptr) {
  CUDA_CHECK(cudaMemset2DAsync(devPtr, sizeof(T) * pitch, value, sizeof(T) * width, height, stream));
}

} // perceptron
} // utils

#endif //PERCEPTRON_COMMON_UTILS_MEMORYUTILS_H
