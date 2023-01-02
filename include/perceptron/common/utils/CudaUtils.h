#ifndef PERCEPTRON_COMMON_UTILS_CUDAUTILS_H
#define PERCEPTRON_COMMON_UTILS_CUDAUTILS_H

#include "perceptron/common/Common.h"

#include <type_traits>
#include <cstdio>
#include <stdexcept>
#include <algorithm>
#include <functional>

#include <cuda_runtime.h>

#define CUDA_CHECK(err)                                                                     \
  do                                                                                        \
  {                                                                                         \
    cudaError_t err__ = (err);                                                              \
    if (err__ != cudaSuccess)                                                               \
    {                                                                                       \
      std::fprintf(stderr, "cuda error %s at %s:%d with message: %s\n",                     \
                   cudaGetErrorName(err__), __FILE__, __LINE__, cudaGetErrorString(err__)); \
      throw std::runtime_error(cudaGetErrorString(err__));                                  \
    }                                                                                       \
  } while (0)

namespace perceptron {
namespace utils {

inline constexpr size_type DEFAULT_BLOCK_SIZE_1D = 256;
inline constexpr size_type DEFAULT_BLOCK_SIZE_2D = 16;

size_type
block_size_by_threads(size_type size, size_type n_threads);

template<class T>
DEVICE_CALLABLE
constexpr T &&
device_forward(typename std::remove_reference_t<T> &t) noexcept {
  return static_cast<T &&>(t);
}

template<class T>
DEVICE_CALLABLE
constexpr T &&
device_forward(typename std::remove_reference_t<T> &&t) noexcept {
  static_assert(!std::is_lvalue_reference<T>::value,
                "Can not forward an rvalue as an lvalue.");
  return static_cast<T &&>(t);
}

} // perceptron
} // utils

#endif //PERCEPTRON_COMMON_UTILS_CUDAUTILS_H
