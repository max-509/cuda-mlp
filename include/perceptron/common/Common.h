#ifndef PERCEPTRON_COMMON_COMMON_H
#define PERCEPTRON_COMMON_COMMON_H

#ifdef __CUDACC__
#define DEVICE_CALLABLE __host__ __device__
#else
#define DEVICE_CALLABLE
#endif

#include <type_traits>
#include <memory>
#include <cstddef>
#include <algorithm>
#include <functional>
#include <cstdio>
#include <stdexcept>

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

using size_type = std::ptrdiff_t;

template<template<typename...> class Template, typename T>
struct is_instantiation_of : std::false_type {};

template<template<typename...> class Template, typename... Args>
struct is_instantiation_of<Template, Template<Args...> > : std::true_type {};

template<template<typename...> class Template, typename... Args>
static constexpr auto is_instantiation_of_v = is_instantiation_of<Template, Template<Args...>>::value;

template<typename T>
struct ptr_type_extract {
  using type = T;
};

template<typename T, typename Deleter>
struct ptr_type_extract<std::unique_ptr<T, Deleter>> {
  using type = typename std::unique_ptr<T, Deleter>::element_type;
};

template<typename T>
struct ptr_type_extract<std::shared_ptr<T>> {
  using type = typename std::shared_ptr<T>::element_type;
};

template<typename T>
using ptr_type_extract_t = typename ptr_type_extract<T>::type;

template<typename T>
struct is_float_or_double : std::integral_constant<
    bool,
    std::is_same<T, float>::value || std::is_same<T, double>::value> {
};

template<typename T>
static constexpr auto is_float_or_double_v = is_float_or_double<T>::value;

template<typename DataType, typename TimeType>
static constexpr auto
    is_float_or_double_data_time_types_v = is_float_or_double<DataType>::value && is_float_or_double<TimeType>::value;

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

template<typename T>
constexpr void is_valid_type() {
  static_assert(is_float_or_double_v<T>, "Tensor types for MLP ops must be a float or double");
}

} // perceptron

#endif //PERCEPTRON_COMMON_COMMON_H
