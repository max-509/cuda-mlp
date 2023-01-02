#ifndef PERCEPTRON_COMMON_FUNCTIONS_MATHFUNCTIONS_H
#define PERCEPTRON_COMMON_FUNCTIONS_MATHFUNCTIONS_H

#include "perceptron/common/Common.h"

#include <cmath>
#include <type_traits>

namespace perceptron {
namespace functions {

template<typename T>
DEVICE_CALLABLE
T
exp(T val) {
  if constexpr (std::is_same_v<T, float>) {
    return ::expf(val);
  } else {
    return ::exp(val);
  }
}

template<typename T>
DEVICE_CALLABLE
T
abs(T val) {
  if constexpr (std::is_same_v<T, float>) {
    return ::fabs(val);
  } else {
    return ::abs(val);
  }
}

template<typename T>
DEVICE_CALLABLE
T
pow(T b, T e) {
  if constexpr (std::is_same_v<T, float>) {
    return ::powf(b, e);
  } else {
    return ::pow(b, e);
  }
}

template<typename T>
__device__
T
uniform(curandState_t *state, T a, T b) {
  if constexpr (std::is_same_v<T, float>) {
    return (b - a) * curand_uniform(state) + a;
  } else {
    return (b - a) * curand_uniform_double(state) + a;
  }
}

template<typename T>
__device__
T
uniform(curandState_t *state) {
  if constexpr (std::is_same_v<T, float>) {
    return curand_uniform(state);
  } else {
    return curand_uniform_double(state);
  }
}

template<typename T>
__device__
T
normal(curandState_t *state, T mean, T stddev) {
  if constexpr (std::is_same_v<T, float>) {
    return mean + stddev * curand_normal(state);
  } else {
    return mean + stddev * curand_normal_double(state);
  }
}

template<typename T>
__device__
T
normal(curandState_t *state) {
  if constexpr (std::is_same_v<T, float>) {
    return curand_normal(state);
  } else {
    return curand_normal_double(state);
  }
}

template<typename T>
__device__
T
log_normal(curandState_t *state, T mean = static_cast<T>(0.0), T stddev = static_cast<T>(1.0)) {
  if constexpr (std::is_same_v<T, float>) {
    return curand_log_normal(state, mean, stddev);
  } else {
    return curand_log_normal_double(state, mean, stddev);
  }
}

} // perceptron
} // functions

#endif //PERCEPTRON_COMMON_FUNCTIONS_MATHFUNCTIONS_H
