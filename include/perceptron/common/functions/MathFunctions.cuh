#ifndef PERCEPTRON_COMMON_FUNCTIONS_MATHFUNCTIONS_H
#define PERCEPTRON_COMMON_FUNCTIONS_MATHFUNCTIONS_H

#include "perceptron/common/Common.h"

#include <cmath>
#include <type_traits>

namespace perceptron {
namespace common {
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
pow(T val) {
  if constexpr (std::is_same_v<T, float>) {
    return ::powf(val);
  } else {
    return ::pow(val);
  }
}

} // perceptron
} // common
} // functions

#endif //PERCEPTRON_COMMON_FUNCTIONS_MATHFUNCTIONS_H
