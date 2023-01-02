#ifndef PERCEPTRON_TENSORS_TENSORGETTER_H
#define PERCEPTRON_TENSORS_TENSORGETTER_H

#include "perceptron/common/Common.h"

namespace perceptron {
namespace tensors {

template<typename T, bool is_transposed = false>
DEVICE_CALLABLE
T *
get_elem(T *ptr, size_type y, size_type x, size_type stride) {
  if constexpr(is_transposed) {
    return ptr + y + (stride * x);
  } else {
    return ptr + (y * stride) + x;
  }
}

template<typename T>
DEVICE_CALLABLE
T *
get_elem(T *ptr, size_type x, size_type stride) {
  return ptr + x * stride;
}

} // perceptron
} // tensors

#endif //PERCEPTRON_TENSORS_TENSORGETTER_H
