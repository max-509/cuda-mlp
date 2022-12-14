#ifndef PERCEPTRON_TENSORS_TENSORHELPER_H
#define PERCEPTRON_TENSORS_TENSORHELPER_H

#include "perceptron/common/Common.h"

namespace perceptron {
namespace tensors {
namespace details {

template<typename T>
struct TensorHelper final {
  using value_type = ptr_type_extract_t<T>;
  using pointer_type = value_type *;
  using reference_type = value_type &;
  using creference_type = const value_type &;
  using vref_type = std::conditional_t<std::is_const<value_type>::value, creference_type, reference_type>;
};

} // perceptron
} // tensors
} // details

#endif //PERCEPTRON_TENSORS_TENSORHELPER_H
