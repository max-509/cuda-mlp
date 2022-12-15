#include "perceptron/activations/ReLUActivation.h"
#include "perceptron/tensors/Tensor2D.h"

namespace perceptron {
namespace activations {
namespace details {

// Reason for non-private methods: nvcc error
// The enclosing parent function ("compute_impl") for an extended __device__ lambda cannot have private or protected access within its class
template<typename T, bool trans>
static tensors::TensorOwnerDevice2D<T>
compute_impl(tensors::TensorReadOnly2D<T, trans> inputs) {
  auto output_owner = tensors::constructTensorOwnerDevice2D<T>(inputs.get_y_dim(),
                                                               inputs.get_x_dim());
  tensors::ops::copy(inputs, output_owner.tensor_view());
  tensors::ops::copy_if(inputs, [] DEVICE_CALLABLE(T value) { return value > 0; }, output_owner.tensor_view());

  return output_owner;
}

template<typename T, bool trans>
static tensors::TensorOwnerDevice2D<T>
derivative_impl(tensors::TensorReadOnly2D<T, trans> inputs) {
  auto output_owner = tensors::constructTensorOwnerDevice2D<T>(inputs.get_y_dim(),
                                                               inputs.get_x_dim());

  tensors::ops::copy(inputs, output_owner.tensor_view());
  tensors::ops::set_if(static_cast<T>(1.0), [] DEVICE_CALLABLE(T value) { return value > 0; }, output_owner.tensor_view());

  return output_owner;
}
} // detaiils

tensors::TensorOwnerDevice2D<float>
activations::ReLUActivation::compute(tensors::TensorReadOnly2D<float, false> inputs) {
  return details::compute_impl(inputs);
}

tensors::TensorOwnerDevice2D<float>
activations::ReLUActivation::compute(tensors::TensorReadOnly2D<float, true> inputs) {
  return details::compute_impl(inputs);
}

tensors::TensorOwnerDevice2D<float>
activations::ReLUActivation::derivative(tensors::TensorReadOnly2D<float, false> inputs) {
  return details::derivative_impl(inputs);
}

tensors::TensorOwnerDevice2D<float>
activations::ReLUActivation::derivative(tensors::TensorReadOnly2D<float, true> inputs) {
  return details::derivative_impl(inputs);
}

} // perceptron
} // activations