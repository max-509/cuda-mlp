#include "perceptron/activations/IdentityActivation.h"

namespace perceptron {
namespace activations {

tensors::TensorOwnerDevice2D<float>
IdentityActivation::compute(tensors::TensorReadOnly2D<float, false> inputs) {
  return compute_impl(inputs);
}

tensors::TensorOwnerDevice2D<float>
IdentityActivation::compute(tensors::TensorReadOnly2D<float, true> inputs) {
  return compute_impl(inputs);
}

tensors::TensorOwnerDevice2D<float>
IdentityActivation::derivative(tensors::TensorReadOnly2D<float, false> inputs) {
  return derivative_impl(inputs);
}

tensors::TensorOwnerDevice2D<float>
IdentityActivation::derivative(tensors::TensorReadOnly2D<float, true> inputs) {
  return derivative_impl(inputs);
}

} // perceptron
} // activations