#include "perceptron/activations/SigmoidActivation.h"

namespace perceptron {
namespace activations {

tensors::TensorOwnerDevice2D<float>
SigmoidActivation::compute(tensors::TensorReadOnly2D<float, false> inputs) {
  return compute_impl(inputs);
}

tensors::TensorOwnerDevice2D<float>
SigmoidActivation::compute(tensors::TensorReadOnly2D<float, true> inputs) {
  return compute_impl(inputs);
}

tensors::TensorOwnerDevice2D<float>
SigmoidActivation::derivative(tensors::TensorReadOnly2D<float, false> inputs) {
  return derivative_impl(inputs);
}

tensors::TensorOwnerDevice2D<float>
SigmoidActivation::derivative(tensors::TensorReadOnly2D<float, true> inputs) {
  return derivative_impl(inputs);
}

} // perceptron
} // activations