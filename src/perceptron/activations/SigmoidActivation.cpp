#include "perceptron/activations/SigmoidActivation.h"

namespace perceptron {
namespace activations {

tensors::TensorOwnerDevice2D<float>
SigmoidActivation::compute(tensors::TensorReadOnly2D<float, false> inputs) {
  return perceptron::tensors::TensorOwnerDevice2D<float>(std::unique_ptr(), 0, 0, 0);
}

tensors::TensorOwnerDevice2D<float>
SigmoidActivation::compute(tensors::TensorReadOnly2D<float, true> inputs) {
  return perceptron::tensors::TensorOwnerDevice2D<float>(std::unique_ptr(), 0, 0, 0);
}

tensors::TensorOwnerDevice2D<float>
SigmoidActivation::derivative(tensors::TensorReadOnly2D<float, false> inputs) {
  return perceptron::tensors::TensorOwnerDevice2D<float>(std::unique_ptr(), 0, 0, 0);
}

tensors::TensorOwnerDevice2D<float>
SigmoidActivation::derivative(tensors::TensorReadOnly2D<float, true> inputs) {
  return perceptron::tensors::TensorOwnerDevice2D<float>(std::unique_ptr(), 0, 0, 0);
}

} // perceptron
} // activations