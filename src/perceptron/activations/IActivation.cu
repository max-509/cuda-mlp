#include "perceptron/activations/IActivation.h"

namespace perceptron {
namespace activations {

tensors::TensorOwner2D<float>
IActivation::compute(tensors::TensorReadOnly2D<float, false> inputs) {
  auto outputs_owner = tensors::constructTensorOwnerDevice2D<float>(inputs.get_nrows(), inputs.get_ncols());
  compute(inputs, outputs_owner.tensor_view());
  return outputs_owner;
}

tensors::TensorOwner2D<float>
IActivation::compute(tensors::TensorReadOnly2D<float, true> inputs) {
  auto outputs_owner = tensors::constructTensorOwnerDevice2D<float>(inputs.get_nrows(), inputs.get_ncols());
  compute(inputs, outputs_owner.tensor_view());
  return outputs_owner;
}

tensors::TensorOwner2D<float>
IActivation::derivative(tensors::TensorReadOnly2D<float, false> inputs) {
  auto outputs_owner = tensors::constructTensorOwnerDevice2D<float>(inputs.get_nrows(), inputs.get_ncols());
  derivative(inputs, outputs_owner.tensor_view());
  return outputs_owner;
}

tensors::TensorOwner2D<float>
IActivation::derivative(tensors::TensorReadOnly2D<float, true> inputs) {
  auto outputs_owner = tensors::constructTensorOwnerDevice2D<float>(inputs.get_nrows(), inputs.get_ncols());
  derivative(inputs, outputs_owner.tensor_view());
  return outputs_owner;
}

} // perceptron
} // activations
