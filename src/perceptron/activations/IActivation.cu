#include "perceptron/activations/IActivation.h"

namespace perceptron {
namespace activations {

tensors::TensorOwnerDevice2D<float>
IActivation::compute(tensors::TensorReadOnly2D<float, false> inputs) {
  auto outputs_owner = tensors::constructTensorOwnerDevice2D<float>(inputs.get_y_dim(), inputs.get_x_dim());
  compute(inputs, outputs_owner.tensor_view());
  return outputs_owner;
}

tensors::TensorOwnerDevice2D<float>
IActivation::compute(tensors::TensorReadOnly2D<float, true> inputs) {
  auto outputs_owner = tensors::constructTensorOwnerDevice2D<float>(inputs.get_y_dim(), inputs.get_x_dim());
  compute(inputs, outputs_owner.tensor_view());
  return outputs_owner;
}

tensors::TensorOwnerDevice2D<float>
IActivation::derivative(tensors::TensorReadOnly2D<float, false> inputs) {
  auto outputs_owner = tensors::constructTensorOwnerDevice2D<float>(inputs.get_y_dim(), inputs.get_x_dim());
  compute(inputs, outputs_owner.tensor_view());
  return outputs_owner;
}

tensors::TensorOwnerDevice2D<float>
IActivation::derivative(tensors::TensorReadOnly2D<float, true> inputs) {
  auto outputs_owner = tensors::constructTensorOwnerDevice2D<float>(inputs.get_y_dim(), inputs.get_x_dim());
  compute(inputs, outputs_owner.tensor_view());
  return outputs_owner;
}

} // perceptron
} // activations
