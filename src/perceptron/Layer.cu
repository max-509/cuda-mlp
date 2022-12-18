#include "perceptron/Layer.h"

#include "perceptron/common/utils/CurandUtils.h"
#include "perceptron/tensors/ops/MathOps.h"

#include <cmath>
#include <optional>

namespace perceptron {
namespace layer {

bool
Layer::with_bias() const {
  return m_bias.has_value();
}

tensors::TensorReadOnly2D<float, false>
Layer::weights() const {
  return m_weights.tensor_view();
}

tensors::TensorReadOnly2D<float, false>
Layer::active_neurons() const {
  if (!m_active_neurons.has_value()) {
    throw std::logic_error{"Active neurons must be a initialized by transform_impl() method"};
  }
  return m_active_neurons->tensor_view();
}

void
Layer::transform(tensors::TensorReadOnly2D<float, false> inputs) {
  transform_impl(inputs);
}

void
Layer::transform(tensors::TensorReadOnly2D<float, true> inputs) {
  transform_impl(inputs);
}

void
Layer::update_gradients(tensors::TensorReadOnly2D<float, false> forwards_transposed,
                        tensors::TensorReadOnly2D<float, false> errors) {
  update_gradients_impl(forwards_transposed, errors);
}

void
Layer::update_gradients(tensors::TensorReadOnly2D<float, false> forwards_transposed,
                        tensors::TensorReadOnly2D<float, true> errors) {
  update_gradients_impl(forwards_transposed, errors);
}

void
Layer::update_gradients(tensors::TensorReadOnly2D<float, true> forwards_transposed,
                        tensors::TensorReadOnly2D<float, false> errors) {
  update_gradients_impl(forwards_transposed, errors);
}

void
Layer::update_gradients(tensors::TensorReadOnly2D<float, true> forwards_transposed,
                        tensors::TensorReadOnly2D<float, true> errors) {
  update_gradients_impl(forwards_transposed, errors);
}

tensors::TensorOwnerDevice2D<float>
Layer::compute_errors(tensors::TensorReadOnly2D<float, false> next_layer_errors,
                      tensors::TensorReadOnly2D<float, false> next_layer_weights) {
  return compute_errors_impl(next_layer_errors, next_layer_weights);
}

tensors::TensorOwnerDevice2D<float>
Layer::compute_errors(tensors::TensorReadOnly2D<float, false> next_layer_errors,
                      tensors::TensorReadOnly2D<float, true> next_layer_weights) {
  return compute_errors_impl(next_layer_errors, next_layer_weights);
}

tensors::TensorOwnerDevice2D<float>
Layer::compute_errors(tensors::TensorReadOnly2D<float, true> next_layer_errors,
                      tensors::TensorReadOnly2D<float, false> next_layer_weights) {
  return compute_errors_impl(next_layer_errors, next_layer_weights);
}

tensors::TensorOwnerDevice2D<float>
Layer::compute_errors(tensors::TensorReadOnly2D<float, true> next_layer_errors,
                      tensors::TensorReadOnly2D<float, true> next_layer_weights) {
  return compute_errors_impl(next_layer_errors, next_layer_weights);
}

tensors::TensorOwnerDevice2D<float>
Layer::compute_errors(tensors::TensorReadOnly2D<float, false> backward_errors) {
  return compute_errors_impl(backward_errors);
}

tensors::TensorOwnerDevice2D<float>
Layer::compute_errors(tensors::TensorReadOnly2D<float, true> backward_errors) {
  return compute_errors_impl(backward_errors);
}

void
Layer::compute_errors(tensors::TensorReadOnly2D<float, false> backward_errors,
                      tensors::TensorWriteable2D<float> outputs) {
  compute_errors_impl(backward_errors, outputs);
}

void
Layer::compute_errors(tensors::TensorReadOnly2D<float, true> backward_errors,
                      tensors::TensorWriteable2D<float> outputs) {
  compute_errors_impl(backward_errors, outputs);
}

void
Layer::compute_errors(tensors::TensorWriteable2D<float> backward_errors) {
  compute_errors_impl(backward_errors);
}

void
Layer::compute_errors_impl(tensors::TensorWriteable2D<float> backward_errors) {
  auto act_der_owner = m_activation->derivative(m_neurons->tensor_view().to_read_only());
  tensors::ops::element_wise_mul(act_der_owner.tensor_view().to_read_only(),
                                 backward_errors);
}

} // perceptron
} // layer