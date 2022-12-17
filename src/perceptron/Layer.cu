#include "perceptron/Layer.h"

#include "perceptron/common/utils/CurandUtils.h"
#include "perceptron/tensors/ops/MathOps.h"

#include <cmath>
#include <optional>

namespace perceptron {
namespace layer {

Layer::Layer(size_type input_layer, size_type n_neurons,
             std::unique_ptr<activations::IActivation> activation,
             std::unique_ptr<optimizers::IOptimizer> optimizer,
             const std::unordered_map<std::string, std::any> &config,
             size_type seed,
             cudaStream_t stream)
    : m_activation(std::move(activation)), m_weights_optimizer(std::move(optimizer)),
      m_weights(tensors::constructTensorOwnerDevice2D<float>(input_layer, n_neurons)),
      m_bias(std::nullopt) {
  auto xavier_lim = std::sqrt(6.0) / std::sqrt(input_layer + n_neurons);

  tensors::ops::generate(utils::curand_uniform_tag{-xavier_lim, xavier_lim},
                         m_weights.tensor_view(), seed, stream);

  if (auto bias_it = config.find("with_bias"); bias_it != config.end() && std::any_cast<bool>(bias_it->second)) {
    // TODO: optimizer for bias
    m_bias = tensors::constructTensorOwnerDevice1D<float>(n_neurons);
    tensors::ops::generate(utils::curand_uniform_tag{-xavier_lim, xavier_lim},
                           m_bias->tensor_view().to_2d(), seed, stream);
  }
}

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

} // perceptron
} // layer