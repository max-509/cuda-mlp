#ifndef PERCEPTRON_LAYER_H
#define PERCEPTRON_LAYER_H

#include "perceptron/common/Common.h"
#include "perceptron/activations/IActivation.h"
#include "perceptron/optimizers//IOptimizer.h"
#include "perceptron/tensors/Tensor2D.h"
#include "perceptron/tensors/Tensor1D.h"
#include "perceptron/tensors/ops/MathOps.h"

#include <cassert>
#include <unordered_map>
#include <string>
#include <any>
#include <optional>

namespace perceptron {
namespace layer {

class Layer {
public:
  Layer(size_type input_layer, size_type n_neurons,
        std::unique_ptr<activations::IActivation> activation,
        std::unique_ptr<optimizers::IOptimizer> optimizer,
        const std::unordered_map<std::string, std::any> &config,
        size_type seed = 42,
        cudaStream_t stream = nullptr);

  [[nodiscard]] bool
  with_bias() const;

  [[nodiscard]] tensors::TensorReadOnly2D<float, false>
  weights() const;

  [[nodiscard]] tensors::TensorReadOnly2D<float, false>
  active_neurons() const;

  void
  transform(tensors::TensorReadOnly2D<float, false> inputs);

  void
  transform(tensors::TensorReadOnly2D<float, true> inputs);

  void
  update_gradients(tensors::TensorReadOnly2D<float, false> forwards,
                   tensors::TensorReadOnly2D<float, false> errors);

  void
  update_gradients(tensors::TensorReadOnly2D<float, false> forwards,
                   tensors::TensorReadOnly2D<float, true> errors);

  void
  update_gradients(tensors::TensorReadOnly2D<float, true> forwards,
                   tensors::TensorReadOnly2D<float, false> errors);

  void
  update_gradients(tensors::TensorReadOnly2D<float, true> forwards,
                   tensors::TensorReadOnly2D<float, true> errors);

  tensors::TensorOwnerDevice2D<float>
  compute_errors(tensors::TensorReadOnly2D<float, false> next_layer_errors,
                 tensors::TensorReadOnly2D<float, false> next_layer_weights);

  tensors::TensorOwnerDevice2D<float>
  compute_errors(tensors::TensorReadOnly2D<float, false> next_layer_errors,
                 tensors::TensorReadOnly2D<float, true> next_layer_weights);

  tensors::TensorOwnerDevice2D<float>
  compute_errors(tensors::TensorReadOnly2D<float, true> next_layer_errors,
                 tensors::TensorReadOnly2D<float, false> next_layer_weights);

  tensors::TensorOwnerDevice2D<float>
  compute_errors(tensors::TensorReadOnly2D<float, true> next_layer_errors,
                 tensors::TensorReadOnly2D<float, true> next_layer_weights);

  tensors::TensorOwnerDevice2D<float>
  compute_errors(tensors::TensorReadOnly2D<float, false> backward_errors);

  tensors::TensorOwnerDevice2D<float>
  compute_errors(tensors::TensorReadOnly2D<float, true> backward_errors);

  void
  compute_errors(tensors::TensorReadOnly2D<float, false> backward_errors,
                 tensors::TensorWriteable2D<float> outputs);

  void
  compute_errors(tensors::TensorReadOnly2D<float, true> backward_errors,
                 tensors::TensorWriteable2D<float> outputs);

  void
  compute_errors(tensors::TensorWriteable2D<float> backward_errors);

private:
  std::unique_ptr<activations::IActivation> m_activation;
  std::unique_ptr<optimizers::IOptimizer> m_weights_optimizer;
  tensors::TensorOwnerDevice2D<float> m_weights;

  std::optional<std::unique_ptr<optimizers::IOptimizer>> m_bias_optimizer{std::nullopt};
  std::optional<tensors::TensorOwnerDevice1D<float>> m_bias{std::nullopt};

  std::optional<tensors::TensorOwnerDevice2D<float>> m_neurons{std::nullopt};
  std::optional<tensors::TensorOwnerDevice2D<float>> m_active_neurons{std::nullopt};

  template<bool trans>
  void
  transform_impl(tensors::TensorReadOnly2D<float, trans> inputs);

  template<bool transa, bool transb>
  void
  update_gradients_impl(tensors::TensorReadOnly2D<float, transa> forwards,
                        tensors::TensorReadOnly2D<float, transb> errors);

  template<bool transa, bool transb>
  tensors::TensorOwnerDevice2D<float>
  compute_errors_impl(tensors::TensorReadOnly2D<float, transa> next_layer_errors,
                      tensors::TensorReadOnly2D<float, transb> next_layer_weights);

  template<bool transa>
  tensors::TensorOwnerDevice2D<float>
  compute_errors_impl(tensors::TensorReadOnly2D<float, transa> backward_errors);

  template<bool transa>
  void
  compute_errors_impl(tensors::TensorReadOnly2D<float, transa> backward_errors,
                      tensors::TensorWriteable2D<float> outputs);

  void
  compute_errors_impl(tensors::TensorWriteable2D<float> backward_errors);
};

template<bool trans>
void
Layer::transform_impl(tensors::TensorReadOnly2D<float, trans> inputs) {
  auto weights_view = m_weights.tensor_view();
  assert(inputs.get_x_dim() == weights_view.get_y_dim());
  if (!m_neurons.has_value()) {
    m_neurons = tensors::constructTensorOwnerDevice2D<float>(inputs.get_y_dim(), weights_view.get_x_dim());
  }
  auto neurons_view = m_neurons->tensor_view();
  tensors::ops::gemm(1.0f, inputs, weights_view.to_read_only(), 0.0f, neurons_view);
  if (with_bias()) {
    tensors::ops::add(1.0f, m_bias->tensor_view().to_read_only(), 1.0f, neurons_view);
  }
  if (!m_active_neurons.has_value()) {
    m_active_neurons = tensors::constructTensorOwnerDevice2D<float>(neurons_view.get_y_dim(), neurons_view.get_x_dim());
  }
  m_activation->compute(neurons_view.to_read_only(), m_active_neurons->tensor_view());
}

template<bool transa, bool transb>
void
Layer::update_gradients_impl(tensors::TensorReadOnly2D<float, transa> forwards,
                             tensors::TensorReadOnly2D<float, transb> errors) {
  assert(forwards.get_y_dim() == errors.get_y_dim());
  auto weights_view = m_weights.tensor_view();
  assert(weights_view.get_y_dim() == forwards.get_x_dim() &&
      weights_view.get_x_dim() == errors.get_x_dim());

  auto grads_owner = tensors::constructTensorOwnerDevice2D<float>(forwards.get_x_dim(), errors.get_x_dim());
  auto grads_view = grads_owner.tensor_view();
  tensors::ops::gemm(1.0f, forwards.t(), errors, 0.0f, grads_view);
  m_weights_optimizer->descent(grads_view.to_read_only(), weights_view);

  if (with_bias()) {
    auto bias_view = m_bias->tensor_view();
    assert(bias_view.get_size() == errors.get_x_dim());
    auto ones_vector = tensors::constructTensorOwnerDevice1D<float>(errors.get_y_dim());
    auto bias_grads_owner = tensors::constructTensorOwnerDevice1D<float>(errors.get_x_dim());
    auto bias_grads_view = bias_grads_owner.tensor_view();
    tensors::ops::gemv(1.0f, errors.t(), ones_vector.tensor_view().to_read_only(), 0.0f, bias_grads_view);
    m_bias_optimizer->get()->descent(bias_grads_view.to_read_only().to_2d(), bias_view.to_2d());
  }
}

template<bool transa, bool transb>
tensors::TensorOwnerDevice2D<float>
Layer::compute_errors_impl(tensors::TensorReadOnly2D<float, transa> next_layer_errors,
                           tensors::TensorReadOnly2D<float, transb> next_layer_weights) {
  auto backward_errors_owner = tensors::constructTensorOwnerDevice2D<float>(next_layer_errors.get_y_dim(),
                                                                            next_layer_weights.get_y_dim());
  auto backward_errors_view = backward_errors_owner.tensor_view();
  tensors::ops::gemm(1.0f, next_layer_errors, next_layer_weights.t(), 0.0f, backward_errors_view);
  compute_errors_impl(backward_errors_view);
  return backward_errors_owner;
}

template<bool transa>
tensors::TensorOwnerDevice2D<float>
Layer::compute_errors_impl(tensors::TensorReadOnly2D<float, transa> backward_errors) {
  auto outputs_owner =
      tensors::constructTensorOwnerDevice2D<float>(backward_errors.get_y_dim(), backward_errors.get_x_dim());
  compute_errors_impl(backward_errors, outputs_owner.tensor_view());
  return outputs_owner;
}

template<bool transa>
void
Layer::compute_errors_impl(tensors::TensorReadOnly2D<float, transa> backward_errors,
                           tensors::TensorWriteable2D<float> outputs) {
  auto act_der_owner = m_activation->derivative(m_neurons->tensor_view().to_read_only());
  tensors::ops::element_wise_mul(backward_errors,
                                 act_der_owner.tensor_view().to_read_only(),
                                 outputs);
}

void
Layer::compute_errors_impl(tensors::TensorWriteable2D<float> backward_errors) {
  auto act_der_owner = m_activation->derivative(m_neurons->tensor_view().to_read_only());
  tensors::ops::element_wise_mul(act_der_owner.tensor_view().to_read_only(),
                                 backward_errors);
}

} // perceptron
} // layer

#endif //PERCEPTRON_LAYER_H
