#ifndef PERCEPTRON_MULTILAYER_PERCEPTRON_H
#define PERCEPTRON_MULTILAYER_PERCEPTRON_H

#include "perceptron/common/Common.h"
#include "perceptron/Layer.h"
#include "perceptron/losses/ILoss.h"
#include "perceptron/activations/IActivation.h"
#include "perceptron/tensors/Tensor2D.h"
#include "perceptron/tensors/ops/MemoryOps.h"
#include "perceptron/common/utils/StreamUtils.h"

#include <vector>
#include <memory>
#include <optional>
#include <unordered_map>
#include <any>
#include <string>
#include <stdexcept>
#include <random>
#include <algorithm>

namespace perceptron {

struct mlp_history_t {
  std::vector<double> train_history;
  std::optional<std::vector<double>> val_history{std::nullopt};
};

template<bool trans_features, bool trans_labels>
struct features_labels_t {
  tensors::TensorReadOnly2D<float, trans_features> features;
  tensors::TensorReadOnly2D<float, trans_labels> labels;
};

class MultilayerPerceptron {
private:
  static constexpr auto STREAMS_BLOCK_SIZE = size_type(8);
public:

  class MultilayerPerceptronBuilder : std::enable_shared_from_this<MultilayerPerceptronBuilder> {
  public:
    std::shared_ptr<MultilayerPerceptronBuilder>
    setBatchSize(size_type batch_size);

    std::shared_ptr<MultilayerPerceptronBuilder>
    setEtol(double etol);

    std::shared_ptr<MultilayerPerceptronBuilder>
    setNEpoch(size_type n_epoch);

    std::shared_ptr<MultilayerPerceptronBuilder>
    setMaxNotChangeIter(size_type max_not_change_iter);

    std::shared_ptr<MultilayerPerceptronBuilder>
    setSeed(size_type seed);

    template<typename OptimizerDescriber>
    MultilayerPerceptron
    build(size_type input_size,
          std::unique_ptr<losses::ILoss> loss,
          OptimizerDescriber optimizer_describer,
          std::vector<layer::layer_params_t> &&layers_params);

  private:
    std::optional<size_type> m_batch_size{std::nullopt};
    double m_etol{1e-5};
    size_type m_n_epoch{1000};
    size_type m_max_not_change_iter{5};
    size_type m_seed{42};
  };

  MultilayerPerceptron() = delete;
  MultilayerPerceptron(const MultilayerPerceptron &) = delete;
  MultilayerPerceptron(MultilayerPerceptron &&) = delete;
  MultilayerPerceptron &operator=(const MultilayerPerceptron &) = delete;
  MultilayerPerceptron &operator=(MultilayerPerceptron &&) = delete;
  ~MultilayerPerceptron() = default;

  static std::shared_ptr<MultilayerPerceptronBuilder>
  builder();

  template<bool trans>
  tensors::TensorOwnerDevice2D<float>
  forward(tensors::TensorReadOnly2D<float, trans> inputs);

  template<bool trans_inputs, bool trans_outputs, bool trans_labels>
  void
  backward(tensors::TensorReadOnly2D<float, trans_inputs> inputs,
           tensors::TensorReadOnly2D<float, trans_outputs> outputs,
           tensors::TensorReadOnly2D<float, trans_labels> labels);

  template<bool trans_train_feat, bool trans_train_labels>
  mlp_history_t
  fit(features_labels_t<trans_train_feat, trans_train_labels> train);

  template<bool trans>
  tensors::TensorOwnerDevice2D<float>
  transform(tensors::TensorReadOnly2D<float, trans> inputs);

  tensors::TensorOwnerDevice2D<float>
  transform(tensors::TensorReadOnly1D<float> inputs);

private:
  std::unique_ptr<losses::ILoss> m_loss;
  std::vector<layer::Layer> m_layers;
  std::optional<size_type> m_batch_size;
  double m_etol;
  size_type m_n_epoch;
  size_type m_max_not_change_iter;
  size_type m_seed;

  template<typename OptimizerDescriber>
  MultilayerPerceptron(size_type input_size,
                       std::unique_ptr<losses::ILoss> loss,
                       OptimizerDescriber optimizer_describer,
                       std::vector<layer::layer_params_t> &&layers_params,
                       std::optional<size_type> batch_size,
                       double etol,
                       size_type n_epoch,
                       size_type max_not_change_iter,
                       size_type seed);
};

template<typename OptimizerDescriber>
MultilayerPerceptron::MultilayerPerceptron(size_type input_size,
                                           std::unique_ptr<losses::ILoss> loss,
                                           OptimizerDescriber optimizer_describer,
                                           std::vector<layer::layer_params_t> &&layers_params,
                                           std::optional<size_type> batch_size,
                                           double etol,
                                           size_type n_epoch,
                                           size_type max_not_change_iter,
                                           size_type seed)
    : m_loss(std::move(loss)), m_batch_size(batch_size), m_etol(etol),
      m_n_epoch(n_epoch), m_max_not_change_iter(max_not_change_iter), m_seed(seed) {
  m_layers = std::vector<layer::Layer>{};
  m_layers.reserve(layers_params.size());

  size_type layer_input = input_size;
  for (auto &&params : layers_params) {
    size_type layer_output = params.layer_size;
    m_layers.emplace_back(layer_input, layer_output,
                          std::move(params.activation),
                          optimizer_describer,
                          params.config, seed);
    layer_input = layer_output;
  }
}

template<typename OptimizerDescriber>
MultilayerPerceptron
MultilayerPerceptron::MultilayerPerceptronBuilder::build(size_type input_size,
                                                         std::unique_ptr<losses::ILoss> loss,
                                                         OptimizerDescriber optimizer_describer,
                                                         std::vector<layer::layer_params_t> &&layers_params) {
  return MultilayerPerceptron(input_size,
                              std::move(loss),
                              optimizer_describer,
                              std::move(layers_params),
                              m_batch_size,
                              m_etol,
                              m_n_epoch,
                              m_max_not_change_iter,
                              m_seed);
}

template<bool trans>
tensors::TensorOwnerDevice2D<float>
MultilayerPerceptron::forward(tensors::TensorReadOnly2D<float, trans> inputs) {
  m_layers[0].transform(inputs);
  auto outputs = m_layers[0].active_neurons();

  for (auto it = m_layers.begin() + 1; it != m_layers.end(); ++it) {
    it->transform(outputs);
    outputs = it->active_neurons();
  }

  auto result_outputs_owner = tensors::constructTensorOwnerDevice2D<float>(outputs.get_nrows(), outputs.get_ncols());
  tensors::ops::copy(outputs, result_outputs_owner.tensor_view());
  return result_outputs_owner;
}

template<bool trans_inputs, bool trans_outputs, bool trans_labels>
void
MultilayerPerceptron::backward(tensors::TensorReadOnly2D<float, trans_inputs> inputs,
                               tensors::TensorReadOnly2D<float, trans_outputs> outputs,
                               tensors::TensorReadOnly2D<float, trans_labels> labels) {
  assert(!m_layers.empty());

  auto curr_backward_errors_owner = m_loss->derivative(outputs, labels);
  auto curr_backward_errors_view = curr_backward_errors_owner.tensor_view();
  m_layers.back().compute_errors(curr_backward_errors_view);

  for (size_type layer_idx = m_layers.size() - 1; layer_idx > 0; --layer_idx) {
    auto &&curr_layer = m_layers[layer_idx];
    auto &&prev_layer = m_layers[layer_idx - 1];

    auto forwards = prev_layer.active_neurons();
    curr_layer.update_gradients(forwards, curr_backward_errors_view().to_read_only());

    auto curr_layer_weights = curr_layer.weights();
    auto curr_layer_weights_t = curr_layer_weights.t();
    auto prev_backward_errors_owner =
        tensors::constructTensorOwnerDevice2D<float>(curr_backward_errors_view.get_nrows(),
                                                     curr_layer_weights_t.get_ncols());
    auto prev_backward_errors_view = prev_backward_errors_owner.tensor_view();
    tensors::ops::gemm(1.0f, curr_backward_errors_view.to_read_only(), curr_layer_weights_t,
                       0.0f, prev_backward_errors_view);
    prev_layer.compute_errors(prev_backward_errors_view);

    curr_backward_errors_owner = std::move(prev_backward_errors_owner);
    curr_backward_errors_view = prev_backward_errors_view;
  }

  m_layers.front().update_gradients(inputs, curr_backward_errors_view.to_read_only());
}

template<bool trans_train_feat, bool trans_train_labels>
mlp_history_t
MultilayerPerceptron::fit(features_labels_t<trans_train_feat, trans_train_labels> train) {
  auto [train_features, train_labels] = train;

  if (train_features.get_nrows() != train_labels.get_nrows()) {
    throw std::invalid_argument{"Number of samples in features must be equal to number of samples in labels"};
  }
  auto n_instances = train_features.get_nrows();
  auto batch_size = m_batch_size.value_or(n_instances);

  std::vector<size_type> batch_indices(batch_size);
  std::vector<size_type> indices(n_instances);
  std::iota(indices.begin(), indices.end(), 0);
  std::random_device rd;
  std::mt19937 g(rd());

  auto batch_train_features_owner = tensors::constructTensorOwnerDevice2D<float>(batch_size, train_features.get_ncols());
  auto batch_train_features_view = batch_train_features_owner.tensor_view();
  auto batch_train_labels_owner = tensors::constructTensorOwnerDevice2D<float>(batch_size, train_labels.get_ncols());
  auto batch_train_labels_view = batch_train_labels_owner.tensor_view();
  auto streams_to_copy = utils::cu_create_streams(STREAMS_BLOCK_SIZE);

  std::vector<double> train_epoch_losses{};
  auto prev_loss = std::numeric_limits<double>::infinity();

  auto curr_epoch = 0;
  size_type not_change_iters = 0;
  while (not_change_iters < m_max_not_change_iter && curr_epoch < m_n_epoch) {
    std::shuffle(indices.begin(), indices.end(), g);
    std::copy_n(indices.cbegin(), batch_size, batch_indices);

    for (size_type stream_block_idx = 0; stream_block_idx < batch_size; stream_block_idx += STREAMS_BLOCK_SIZE) {
      auto current_streams_block_size = std::min(STREAMS_BLOCK_SIZE, batch_size - stream_block_idx);
      for (size_type stream_idx = 0; stream_idx < current_streams_block_size; ++stream_idx) {
        auto row_idx = stream_block_idx + stream_idx;
        tensors::ops::copy(train_features.get_row(batch_indices[row_idx]).to_2d(),
                           batch_train_features_view.get_row(row_idx).to_2d(),
                           streams_to_copy[stream_idx]);

        tensors::ops::copy(train_labels.get_row(batch_indices[row_idx]).to_2d(),
                           batch_train_labels_view.get_row(row_idx).to_2d(),
                           streams_to_copy[stream_idx]);
      }
    }

    utils::cu_wait_streams(streams_to_copy);

    // TODO: Reuse memory
    auto batch_outputs_owner = forward(batch_train_features_view.to_read_only());
    auto batch_outputs_view = batch_outputs_owner.to_read_only();
    backward(batch_train_features_view.to_read_only(),
             batch_outputs_view.to_read_only(),
             batch_train_labels_view.to_read_only());

    train_epoch_losses.push_back(m_loss->compute(batch_outputs_view.to_read_only(),
                                                 batch_train_labels_view.to_read_only()));

    // TODO: Val test

    ++curr_epoch;
  }

  return mlp_history_t{train_epoch_losses};
}

template<bool trans>
tensors::TensorOwnerDevice2D<float>
MultilayerPerceptron::transform(tensors::TensorReadOnly2D<float, trans> inputs) {
  // TODO: If loss with activation, call activate method
  return forward(inputs);
}

} // perceptron

#endif // PERCEPTRON_MULTILAYER_PERCEPTRON_H
