#ifndef PERCEPTRON_MULTILAYER_PERCEPTRON_H
#define PERCEPTRON_MULTILAYER_PERCEPTRON_H

#include "perceptron/common/Common.h"
#include "perceptron/Layer.h"
#include "perceptron/losses/ILoss.h"
#include "perceptron/activations/IActivation.h"
#include "perceptron/tensors/Tensor2D.h"
#include "perceptron/tensors/ops/MemoryOps.h"
#include "perceptron/tensors/shufflers/IShuffler.h"
#include "perceptron/tensors/shufflers/DummyShuffler.h"
#include "perceptron/tensors/shufflers/BatchShuffler.h"
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
struct features2D_labels2D_pair_t {
  tensors::TensorReadOnly2D<float, trans_features> features;
  tensors::TensorReadOnly2D<float, trans_labels> labels;
};

template<bool trans_features>
struct features2D_labels1D_pair_t {
  tensors::TensorReadOnly2D<float, trans_features> features;
  tensors::TensorReadOnly1D<float> labels;
};

template<bool trans_labels>
struct features1D_labels2D_pair_t {
  tensors::TensorReadOnly1D<float> features;
  tensors::TensorReadOnly2D<float, trans_labels> labels;
};

struct features1D_labels1D_pair_t {
  tensors::TensorReadOnly1D<float> features;
  tensors::TensorReadOnly1D<float> labels;
};

class MultilayerPerceptron {
private:
  static constexpr size_type STREAMS_BLOCK_SIZE = 32;
public:

  class MultilayerPerceptronBuilder : public std::enable_shared_from_this<MultilayerPerceptronBuilder> {
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
          std::shared_ptr<losses::ILoss> loss,
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
  tensors::TensorOwner2D<float>
  forward(tensors::TensorReadOnly2D<float, trans> inputs);

  template<bool trans>
  void
  forward(tensors::TensorReadOnly2D<float, trans> inputs,
          tensors::TensorWriteable2D<float> outputs);

  template<bool trans_inputs, bool trans_outputs, bool trans_labels>
  void
  backward(tensors::TensorReadOnly2D<float, trans_inputs> inputs,
           tensors::TensorReadOnly2D<float, trans_outputs> outputs,
           tensors::TensorReadOnly2D<float, trans_labels> labels);

  template<bool trans_train_feat, bool trans_train_labels, bool trans_val_feat, bool trans_val_labels>
  mlp_history_t
  fit(features2D_labels2D_pair_t<trans_train_feat, trans_train_labels> train,
      features2D_labels2D_pair_t<trans_val_feat, trans_val_labels> val);

  template<bool trans_train_feat, bool trans_train_labels>
  mlp_history_t
  fit(features2D_labels2D_pair_t<trans_train_feat, trans_train_labels> train);

  template<bool trans_train_feat>
  mlp_history_t
  fit(features2D_labels1D_pair_t<trans_train_feat> train);

  template<bool trans_train_labels>
  mlp_history_t
  fit(features1D_labels2D_pair_t<trans_train_labels> train);

  mlp_history_t
  fit(features1D_labels1D_pair_t train);

  template<bool trans>
  tensors::TensorOwner2D<float>
  transform(tensors::TensorReadOnly2D<float, trans> inputs);

  tensors::TensorOwner2D<float>
  transform(tensors::TensorReadOnly1D<float> inputs);

private:
  std::shared_ptr<losses::ILoss> m_loss;
  std::vector<layer::Layer> m_layers;
  std::optional<size_type> m_batch_size;
  double m_etol;
  size_type m_n_epoch;
  size_type m_max_not_change_iter;
  size_type m_seed;

  template<typename OptimizerDescriber>
  MultilayerPerceptron(size_type input_size,
                       std::shared_ptr<losses::ILoss> loss,
                       OptimizerDescriber optimizer_describer,
                       std::vector<layer::layer_params_t> &&layers_params,
                       std::optional<size_type> batch_size,
                       double etol,
                       size_type n_epoch,
                       size_type max_not_change_iter,
                       size_type seed);

  template<bool trans_train_feat, bool trans_train_labels, bool trans_val_feat, bool trans_val_labels>
  mlp_history_t
  fit_impl(features2D_labels2D_pair_t<trans_train_feat, trans_train_labels> train,
           std::optional<features2D_labels2D_pair_t<trans_val_feat, trans_val_labels>> val);
};

template<typename OptimizerDescriber>
MultilayerPerceptron::MultilayerPerceptron(size_type input_size,
                                           std::shared_ptr<losses::ILoss> loss,
                                           OptimizerDescriber optimizer_describer,
                                           std::vector<layer::layer_params_t> &&layers_params,
                                           std::optional<size_type> batch_size,
                                           double etol,
                                           size_type n_epoch,
                                           size_type max_not_change_iter,
                                           size_type seed)
    : m_loss(loss), m_batch_size(batch_size), m_etol(etol),
      m_n_epoch(n_epoch), m_max_not_change_iter(max_not_change_iter), m_seed(seed) {
  m_layers = std::vector<layer::Layer>{};
  m_layers.reserve(layers_params.size());

  auto layer_input = input_size;
  auto streams = utils::cu_create_streams(layers_params.size());
  for (size_type i = 0; i < layers_params.size(); ++i) {
    auto params = layers_params[i];
    size_type layer_output = params.layer_size;
    m_layers.emplace_back(layer_input, layer_output,
                          std::move(params.activation),
                          optimizer_describer,
                          params.config, seed + i, *streams[i]);
    layer_input = layer_output;
  }

  utils::cu_wait_streams(streams);
}

template<typename OptimizerDescriber>
MultilayerPerceptron
MultilayerPerceptron::MultilayerPerceptronBuilder::build(size_type input_size,
                                                         std::shared_ptr<losses::ILoss> loss,
                                                         OptimizerDescriber optimizer_describer,
                                                         std::vector<layer::layer_params_t> &&layers_params) {
  return MultilayerPerceptron(input_size,
                              loss,
                              optimizer_describer,
                              std::move(layers_params),
                              m_batch_size,
                              m_etol,
                              m_n_epoch,
                              m_max_not_change_iter,
                              m_seed);
}

template<bool trans>
tensors::TensorOwner2D<float>
MultilayerPerceptron::forward(tensors::TensorReadOnly2D<float, trans> inputs) {
  m_layers.front().transform(inputs);
  auto layer_outputs = m_layers.front().active_neurons();

  for (auto it = m_layers.begin() + 1; it != m_layers.end(); ++it) {
    it->transform(layer_outputs);
    layer_outputs = it->active_neurons();
  }

  return tensors::ops::copy(layer_outputs);
}

template<bool trans>
void
MultilayerPerceptron::forward(tensors::TensorReadOnly2D<float, trans> inputs,
                              tensors::TensorWriteable2D<float> outputs) {
  m_layers.front().transform(inputs);
  auto layer_outputs = m_layers.front().active_neurons();

  for (auto it = m_layers.begin() + 1; it != m_layers.end(); ++it) {
    it->transform(layer_outputs);
    layer_outputs = it->active_neurons();
  }

  tensors::ops::copy(layer_outputs, outputs);
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
    auto &curr_layer = m_layers[layer_idx];
    auto &prev_layer = m_layers[layer_idx - 1];

    auto forwards = prev_layer.active_neurons();
    curr_layer.update_gradients(forwards, curr_backward_errors_view.to_read_only());

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
    curr_backward_errors_view = curr_backward_errors_owner.tensor_view();
  }

  m_layers.front().update_gradients(inputs, curr_backward_errors_view.to_read_only());
}

template<bool trans_train_feat, bool trans_train_labels, bool trans_val_feat, bool trans_val_labels>
mlp_history_t
MultilayerPerceptron::fit(features2D_labels2D_pair_t<trans_train_feat, trans_train_labels> train,
                          features2D_labels2D_pair_t<trans_val_feat, trans_val_labels> val) {
  return fit_impl(train, std::make_optional(val));
}

template<bool trans_train_feat, bool trans_train_labels>
mlp_history_t
MultilayerPerceptron::fit(features2D_labels2D_pair_t<trans_train_feat, trans_train_labels> train) {
  return fit_impl<trans_train_feat, trans_train_labels, false, false>(train, std::nullopt);
}

template<bool trans_train_feat>
mlp_history_t
MultilayerPerceptron::fit(features2D_labels1D_pair_t<trans_train_feat> train) {
  return fit(features2D_labels2D_pair_t<trans_train_feat, false>{train.features, train.labels.to_2d()});
}

template<bool trans_train_labels>
mlp_history_t
MultilayerPerceptron::fit(features1D_labels2D_pair_t<trans_train_labels> train) {
  return fit(features2D_labels2D_pair_t<false, trans_train_labels>{train.features.to_2d(), train.labels});
}

template<bool trans_train_feat, bool trans_train_labels, bool trans_val_feat, bool trans_val_labels>
mlp_history_t
MultilayerPerceptron::fit_impl(features2D_labels2D_pair_t<trans_train_feat, trans_train_labels> train,
                               std::optional<features2D_labels2D_pair_t<trans_val_feat, trans_val_labels>> val) {
  auto [train_features, train_labels] = train;

  if (train_features.get_nrows() != train_labels.get_nrows()) {
    throw std::invalid_argument{"Number of samples in features must be equal to number of samples in labels"};
  }
  auto n_instances = train_features.get_nrows();

  size_type shuffled_buffer_nrows;
  std::unique_ptr<tensors::shufflers::IShuffler> shuffler;
  if (m_batch_size.has_value()) {
    auto batch_size = m_batch_size.value();
    shuffler = std::unique_ptr<tensors::shufflers::IShuffler>(new tensors::shufflers::BatchShuffler(n_instances,
                                                                                                    batch_size,
                                                                                                    m_seed));
    shuffled_buffer_nrows = batch_size;
  } else {
    shuffler = std::unique_ptr<tensors::shufflers::IShuffler>(new tensors::shufflers::DummyShuffler);
    shuffled_buffer_nrows = n_instances;
  }

  auto
      batch_train_features_owner =
      tensors::constructTensorOwnerDevice2D<float>(shuffled_buffer_nrows, train_features.get_ncols());
  auto batch_train_features_view = batch_train_features_owner.tensor_view();
  auto batch_train_labels_owner =
      tensors::constructTensorOwnerDevice2D<float>(shuffled_buffer_nrows, train_labels.get_ncols());
  auto batch_train_labels_view = batch_train_labels_owner.tensor_view();
  auto batch_outputs_owner =
      tensors::constructTensorOwnerDevice2D<float>(shuffled_buffer_nrows, train_labels.get_ncols());
  auto batch_outputs_view = batch_outputs_owner.tensor_view();
  std::optional<tensors::TensorOwner2D<float>> val_outputs_owner{std::nullopt};
  if (val.has_value()) {
    auto [val_features, val_labels] = val.value();
    val_outputs_owner =
        std::make_optional(tensors::constructTensorOwnerDevice2D<float>(val_labels.get_nrows(),
                                                                        val_labels.get_ncols()));
  }

  std::vector<double> train_epoch_losses{};
  std::optional<std::vector<double>> val_epoch_losses{std::nullopt};
  if (val.has_value()) {
    val_epoch_losses = std::make_optional(std::vector<double>{});
  }
  auto prev_loss = std::numeric_limits<double>::infinity();

  auto curr_epoch = 0;
  size_type not_change_iters = 0;
  while (not_change_iters < m_max_not_change_iter && curr_epoch < m_n_epoch) {
    shuffler->shuffle();
    shuffler->get_shuffled(train_features, batch_train_features_view);
    shuffler->get_shuffled(train_labels, batch_train_labels_view);

    forward(batch_train_features_view.to_read_only(), batch_outputs_view);
    backward(batch_train_features_view.to_read_only(),
             batch_outputs_view.to_read_only(),
             batch_train_labels_view.to_read_only());

    train_epoch_losses.push_back(m_loss->compute(batch_outputs_view.to_read_only(),
                                                 batch_train_labels_view.to_read_only()));

    // TODO: Val test
    if (val.has_value()) {
      auto [val_features, val_labels] = val.value();
      auto val_outputs = val_outputs_owner.value().tensor_view();
      forward(val_features, val_outputs);
      auto curr_loss = m_loss->compute(val_outputs.to_read_only(),
                                       val_labels);
      val_epoch_losses.value().push_back(curr_loss);

      if (std::abs(curr_loss - prev_loss) < m_etol) {
        ++not_change_iters;
      }

      prev_loss = curr_loss;
    }

    ++curr_epoch;
  }

  return mlp_history_t{train_epoch_losses, val_epoch_losses};
}

template<bool trans>
tensors::TensorOwner2D<float>
MultilayerPerceptron::transform(tensors::TensorReadOnly2D<float, trans> inputs) {
  // TODO: If loss with activation, call activate method
  return forward(inputs);
}

} // perceptron

#endif // PERCEPTRON_MULTILAYER_PERCEPTRON_H
