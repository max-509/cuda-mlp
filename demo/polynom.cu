#include "perceptron/MultilayerPerceptron.h"
#include "perceptron/optimizers/SGD.h"
#include "perceptron/tensors/ops/Ops.h"
#include "perceptron/tensors/shufflers/BatchShuffler.h"
#include "perceptron/losses/SquaredLoss.h"
#include "perceptron/activations/ReLUActivation.h"
#include "perceptron/activations/IdentityActivation.h"

#include "results_saver.h"

#include <memory>
#include <iostream>

using namespace perceptron;

int main() {
  auto loss_function = std::make_shared<losses::SquaredLoss>();
  auto builder = MultilayerPerceptron::builder()->setEtol(1e-8)->setNEpoch(10000)->setBatchSize(256);
  auto mlp = builder->build(1, loss_function, optimizers::SGD::describer_t{}.set_lr(1e-3).set_momentum(0.9), {
      layer::layer_params_t{10, std::make_shared<activations::ReLUActivation>()},
      layer::layer_params_t{10, std::make_shared<activations::ReLUActivation>()},
      layer::layer_params_t{1, std::make_shared<activations::IdentityActivation>()}
  });

  size_type dataset_size = 10000;
  auto features = tensors::ops::linspace(-2.0f, 2.0f, dataset_size);

  auto labels = tensors::ops::scal(5.0f,
                                   features.tensor_view().to_read_only().to_2d());
  auto features_sqr = tensors::ops::element_wise_mul(features.tensor_view().to_read_only().to_2d(),
                                                     features.tensor_view().to_read_only().to_2d());
  tensors::ops::element_wise_mul(features_sqr.tensor_view().to_read_only(),
                                 labels.tensor_view());
  tensors::ops::geam(features_sqr.tensor_view().to_read_only(), 1.0f,
                     labels.tensor_view(), 1.0f);
  tensors::ops::add(5.0f, labels.tensor_view());

  float train_test_fraction = 0.7;
  auto train_size = static_cast<size_type>(static_cast<double>(dataset_size) * train_test_fraction);

  tensors::shufflers::BatchShuffler train_test_shuffler{dataset_size, train_size};
  train_test_shuffler.shuffle();
  auto [train_features, test_features] =
      train_test_shuffler.train_test_split(features.tensor_view().to_read_only().to_2d());
  auto [train_labels, test_labels] = train_test_shuffler.train_test_split(labels.tensor_view().to_read_only());

  auto history = mlp.fit(features2D_labels2D_pair_t<false, false>{train_features.tensor_view().to_read_only(),
                                                                  train_labels.tensor_view().to_read_only()});
  auto transformed_owner = mlp.transform(test_features.tensor_view().to_read_only());
  auto loss =
      loss_function->compute(transformed_owner.tensor_view().to_read_only(), test_labels.tensor_view().to_read_only());
  std::cerr << "Loss equals to: " << loss << std::endl;

  test_features.to_host();
  transformed_owner.to_host();
  tensor_to_csv(test_features.tensor_view().to_read_only(), "polynom_test_features.csv", {"f1"});
  tensor_to_csv(transformed_owner.tensor_view().to_read_only(), "polynom_test_labels.csv", {"label"});

  return 0;
}
