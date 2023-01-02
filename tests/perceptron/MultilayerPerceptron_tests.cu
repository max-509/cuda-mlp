#define CATCH_CONFIG_MAIN

#include <catch2/catch.hpp>

#include "perceptron/MultilayerPerceptron.h"
#include "perceptron/Layer.h"
#include "perceptron/tensors/ops/Ops.h"
#include "perceptron/activations/ReLUActivation.h"
#include "perceptron/activations/LeakyReLUActivation.h"
#include "perceptron/activations/IdentityActivation.h"
#include "perceptron/losses/SquaredLoss.h"
#include "perceptron/optimizers/SGD.h"
#include "perceptron/activations/SigmoidActivation.h"

#include <memory>

using namespace perceptron;

TEST_CASE("Create MLP", "[perceptron]") {
  auto builder = MultilayerPerceptron::builder()->setEtol(1e-4)->setNEpoch(1000)->setBatchSize(256);
  auto mlp = builder->build(1, std::make_shared<losses::SquaredLoss>(), optimizers::SGD::describer_t{}.set_lr(1e-2), {
      layer::layer_params_t{10, std::make_shared<activations::LeakyReLUActivation>()},
      layer::layer_params_t{10, std::make_shared<activations::LeakyReLUActivation>()},
      layer::layer_params_t{10, std::make_shared<activations::LeakyReLUActivation>()},
      layer::layer_params_t{1, std::make_shared<activations::IdentityActivation>()}
  });

  size_type dataset_size = 5000;
  auto train_features = tensors::ops::linspace(-2.0f, 2.0f, dataset_size);

  auto train_labels = tensors::ops::element_wise_mul(train_features.tensor_view().to_read_only().to_2d(),
                                                     train_features.tensor_view().to_read_only().to_2d());

  auto history = mlp.fit(features_labels_pair_t<false, false>{train_features.tensor_view().to_read_only().to_2d(),
                                                              train_labels.tensor_view().to_read_only()}).train_history;
  auto transformed_owner = mlp.transform(train_features.tensor_view().to_read_only());
  transformed_owner.to_host();
  auto transformed_view = transformed_owner.tensor_view();
  std::cout << "history" << std::endl;
  for (auto &&l : history) {
    std::cout << l << std::endl;
  }
  std::cout << "values" << std::endl;
  for (size_type i = 0; i < transformed_view.get_nrows(); ++i) {
    std::cout << transformed_view(i, 0) << std::endl;
  }
}
