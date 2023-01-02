#include "perceptron/MultilayerPerceptron.h"

namespace perceptron {

std::shared_ptr<MultilayerPerceptron::MultilayerPerceptronBuilder>
MultilayerPerceptron::MultilayerPerceptronBuilder::setBatchSize(
    size_type batch_size) {
  m_batch_size = batch_size;
  return shared_from_this();
}

std::shared_ptr<MultilayerPerceptron::MultilayerPerceptronBuilder>
MultilayerPerceptron::MultilayerPerceptronBuilder::setEtol(double etol) {
  m_etol = etol;
  return shared_from_this();
}

std::shared_ptr<MultilayerPerceptron::MultilayerPerceptronBuilder>
MultilayerPerceptron::MultilayerPerceptronBuilder::setNEpoch(size_type n_epoch) {
  m_n_epoch = n_epoch;
  return shared_from_this();
}

std::shared_ptr<MultilayerPerceptron::MultilayerPerceptronBuilder>
MultilayerPerceptron::MultilayerPerceptronBuilder::setMaxNotChangeIter(size_type max_not_change_iter) {
  m_max_not_change_iter = max_not_change_iter;
  return shared_from_this();
}

std::shared_ptr<MultilayerPerceptron::MultilayerPerceptronBuilder>
MultilayerPerceptron::MultilayerPerceptronBuilder::setSeed(size_type seed) {
  m_seed = seed;
  return shared_from_this();
}

std::shared_ptr<MultilayerPerceptron::MultilayerPerceptronBuilder>
MultilayerPerceptron::builder() {
  return std::make_shared<MultilayerPerceptronBuilder>();
}

tensors::TensorOwner2D<float>
MultilayerPerceptron::transform(tensors::TensorReadOnly1D<float> inputs) {
  return transform(inputs.to_2d());
}

} // perceptron
