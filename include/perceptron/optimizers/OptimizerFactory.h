#ifndef PERCEPTRON_OPTIMIZERS_OPTIMIZERFACTORY_H
#define PERCEPTRON_OPTIMIZERS_OPTIMIZERFACTORY_H

#include "perceptron/optimizers/IOptimizer.h"

#include <memory>

namespace perceptron {
namespace optimizers {

class OptimizerFactory {
public:
  template<typename OptimizerDescriber>
  std::unique_ptr<IOptimizer>
  build(OptimizerDescriber describer);
};

template<typename OptimizerDescriber>
std::unique_ptr<IOptimizer>
OptimizerFactory::build(OptimizerDescriber describer) {
  return describer.build();
}

} // perceptron
} // optimizers

#endif //PERCEPTRON_OPTIMIZERS_OPTIMIZERFACTORY_H
