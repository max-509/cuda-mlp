#ifndef PERCEPTRON_OPTIMIZERS_IOPTIMIZER_H
#define PERCEPTRON_OPTIMIZERS_IOPTIMIZER_H

#include "perceptron/common/Common.h"
#include "perceptron/tensors/Tensor2D.h"

namespace perceptron {
namespace optimizers {

class IOptimizer {
public:
  IOptimizer() = default;
  IOptimizer(const IOptimizer &) = default;
  IOptimizer(IOptimizer &&) = default;
  IOptimizer &operator=(const IOptimizer &) = default;
  IOptimizer &operator=(IOptimizer &&) = default;

  virtual void
  descent(tensors::TensorReadOnly2D<float, false> grads,
          tensors::TensorWriteable2D<float> weigths) = 0;

  virtual void
  descent(tensors::TensorReadOnly2D<float, true> grads,
          tensors::TensorWriteable2D<float> weigths) = 0;

  virtual void
  reset() = 0;

  virtual ~IOptimizer() = default;
};

} // perceptron
} // optimizers

#endif //PERCEPTRON_OPTIMIZERS_IOPTIMIZER_H
