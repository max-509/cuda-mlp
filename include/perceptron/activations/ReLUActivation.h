#ifndef PERCEPTRON_ACTIVATIONS_RELUACTIVATION_H
#define PERCEPTRON_ACTIVATIONS_RELUACTIVATION_H

#include "perceptron/common/Common.h"
#include "perceptron/activations/IActivation.h"
#include "perceptron/tensors/Tensor2D.h"
#include "perceptron/tensors/ops/MathOps.h"
#include "perceptron/tensors/ops/MemoryOps.h"

namespace perceptron {
namespace activations {

class ReLUActivation : public IActivation {
public:
  using IActivation::compute;
  void
  compute(tensors::TensorReadOnly2D<float, false> inputs,
          tensors::TensorWriteable2D<float> outputs) override;

  void
  compute(tensors::TensorReadOnly2D<float, true> inputs,
          tensors::TensorWriteable2D<float> outputs) override;

  void
  derivative(tensors::TensorReadOnly2D<float, false> inputs,
             tensors::TensorWriteable2D<float> outputs) override;

  void
  derivative(tensors::TensorReadOnly2D<float, true> inputs,
             tensors::TensorWriteable2D<float> outputs) override;
};

} // perceptron
} // activations

#endif //PERCEPTRON_ACTIVATIONS_RELUACTIVATION_H
