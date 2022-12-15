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
  tensors::TensorOwnerDevice2D<float>
  compute(tensors::TensorReadOnly2D<float, false> inputs) override;

  tensors::TensorOwnerDevice2D<float>
  compute(tensors::TensorReadOnly2D<float, true> inputs) override;

  tensors::TensorOwnerDevice2D<float>
  derivative(tensors::TensorReadOnly2D<float, false> inputs) override;

  tensors::TensorOwnerDevice2D<float>
  derivative(tensors::TensorReadOnly2D<float, true> inputs) override;
};

} // perceptron
} // activations

#endif //PERCEPTRON_ACTIVATIONS_RELUACTIVATION_H
