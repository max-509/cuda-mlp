#ifndef PERCEPTRON_ACTIVATIONS_LEAKYRELUACTIVATION_H
#define PERCEPTRON_ACTIVATIONS_LEAKYRELUACTIVATION_H

#include "perceptron/common/Common.h"
#include "perceptron/activations/IActivation.h"
#include "perceptron/tensors/Tensor2D.h"
#include "perceptron/tensors/ops/MathOps.h"
#include "perceptron/tensors/ops/MemoryOps.h"

namespace perceptron {
namespace activations {

class LeakyReLUActivation : public IActivation {
public:
  using IActivation::compute;

  LeakyReLUActivation() = default;
  explicit LeakyReLUActivation(float negative_slope);

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

private:
  float m_negative_slope{0.01f};
};

} // perceptron
} // activations

#endif //PERCEPTRON_ACTIVATIONS_LEAKYRELUACTIVATION_H
