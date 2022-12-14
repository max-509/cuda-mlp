#ifndef PERCEPTRON_ACTIVATIONS_IACTIVATION_H
#define PERCEPTRON_ACTIVATIONS_IACTIVATION_H

#include "perceptron/common/Common.h"
#include "perceptron/tensors/Tensor2D.h"

namespace perceptron {
namespace activations {

class IActivation {
public:
  IActivation() = default;
  IActivation(const IActivation &) = default;
  IActivation(IActivation &&) = default;
  IActivation &operator=(const IActivation &) = default;
  IActivation &operator=(IActivation &&) = default;

  virtual tensors::TensorOwnerDevice2D<float>
  compute(tensors::TensorReadOnly2D<float, false> inputs) = 0;

  virtual tensors::TensorOwnerDevice2D<float>
  compute(tensors::TensorReadOnly2D<float, true> inputs) = 0;

  virtual tensors::TensorOwnerDevice2D<float>
  derivative(tensors::TensorReadOnly2D<float, false> inputs) = 0;

  virtual tensors::TensorOwnerDevice2D<float>
  derivative(tensors::TensorReadOnly2D<float, true> inputs) = 0;

  virtual ~IActivation() = default;
};

} // perceptron
} // activations

#endif //PERCEPTRON_ACTIVATIONS_IACTIVATION_H
