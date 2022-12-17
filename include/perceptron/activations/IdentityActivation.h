#ifndef PERCEPTRON_ACTIVATIONS_IDENTITYACTIVATION_H
#define PERCEPTRON_ACTIVATIONS_IDENTITYACTIVATION_H

#include "perceptron/common/Common.h"
#include "perceptron/activations/IActivation.h"
#include "perceptron/tensors/Tensor2D.h"
#include "perceptron/tensors/ops/MemoryOps.h"

namespace perceptron {
namespace activations {

class IdentityActivation : public IActivation {
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

private:
  template<typename T, bool trans>
  void
  compute_impl(tensors::TensorReadOnly2D<T, trans> inputs,
               tensors::TensorWriteable2D<T> outputs);

  template<typename T, bool trans>
  void
  derivative_impl(tensors::TensorReadOnly2D<T, trans> inputs,
                  tensors::TensorWriteable2D<T> outputs);
};

template<typename T, bool trans>
void
IdentityActivation::compute_impl(tensors::TensorReadOnly2D<T, trans> inputs,
                                 tensors::TensorWriteable2D<T> outputs) {
  tensors::ops::copy(inputs, outputs);
}

template<typename T, bool trans>
void
IdentityActivation::derivative_impl(tensors::TensorReadOnly2D<T, trans> inputs,
                                    tensors::TensorWriteable2D<T> outputs) {
  tensors::ops::set(static_cast<T>(1.0), outputs);
}

} // perceptron
} // activations

#endif //PERCEPTRON_ACTIVATIONS_IDENTITYACTIVATION_H
