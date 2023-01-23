#ifndef PERCEPTRON_ACTIVATIONS_SOFTMAXACTIVATION_H
#define PERCEPTRON_ACTIVATIONS_SOFTMAXACTIVATION_H

#include "perceptron/common/Common.h"
#include "perceptron/activations/IActivation.h"
#include "perceptron/tensors/ops/Ops.h"

namespace perceptron {
namespace activations {

class SoftMaxActivation : public IActivation {
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
SoftMaxActivation::compute_impl(tensors::TensorReadOnly2D<T, trans> inputs,
                                tensors::TensorWriteable2D<T> outputs) {
  // TODO: Finish
}

template<typename T, bool trans>
void
SoftMaxActivation::derivative_impl(tensors::TensorReadOnly2D<T, trans> inputs,
                                   tensors::TensorWriteable2D<T> outputs) {
  // TODO: Finish
}

} // perceptron
} // activations

#endif //PERCEPTRON_ACTIVATIONS_SOFTMAXACTIVATION_H
