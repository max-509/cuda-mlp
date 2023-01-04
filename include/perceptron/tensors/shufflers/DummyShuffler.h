#ifndef PERCEPTRON_TENSORS_SHUFFLERS_DUMMYSHUFFLER_H
#define PERCEPTRON_TENSORS_SHUFFLERS_DUMMYSHUFFLER_H

#include "perceptron/common/Common.h"
#include "perceptron/tensors/shufflers/IShuffler.h"
#include "perceptron/tensors/ops/Ops.h"

namespace perceptron {
namespace tensors {
namespace shufflers {

class DummyShuffler final : public IShuffler {
public:
  using IShuffler::get_shuffled;

  void
  shuffle() override;

  TensorOwner2D<float>
  get_shuffled(TensorReadOnly2D<float, false> tensor_to_shuffle,
               cudaStream_t stream) override;

  TensorOwner2D<float>
  get_shuffled(TensorReadOnly2D<float, true> tensor_to_shuffle,
               cudaStream_t stream) override;

  TensorOwner2D<double>
  get_shuffled(TensorReadOnly2D<double, false> tensor_to_shuffle,
               cudaStream_t stream) override;

  TensorOwner2D<double>
  get_shuffled(TensorReadOnly2D<double, true> tensor_to_shuffle,
               cudaStream_t stream) override;

  void
  get_shuffled(TensorReadOnly2D<float, false> tensor_to_shuffle,
               TensorWriteable2D<float> shuffled,
               cudaStream_t stream) override;

  void
  get_shuffled(TensorReadOnly2D<float, true> tensor_to_shuffle,
               TensorWriteable2D<float> shuffled,
               cudaStream_t stream) override;

  void
  get_shuffled(TensorReadOnly2D<double, false> tensor_to_shuffle,
               TensorWriteable2D<double> shuffled,
               cudaStream_t stream) override;

  void
  get_shuffled(TensorReadOnly2D<double, true> tensor_to_shuffle,
               TensorWriteable2D<double> shuffled,
               cudaStream_t stream) override;
private:

  template<typename T, bool trans>
  void
  get_shuffled_impl(TensorReadOnly2D<T, trans> tensor_to_shuffle,
                    TensorWriteable2D<T> shuffled,
                    cudaStream_t stream);
};

template<typename T, bool trans>
void
DummyShuffler::get_shuffled_impl(TensorReadOnly2D<T, trans> tensor_to_shuffle,
                                 TensorWriteable2D<T> shuffled,
                                 cudaStream_t stream) {
  ops::copy(tensor_to_shuffle, shuffled, stream);
}

} // perceptron
} // tensors
} // shufflers

#endif //PERCEPTRON_TENSORS_SHUFFLERS_DUMMYSHUFFLER_H
