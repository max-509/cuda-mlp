#ifndef PERCEPTRON_TENSORS_SHUFFLER_ISHUFFLER_H
#define PERCEPTRON_TENSORS_SHUFFLER_ISHUFFLER_H

#include "perceptron/common/Common.h"
#include "perceptron/tensors/Tensor1D.h"
#include "perceptron/tensors/Tensor2D.h"

namespace perceptron {
namespace tensors {
namespace shufflers {

class IShuffler {
public:

  virtual void
  shuffle() = 0;

  virtual TensorOwner2D<float>
  get_shuffled(TensorReadOnly2D<float, false> tensor_to_shuffle,
               cudaStream_t stream) = 0;

  virtual TensorOwner2D<float>
  get_shuffled(TensorReadOnly2D<float, true> tensor_to_shuffle,
               cudaStream_t stream) = 0;

  virtual TensorOwner2D<double>
  get_shuffled(TensorReadOnly2D<double, false> tensor_to_shuffle,
               cudaStream_t stream) = 0;

  virtual TensorOwner2D<double>
  get_shuffled(TensorReadOnly2D<double, true> tensor_to_shuffle,
               cudaStream_t stream) = 0;

  virtual void
  get_shuffled(TensorReadOnly2D<float, false> tensor_to_shuffle,
               TensorWriteable2D<float> shuffled,
               cudaStream_t stream) = 0;

  virtual void
  get_shuffled(TensorReadOnly2D<float, true> tensor_to_shuffle,
               TensorWriteable2D<float> shuffled,
               cudaStream_t stream) = 0;

  virtual void
  get_shuffled(TensorReadOnly2D<double, false> tensor_to_shuffle,
               TensorWriteable2D<double> shuffled,
               cudaStream_t stream) = 0;

  virtual void
  get_shuffled(TensorReadOnly2D<double, true> tensor_to_shuffle,
               TensorWriteable2D<double> shuffled,
               cudaStream_t stream) = 0;

  TensorOwner2D<float>
  get_shuffled(TensorReadOnly2D<float, false> tensor_to_shuffle);

  TensorOwner2D<float>
  get_shuffled(TensorReadOnly2D<float, true> tensor_to_shuffle);

  TensorOwner2D<double>
  get_shuffled(TensorReadOnly2D<double, false> tensor_to_shuffle);

  TensorOwner2D<double>
  get_shuffled(TensorReadOnly2D<double, true> tensor_to_shuffle);

  void
  get_shuffled(TensorReadOnly2D<float, false> tensor_to_shuffle,
               TensorWriteable2D<float> shuffled);

  void
  get_shuffled(TensorReadOnly2D<float, true> tensor_to_shuffle,
               TensorWriteable2D<float> shuffled);

  void
  get_shuffled(TensorReadOnly2D<double, false> tensor_to_shuffle,
               TensorWriteable2D<double> shuffled);

  void
  get_shuffled(TensorReadOnly2D<double, true> tensor_to_shuffle,
               TensorWriteable2D<double> shuffled);

};

} // perceptron
} // tensors
} // shufflers

#endif //PERCEPTRON_TENSORS_SHUFFLER_ISHUFFLER_H
