#ifndef MLP_INCLUDE_PERCEPTRON_TENSORS_SHUFFLER_ISHUFFLER_H
#define MLP_INCLUDE_PERCEPTRON_TENSORS_SHUFFLER_ISHUFFLER_H

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
};

} // perceptron
} // tensors
} // shufflers

#endif //MLP_INCLUDE_PERCEPTRON_TENSORS_SHUFFLER_ISHUFFLER_H
