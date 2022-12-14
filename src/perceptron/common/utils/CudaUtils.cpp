#include "perceptron/common/utils/CudaUtils.h"

namespace perceptron {
namespace utils {

size_type
block_size_by_threads(size_type size, size_type n_threads) {
  return (size / n_threads) + static_cast<size_type>((size % n_threads) != 0);
}

} // perceptron
} // utils
