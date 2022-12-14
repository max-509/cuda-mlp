#ifndef PERCEPTRON_COMMON_UTILS_CUDAUTILS_H
#define PERCEPTRON_COMMON_UTILS_CUDAUTILS_H

#include "perceptron/common/Common.h"

namespace perceptron {
namespace utils {

static constexpr size_type DEFAULT_BLOCK_SIZE = 16;

size_type
block_size_by_threads(size_type size, size_type n_threads);

} // perceptron
} // utils

#endif //PERCEPTRON_COMMON_UTILS_CUDAUTILS_H
