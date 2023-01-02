#include "perceptron/common/utils/MemoryUtils.h"

namespace perceptron {
namespace utils {

bool
is_unregistered_host(const cudaPointerAttributes &attrs) {
  return attrs.type == cudaMemoryTypeUnregistered;
}

bool
is_device(const cudaPointerAttributes &attrs) {
  return attrs.type == cudaMemoryTypeDevice;
}

bool
is_host(const cudaPointerAttributes &attrs) {
  return attrs.type == cudaMemoryTypeHost;
}

bool
is_managed(const cudaPointerAttributes &attrs) {
  return attrs.type == cudaMemoryTypeManaged;
}

} // perceptron
} // utils
