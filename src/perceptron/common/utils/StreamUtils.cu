#include "perceptron/common/utils/StreamUtils.h"

namespace perceptron {
namespace utils {

CudaStreamOwner
cu_create_stream() {
  std::unique_ptr<cudaStream_t> stream_ptr{new cudaStream_t};
  CUDA_CHECK(cudaStreamCreate(stream_ptr.get()));
  return CudaStreamOwner{stream_ptr.release(), cu_stream_deleter{}};
}

CudaStreamsOwner
cu_create_streams(size_type size) {
  CudaStreamsOwner streams{};
  streams.reserve(size);

  for (size_type i = 0; i < size; ++i) {
    streams.push_back(cu_create_stream());
  }

  return streams;
}

void
cu_wait_streams(const CudaStreamsOwner &streams) {
  for (auto &&stream : streams) {
    CUDA_CHECK(cudaStreamSynchronize(*stream));
  }
}

} // perceptron
} // utils
