#include "perceptron/common/utils/CurandUtils.h"

namespace perceptron {
namespace utils {
namespace details {

__global__
static void
curand_states_init(size_type seed, size_type size, curandState *state) {
  auto id = threadIdx.x + blockIdx.x * blockDim.x;
  if (id < size) {
    curand_init(seed, id, 0, &state[id]);
  }
}

} // details

const char *
curandGetErrorString(curandStatus_t error) {
  switch (error) {
  case CURAND_STATUS_SUCCESS:return "CURAND_STATUS_SUCCESS";
  case CURAND_STATUS_VERSION_MISMATCH:return "CURAND_STATUS_VERSION_MISMATCH";
  case CURAND_STATUS_NOT_INITIALIZED:return "CURAND_STATUS_NOT_INITIALIZED";
  case CURAND_STATUS_ALLOCATION_FAILED:return "CURAND_STATUS_ALLOCATION_FAILED";
  case CURAND_STATUS_TYPE_ERROR:return "CURAND_STATUS_TYPE_ERROR";
  case CURAND_STATUS_OUT_OF_RANGE:return "CURAND_STATUS_OUT_OF_RANGE";
  case CURAND_STATUS_LENGTH_NOT_MULTIPLE:return "CURAND_STATUS_LENGTH_NOT_MULTIPLE";
  case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED:return "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED";
  case CURAND_STATUS_LAUNCH_FAILURE:return "CURAND_STATUS_LAUNCH_FAILURE";
  case CURAND_STATUS_PREEXISTING_FAILURE:return "CURAND_STATUS_PREEXISTING_FAILURE";
  case CURAND_STATUS_INITIALIZATION_FAILED:return "CURAND_STATUS_INITIALIZATION_FAILED";
  case CURAND_STATUS_ARCH_MISMATCH:return "CURAND_STATUS_ARCH_MISMATCH";
  case CURAND_STATUS_INTERNAL_ERROR:return "CURAND_STATUS_INTERNAL_ERROR";
  }

  return "<unknown>";
}

CurandGeneratorOwner
curand_create_generator(size_type seed, curandRngType rng_type) {
  std::unique_ptr<curandGenerator_t> gen_ptr{new curandGenerator_t};
  CURAND_CHECK(curandCreateGenerator(gen_ptr.get(), rng_type));
  CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(*gen_ptr, seed));
  return CurandGeneratorOwner{gen_ptr.release(), curand_generator_deleter_t{}};
}

CurandStatesOwner
curand_create_states(size_type seed, size_type size, cudaStream_t stream) {
  auto states_ptr = cu_make_memory_unique<curandState_t>(size, stream);
  dim3 threads(utils::DEFAULT_BLOCK_SIZE_2D * utils::DEFAULT_BLOCK_SIZE_2D);
  dim3 blocks(utils::block_size_by_threads(size, threads.x));
  details::curand_states_init<<<blocks, threads, 0, stream>>>(seed, size, states_ptr.get());
  return states_ptr;
}

void
curand_set_stream(const CurandGeneratorOwner &gen,
                  cudaStream_t stream) {
  CURAND_CHECK(curandSetStream(*gen, stream));
}

void
curand_generate(const CurandGeneratorOwner &gen,
                curand_uniform_tag tag,
                size_type n,
                float *ptr) {
  CURAND_CHECK(curandGenerateUniform(*gen, ptr, n));
}

void
curand_generate(const CurandGeneratorOwner &gen,
                curand_log_normal_tag tag,
                size_type n,
                float *ptr) {
  CURAND_CHECK(curandGenerateLogNormal(*gen, ptr, n, tag.mean, tag.stddev));
}

void
curand_generate(const CurandGeneratorOwner &gen,
                curand_normal_tag tag,
                size_type n,
                float *ptr) {
  CURAND_CHECK(curandGenerateNormal(*gen, ptr, n, tag.mean, tag.stddev));
}

void
curand_generate(const CurandGeneratorOwner &gen,
                curand_uniform_tag tag,
                size_type n,
                double *ptr) {
  CURAND_CHECK(curandGenerateUniformDouble(*gen, ptr, n));
}

void
curand_generate(const CurandGeneratorOwner &gen,
                curand_log_normal_tag tag,
                size_type n,
                double *ptr) {
  CURAND_CHECK(curandGenerateLogNormalDouble(*gen, ptr, n, tag.mean, tag.stddev));
}

void
curand_generate(const CurandGeneratorOwner &gen,
                curand_normal_tag tag,
                size_type n,
                double *ptr) {
  CURAND_CHECK(curandGenerateNormalDouble(*gen, ptr, n, tag.mean, tag.stddev));
}

} // perceptron
} // utils
