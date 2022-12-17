#ifndef PERCEPTRON_COMMON_UTILS_CURANDUTILS_H
#define PERCEPTRON_COMMON_UTILS_CURANDUTILS_H

#include "perceptron/common/Common.h"
#include "perceptron/common/utils/MemoryUtils.h"

#include <cstdio>
#include <stdexcept>
#include <memory>

#include <curand.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#define CURAND_CHECK(err)                                                                       \
  do                                                                                            \
  {                                                                                             \
    curandStatus_t err__ = (err);                                                               \
    if (err__ != CURAND_STATUS_SUCCESS)                                                         \
    {                                                                                           \
      std::fprintf(stderr, "curand error %s at %s:%d\n",                                        \
                   perceptron::utils::curandGetErrorString(err__), __FILE__, __LINE__);         \
      throw std::runtime_error(curandGetErrorString(err__));                                    \
    }                                                                                           \
  } while (0)

namespace perceptron {
namespace utils {

const char *
curandGetErrorString(curandStatus_t error);

struct curand_generator_deleter_t {
  void
  operator()(void *ptr) const {
    CURAND_CHECK(curandDestroyGenerator(static_cast<curandGenerator_t>(ptr)));
  }
};

struct curand_uniform_tag {
  double a{0.0};
  double b{1.0};
};
struct curand_log_normal_tag {
  double mean{0.0};
  double stddev{1.0};
};
struct curand_normal_tag {
  double mean{0.0};
  double stddev{1.0};
};
struct curand_binomial_tag {};
struct curand_poisonn_tag {};

using CurandGeneratorOwner = std::unique_ptr<curandGenerator_t, curand_generator_deleter_t>;
using CurandStatesOwner = std::unique_ptr<curandState, cu_memory_deleter_t>;

CurandGeneratorOwner
curand_create_generator(size_type seed, curandRngType rng_type = CURAND_RNG_PSEUDO_DEFAULT);

CurandStatesOwner
curand_create_states(size_type seed, size_type size, cudaStream_t stream = nullptr);

void
curand_set_stream(const CurandGeneratorOwner &gen,
                  cudaStream_t stream);

void
curand_generate(const CurandGeneratorOwner &gen,
                curand_uniform_tag tag,
                size_type n,
                float *ptr);

void
curand_generate(const CurandGeneratorOwner &gen,
                curand_log_normal_tag tag,
                size_type n,
                float *ptr);

void
curand_generate(const CurandGeneratorOwner &gen,
                curand_normal_tag tag,
                size_type n,
                float *ptr);

void
curand_generate(const CurandGeneratorOwner &gen,
                curand_uniform_tag tag,
                size_type n,
                double *ptr);

void
curand_generate(const CurandGeneratorOwner &gen,
                curand_log_normal_tag tag,
                size_type n,
                double *ptr);

void
curand_generate(const CurandGeneratorOwner &gen,
                curand_normal_tag tag,
                size_type n,
                double *ptr);

} // perceptron
} // utils

#endif //PERCEPTRON_COMMON_UTILS_CURANDUTILS_H
