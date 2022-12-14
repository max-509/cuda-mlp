#include "perceptron/tensors/ops/MathOps.h"

#include <cublas_v2.h>

#include <cassert>

namespace perceptron {
namespace tensors {
namespace ops {

static cublasOperation_t
cu_is_transposed(TensorReadOnly2D<float> M);

void geam(cublasHandle_t handle,
          TensorReadOnly2D<float> A,
          const float alpha,
          TensorReadOnly2D<float> B,
          const float beta,
          TensorWriteable2D<float> &C) {
}

void geam(cublasHandle_t handle,
          TensorReadOnly2D<double> A,
          const double alpha,
          TensorReadOnly2D<double> B,
          const double beta,
          TensorWriteable2D<double> &C);

void geam(cublasHandle_t handle,
          TensorReadOnly2D<double> A,
          const float alpha,
          TensorWriteable2D<double> C,
          const float beta);

static cublasOperation_t
cu_is_transposed(TensorReadOnly2D<float> M) {
  auto transm = CUBLAS_OP_N;
  if (M.transposed()) {
    transm = CUBLAS_OP_T;
  }

  return transm;
}

} // perceptron
} // tensors
} // ops
