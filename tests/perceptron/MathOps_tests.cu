#define CATCH_CONFIG_MAIN

#include <catch2/catch.hpp>

#include "perceptron/tensors/ops/Ops.h"

#include <memory>

using namespace perceptron::tensors;
using namespace perceptron::tensors::ops;

TEMPLATE_TEST_CASE("gemm_common", "[perceptron][tensors][ops]", float, double) {
  using T = TestType;
  auto alpha = static_cast<T>(1.0);
  auto A_owner = constructTensorOwner2D(linspace(static_cast<T>(1.0), static_cast<T>(9.0), 9).release(), 3, 3);
  auto B_owner = constructTensorOwner2D(linspace(static_cast<T>(1.0), static_cast<T>(9.0), 9).release(), 3, 3);
  auto beta = static_cast<T>(1.0);
  auto C_owner = set(static_cast<T>(1.0), 3, 3, DEFAULT_2D_STRIDE);

  gemm(alpha, A_owner.tensor_view().to_read_only(), B_owner.tensor_view().to_read_only(),
       beta, C_owner.tensor_view());

  C_owner.to_host();
  auto C_res_view = C_owner.tensor_view();
  REQUIRE(C_res_view(0, 0) == 31);
  REQUIRE(C_res_view(0, 1) == 37);
  REQUIRE(C_res_view(0, 2) == 43);
  REQUIRE(C_res_view(1, 0) == 67);
  REQUIRE(C_res_view(1, 1) == 82);
  REQUIRE(C_res_view(1, 2) == 97);
  REQUIRE(C_res_view(2, 0) == 103);
  REQUIRE(C_res_view(2, 1) == 127);
  REQUIRE(C_res_view(2, 2) == 151);
}

TEMPLATE_TEST_CASE("gemm_trans_A", "[perceptron][tensors][ops]", float, double) {
  using T = TestType;
  auto alpha = static_cast<T>(1.0);
  auto A_owner = constructTensorOwner2D(linspace(static_cast<T>(1.0), static_cast<T>(9.0), 9).release(), 3, 3);
  auto B_owner = constructTensorOwner2D(linspace(static_cast<T>(1.0), static_cast<T>(9.0), 9).release(), 3, 3);
  auto beta = static_cast<T>(1.0);
  auto C_owner = set(static_cast<T>(1.0), 3, 3, DEFAULT_2D_STRIDE);

  gemm(alpha, A_owner.tensor_view().to_read_only().t(), B_owner.tensor_view().to_read_only(),
       beta, C_owner.tensor_view());

  C_owner.to_host();
  auto C_res_view = C_owner.tensor_view();
  REQUIRE(C_res_view(0, 0) == 67);
  REQUIRE(C_res_view(0, 1) == 79);
  REQUIRE(C_res_view(0, 2) == 91);
  REQUIRE(C_res_view(1, 0) == 79);
  REQUIRE(C_res_view(1, 1) == 94);
  REQUIRE(C_res_view(1, 2) == 109);
  REQUIRE(C_res_view(2, 0) == 91);
  REQUIRE(C_res_view(2, 1) == 109);
  REQUIRE(C_res_view(2, 2) == 127);
}

TEMPLATE_TEST_CASE("gemm_trans_B", "[perceptron][tensors][ops]", float, double) {
  using T = TestType;
  auto alpha = static_cast<T>(1.0);
  auto A_owner = constructTensorOwner2D(linspace(static_cast<T>(1.0), static_cast<T>(9.0), 9).release(), 3, 3);
  auto B_owner = constructTensorOwner2D(linspace(static_cast<T>(1.0), static_cast<T>(9.0), 9).release(), 3, 3);
  auto beta = static_cast<T>(1.0);
  auto C_owner = set(static_cast<T>(1.0), 3, 3, DEFAULT_2D_STRIDE);

  gemm(alpha, A_owner.tensor_view().to_read_only(), B_owner.tensor_view().to_read_only().t(),
       beta, C_owner.tensor_view());

  C_owner.to_host();
  auto C_res_view = C_owner.tensor_view();
  REQUIRE(C_res_view(0, 0) == 15);
  REQUIRE(C_res_view(0, 1) == 33);
  REQUIRE(C_res_view(0, 2) == 51);
  REQUIRE(C_res_view(1, 0) == 33);
  REQUIRE(C_res_view(1, 1) == 78);
  REQUIRE(C_res_view(1, 2) == 123);
  REQUIRE(C_res_view(2, 0) == 51);
  REQUIRE(C_res_view(2, 1) == 123);
  REQUIRE(C_res_view(2, 2) == 195);
}

TEMPLATE_TEST_CASE("gemm_trans_AB", "[perceptron][tensors][ops]", float, double) {
  using T = TestType;
  auto alpha = static_cast<T>(1.0);
  auto A_owner = constructTensorOwner2D(linspace(static_cast<T>(1.0), static_cast<T>(9.0), 9).release(), 3, 3);
  auto B_owner = constructTensorOwner2D(linspace(static_cast<T>(1.0), static_cast<T>(9.0), 9).release(), 3, 3);
  auto beta = static_cast<T>(1.0);
  auto C_owner = set(static_cast<T>(1.0), 3, 3, DEFAULT_2D_STRIDE);

  gemm(alpha, A_owner.tensor_view().to_read_only().t(), B_owner.tensor_view().to_read_only().t(),
       beta, C_owner.tensor_view());

  C_owner.to_host();
  auto C_res_view = C_owner.tensor_view();
  REQUIRE(C_res_view(0, 0) == 31);
  REQUIRE(C_res_view(0, 1) == 67);
  REQUIRE(C_res_view(0, 2) == 103);
  REQUIRE(C_res_view(1, 0) == 37);
  REQUIRE(C_res_view(1, 1) == 82);
  REQUIRE(C_res_view(1, 2) == 127);
  REQUIRE(C_res_view(2, 0) == 43);
  REQUIRE(C_res_view(2, 1) == 97);
  REQUIRE(C_res_view(2, 2) == 151);
}

TEMPLATE_TEST_CASE("geam", "[perceptron][tensors][ops]", float, double) {
  using T = TestType;
  auto alpha = static_cast<T>(1.0);
  auto A_owner = constructTensorOwner2D(linspace(static_cast<T>(1.0), static_cast<T>(9.0), 9).release(), 3, 3);
  auto beta = static_cast<T>(1.0);
  auto B_owner = constructTensorOwner2D(linspace(static_cast<T>(1.0), static_cast<T>(9.0), 9).release(), 3, 3);
  auto C_owner = constructTensorOwnerDevice2D<T>(3, 3);
  geam(A_owner.tensor_view().to_read_only(), alpha,
       B_owner.tensor_view().to_read_only(), beta,
       C_owner.tensor_view());

  C_owner.to_host();
  auto C_res_view = C_owner.tensor_view();
  REQUIRE(C_res_view(0, 0) == 2);
  REQUIRE(C_res_view(0, 1) == 4);
  REQUIRE(C_res_view(0, 2) == 6);
  REQUIRE(C_res_view(1, 0) == 8);
  REQUIRE(C_res_view(1, 1) == 10);
  REQUIRE(C_res_view(1, 2) == 12);
  REQUIRE(C_res_view(2, 0) == 14);
  REQUIRE(C_res_view(2, 1) == 16);
  REQUIRE(C_res_view(2, 2) == 18);
}

TEMPLATE_TEST_CASE("geam_trans_A", "[perceptron][tensors][ops]", float, double) {
  using T = TestType;
  auto alpha = static_cast<T>(1.0);
  auto A_owner = constructTensorOwner2D(linspace(static_cast<T>(1.0), static_cast<T>(9.0), 9).release(), 3, 3);
  auto beta = static_cast<T>(1.0);
  auto B_owner = constructTensorOwner2D(linspace(static_cast<T>(1.0), static_cast<T>(9.0), 9).release(), 3, 3);
  auto C_owner = constructTensorOwnerDevice2D<T>(3, 3);
  geam(A_owner.tensor_view().to_read_only().t(), alpha,
       B_owner.tensor_view().to_read_only(), beta,
       C_owner.tensor_view());

  C_owner.to_host();
  auto C_res_view = C_owner.tensor_view();
  REQUIRE(C_res_view(0, 0) == 2);
  REQUIRE(C_res_view(0, 1) == 6);
  REQUIRE(C_res_view(0, 2) == 10);
  REQUIRE(C_res_view(1, 0) == 6);
  REQUIRE(C_res_view(1, 1) == 10);
  REQUIRE(C_res_view(1, 2) == 14);
  REQUIRE(C_res_view(2, 0) == 10);
  REQUIRE(C_res_view(2, 1) == 14);
  REQUIRE(C_res_view(2, 2) == 18);
}

TEMPLATE_TEST_CASE("geam_trans_B", "[perceptron][tensors][ops]", float, double) {
  using T = TestType;
  auto alpha = static_cast<T>(1.0);
  auto A_owner = constructTensorOwner2D(linspace(static_cast<T>(1.0), static_cast<T>(9.0), 9).release(), 3, 3);
  auto beta = static_cast<T>(1.0);
  auto B_owner = constructTensorOwner2D(linspace(static_cast<T>(1.0), static_cast<T>(9.0), 9).release(), 3, 3);
  auto C_owner = constructTensorOwnerDevice2D<T>(3, 3);
  geam(A_owner.tensor_view().to_read_only(), alpha,
       B_owner.tensor_view().to_read_only().t(), beta,
       C_owner.tensor_view());

  C_owner.to_host();
  auto C_res_view = C_owner.tensor_view();
  REQUIRE(C_res_view(0, 0) == 2);
  REQUIRE(C_res_view(0, 1) == 6);
  REQUIRE(C_res_view(0, 2) == 10);
  REQUIRE(C_res_view(1, 0) == 6);
  REQUIRE(C_res_view(1, 1) == 10);
  REQUIRE(C_res_view(1, 2) == 14);
  REQUIRE(C_res_view(2, 0) == 10);
  REQUIRE(C_res_view(2, 1) == 14);
  REQUIRE(C_res_view(2, 2) == 18);
}

TEMPLATE_TEST_CASE("geam_trans_AB", "[perceptron][tensors][ops]", float, double) {
  using T = TestType;
  auto alpha = static_cast<T>(1.0);
  auto A_owner = constructTensorOwner2D(linspace(static_cast<T>(1.0), static_cast<T>(9.0), 9).release(), 3, 3);
  auto beta = static_cast<T>(1.0);
  auto B_owner = constructTensorOwner2D(linspace(static_cast<T>(1.0), static_cast<T>(9.0), 9).release(), 3, 3);
  auto C_owner = constructTensorOwnerDevice2D<T>(3, 3);
  geam(A_owner.tensor_view().to_read_only().t(), alpha,
       B_owner.tensor_view().to_read_only().t(), beta,
       C_owner.tensor_view());

  C_owner.to_host();
  auto C_res_view = C_owner.tensor_view();
  REQUIRE(C_res_view(0, 0) == 2);
  REQUIRE(C_res_view(0, 1) == 8);
  REQUIRE(C_res_view(0, 2) == 14);
  REQUIRE(C_res_view(1, 0) == 4);
  REQUIRE(C_res_view(1, 1) == 10);
  REQUIRE(C_res_view(1, 2) == 16);
  REQUIRE(C_res_view(2, 0) == 6);
  REQUIRE(C_res_view(2, 1) == 12);
  REQUIRE(C_res_view(2, 2) == 18);
}

TEMPLATE_TEST_CASE_SIG("nrm2_2D", "[perceptron][tensors][ops]", ((typename T, bool trans), T, trans),
                       (float, false), (float, true), (double, false), (double, true)) {
  auto A_owner = constructTensorOwner2D(linspace(static_cast<T>(1.0), static_cast<T>(4.0), 4).release(), 2, 2);
  auto expected = 5.477225575051661;
  T actual;
  if constexpr (trans) {
    actual = nrm2(A_owner.tensor_view().to_read_only().t());
  } else {
    actual = nrm2(A_owner.tensor_view().to_read_only());
  }
  REQUIRE(std::fabs(expected - actual) < 1e-3);
}

TEMPLATE_TEST_CASE("scal", "[perceptron][tensors][ops]", double, float) {
  using T = TestType;
  auto A_owner = constructTensorOwner2D(linspace(static_cast<T>(1.0), static_cast<T>(4.0), 4).release(), 2, 2);
  auto scal_val = static_cast<T>(2.0);
  scal(scal_val, A_owner.tensor_view());
  A_owner.to_host();
  auto A_scaled_view = A_owner.tensor_view();
  REQUIRE(A_scaled_view(0, 0) == 2.0);
  REQUIRE(A_scaled_view(0, 1) == 4.0);
  REQUIRE(A_scaled_view(1, 0) == 6.0);
  REQUIRE(A_scaled_view(1, 1) == 8.0);
}

TEMPLATE_TEST_CASE("reverse_scal", "[perceptron][tensors][ops]", double, float) {
  using T = TestType;
  auto A_owner =
      constructTensorOwner2D(arange(static_cast<T>(2.0), static_cast<T>(8.0), static_cast<T>(2.0), true).release(),
                             2,
                             2);
  auto scal_val = static_cast<T>(1.0);
  reverse_scal(scal_val, A_owner.tensor_view());
  A_owner.to_host();
  auto A_scaled_view = A_owner.tensor_view();
  REQUIRE(A_scaled_view(0, 0) == 0.5);
  REQUIRE(A_scaled_view(0, 1) == 0.25);
  REQUIRE(std::fabs(A_scaled_view(1, 0) - 0.1666666667) < 1e-4);
  REQUIRE(A_scaled_view(1, 1) == 0.125);
}

TEMPLATE_TEST_CASE("add_row", "[perceptron][tensors][ops]", float, double) {
  using T = TestType;
  auto dst_owner = constructTensorOwner2D(linspace(static_cast<T>(1.0), static_cast<T>(4.0), 4).release(), 2, 2);
  auto row_owner = constructTensorOwner1D(linspace(static_cast<T>(1.0), static_cast<T>(2.0), 2).release(), 2);
  auto alpha = static_cast<T>(1.0);
  auto beta = static_cast<T>(1.0);
  add_row(alpha, row_owner.tensor_view().to_read_only(),
          beta, dst_owner.tensor_view());
  dst_owner.to_host();
  auto dst_view_actual = dst_owner.tensor_view();
  REQUIRE(dst_view_actual(0, 0) == 2);
  REQUIRE(dst_view_actual(0, 1) == 4);
  REQUIRE(dst_view_actual(1, 0) == 4);
  REQUIRE(dst_view_actual(1, 1) == 6);
}

TEMPLATE_TEST_CASE("add_col", "[perceptron][tensors][ops]", float, double) {
  using T = TestType;
  auto dst_owner = constructTensorOwner2D(linspace(static_cast<T>(1.0), static_cast<T>(4.0), 4).release(), 2, 2);
  auto col_owner = constructTensorOwner1D(linspace(static_cast<T>(1.0), static_cast<T>(2.0), 2).release(), 2);
  auto alpha = static_cast<T>(1.0);
  auto beta = static_cast<T>(1.0);
  add_col(alpha, col_owner.tensor_view().to_read_only(),
          beta, dst_owner.tensor_view());
  dst_owner.to_host();
  auto dst_view_actual = dst_owner.tensor_view();
  REQUIRE(dst_view_actual(0, 0) == 2);
  REQUIRE(dst_view_actual(0, 1) == 3);
  REQUIRE(dst_view_actual(1, 0) == 5);
  REQUIRE(dst_view_actual(1, 1) == 6);
}

TEMPLATE_TEST_CASE("add", "[perceptron][tensors][ops]", float, double) {
  using T = TestType;
  auto dst_owner = constructTensorOwner2D(linspace(static_cast<T>(1.0), static_cast<T>(4.0), 4).release(), 2, 2);
  auto scale = static_cast<T>(1.0);
  add(scale, dst_owner.tensor_view());
  dst_owner.to_host();
  auto dst_view_actual = dst_owner.tensor_view();
  REQUIRE(dst_view_actual(0, 0) == 2);
  REQUIRE(dst_view_actual(0, 1) == 3);
  REQUIRE(dst_view_actual(1, 0) == 4);
  REQUIRE(dst_view_actual(1, 1) == 5);
}

TEMPLATE_TEST_CASE("add_negative", "[perceptron][tensors][ops]", float, double) {
  using T = TestType;
  auto dst_owner = constructTensorOwner2D(linspace(static_cast<T>(1.0), static_cast<T>(4.0), 4).release(), 2, 2);
  auto scale = static_cast<T>(1.0);
  add_negative(scale, dst_owner.tensor_view());

  dst_owner.to_host();
  auto dst_view_actual = dst_owner.tensor_view();
  REQUIRE(dst_view_actual(0, 0) == 0);
  REQUIRE(dst_view_actual(0, 1) == -1);
  REQUIRE(dst_view_actual(1, 0) == -2);
  REQUIRE(dst_view_actual(1, 1) == -3);
}

TEMPLATE_TEST_CASE("element_wise_mul_t1_t2", "[perceptron][tensors][ops]", float, double) {
  using T = TestType;
  auto A_owner = constructTensorOwner2D(linspace(static_cast<T>(1.0), static_cast<T>(4.0), 4).release(), 2, 2);
  auto B_owner = constructTensorOwner2D(linspace(static_cast<T>(1.0), static_cast<T>(4.0), 4).release(), 2, 2);

  auto dst_owner = element_wise_mul(A_owner.tensor_view().to_read_only(),
                                    B_owner.tensor_view().to_read_only());

  dst_owner.to_host();
  auto dst_owner_result = dst_owner.tensor_view();
  REQUIRE(dst_owner_result(0, 0) == 1);
  REQUIRE(dst_owner_result(0, 1) == 4);
  REQUIRE(dst_owner_result(1, 0) == 9);
  REQUIRE(dst_owner_result(1, 1) == 16);
}

TEMPLATE_TEST_CASE("element_wise_mul_t1_t2_trans", "[perceptron][tensors][ops]", float, double) {
  using T = TestType;
  auto A_owner = constructTensorOwner2D(linspace(static_cast<T>(1.0), static_cast<T>(4.0), 4).release(), 2, 2);
  auto B_owner = constructTensorOwner2D(linspace(static_cast<T>(1.0), static_cast<T>(4.0), 4).release(), 2, 2);

  auto dst_owner = element_wise_mul(A_owner.tensor_view().to_read_only(),
                                    B_owner.tensor_view().to_read_only().t());

  dst_owner.to_host();
  auto dst_owner_result = dst_owner.tensor_view();
  REQUIRE(dst_owner_result(0, 0) == 1);
  REQUIRE(dst_owner_result(0, 1) == 6);
  REQUIRE(dst_owner_result(1, 0) == 6);
  REQUIRE(dst_owner_result(1, 1) == 16);
}

TEMPLATE_TEST_CASE("element_wise_mul_t1_trans_t2", "[perceptron][tensors][ops]", float, double) {
  using T = TestType;
  auto A_owner = constructTensorOwner2D(linspace(static_cast<T>(1.0), static_cast<T>(4.0), 4).release(), 2, 2);
  auto B_owner = constructTensorOwner2D(linspace(static_cast<T>(1.0), static_cast<T>(4.0), 4).release(), 2, 2);

  auto dst_owner = element_wise_mul(A_owner.tensor_view().to_read_only().t(),
                                    B_owner.tensor_view().to_read_only());

  dst_owner.to_host();
  auto dst_owner_result = dst_owner.tensor_view();
  REQUIRE(dst_owner_result(0, 0) == 1);
  REQUIRE(dst_owner_result(0, 1) == 6);
  REQUIRE(dst_owner_result(1, 0) == 6);
  REQUIRE(dst_owner_result(1, 1) == 16);
}

TEMPLATE_TEST_CASE("element_wise_mul_t1_trans_t2_trans", "[perceptron][tensors][ops]", float, double) {
  using T = TestType;
  auto A_owner = constructTensorOwner2D(linspace(static_cast<T>(1.0), static_cast<T>(4.0), 4).release(), 2, 2);
  auto B_owner = constructTensorOwner2D(linspace(static_cast<T>(1.0), static_cast<T>(4.0), 4).release(), 2, 2);

  auto dst_owner = element_wise_mul(A_owner.tensor_view().to_read_only().t(),
                                    B_owner.tensor_view().to_read_only().t());

  dst_owner.to_host();
  auto dst_owner_result = dst_owner.tensor_view();
  REQUIRE(dst_owner_result(0, 0) == 1);
  REQUIRE(dst_owner_result(0, 1) == 9);
  REQUIRE(dst_owner_result(1, 0) == 4);
  REQUIRE(dst_owner_result(1, 1) == 16);
}

TEMPLATE_TEST_CASE("element_wise_mul_t1", "[perceptron][tensors][ops]", float, double) {
  using T = TestType;
  auto A_owner = constructTensorOwner2D(linspace(static_cast<T>(1.0), static_cast<T>(4.0), 4).release(), 2, 2);
  auto dst_owner = constructTensorOwner2D(linspace(static_cast<T>(1.0), static_cast<T>(4.0), 4).release(), 2, 2);

  element_wise_mul(A_owner.tensor_view().to_read_only(), dst_owner.tensor_view());

  dst_owner.to_host();
  auto dst_owner_result = dst_owner.tensor_view();
  REQUIRE(dst_owner_result(0, 0) == 1);
  REQUIRE(dst_owner_result(0, 1) == 4);
  REQUIRE(dst_owner_result(1, 0) == 9);
  REQUIRE(dst_owner_result(1, 1) == 16);
}

TEMPLATE_TEST_CASE("element_wise_mul_t1_trans", "[perceptron][tensors][ops]", float, double) {
  using T = TestType;
  auto A_owner = constructTensorOwner2D(linspace(static_cast<T>(1.0), static_cast<T>(4.0), 4).release(), 2, 2);
  auto dst_owner = constructTensorOwner2D(linspace(static_cast<T>(1.0), static_cast<T>(4.0), 4).release(), 2, 2);

  element_wise_mul(A_owner.tensor_view().to_read_only().t(), dst_owner.tensor_view());

  dst_owner.to_host();
  auto dst_owner_result = dst_owner.tensor_view();
  REQUIRE(dst_owner_result(0, 0) == 1);
  REQUIRE(dst_owner_result(0, 1) == 6);
  REQUIRE(dst_owner_result(1, 0) == 6);
  REQUIRE(dst_owner_result(1, 1) == 16);
}
