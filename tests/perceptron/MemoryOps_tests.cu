#define CATCH_CONFIG_MAIN

#include <catch2/catch.hpp>

#include "perceptron/tensors/ops/Ops.h"

#include <memory>

using namespace perceptron::tensors;
using namespace perceptron::tensors::ops;

TEMPLATE_TEST_CASE_SIG("copy", "[perceptron][tensors][ops]", ((typename T, bool trans), T, trans),
                       (float, false), (float, true), (double, false), (double, true)) {
  auto src_owner = constructTensorOwner2D(linspace(static_cast<T>(1.0), static_cast<T>(4.0), 4).release(), 2, 2);
  auto dst_owner = constructTensorOwnerDevice2D<T>(2, 2);
  T r11 = 1.0, r22 = 4.0, r12, r21;
  if constexpr (trans) {
    copy(src_owner.tensor_view().to_read_only().t(),
         dst_owner.tensor_view());
    r12 = 3.0;
    r21 = 2.0;
  } else {
    copy(src_owner.tensor_view().to_read_only(),
         dst_owner.tensor_view());
    r12 = 2.0;
    r21 = 3.0;
  }
  dst_owner.to_host();
  auto dst_view_result = dst_owner.tensor_view();
  REQUIRE(dst_view_result(0, 0) == r11);
  REQUIRE(dst_view_result(0, 1) == r12);
  REQUIRE(dst_view_result(1, 0) == r21);
  REQUIRE(dst_view_result(1, 1) == r22);
}

TEMPLATE_TEST_CASE("copy_or_zero", "[perceptron][tensors][ops]", float, double) {
  using T = TestType;
  auto src_owner = constructTensorOwner2D(linspace(static_cast<T>(-4.0), static_cast<T>(4.0), 9).release(), 3, 3);
  auto dst_owner = constructTensorOwnerDevice2D<T>(3, 3);
  T r11 = 0.0, r22 = 0.0, r33 = 4.0, r12, r13, r21, r23, r31, r32;
  copy_or_zero(src_owner.tensor_view().to_read_only(),
               [] DEVICE_CALLABLE(T value) { return value >= static_cast<T>(0.0); },
               dst_owner.tensor_view());
  r12 = 0.0;
  r13 = 0.0;
  r21 = 0.0;
  r23 = 1.0;
  r31 = 2.0;
  r32 = 3.0;

  dst_owner.to_host();
  auto dst_view_result = dst_owner.tensor_view();
  REQUIRE(dst_view_result(0, 0) == r11);
  REQUIRE(dst_view_result(0, 1) == r12);
  REQUIRE(dst_view_result(0, 2) == r13);
  REQUIRE(dst_view_result(1, 0) == r21);
  REQUIRE(dst_view_result(1, 1) == r22);
  REQUIRE(dst_view_result(1, 2) == r23);
  REQUIRE(dst_view_result(2, 0) == r31);
  REQUIRE(dst_view_result(2, 1) == r32);
  REQUIRE(dst_view_result(2, 2) == r33);
}

TEMPLATE_TEST_CASE("copy_or_zero_trans", "[perceptron][tensors][ops]", float, double) {
  using T = TestType;
  auto src_owner = constructTensorOwner2D(linspace(static_cast<T>(-4.0), static_cast<T>(4.0), 9).release(), 3, 3);
  auto dst_owner = constructTensorOwnerDevice2D<T>(3, 3);
  T r11 = 0.0, r22 = 0.0, r33 = 4.0, r12, r13, r21, r23, r31, r32;
  copy_or_zero(src_owner.tensor_view().to_read_only().t(),
               [] DEVICE_CALLABLE(T value) { return value >= static_cast<T>(0.0); },
               dst_owner.tensor_view());
  r21 = 0.0;
  r12 = 0.0;
  r31 = 0.0;
  r32 = 1.0;
  r13 = 2.0;
  r23 = 3.0;

  dst_owner.to_host();
  auto dst_view_result = dst_owner.tensor_view();
  REQUIRE(dst_view_result(0, 0) == r11);
  REQUIRE(dst_view_result(0, 1) == r12);
  REQUIRE(dst_view_result(0, 2) == r13);
  REQUIRE(dst_view_result(1, 0) == r21);
  REQUIRE(dst_view_result(1, 1) == r22);
  REQUIRE(dst_view_result(1, 2) == r23);
  REQUIRE(dst_view_result(2, 0) == r31);
  REQUIRE(dst_view_result(2, 1) == r32);
  REQUIRE(dst_view_result(2, 2) == r33);
}

TEMPLATE_TEST_CASE("set_or_zero", "[perceptron][tensors][ops]", float, double) {
  using T = TestType;
  auto dst_owner = constructTensorOwner2D(linspace(static_cast<T>(-4.0), static_cast<T>(4.0), 9).release(), 3, 3);
  T value = -1.0;
  T r11 = 0.0, r22 = value, r33 = value, r12 = 0.0, r13 = 0.0, r21 = 0.0, r23 = value, r31 = value, r32 = value;
  set_or_zero(value,
              [] DEVICE_CALLABLE(T value) { return value >= static_cast<T>(0.0); },
              dst_owner.tensor_view());
  dst_owner.to_host();
  auto dst_view_result = dst_owner.tensor_view();

  REQUIRE(dst_view_result(0, 0) == r11);
  REQUIRE(dst_view_result(0, 1) == r12);
  REQUIRE(dst_view_result(0, 2) == r13);
  REQUIRE(dst_view_result(1, 0) == r21);
  REQUIRE(dst_view_result(1, 1) == r22);
  REQUIRE(dst_view_result(1, 2) == r23);
  REQUIRE(dst_view_result(2, 0) == r31);
  REQUIRE(dst_view_result(2, 1) == r32);
  REQUIRE(dst_view_result(2, 2) == r33);
}
