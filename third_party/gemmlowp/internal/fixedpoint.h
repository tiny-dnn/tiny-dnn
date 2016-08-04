// Copyright 2015 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// fixedpoint.h: fixed-point arithmetic, with basic operations and
// a few math functions such as tanh.

// This is only used in output.h
// for some specific output pipeline stages (tanh); most of gemmlowp
// uses only plain integer arithmetic, not fixed-point arithmetic.
// At the most basic level, we distinguish between plain integer
// arithmetic and fixed-point arithmetic by the type of multiplication
// that is used: plain integer arithmetic uses plain (overflowing)
// integer multiplication, whereas fixed-point arithmetic uses
// "multiply-high" instructions, which means using only the most
// significant bits of the product, or equivalently, multiplying
// fixed-point numbers in the [-1 .. +1] interval.

#ifndef GEMMLOWP_INTERNAL_FIXEDPOINT_H_
#define GEMMLOWP_INTERNAL_FIXEDPOINT_H_

#include "common.h"

#include <limits>
#include <cassert>

namespace gemmlowp {

template <typename tIntegerType>
tIntegerType BitAnd(tIntegerType a, tIntegerType b) {
  return a & b;
}

template <typename tIntegerType>
tIntegerType BitOr(tIntegerType a, tIntegerType b) {
  return a | b;
}

template <typename tIntegerType>
tIntegerType BitXor(tIntegerType a, tIntegerType b) {
  return a ^ b;
}

template <typename tIntegerType>
tIntegerType BitNot(tIntegerType a) {
  return ~a;
}

template <typename tIntegerType>
tIntegerType Add(tIntegerType a, tIntegerType b) {
  return a + b;
}

template <typename tIntegerType>
tIntegerType Sub(tIntegerType a, tIntegerType b) {
  return a - b;
}

template <typename tIntegerType>
tIntegerType Neg(tIntegerType a) {
  return -a;
}

template <typename tIntegerType>
tIntegerType ShiftLeft(tIntegerType a, int offset) {
  return a * (1 << offset);
}

template <typename tIntegerType>
tIntegerType ShiftRight(tIntegerType a, int offset) {
  return a / (1 << offset);
}

template <typename tIntegerType>
tIntegerType SelectUsingMask(tIntegerType if_mask, tIntegerType then_val,
                             tIntegerType else_val) {
  return BitXor(BitAnd(if_mask, then_val), BitAnd(BitNot(if_mask), else_val));
}

template <typename tIntegerType>
tIntegerType MaskIfNonZero(tIntegerType a) {
  static const tIntegerType zero = 0;
  return a ? BitNot(zero) : zero;
}

template <typename tIntegerType>
tIntegerType MaskIfZero(tIntegerType a) {
  return MaskIfNonZero<tIntegerType>(!a);
}

template <typename tIntegerType>
tIntegerType MaskIfEqual(tIntegerType a, tIntegerType b) {
  return MaskIfNonZero<tIntegerType>(a == b);
}

template <typename tIntegerType>
tIntegerType MaskIfNotEqual(tIntegerType a, tIntegerType b) {
  return MaskIfNonZero<tIntegerType>(a != b);
}

template <typename tIntegerType>
tIntegerType MaskIfGreaterThan(tIntegerType a, tIntegerType b) {
  return MaskIfNonZero<tIntegerType>(a > b);
}

template <typename tIntegerType>
tIntegerType MaskIfGreaterThanOrEqual(tIntegerType a, tIntegerType b) {
  return MaskIfNonZero<tIntegerType>(a >= b);
}

template <typename tIntegerType>
tIntegerType MaskIfLessThan(tIntegerType a, tIntegerType b) {
  return MaskIfNonZero<tIntegerType>(a < b);
}

template <typename tIntegerType>
tIntegerType MaskIfLessThanOrEqual(tIntegerType a, tIntegerType b) {
  return MaskIfNonZero<tIntegerType>(a <= b);
}

template <typename tIntegerType>
bool All(tIntegerType a) {
  return a;
}

template <typename tIntegerType>
bool Any(tIntegerType a) {
  return a;
}

template <typename IntegerType>
IntegerType RoundingHalfSum(IntegerType a, IntegerType b) {
  static_assert(std::is_same<IntegerType, void>::value, "unimplemented");
  return a;
}

template <>
inline int32_t RoundingHalfSum(int32_t a, int32_t b) {
  int64_t a64 = a;
  int64_t b64 = b;
  int64_t sum = a64 + b64;
  int64_t sign = sum >= 0 ? 1 : -1;
  return static_cast<int32_t>((sum + sign) / 2);
}

template <typename IntegerType>
IntegerType SaturatingRoundingDoublingHighMul(IntegerType a, IntegerType b) {
  static_assert(std::is_same<IntegerType, void>::value, "unimplemented");
  return a;
}

// This function implements the same computation as the ARMv7 NEON VQRDMULH
// instruction.
template <>
inline int32_t SaturatingRoundingDoublingHighMul(int32_t a, int32_t b) {
  bool overflow = a == b && a == std::numeric_limits<int32_t>::min();
  int64_t a_64(a);
  int64_t b_64(b);
  int64_t ab_64 = a_64 * b_64;
  int32_t nudge = ab_64 >= 0 ? (1 << 30) : (1 - (1 << 30));
  int32_t ab_x2_high32 = static_cast<int32_t>((ab_64 + nudge) / (1ll << 31));
  return overflow ? std::numeric_limits<int32_t>::max() : ab_x2_high32;
}

template <int Exponent, typename IntegerType,
          int ExponentSign = (Exponent > 0 ? 1 : Exponent < 0 ? -1 : 0)>
struct ImplSaturatingRoundingMultiplyByPOT {};

template <int Exponent, typename IntegerType>
struct ImplSaturatingRoundingMultiplyByPOT<Exponent, IntegerType, 0> {
  static IntegerType eval(IntegerType x) { return x; }
};

template <int Exponent>
struct ImplSaturatingRoundingMultiplyByPOT<Exponent, int32_t, 1> {
  static int32_t eval(int32_t x) {
    const int64_t min = std::numeric_limits<int32_t>::min();
    const int64_t max = std::numeric_limits<int32_t>::max();
    return x >= (1 << (31 - Exponent)) ? max : x <= -(1 << (31 - Exponent))
                                                   ? min
                                                   : x * (1 << Exponent);
  }
};

template <int Exponent>
struct ImplSaturatingRoundingMultiplyByPOT<Exponent, int32_t, -1> {
  static int32_t eval(int32_t x) {
    int32_t b = (std::abs(x) & (1 << (-Exponent - 1))) >> (-Exponent - 1);
    int32_t nudge = x >= 0 ? b : -b;
    return x / (1 << -Exponent) + nudge;
  }
};

template <int Exponent, typename IntegerType>
IntegerType SaturatingRoundingMultiplyByPOT(IntegerType x) {
  return ImplSaturatingRoundingMultiplyByPOT<Exponent, IntegerType>::eval(x);
}

template <typename tIntegerType>
struct FixedPointRawTypeTraits {};

template <>
struct FixedPointRawTypeTraits<int32_t> {
  typedef int32_t ScalarRawType;
  static const int kLanes = 1;
};

template <typename tRawType>
tRawType Dup(typename FixedPointRawTypeTraits<tRawType>::ScalarRawType x) {
  return x;
}

template <typename tRawType, int tIntegerBits>
class FixedPoint {
 public:
  typedef tRawType RawType;

  typedef FixedPointRawTypeTraits<RawType> RawTypeTraits;
  typedef typename RawTypeTraits::ScalarRawType ScalarRawType;

  static const int kTotalBits = 8 * sizeof(ScalarRawType);
  static const int kIntegerBits = tIntegerBits;
  static const int kFractionalBits = kTotalBits - 1 - kIntegerBits;
  static_assert(kIntegerBits >= 0 && kIntegerBits < kTotalBits,
                "bad IntegerBits");

  typedef FixedPoint<ScalarRawType, kIntegerBits> ScalarFixedPointType;

  static const ScalarRawType ScalarRawMin() {
    return std::numeric_limits<ScalarRawType>::min();
  }

  static const ScalarRawType ScalarRawMax() {
    return std::numeric_limits<ScalarRawType>::max();
  }

  static const ScalarRawType RawMin() {
    return VectorFromScalar(ScalarRawMin());
  }

  static const ScalarRawType RawMax() {
    return VectorFromScalar(ScalarRawMax());
  }

  static FixedPoint FromRaw(RawType x) {
    FixedPoint retval;
    retval.raw() = x;
    return retval;
  }

  static FixedPoint FromScalarRaw(ScalarRawType x) {
    FixedPoint retval;
    retval.raw() = Dup<RawType>(x);
    return retval;
  }

  static FixedPoint FromScalarFixedPoint(ScalarFixedPointType x) {
    return FromScalarRaw(x.raw());
  }

  template <int Exponent>
  static FixedPoint ConstantPOT() {
    static const int kOffset = kFractionalBits + Exponent;
    static_assert(
        kOffset < 31,
        "Constant not exactly representable in this fixed-point format");
    return FromScalarRaw(ScalarRawType(1) << kOffset);
  }

  static FixedPoint Zero() { return FromScalarRaw(0); }

  static FixedPoint One() {
    return FromScalarRaw(kIntegerBits == 0
                             ? ScalarRawMax()
                             : (ScalarRawType(1) << kFractionalBits));
  }

  RawType raw() const { return i_; }
  RawType& raw() { return i_; }

 private:
  RawType i_;
};

template <typename tRawType, int tIntegerBits_a, int tIntegerBits_b>
FixedPoint<tRawType, tIntegerBits_a + tIntegerBits_b> operator*(
    FixedPoint<tRawType, tIntegerBits_a> a,
    FixedPoint<tRawType, tIntegerBits_b> b) {
  FixedPoint<tRawType, tIntegerBits_a + tIntegerBits_b> c;
  c.raw() = SaturatingRoundingDoublingHighMul(a.raw(), b.raw());
  return c;
}

template <int tExponent, typename tRawType, int tIntegerBits>
FixedPoint<tRawType, tExponent + tIntegerBits> ExactMulByPot(
    FixedPoint<tRawType, tIntegerBits> a) {
  FixedPoint<tRawType, tExponent + tIntegerBits> c;
  c.raw() = a.raw();
  return c;
}

template <int tExponent, typename tRawType, int tIntegerBits>
FixedPoint<tRawType, tIntegerBits> SaturatingRoundingMultiplyByPOT(
    FixedPoint<tRawType, tIntegerBits> a) {
  return FixedPoint<tRawType, tIntegerBits>::FromRaw(
      SaturatingRoundingMultiplyByPOT<tExponent>(a.raw()));
}

#define MAKE_FIXEDPOINT_UNARY_FUNC(FuncName, ImplFuncName)                     \
  template <typename tRawType, int tIntegerBits>                               \
  FixedPoint<tRawType, tIntegerBits> FuncName(                                 \
      FixedPoint<tRawType, tIntegerBits> a) {                                  \
    return FixedPoint<tRawType, tIntegerBits>::FromRaw(ImplFuncName(a.raw())); \
  }

#define MAKE_FIXEDPOINT_BINARY_FUNC(FuncName, ImplFuncName) \
  template <typename tRawType, int tIntegerBits>            \
  FixedPoint<tRawType, tIntegerBits> FuncName(              \
      FixedPoint<tRawType, tIntegerBits> a,                 \
      FixedPoint<tRawType, tIntegerBits> b) {               \
    return FixedPoint<tRawType, tIntegerBits>::FromRaw(     \
        ImplFuncName(a.raw(), b.raw()));                    \
  }

MAKE_FIXEDPOINT_UNARY_FUNC(operator-, Neg)
MAKE_FIXEDPOINT_UNARY_FUNC(operator~, BitNot)
MAKE_FIXEDPOINT_BINARY_FUNC(operator+, Add)
MAKE_FIXEDPOINT_BINARY_FUNC(operator-, Sub)
MAKE_FIXEDPOINT_BINARY_FUNC(operator&, BitAnd)
MAKE_FIXEDPOINT_BINARY_FUNC(operator^, BitXor)
MAKE_FIXEDPOINT_BINARY_FUNC(operator|, BitOr)
MAKE_FIXEDPOINT_BINARY_FUNC(RoundingHalfSum, RoundingHalfSum)

#undef MAKE_FIXEDPOINT_UNARY_FUNC
#undef MAKE_FIXEDPOINT_BINARY_FUNC

#define MAKE_FIXEDPOINT_UNARY_FUNC_RETURNING_RAW(FuncName)  \
  template <typename tRawType, int tIntegerBits>            \
  tRawType FuncName(FixedPoint<tRawType, tIntegerBits> a) { \
    return FuncName(a.raw());                               \
  }

#define MAKE_FIXEDPOINT_BINARY_FUNC_RETURNING_RAW(FuncName) \
  template <typename tRawType, int tIntegerBits>            \
  tRawType FuncName(FixedPoint<tRawType, tIntegerBits> a,   \
                    FixedPoint<tRawType, tIntegerBits> b) { \
    return FuncName(a.raw(), b.raw());                      \
  }

MAKE_FIXEDPOINT_UNARY_FUNC_RETURNING_RAW(MaskIfZero)
MAKE_FIXEDPOINT_UNARY_FUNC_RETURNING_RAW(MaskIfNonZero)
MAKE_FIXEDPOINT_BINARY_FUNC_RETURNING_RAW(MaskIfEqual)
MAKE_FIXEDPOINT_BINARY_FUNC_RETURNING_RAW(MaskIfNotEqual)
MAKE_FIXEDPOINT_BINARY_FUNC_RETURNING_RAW(MaskIfGreaterThan)
MAKE_FIXEDPOINT_BINARY_FUNC_RETURNING_RAW(MaskIfGreaterThanOrEqual)
MAKE_FIXEDPOINT_BINARY_FUNC_RETURNING_RAW(MaskIfLessThan)
MAKE_FIXEDPOINT_BINARY_FUNC_RETURNING_RAW(MaskIfLessThanOrEqual)

#undef MAKE_FIXEDPOINT_UNARY_FUNC_RETURNING_RAW
#undef MAKE_FIXEDPOINT_BINARY_FUNC_RETURNING_RAW

template <typename tRawType, int tIntegerBits>
FixedPoint<tRawType, tIntegerBits> SelectUsingMask(
    tRawType if_mask, FixedPoint<tRawType, tIntegerBits> then_val,
    FixedPoint<tRawType, tIntegerBits> else_val) {
  return FixedPoint<tRawType, tIntegerBits>::FromRaw(
      SelectUsingMask(if_mask, then_val.raw(), else_val.raw()));
}

template <typename tRawType, int tIntegerBits>
bool operator==(FixedPoint<tRawType, tIntegerBits> a,
                FixedPoint<tRawType, tIntegerBits> b) {
  return All(MaskIfEqual(a.raw(), b.raw()));
}

template <typename tRawType, int tIntegerBits>
bool operator!=(FixedPoint<tRawType, tIntegerBits> a,
                FixedPoint<tRawType, tIntegerBits> b) {
  return !(a == b);
}

template <typename tRawType, int tIntegerBits>
double ToDouble(FixedPoint<tRawType, tIntegerBits> x) {
  static_assert(FixedPointRawTypeTraits<tRawType>::kLanes == 1,
                "not applicable to SIMD types");
  typedef FixedPoint<tRawType, tIntegerBits> F;
  return x.raw() / double(1ll << F::kFractionalBits);
}

template <typename tRawType, int tIntegerBits>
FixedPoint<tRawType, tIntegerBits> ToFixedPoint(double x) {
  typedef FixedPoint<tRawType, tIntegerBits> F;
  return F::FromScalarRaw(static_cast<int32_t>(
      std::min(std::max(round(x * double(1ll << F::kFractionalBits)),
                        double(F::ScalarRawMin())),
               double(F::ScalarRawMax()))));
}

template <int tIntegerBitsDst, typename tRawType, int tIntegerBitsSrc>
FixedPoint<tRawType, tIntegerBitsDst> Rescale(
    FixedPoint<tRawType, tIntegerBitsSrc> x) {
  static const int kExponent = tIntegerBitsSrc - tIntegerBitsDst;
  FixedPoint<tRawType, tIntegerBitsDst> result;
  result.raw() = SaturatingRoundingMultiplyByPOT<kExponent>(x.raw());
  return result;
}

#ifdef GEMMLOWP_ENABLE_FIXEDPOINT_CONSTANTS_CHECKS
template <typename FixedPointType>
FixedPointType CheckedFixedPointConstant(
    typename FixedPointType::ScalarRawType raw_value, double double_value) {
  typedef typename FixedPointType::RawType RawType;
  static const int kIntegerBits = FixedPointType::kIntegerBits;
  FixedPointType ref = FixedPointType::FromScalarRaw(raw_value);
  FixedPointType check = ToFixedPoint<RawType, kIntegerBits>(double_value);
  assert(ref == check);
  return ref;
}
#define GEMMLOWP_CHECKED_FIXEDPOINT_CONSTANT(FixedPointType, ScalarRawValue, \
                                             DoubleValue)                    \
  (CheckedFixedPointConstant<FixedPointType>(ScalarRawValue, DoubleValue))

#else
#define GEMMLOWP_CHECKED_FIXEDPOINT_CONSTANT(FixedPointType, ScalarRawValue, \
                                             DoubleValue)                    \
  (FixedPointType::FromScalarRaw(ScalarRawValue))
#endif

template <typename tRawType>
FixedPoint<tRawType, 0> exp_on_interval_between_negative_one_quarter_and_0_excl(
    FixedPoint<tRawType, 0> a) {
  typedef FixedPoint<tRawType, 0> F;
  const F constant_term =
      GEMMLOWP_CHECKED_FIXEDPOINT_CONSTANT(F, 1895147668, std::exp(-1.0 / 8.0));
  const F constant_1_over_3 =
      GEMMLOWP_CHECKED_FIXEDPOINT_CONSTANT(F, 715827883, 1.0 / 3.0);
  // We're evaluating a Taylor expansion around -1/8, so we do the change of
  // variable: x = a + 1/8.
  // In fixed-point with 0 integer bits, 1/8 is represented by 1 << 28.
  F x = a + F::template ConstantPOT<-3>();
  F x2 = x * x;
  F x3 = x2 * x;
  F x4 = x2 * x2;
  F x4_over_4 = SaturatingRoundingMultiplyByPOT<-2>(x4);
  F x4_over_24_plus_x3_over_6_plus_x2_over_2 =
      SaturatingRoundingMultiplyByPOT<-1>(
          ((x4_over_4 + x3) * constant_1_over_3) + x2);
  return constant_term +
         constant_term * (x + x4_over_24_plus_x3_over_6_plus_x2_over_2);
}

template <typename tRawType, int tIntegerBits>
FixedPoint<tRawType, 0> exp_on_negative_values(
    FixedPoint<tRawType, tIntegerBits> a) {
  typedef FixedPoint<tRawType, tIntegerBits> InputF;
  typedef FixedPoint<tRawType, 0> ResultF;
  static const int kFractionalBits = InputF::kFractionalBits;
  static const int kIntegerBits = InputF::kIntegerBits;
  static const InputF kOneQuarter = InputF::template ConstantPOT<-2>();
  InputF mask = kOneQuarter - InputF::FromScalarRaw(1);
  InputF a_mod_quarter_minus_one_quarter = (a & mask) - kOneQuarter;
  ResultF result = exp_on_interval_between_negative_one_quarter_and_0_excl(
      Rescale<0>(a_mod_quarter_minus_one_quarter));
  tRawType remainder = (a_mod_quarter_minus_one_quarter - a).raw();

#define GEMMLOWP_EXP_BARREL_SHIFTER(Exponent, FixedPointMultiplier)         \
  if (kIntegerBits > Exponent) {                                            \
    const ResultF kMultiplier = GEMMLOWP_CHECKED_FIXEDPOINT_CONSTANT(       \
        ResultF, FixedPointMultiplier, std::exp(-std::pow(2.0, Exponent))); \
    result = SelectUsingMask(                                               \
        MaskIfNonZero(BitAnd(                                               \
            remainder, Dup<tRawType>(1 << (kFractionalBits + Exponent)))),  \
        result * kMultiplier, result);                                      \
  }

  GEMMLOWP_EXP_BARREL_SHIFTER(-2, 1672461947);
  GEMMLOWP_EXP_BARREL_SHIFTER(-1, 1302514674);
  GEMMLOWP_EXP_BARREL_SHIFTER(+0, 790015084);
  GEMMLOWP_EXP_BARREL_SHIFTER(+1, 290630308);
  GEMMLOWP_EXP_BARREL_SHIFTER(+2, 39332535);
  GEMMLOWP_EXP_BARREL_SHIFTER(+3, 720401);
  GEMMLOWP_EXP_BARREL_SHIFTER(+4, 242);

#undef GEMMLOWP_EXP_BARREL_SHIFTER

  if (kIntegerBits > 5) {
    static const int b = kIntegerBits > 5 ? kFractionalBits + 5 : 0;
    const InputF clamp =
        GEMMLOWP_CHECKED_FIXEDPOINT_CONSTANT(InputF, -(1 << b), -32.0);
    result = SelectUsingMask(MaskIfLessThan(a, clamp), ResultF::Zero(), result);
  }

  result = SelectUsingMask(MaskIfZero(a), ResultF::One(), result);
  return result;
}

template <typename tRawType>
FixedPoint<tRawType, 0> one_minus_x_over_one_plus_x_for_x_in_0_1(
    FixedPoint<tRawType, 0> a) {
  typedef FixedPoint<tRawType, 0> F0;
  typedef FixedPoint<tRawType, 2> F2;
  F0 half_denominator = RoundingHalfSum(a, F0::One());
  const F2 constant_48_over_17 =
      GEMMLOWP_CHECKED_FIXEDPOINT_CONSTANT(F2, 1515870810, 48.0 / 17.0);
  const F2 constant_neg_32_over_17 =
      GEMMLOWP_CHECKED_FIXEDPOINT_CONSTANT(F2, -1010580540, -32.0 / 17.0);
  F2 x = constant_48_over_17 + half_denominator * constant_neg_32_over_17;
  for (int i = 0; i < 3; i++) {
    F2 half_denominator_times_x = half_denominator * x;
    F2 one_minus_half_denominator_times_x =
        F2::One() - half_denominator_times_x;
    x = x + Rescale<2>(x * one_minus_half_denominator_times_x);
  }
  return Rescale<0>(x - F2::One());
}

template <typename tRawType, int tIntegerBits>
FixedPoint<tRawType, 0> neg_tanh_on_negative_values(
    FixedPoint<tRawType, tIntegerBits> a) {
  return one_minus_x_over_one_plus_x_for_x_in_0_1(
      exp_on_negative_values(ExactMulByPot<1>(a)));
}

template <typename tRawType, int tIntegerBits>
FixedPoint<tRawType, 0> tanh(FixedPoint<tRawType, tIntegerBits> a) {
  typedef FixedPoint<tRawType, tIntegerBits> InputF;
  typedef FixedPoint<tRawType, 0> ResultF;
  tRawType mask_if_negative = MaskIfLessThan(a, InputF::Zero());
  tRawType mask_if_zero = MaskIfZero(a);
  InputF n = SelectUsingMask(mask_if_negative, a, -a);
  ResultF t = neg_tanh_on_negative_values(n);
  return SelectUsingMask(mask_if_zero, ResultF::Zero(),
                         SelectUsingMask(mask_if_negative, -t, t));
}

}  // end namespace gemmlowp

#ifdef GEMMLOWP_NEON
#include "fixedpoint_neon.h"
#endif

#endif  // GEMMLOWP_INTERNAL_FIXEDPOINT_H_
