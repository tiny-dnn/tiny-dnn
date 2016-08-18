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

// fixedpoint_neon.h: optimized NEON specializations of the templates
// in fixedpoint.h.

#ifndef GEMMLOWP_INTERNAL_FIXEDPOINT_NEON_H_
#define GEMMLOWP_INTERNAL_FIXEDPOINT_NEON_H_

#include "fixedpoint.h"

#include <arm_neon.h>

namespace gemmlowp {

template <>
inline int32x4_t BitAnd(int32x4_t a, int32x4_t b) {
  return vandq_s32(a, b);
}

template <>
inline int32x4_t BitOr(int32x4_t a, int32x4_t b) {
  return vorrq_s32(a, b);
}

template <>
inline int32x4_t BitXor(int32x4_t a, int32x4_t b) {
  return veorq_s32(a, b);
}

template <>
inline int32x4_t BitNot(int32x4_t a) {
  return veorq_s32(a, vdupq_n_s32(-1));
}

template <>
inline int32x4_t Add(int32x4_t a, int32x4_t b) {
  return vaddq_s32(a, b);
}

template <>
inline int32x4_t Sub(int32x4_t a, int32x4_t b) {
  return vsubq_s32(a, b);
}

template <>
inline int32x4_t Neg(int32x4_t a) {
  return vnegq_s32(a);
}

template <>
inline int32x4_t ShiftLeft(int32x4_t a, int offset) {
  return vshlq_s32(a, vdupq_n_s32(offset));
}

template <>
inline int32x4_t ShiftRight(int32x4_t a, int offset) {
  return vshlq_s32(a, vdupq_n_s32(-offset));
}

template <>
inline int32x4_t SelectUsingMask(int32x4_t if_mask, int32x4_t then_val,
                                 int32x4_t else_val) {
  return vbslq_s32(vreinterpretq_u32_s32(if_mask), then_val, else_val);
}

template <>
inline int32x4_t MaskIfEqual(int32x4_t a, int32x4_t b) {
  return vreinterpretq_s32_u32(vceqq_s32(a, b));
}

template <>
inline int32x4_t MaskIfNotEqual(int32x4_t a, int32x4_t b) {
  return BitNot(MaskIfEqual(a, b));
}

template <>
inline int32x4_t MaskIfZero(int32x4_t a) {
  return MaskIfEqual(a, vdupq_n_s32(0));
}

template <>
inline int32x4_t MaskIfNonZero(int32x4_t a) {
  return vreinterpretq_s32_u32(vtstq_s32(a, a));
}

template <>
inline int32x4_t MaskIfGreaterThan(int32x4_t a, int32x4_t b) {
  return vreinterpretq_s32_u32(vcgtq_s32(a, b));
}

template <>
inline int32x4_t MaskIfGreaterThanOrEqual(int32x4_t a, int32x4_t b) {
  return vreinterpretq_s32_u32(vcgeq_s32(a, b));
}

template <>
inline int32x4_t MaskIfLessThan(int32x4_t a, int32x4_t b) {
  return vreinterpretq_s32_u32(vcltq_s32(a, b));
}

template <>
inline int32x4_t MaskIfLessThanOrEqual(int32x4_t a, int32x4_t b) {
  return vreinterpretq_s32_u32(vcleq_s32(a, b));
}

template <>
inline bool All(int32x4_t a) {
  a = vandq_s32(a, vextq_s32(a, a, 1));
  a = vandq_s32(a, vextq_s32(a, a, 2));
  return vgetq_lane_s32(a, 0);
}

template <>
inline bool Any(int32x4_t a) {
  a = vorrq_s32(a, vextq_s32(a, a, 1));
  a = vorrq_s32(a, vextq_s32(a, a, 2));
  return vgetq_lane_s32(a, 0);
}

template <>
inline int32x4_t RoundingHalfSum(int32x4_t a, int32x4_t b) {
  return vrhaddq_s32(a, b);
}

template <>
inline int32x4_t SaturatingRoundingDoublingHighMul(int32x4_t a, int32x4_t b) {
  return vqrdmulhq_s32(a, b);
}

template <int Exponent>
struct ImplSaturatingRoundingMultiplyByPOT<Exponent, int32x4_t, 1> {
  static int32x4_t eval(int32x4_t x) { return vqshlq_n_s32(x, Exponent); }
};

template <int Exponent>
struct ImplSaturatingRoundingMultiplyByPOT<Exponent, int32x4_t, -1> {
  static int32x4_t eval(int32x4_t x) { return vrshrq_n_s32(x, -Exponent); }
};

template <>
struct FixedPointRawTypeTraits<int32x4_t> {
  typedef int32_t ScalarRawType;
  static const int kLanes = 4;
};

template <>
inline int32x4_t Dup<int32x4_t>(int32_t x) {
  return vdupq_n_s32(x);
}

}  // end namespace gemmlowp

#endif  // GEMMLOWP_INTERNAL_FIXEDPOINT_NEON_H_
