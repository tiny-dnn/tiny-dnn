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

// common.h: contains stuff that's used throughout gemmlowp
// and should always be available.

#ifndef GEMMLOWP_INTERNAL_COMMON_H_
#define GEMMLOWP_INTERNAL_COMMON_H_

#include <thread>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>

#include "../profiling/instrumentation.h"

// Our inline assembly path assume GCC/Clang syntax.
// Native Client doesn't seem to support inline assembly(?).
#if defined(__GNUC__) && !defined(__native_client__)
#define GEMMLOWP_ALLOW_INLINE_ASM
#endif

// Define macro statement that avoids inlining for GCC.
// For non-GCC, define as empty macro.
#if defined(__GNUC__)
#define GEMMLOWP_NOINLINE __attribute__((noinline))
#else
#define GEMMLOWP_NOINLINE
#endif

// Detect ARM, 32-bit or 64-bit
#ifdef __arm__
#define GEMMLOWP_ARM_32
#endif

#ifdef __aarch64__
#define GEMMLOWP_ARM_64
#endif

#if defined(GEMMLOWP_ARM_32) || defined(GEMMLOWP_ARM_64)
#define GEMMLOWP_ARM
#endif

// Detect x86, 32-bit or 64-bit
#if defined(__i386__) || defined(_M_IX86) || defined(_X86_) || defined(__i386)
#define GEMMLOWP_X86_32
#endif

#if defined(__x86_64__) || defined(_M_X64) || defined(__amd64)
#define GEMMLOWP_X86_64
#endif

#if defined(GEMMLOWP_X86_32) || defined(GEMMLOWP_X86_64)
#define GEMMLOWP_X86
#endif

// Some of our optimized paths use inline assembly and for
// now we don't bother enabling some other optimized paths using intrinddics
// where we can't use inline assembly paths.
#ifdef GEMMLOWP_ALLOW_INLINE_ASM

// Detect NEON. It's important to check for both tokens.
#if (defined __ARM_NEON) || (defined __ARM_NEON__)
#define GEMMLOWP_NEON
#endif

// Convenience NEON tokens for 32-bit or 64-bit
#if defined(GEMMLOWP_NEON) && defined(GEMMLOWP_ARM_32)
#define GEMMLOWP_NEON_32
#endif

#if defined(GEMMLOWP_NEON) && defined(GEMMLOWP_ARM_64)
#define GEMMLOWP_NEON_64
#endif

// Detect SSE.
#ifdef __SSE4_1__
#define GEMMLOWP_SSE4
#endif

#ifdef __SSE3__
#define GEMMLOWP_SSE3
#endif

// Convenience SSE4 tokens for 32-bit or 64-bit
#if defined(GEMMLOWP_SSE4) && defined(GEMMLOWP_X86_32)
#define GEMMLOWP_SSE4_32
#endif

#if defined(GEMMLOWP_SSE3) && defined(GEMMLOWP_X86_32)
#define GEMMLOWP_SSE3_32
#endif

#if defined(GEMMLOWP_SSE4) && defined(GEMMLOWP_X86_64)
#define GEMMLOWP_SSE4_64
#endif

#if defined(GEMMLOWP_SSE3) && defined(GEMMLOWP_X86_64)
#define GEMMLOWP_SSE3_64
#endif


#endif  // GEMMLOWP_ALLOW_INLINE_ASM

// Detect Android. Don't conflate with ARM - we care about tuning
// for non-ARM Android devices too. This can be used in conjunction
// with x86 to tune differently for mobile x86 CPUs (Atom) vs. desktop x86 CPUs.
#if defined(__ANDROID__) || defined(ANDROID)
#define GEMMLOWP_ANDROID
#endif

namespace gemmlowp {

// Standard cache line size. Useful to optimize alignment and
// prefetches. Ideally we would query this at runtime, however
// 64 byte cache lines are the vast majority, and even if it's
// wrong on some device, it will be wrong by no more than a 2x factor,
// which should be acceptable.
const int kDefaultCacheLineSize = 64;

// Default L1 and L2 data cache sizes.
// The L1 cache size is assumed to be for each core.
// The L2 cache size is assumed to be shared among all cores. What
// we call 'L2' here is effectively top-level cache.
//
// On x86, we should ideally query this at
// runtime. On ARM, the instruction to query this is privileged and
// Android kernels do not expose it to userspace. Fortunately, the majority
// of ARM devices have roughly comparable values:
//   Nexus 5: L1 16k, L2 1M
//   Android One: L1 32k, L2 512k
// The following values are equal to or somewhat lower than that, and were
// found to perform well on both the Nexus 5 and Android One.
// Of course, these values are in principle too low for typical x86 CPUs
// where we should set the L2 value to (L3 cache size / number of cores) at
// least.
#if defined(GEMMLOWP_ARM) || defined(GEMMLOWP_ANDROID)
// ARM or ARM-like hardware (Android implies ARM-like) so here it's OK
// to tune for ARM, although on x86 Atom we might be able to query
// cache sizes at runtime, which would be better.
const int kDefaultL1CacheSize = 16 * 1024;
const int kDefaultL2CacheSize = 384 * 1024;
#elif defined(GEMMLOWP_X86_64)
// x86-64 and not Android. Therefore, likely desktop-class x86 hardware.
// Thus we assume larger cache sizes, though we really should query
// them at runtime.
const int kDefaultL1CacheSize = 32 * 1024;
const int kDefaultL2CacheSize = 4 * 1024 * 1024;
#elif defined(GEMMLOWP_X86_32)
// x86-32 and not Android. Same as x86-64 but less bullish.
const int kDefaultL1CacheSize = 32 * 1024;
const int kDefaultL2CacheSize = 2 * 1024 * 1024;
#else
// Less common hardware. Maybe some unusual or older or embedded thing.
// Assume smaller caches, but don't depart too far from what we do
// on ARM/Android to avoid accidentally exposing unexpected behavior.
const int kDefaultL1CacheSize = 16 * 1024;
const int kDefaultL2CacheSize = 256 * 1024;
#endif

// The proportion of the cache that we intend to use for storing
// RHS blocks. This should be between 0 and 1, and typically closer to 1,
// as we typically want to use most of the L2 cache for storing a large
// RHS block.
#if defined(GEMMLOWP_X86)
// For IA, use the entire L2 cache for the RHS matrix. LHS matrix is not blocked
// for L2 cache.
const float kDefaultL2RhsFactor = 1.00f;
#else
const float kDefaultL2RhsFactor = 0.75f;
#endif

// The number of bytes in a SIMD register. This is used to determine
// the dimensions of PackingRegisterBlock so that such blocks can
// be efficiently loaded into registers, so that packing code can
// work within registers as much as possible.
// In the non-SIMD generic fallback code, this is just a generic array
// size, so any size would work there. Different platforms may set this
// to different values but must ensure that their own optimized packing paths
// are consistent with this value.
const int kRegisterSize = 16;

// Requantization to less-than-8-bit is costly, so it only worth
// doing if the GEMM width is large enough
const int kMinimumWidthForRequantization = 100;

// Hints the CPU to prefetch the cache line containing ptr.
inline void Prefetch(const void* ptr) {
#ifdef __GNUC__  // Clang and GCC define __GNUC__ and have __builtin_prefetch.
  __builtin_prefetch(ptr);
#else
  (void)ptr;
#endif
}

// Returns the runtime argument rounded down to the nearest multiple of
// the fixed Modulus.
template <unsigned Modulus, typename Integer>
Integer RoundDown(Integer i) {
  return i - (i % Modulus);
}

// Returns the runtime argument rounded up to the nearest multiple of
// the fixed Modulus.
template <unsigned Modulus, typename Integer>
Integer RoundUp(Integer i) {
  return RoundDown<Modulus>(i + Modulus - 1);
}

// Returns the quotient a / b rounded up ('ceil') to the nearest integer.
template <typename Integer>
Integer CeilQuotient(Integer a, Integer b) {
  return (a + b - 1) / b;
}

// Returns the argument rounded up to the nearest power of two.
template <typename Integer>
Integer RoundUpToPowerOfTwo(Integer n) {
  Integer i = n - 1;
  i |= i >> 1;
  i |= i >> 2;
  i |= i >> 4;
  i |= i >> 8;
  i |= i >> 16;
  return i + 1;
}

template <int N>
struct IsPowerOfTwo {
  static const bool value = !(N & (N - 1));
};

}  // namespace gemmlowp

#endif  // GEMMLOWP_INTERNAL_COMMON_H_
