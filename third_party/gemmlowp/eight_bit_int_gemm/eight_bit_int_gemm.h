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

// eight_bit_int_gemm.h: exposes the standard EightBitIntGemm interface.

#ifndef GEMMLOWP_EIGHT_BIT_INT_GEMM_EIGHT_BIT_INT_GEMM_H_
#define GEMMLOWP_EIGHT_BIT_INT_GEMM_EIGHT_BIT_INT_GEMM_H_

#ifndef GEMMLOWP_USE_STLPORT
#include <cstdint>
#else
#include <stdint.h>
namespace std {
using ::uint8_t;
using ::int32_t;
}
#endif

namespace gemmlowp {

namespace eight_bit_int_gemm {

// Concurrency / reentrancy notice
// ===============================
//
// This eight_bit_int_gemm has global singleton persistent state.
// A global lock ensures serialization of calls, so this library
// is fully reentrant but only one calling thread gets to actually run
// at a time, while other calling threads would wait. So it is safe
// albeit potentially slow to call the functions exposed here on
// multiple threads concurrently.
//
// Users who prefer a state-less, singleton-less interface,
// should use the main gemmlowp interface (public/gemmlowp.h) instead.

// The BitDepthSetting enum lists supported a/b bit-depth combinations.
enum class BitDepthSetting {
  A8B8,  // 8-bit a, 8-bit b
  A5B7   // 5-bit a, 7-bit b
};

// The main entry point to compute a Gemm. This is the standard
// EightBitIntGemm interface.
void EightBitIntGemm(bool transpose_a, bool transpose_b, bool transpose_c,
                     int m, int n, int k, const std::uint8_t *a,
                     std::int32_t a_offset, int lda, const std::uint8_t *b,
                     std::int32_t b_offset, int ldb, std::uint8_t *c,
                     std::int32_t c_offset, std::int32_t c_mult_int,
                     std::int32_t c_shift, int ldc, BitDepthSetting bit_depth);

void EightBitIntGemm(bool transpose_a, bool transpose_b, bool transpose_c,
                     int m, int n, int k, const std::uint8_t *a,
                     std::int32_t a_offset, int lda, const std::uint8_t *b,
                     std::int32_t b_offset, int ldb, float *c, float c_offset,
                     int ldc, BitDepthSetting bit_depth);

// Frees any persistent resources
// (threads, thread pools, allocators, buffers, ...)
// that gemmlowp might hold. This is called automatically
// on thread exit, but one may also call it earlier, at any time.
void FreePersistentResources();

// Allows specifying the number of hardware threads, as a hint as to
// how many worker threads to use for sufficiently large Gemm's.
// We will never use more threads than that, but may use fewer,
// for instance on Gemm's that are too small to benefit from all
// available threads. The value 0 lets the implementation query
// the system to determine the number of hardware threads.
// Default value: 0.
void SetMaxNumThreads(int n);

}  // namespace eight_bit_int_gemm

}  // namespace gemmlowp

#endif  // GEMMLOWP_EIGHT_BIT_INT_GEMM_EIGHT_BIT_INT_GEMM_H_
