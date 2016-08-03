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

#include "eight_bit_int_gemm.h"

#include <memory>

// gemmlowp symbols should have hidden visibility.
// currently this is ensured in the build system by
// passing -finlines-visibility-hidden. TODO: it would be
// safer to hardcode it here with some #pragma's.
#include "../public/gemmlowp.h"

// Define GEMMLOWP_USE_META_FASTPATH in order to use the fastpath ARM/NEON
// code. This code path consists of a number of meta-programmed, automatically
// generated GEMM kernels that are suitable for some sizes of input matrices.
// Due to the fact that the generated code relies heavily on loop unrolling,
// inling and currying of runtime parameters the size of the generated binary
// is quite significant (approx. 200kb) which might be prohibitive in
// low-memory situations.

#if defined(GEMMLOWP_USE_META_FASTPATH) && defined(GEMMLOWP_NEON)
#include "../meta/multi_thread_gemm.h"
#else

#if defined(GEMMLOWP_USE_META_FASTPATH)
#warning "META fast path turned on without NEON!"
#endif

#endif

namespace gemmlowp {
namespace eight_bit_int_gemm {
namespace {

// To be used as template parameter for GlobalLock.
// GlobalLock<EightBitIntGemmLockId> is the global lock
// on EightBitIntGemm entry points, protecting
// EightBitIntGemm's global state.
struct EightBitIntGemmLockId;

// Global state: consists of one global GemmContext instance.
GemmContext* global_context;

GemmContext* GetOrCreateGlobalContext() {
  if (!global_context) {
    global_context = new GemmContext;
  }
  return global_context;
}

void DestroyGlobalContext() {
  delete global_context;
  global_context = nullptr;
}

template <bool transpose_a, bool transpose_b, bool transpose_c>
void EightBitIntGemmImpl(GemmContext* context, int m, int n, int k,
                         const std::uint8_t* a, std::int32_t a_offset, int lda,
                         const std::uint8_t* b, std::int32_t b_offset, int ldb,
                         std::uint8_t* c, std::int32_t c_offset,
                         std::int32_t c_mult_int, std::int32_t c_shift, int ldc,
                         BitDepthSetting bit_depth) {
  const int lhs_offset = a_offset;
  const int rhs_offset = b_offset;
  const int result_offset = c_offset;
  const int result_mult_int = c_mult_int;
  const int result_shift = c_shift;

  static const MapOrder ResultOrder =
      transpose_c ? MapOrder::RowMajor : MapOrder::ColMajor;
  static const MapOrder LhsOrder =
      transpose_a ? MapOrder::RowMajor : MapOrder::ColMajor;
  static const MapOrder RhsOrder =
      transpose_b ? MapOrder::RowMajor : MapOrder::ColMajor;

  MatrixMap<const std::uint8_t, LhsOrder> lhs(a, m, k, lda);
  MatrixMap<const std::uint8_t, RhsOrder> rhs(b, k, n, ldb);
  MatrixMap<std::uint8_t, ResultOrder> result(c, m, n, ldc);

  switch (bit_depth) {
#define GEMMLOWP_HANDLE_BIT_DEPTH(BIT_DEPTH_SETTING, BIT_DEPTH_PARAMS)     \
  case BitDepthSetting::BIT_DEPTH_SETTING:                                 \
    Gemm<std::uint8_t, BIT_DEPTH_PARAMS>(                                  \
        context, lhs, rhs, &result, lhs_offset, rhs_offset, result_offset, \
        result_mult_int, result_shift);                                    \
    return;
    GEMMLOWP_HANDLE_BIT_DEPTH(A8B8, DefaultL8R8BitDepthParams)
    GEMMLOWP_HANDLE_BIT_DEPTH(A5B7, DefaultL7R5BitDepthParams)
    default:
      abort();
#undef GEMMLOWP_HANDLE_BIT_DEPTH
  }
}

template <bool transpose_a, bool transpose_b, bool transpose_c>
void EightBitIntGemmInt32Impl(GemmContext* context, int m, int n, int k,
                              const std::uint8_t* a, std::int32_t a_offset,
                              int lda, const std::uint8_t* b,
                              std::int32_t b_offset, int ldb, std::int32_t* c,
                              int ldc, BitDepthSetting bit_depth) {
  const int lhs_offset = a_offset;
  const int rhs_offset = b_offset;

  static const MapOrder ResultOrder =
      transpose_c ? MapOrder::RowMajor : MapOrder::ColMajor;
  static const MapOrder LhsOrder =
      transpose_a ? MapOrder::RowMajor : MapOrder::ColMajor;
  static const MapOrder RhsOrder =
      transpose_b ? MapOrder::RowMajor : MapOrder::ColMajor;

  MatrixMap<const std::uint8_t, LhsOrder> lhs(a, m, k, lda);
  MatrixMap<const std::uint8_t, RhsOrder> rhs(b, k, n, ldb);
  MatrixMap<std::int32_t, ResultOrder> result(c, m, n, ldc);

  auto empty_pipeline = std::make_tuple();

  switch (bit_depth) {
#define GEMMLOWP_HANDLE_BIT_DEPTH_INT32(BIT_DEPTH_SETTING, BIT_DEPTH_PARAMS) \
  case BitDepthSetting::BIT_DEPTH_SETTING:                                   \
    GemmWithOutputPipeline<std::uint8_t, std::int32_t, BIT_DEPTH_PARAMS>(    \
        context, lhs, rhs, &result, lhs_offset, rhs_offset, empty_pipeline); \
    return;
    GEMMLOWP_HANDLE_BIT_DEPTH_INT32(A8B8, DefaultL8R8BitDepthParams)
    GEMMLOWP_HANDLE_BIT_DEPTH_INT32(A5B7, DefaultL7R5BitDepthParams)
    default:
      abort();
#undef GEMMLOWP_HANDLE_BIT_DEPTH_INT32
  }
}

class Scratch {
 public:
  Scratch() : buffer_(), size_(0) {}

  void AssureSize(std::int32_t required_size) {
    if (size_ >= required_size) {
      return;
    }
    buffer_.reset(new std::uint8_t[required_size]);
    size_ = required_size;
  }

  void Clear() {
    buffer_.reset(nullptr);
    size_ = 0;
  }

  std::uint8_t* buffer() { return buffer_.get(); }

 private:
  std::unique_ptr<std::uint8_t[]> buffer_;
  std::int32_t size_;
};

Scratch* global_scratch = nullptr;

Scratch* GetOrCreateGlobalScratch() {
  if (global_scratch == nullptr) {
    global_scratch = new Scratch();
  }
  return global_scratch;
}

void DestroyGlobalScratch() {
  delete global_scratch;
  global_scratch = nullptr;
}

#if defined(GEMMLOWP_USE_META_FASTPATH) && defined(GEMMLOWP_NEON)

bool IsRowMajorOrVector(bool transpose, int stride, int rows, int cols) {
  // Is it row major and nicely packed?
  if (transpose && stride == cols) {
    return true;
  }

  // Is it a one row vector? (a vector is both row and column major)
  if (rows == 1) {
    return true;
  }

  return false;
}

bool IsColumnMajorOrVector(bool transpose, int stride, int rows, int cols) {
  // Is it column major and nicely packed?
  if (!transpose && stride == rows) {
    return true;
  }

  // Is it a one column vector? (a vector is both row and column major)
  if (cols == 1) {
    return true;
  }

  return false;
}

bool CanHandleMetaFastpath(bool transpose_a, bool transpose_b, bool transpose_c,
                           int m, int n, int k, int lda, int ldb, int ldc,
                           BitDepthSetting depth_setting) {
  // Meta fastpath only supports 8bit x 8bit and k between 8 and 2048.
  if (depth_setting != BitDepthSetting::A8B8 || k < 8 || k > 2048) {
    return false;
  }

  // The first operand needs to be a row major matrix or a vector.
  if (!IsRowMajorOrVector(transpose_a, lda, m, k)) {
    return false;
  }

  // The second operand needs to be a column major matrix or a vector.
  if (!IsColumnMajorOrVector(transpose_b, ldb, k, n)) {
    return false;
  }

  // The result can either be a row major matrix, a column major matrix or
  // a vector.
  if (IsRowMajorOrVector(transpose_c, ldc, m, n)) {
    return true;
  }

  if (IsColumnMajorOrVector(transpose_c, ldc, m, n)) {
    return true;
  }

  return false;
}

// Assure enough scratch memory is allocated and run the fast path gemm.
void MetaGemmQuantized8Bit(GemmContext* context, const std::uint8_t* lhs,
                           const std::uint8_t* rhs, int m, int n, int k,
                           std::int32_t lhs_offset, std::int32_t rhs_offset,
                           std::int32_t sum_offset,
                           std::int32_t multiplicative_offset,
                           std::int32_t shift, bool result_transpose,
                           std::int32_t result_stride, std::uint8_t* result) {
  Scratch* scratch = GetOrCreateGlobalScratch();
  const std::int32_t max_num_threads = context->max_num_threads();
  if (IsRowMajorOrVector(result_transpose, result_stride, m, n)) {
    scratch->AssureSize(meta::gemm_q8_scratch(m, n, k, max_num_threads));
    meta::multi_thread_gemm_q8(
        context->workers_pool(), max_num_threads, scratch->buffer(),
        lhs, rhs, m, n, k, lhs_offset, rhs_offset, sum_offset,
        multiplicative_offset, shift, result);
  } else {
    scratch->AssureSize(meta::gemm_q8_scratch(n, m, k, max_num_threads));
    meta::multi_thread_gemm_q8(
        context->workers_pool(), max_num_threads, scratch->buffer(),
        rhs, lhs, n, m, k, rhs_offset, lhs_offset, sum_offset,
        multiplicative_offset, shift, result);
  }
}

// Assure enough scratch memory is allocated and run the 8bit to float fast
// path gemm.
void MetaGemmFloat(GemmContext* context, const std::uint8_t* lhs,
                   const std::uint8_t* rhs, int m, int n, int k,
                   std::int32_t lhs_offset, std::int32_t rhs_offset,
                   float result_offset, bool result_transpose,
                   std::int32_t result_stride, float* result) {
  Scratch* scratch = GetOrCreateGlobalScratch();
  const std::int32_t max_num_threads = context->max_num_threads();
  if (IsRowMajorOrVector(result_transpose, result_stride, m, n)) {
    scratch->AssureSize(meta::gemm_f_scratch(m, n, k, max_num_threads));
    meta::multi_thread_gemm_f(
        context->workers_pool(), max_num_threads, scratch->buffer(),
        lhs, rhs, m, n, k, lhs_offset, rhs_offset, result_offset, result);
  } else {
    scratch->AssureSize(meta::gemm_f_scratch(n, m, k, max_num_threads));
    meta::multi_thread_gemm_f(
        context->workers_pool(), max_num_threads, scratch->buffer(),
        rhs, lhs, n, m, k, rhs_offset, lhs_offset, result_offset, result);
  }
}

#endif

}  // end anonymous namespace

// Public interface entry points

void EightBitIntGemm(bool transpose_a, bool transpose_b, bool transpose_c,
                     int m, int n, int k, const std::uint8_t* a,
                     std::int32_t a_offset, int lda, const std::uint8_t* b,
                     std::int32_t b_offset, int ldb, std::uint8_t* c,
                     std::int32_t c_offset, std::int32_t c_mult_int,
                     std::int32_t c_shift, int ldc, BitDepthSetting bit_depth) {
  AutoGlobalLock<EightBitIntGemmLockId> lock;
  GemmContext* context = GetOrCreateGlobalContext();

#if defined(GEMMLOWP_USE_META_FASTPATH) && defined(GEMMLOWP_NEON)
  if (CanHandleMetaFastpath(transpose_a, transpose_b, transpose_c, m, n, k, lda,
                            ldb, ldc, bit_depth)) {
    MetaGemmQuantized8Bit(context, a, b, m, n, k, a_offset, b_offset, c_offset,
                          c_mult_int, c_shift, transpose_c, ldc, c);
    return;
  }
#endif

#define GEMMLOWP_HANDLE_CASE(ta, tb, tc)                                    \
  if (transpose_a == ta && transpose_b == tb && transpose_c == tc) {        \
    EightBitIntGemmImpl<ta, tb, tc>(context, m, n, k, a, a_offset, lda, b,  \
                                    b_offset, ldb, c, c_offset, c_mult_int, \
                                    c_shift, ldc, bit_depth);               \
  }

  GEMMLOWP_HANDLE_CASE(false, false, false)
  GEMMLOWP_HANDLE_CASE(false, false, true)
  GEMMLOWP_HANDLE_CASE(false, true, false)
  GEMMLOWP_HANDLE_CASE(false, true, true)
  GEMMLOWP_HANDLE_CASE(true, false, false)
  GEMMLOWP_HANDLE_CASE(true, false, true)
  GEMMLOWP_HANDLE_CASE(true, true, false)
  GEMMLOWP_HANDLE_CASE(true, true, true)

#undef GEMMLOWP_HANDLE_CASE
}

void EightBitIntGemm(bool transpose_a, bool transpose_b, bool transpose_c,
                     int m, int n, int k, const std::uint8_t* a,
                     std::int32_t a_offset, std::int32_t lda,
                     const std::uint8_t* b, std::int32_t b_offset,
                     std::int32_t ldb, float* c, float c_offset,
                     std::int32_t ldc, BitDepthSetting bit_depth) {
  AutoGlobalLock<EightBitIntGemmLockId> lock;
  GemmContext* context = GetOrCreateGlobalContext();

#if defined(GEMMLOWP_USE_META_FASTPATH) && defined(GEMMLOWP_NEON)
  if (CanHandleMetaFastpath(transpose_a, transpose_b, transpose_c, m, n, k, lda,
                            ldb, ldc, bit_depth)) {
    MetaGemmFloat(context, a, b, m, n, k, a_offset, b_offset, c_offset,
                  transpose_c, ldc, c);
    return;
  }
#endif

  // TODO(maciekc): implement a float output stage, get rid of scratch memory.
  Scratch* scratch = GetOrCreateGlobalScratch();
  if (transpose_c) {
    scratch->AssureSize(m * ldc * sizeof(std::int32_t));
  } else {
    scratch->AssureSize(n * ldc * sizeof(std::int32_t));
  }
  std::int32_t* temp_c = reinterpret_cast<std::int32_t*>(scratch->buffer());

#define GEMMLOWP_HANDLE_INT32_CASE(ta, tb, tc)                               \
  if (transpose_a == ta && transpose_b == tb && transpose_c == tc) {         \
    EightBitIntGemmInt32Impl<ta, tb, tc>(context, m, n, k, a, a_offset, lda, \
                                         b, b_offset, ldb, temp_c, ldc,      \
                                         bit_depth);                         \
  }

  GEMMLOWP_HANDLE_INT32_CASE(false, false, false)
  GEMMLOWP_HANDLE_INT32_CASE(false, false, true)
  GEMMLOWP_HANDLE_INT32_CASE(false, true, false)
  GEMMLOWP_HANDLE_INT32_CASE(false, true, true)
  GEMMLOWP_HANDLE_INT32_CASE(true, false, false)
  GEMMLOWP_HANDLE_INT32_CASE(true, false, true)
  GEMMLOWP_HANDLE_INT32_CASE(true, true, false)
  GEMMLOWP_HANDLE_INT32_CASE(true, true, true)

#undef GEMMLOWP_HANDLE_INT32_CASE

  if (transpose_c) {
    // Row major.
    for (int i = 0; i < m; ++i) {
      float* dest_row = c + i * ldc;
      std::int32_t* src_row = temp_c + i * ldc;
      for (int j = 0; j < n; ++j) {
        dest_row[j] = static_cast<float>(src_row[j]) * c_offset;
      }
    }
  } else {
    // Column major.
    for (int i = 0; i < n; ++i) {
      float* dest_column = c + i * ldc;
      std::int32_t* src_column = temp_c + i * ldc;
      for (int j = 0; j < m; ++j) {
        dest_column[j] = static_cast<float>(src_column[j]) * c_offset;
      }
    }
  }
}

void SetMaxNumThreads(int n) {
  AutoGlobalLock<EightBitIntGemmLockId> lock;
  GemmContext* context = GetOrCreateGlobalContext();
  context->set_max_num_threads(n);
}

void FreePersistentResources() {
  AutoGlobalLock<EightBitIntGemmLockId> lock;
  DestroyGlobalContext();
  DestroyGlobalScratch();
}

}  // namespace eight_bit_int_gemm
}  // namespace gemmlowp
