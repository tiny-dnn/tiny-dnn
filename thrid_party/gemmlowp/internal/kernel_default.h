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

// kernel_default.h: Chooses default GEMM and GEMV kernels for the
// host platform.

#ifndef GEMMLOWP_INTERNAL_KERNEL_DEFAULT_H_
#define GEMMLOWP_INTERNAL_KERNEL_DEFAULT_H_

#include "../public/bit_depth.h"
#include "common.h"

namespace gemmlowp {

enum class KernelFamily { Gemm, Gemv };

template <KernelFamily Family, int ProductBits>
struct DefaultKernelImpl : DefaultKernelImpl<Family, ProductBits + 1> {
  static_assert(ProductBits <= 16, "Bit depth too large");
};

template <KernelFamily Family, typename BitDepthParams>
struct DefaultKernel
    : DefaultKernelImpl<Family, BitDepthParams::LhsBitDepth::kBits +
                                    BitDepthParams::RhsBitDepth::kBits> {};

}  // end namespace gemmlowp

#define GEMMLOWP_SET_DEFAULT_KERNEL(op, max_product_bits, kernel)           \
  namespace gemmlowp {                                                      \
  template <>                                                               \
  struct DefaultKernelImpl<KernelFamily::op, max_product_bits> : kernel {}; \
  }

#if defined GEMMLOWP_NEON_32
#include "kernel_neon.h"
GEMMLOWP_SET_DEFAULT_KERNEL(Gemm, 16, NEON_32_Kernel12x4Depth2)
GEMMLOWP_SET_DEFAULT_KERNEL(Gemm, 12,
                            NEON_32_Kernel12x4Depth2Assuming12BitProducts)
GEMMLOWP_SET_DEFAULT_KERNEL(Gemv, 16, NEONKernel4Nx1Depth2<3>)
#elif defined GEMMLOWP_NEON_64
#include "kernel_neon.h"
GEMMLOWP_SET_DEFAULT_KERNEL(Gemm, 16, NEON_64_Kernel12x8Depth2)
GEMMLOWP_SET_DEFAULT_KERNEL(Gemv, 16, NEONKernel4Nx1Depth2<3>)
#elif defined GEMMLOWP_SSE4_32
#include "kernel_SSE.h"
GEMMLOWP_SET_DEFAULT_KERNEL(Gemm, 16, SSE4_32_Kernel4x4Depth2)
GEMMLOWP_SET_DEFAULT_KERNEL(Gemv, 16, SSE4_32_Kernel4x4Depth2)
#elif defined GEMMLOWP_SSE4_64
#include "kernel_SSE.h"
GEMMLOWP_SET_DEFAULT_KERNEL(Gemm, 16, SSE4_64_Kernel12x4Depth2)
GEMMLOWP_SET_DEFAULT_KERNEL(Gemv, 16, SSE4_64_Kernel12x4Depth2)
#else
#include "kernel_reference.h"
namespace gemmlowp {
typedef ReferenceKernel<KernelFormat<KernelSideFormat<CellFormat<4, 4>, 2>,
                                     KernelSideFormat<CellFormat<4, 4>, 2> > >
    DefaultReferenceKernel;
}
GEMMLOWP_SET_DEFAULT_KERNEL(Gemm, 16, DefaultReferenceKernel)
GEMMLOWP_SET_DEFAULT_KERNEL(Gemv, 16, DefaultReferenceKernel)
#endif

#endif  // GEMMLOWP_INTERNAL_KERNEL_DEFAULT_H_
