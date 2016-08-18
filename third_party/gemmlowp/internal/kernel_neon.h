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

// kernel_neon.h: a collection of NEON optimized kernels.
// Check in kernel_default.h which one(s) are actually used by default.
// Others are mere experiments; they are still covered by tests
// in case they might be useful some day.

#ifndef GEMMLOWP_INTERNAL_KERNEL_NEON_H_
#define GEMMLOWP_INTERNAL_KERNEL_NEON_H_

#include "kernel.h"

#include <arm_neon.h>
#include <cassert>

namespace gemmlowp {

// The kernels here are specifically arm 32bit assembly, not arm 64bit.
#ifdef GEMMLOWP_NEON_32

// Our main GEMM kernel.
struct NEON_32_Kernel12x4Depth2 : KernelBase {
  typedef KernelFormat<KernelSideFormat<CellFormat<4, 2>, 3>,
                       KernelSideFormat<CellFormat<4, 2>, 1> >
      Format;

  const char* Name() const override { return "NEON, 12x4, depth 2"; }

  // TODO(benoitjacob): reorder function arguments so dst comes last
  void Run(std::int32_t* dst_ptr, std::size_t dst_row_stride,
           std::size_t dst_col_stride, const std::uint8_t* lhs_ptr,
           const std::uint8_t* rhs_ptr, std::size_t start_depth,
           std::size_t run_depth) const override {
    ScopedProfilingLabel label("optimized kernel (NEON 12x4)");

// For iOS assembler, the %= style of local labels cause compilation errors,
//  so use numerical ones instead. See
// http://stackoverflow.com/questions/3898435/labels-in-gcc-inline-assembly
// If you add any labels, remember to undef them at the end.
#define GEMMLOWP_LOOP_NEON_KERNEL_12X4_DEPTH2 "1"
#define GEMMLOWP_STORE_RESULT_NEON_KERNEL_12X4_DEPTH2 "2"

    assert(dst_row_stride == 1);
    asm volatile(
        // Clear accumulator registers (see layout below)
        "vmov.s32 q4, #0\n"
        "vmov.s32 q8, q4\n"
        "vmov.s32 q12, q4\n"
        "vmov.s32 q5, q4\n"
        "vmov.s32 q9, q4\n"
        "vmov.s32 q13, q4\n"
        "vmov.s32 q6, q4\n"
        "vmov.s32 q10, q4\n"
        "vmov.s32 q14, q4\n"
        "vmov.s32 q7, q4\n"
        "vmov.s32 q11, q4\n"
        "vmov.s32 q15, q4\n"

        /* Main loop */

        GEMMLOWP_LOOP_NEON_KERNEL_12X4_DEPTH2
        ":\n"

        // Overview of register layout:
        //
        // A 2x4 cell of Rhs is stored in 16bit in d0--d1 (q0).
        // A 12x2 block of 3 4x2 cells Lhs is stored in 16bit in d2--d7
        // (q1--q3).
        // A 12x4 block of accumulators is stored in 32bit in q4--q15.
        //
        //                   +-----+-----+-----+-----+
        //                   |d0[0]|d0[1]|d0[2]|d0[3]|
        //              Rhs  +-----+-----+-----+-----+
        //                   |d1[0]|d1[1]|d1[2]|d1[3]|
        //                   +-----+-----+-----+-----+
        //
        //                   |     |     |     |     |
        //
        //    Lhs            |     |     |     |     |
        //
        //  +--+--+ - - - -  +-----+-----+-----+-----+
        //  |d2|d3|          | q4  | q5  | q6  | q7  |
        //  |d2|d3|          | q4  | q5  | q6  | q7  |
        //  |d2|d3|          | q4  | q5  | q6  | q7  |
        //  |d2|d3|          | q4  | q5  | q6  | q7  |
        //  +--+--+ - - - -  +-----+-----+-----+-----+
        //  |d4|d5|          | q8  | q9  | q10 | q11 |
        //  |d4|d5|          | q8  | q9  | q10 | q11 |
        //  |d4|d5|          | q8  | q9  | q10 | q11 |
        //  |d4|d5|          | q8  | q9  | q10 | q11 |
        //  +--+--+ - - - -  +-----+-----+-----+-----+
        //  |d6|d7|          | q12 | q13 | q14 | q15 |
        //  |d6|d7|          | q12 | q13 | q14 | q15 |
        //  |d6|d7|          | q12 | q13 | q14 | q15 |
        //  |d6|d7|          | q12 | q13 | q14 | q15 |
        //  +--+--+ - - - -  +-----+-----+-----+-----+
        //
        //                            Accumulator

        // Load 1 Rhs cell of size 2x4
        "vld1.8 {d0}, [%[rhs_ptr]:64]!\n"

        // Load 3 Lhs cells of size 4x2 each
        "vld1.8 {d2}, [%[lhs_ptr]:64]!\n"
        "vld1.8 {d4}, [%[lhs_ptr]:64]!\n"
        "vld1.8 {d6}, [%[lhs_ptr]:64]!\n"

        // Expand Lhs/Rhs cells to 16 bit.
        "vmovl.u8 q0, d0\n"
        "vmovl.u8 q1, d2\n"
        "vmovl.u8 q2, d4\n"
        "vmovl.u8 q3, d6\n"

        // Multiply-accumulate, level of depth 0
        "vmlal.u16 q4, d2, d0[0]\n"
        "vmlal.u16 q5, d2, d0[1]\n"
        "vmlal.u16 q6, d2, d0[2]\n"
        "vmlal.u16 q7, d2, d0[3]\n"
        "vmlal.u16 q8, d4, d0[0]\n"
        "vmlal.u16 q9, d4, d0[1]\n"
        "vmlal.u16 q10, d4, d0[2]\n"
        "vmlal.u16 q11, d4, d0[3]\n"
        "vmlal.u16 q12, d6, d0[0]\n"
        "vmlal.u16 q13, d6, d0[1]\n"
        "vmlal.u16 q14, d6, d0[2]\n"
        "vmlal.u16 q15, d6, d0[3]\n"

        // Multiply-accumulate, level of depth 1
        "vmlal.u16 q4, d3, d1[0]\n"
        "vmlal.u16 q5, d3, d1[1]\n"
        "vmlal.u16 q6, d3, d1[2]\n"
        "vmlal.u16 q7, d3, d1[3]\n"
        "vmlal.u16 q8, d5, d1[0]\n"
        "vmlal.u16 q9, d5, d1[1]\n"
        "vmlal.u16 q10, d5, d1[2]\n"
        "vmlal.u16 q11, d5, d1[3]\n"
        "vmlal.u16 q12, d7, d1[0]\n"
        "vmlal.u16 q13, d7, d1[1]\n"
        "vmlal.u16 q14, d7, d1[2]\n"
        "vmlal.u16 q15, d7, d1[3]\n"

        // Loop. Decrement loop index (depth) by 2, since we just handled 2
        // levels of depth (Kernel::kDepth=2).
        "subs %[run_depth], #2\n"
        "bne " GEMMLOWP_LOOP_NEON_KERNEL_12X4_DEPTH2
        "b\n"

        /* end of main loop */

        /* Accumulate our local accumulator registers into the destination block
           */

        // Compute stride between consecutive columns, in bytes
        "mov r0, #4\n"  // multiply by 4 = sizeof(int32)
        "mul %[dst_col_stride], r0\n"

        // If start_depth == 0, then there is no preexisting accumulator
        // to accumulate, so we can simply store our result.
        "cmp %[start_depth], #0\n"
        "beq " GEMMLOWP_STORE_RESULT_NEON_KERNEL_12X4_DEPTH2
        "f\n"

        "mov r0, %[dst_ptr]\n"

        // Load a column
        "mov r1, r0\n"
        "vld1.32 {d0, d1}, [r1]!\n"
        "vld1.32 {d2, d3}, [r1]!\n"
        "vld1.32 {d4, d5}, [r1]!\n"
        // Accumulate a column
        "vadd.s32 q4, q4, q0\n"
        "vadd.s32 q8, q8, q1\n"
        "vadd.s32 q12, q12, q2\n"

        "add r0, %[dst_col_stride]\n"
        // Load a column
        "mov r1, r0\n"
        "vld1.32 {d0, d1}, [r1]!\n"
        "vld1.32 {d2, d3}, [r1]!\n"
        "vld1.32 {d4, d5}, [r1]!\n"
        // Accumulate a column
        "vadd.s32 q5, q5, q0\n"
        "vadd.s32 q9, q9, q1\n"
        "vadd.s32 q13, q13, q2\n"

        "add r0, %[dst_col_stride]\n"
        // Load a column
        "mov r1, r0\n"
        "vld1.32 {d0, d1}, [r1]!\n"
        "vld1.32 {d2, d3}, [r1]!\n"
        "vld1.32 {d4, d5}, [r1]!\n"
        // Accumulate a column
        "vadd.s32 q6, q6, q0\n"
        "vadd.s32 q10, q10, q1\n"
        "vadd.s32 q14, q14, q2\n"

        "add r0, %[dst_col_stride]\n"
        // Load a column
        "mov r1, r0\n"
        "vld1.32 {d0, d1}, [r1]!\n"
        "vld1.32 {d2, d3}, [r1]!\n"
        "vld1.32 {d4, d5}, [r1]!\n"
        // Accumulate a column
        "vadd.s32 q7, q7, q0\n"
        "vadd.s32 q11, q11, q1\n"
        "vadd.s32 q15, q15, q2\n"

        GEMMLOWP_STORE_RESULT_NEON_KERNEL_12X4_DEPTH2
        ":\n"

        "mov r0, %[dst_ptr]\n"
        // Store a column
        "mov r1, r0\n"
        "vst1.32 {d8, d9}, [r1]!\n"
        "vst1.32 {d16, d17}, [r1]!\n"
        "vst1.32 {d24, d25}, [r1]!\n"
        // Store a column
        "add r0, %[dst_col_stride]\n"
        "mov r1, r0\n"
        "vst1.32 {d10, d11}, [r1]!\n"
        "vst1.32 {d18, d19}, [r1]!\n"
        "vst1.32 {d26, d27}, [r1]!\n"
        // Store a column
        "add r0, %[dst_col_stride]\n"
        "mov r1, r0\n"
        "vst1.32 {d12, d13}, [r1]!\n"
        "vst1.32 {d20, d21}, [r1]!\n"
        "vst1.32 {d28, d29}, [r1]!\n"
        // Store a column
        "add r0, %[dst_col_stride]\n"
        "mov r1, r0\n"
        "vst1.32 {d14, d15}, [r1]!\n"
        "vst1.32 {d22, d23}, [r1]!\n"
        "vst1.32 {d30, d31}, [r1]!\n"
        :  // outputs
        [lhs_ptr] "+r"(lhs_ptr), [rhs_ptr] "+r"(rhs_ptr),
        [dst_ptr] "+r"(dst_ptr),
        [run_depth] "+r"(run_depth)
        :  // inputs
        [start_depth] "r"(start_depth),
        [dst_col_stride] "r"(dst_col_stride)
        :  // clobbers
        "cc", "memory", "r0", "r1",
        // note: someone on internet says that quad registers are
        // unsupported in the clobber list!
        "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10",
        "d11", "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19", "d20",
        "d21", "d22", "d23", "d24", "d25", "d26", "d27", "d28", "d29", "d30",
        "d31");
#undef GEMMLOWP_LOOP_NEON_KERNEL_12X4_DEPTH2
#undef GEMMLOWP_STORE_RESULT_NEON_KERNEL_12X4_DEPTH2
  }
};

struct NEON_32_Kernel12x4Depth2Assuming12BitProducts : KernelBase {
  typedef KernelFormat<
      KernelSideFormat<CellFormat<4, 2, CellOrder::WidthMajor>, 3>,
      KernelSideFormat<CellFormat<4, 2, CellOrder::WidthMajor>, 1> >
      Format;

  const char* Name() const override {
    return "NEON, 12x4, depth 2, assuming 12-bit products";
  }

  // TODO(benoitjacob): reorder function arguments so dst comes last
  void Run(std::int32_t* dst_ptr, std::size_t dst_row_stride,
           std::size_t dst_col_stride, const std::uint8_t* lhs_ptr,
           const std::uint8_t* rhs_ptr, std::size_t start_depth,
           std::size_t run_depth) const override {
    ScopedProfilingLabel label(
        "optimized kernel (NEON 12x4, assuming 12-bit products)");
    assert(dst_row_stride == 1);

// See comments above for why we need local numerical labels in our asm.
#define GEMMLOWP_LOOP_NEON_32_KERNEL_12X4_DEPTH2_ASSUMING_12BIT_PRODUCTS "1"
#define GEMMLOWP_LOAD_GLOBAL_ACCUMULATORS_NEON_32_KERNEL_12X4_DEPTH2_12BIT "2"
#define GEMMLOWP_LABEL_32 "3"
#define GEMMLOWP_LABEL_24 "4"
#define GEMMLOWP_LABEL_16 "5"
#define GEMMLOWP_LABEL_8 "6"
#define GEMMLOWP_LABEL_2 "7"

    // This kernel is special in that it uses local 16-bit accumulators.
    // Because it assumes that each product fits in 12 bits, it can accumulate
    // 16 products into a local 16-bit accumulator without risking overflow.
    // At that point, it must accumulate these local 16-bit accumulators back
    // into global 32-bit accumulators, which have to be stored in memory for
    // lack of register space.
    // This 12x4 block of global accumulators is laid out as 3 cells of size 4x4
    // stored in diagonal-major order like this for the first 4x4 cell:
    //
    //   0   4   8  12
    //  13   1   5   9
    //  10  14   2   6
    //   7  11  15   3
    //
    // and likewise for the 2nd  cell (16--31) and 3rd cell (32--47)
    std::int32_t global_accumulators[3 * 4 * 4];
    asm volatile(
        // Compute stride between consecutive columns, in bytes
        "mov r0, #4\n"  // multiply by 4 = sizeof(int32)
        "mul %[dst_col_stride], r0\n"

        "cmp %[start_depth], #0\n"
        "bne"
        " " GEMMLOWP_LOAD_GLOBAL_ACCUMULATORS_NEON_32_KERNEL_12X4_DEPTH2_12BIT
        "f\n"

        // If start_depth==0, we need to clear our global accumulators
        "mov r0, %[global_accumulators]\n"
        "vmov.s32 q8, #0\n"
        "vmov.s32 q9, q8\n"
        "vst1.32 {d16,d17,d18,d19}, [r0]!\n"
        "vst1.32 {d16,d17,d18,d19}, [r0]!\n"
        "vst1.32 {d16,d17,d18,d19}, [r0]!\n"
        "vst1.32 {d16,d17,d18,d19}, [r0]!\n"
        "vst1.32 {d16,d17,d18,d19}, [r0]!\n"
        "vst1.32 {d16,d17,d18,d19}, [r0]!\n"
        "b " GEMMLOWP_LOOP_NEON_32_KERNEL_12X4_DEPTH2_ASSUMING_12BIT_PRODUCTS
        "f\n"

        // If start_depth!=0, we need to load our existing global accumulators
        GEMMLOWP_LOAD_GLOBAL_ACCUMULATORS_NEON_32_KERNEL_12X4_DEPTH2_12BIT
        ":\n"
        // Load global accumulators from destination matrix, column-major
        "mov r1, %[dst_ptr]\n"
        "mov r0, %[dst_col_stride]\n"
        "sub r0, #32\n"
        "vld1.32 {d0,d1}, [r1]!\n"
        "vld1.32 {d8,d9}, [r1]!\n"
        "vld1.32 {d16,d17}, [r1], r0\n"
        "vld1.32 {d2,d3}, [r1]!\n"
        "vld1.32 {d10,d11}, [r1]!\n"
        "vld1.32 {d18,d19}, [r1], r0\n"
        "vld1.32 {d4,d5}, [r1]!\n"
        "vld1.32 {d12,d13}, [r1]!\n"
        "vld1.32 {d20,d21}, [r1], r0\n"
        "vld1.32 {d6,d7}, [r1]!\n"
        "vld1.32 {d14,d15}, [r1]!\n"
        "vld1.32 {d22,d23}, [r1], r0\n"
        // Now we need to convert the global accumulator registers to
        // 4x4-block-wise diagonal-major order. What we effectively want to do
        // is to rotate the rows, however the accumulators are stored in
        // column-major order in registers. So we achieve this by
        // transposing, rotating the registers, and transposing again each
        // 4x4 block.
        //
        // Transpose 3 4x4 blocks separately
        "vtrn.32 q0, q1\n"
        "vtrn.32 q2, q3\n"
        "vswp d1, d4\n"
        "vswp d3, d6\n"
        "vtrn.32 q4, q5\n"
        "vtrn.32 q6, q7\n"
        "vswp d9, d12\n"
        "vswp d11, d14\n"
        "vtrn.32 q8, q9\n"
        "vtrn.32 q10, q11\n"
        "vswp d17, d20\n"
        "vswp d19, d22\n"
        // Rotate the registers
        "vext.32 q1, q1, q1, #1\n"
        "vext.32 q2, q2, q2, #2\n"
        "vext.32 q3, q3, q3, #3\n"
        "vext.32 q5, q5, q5, #1\n"
        "vext.32 q6, q6, q6, #2\n"
        "vext.32 q7, q7, q7, #3\n"
        "vext.32 q9, q9, q9, #1\n"
        "vext.32 q10, q10, q10, #2\n"
        "vext.32 q11, q11, q11, #3\n"
        // Transpose again and store into our global accumulators
        // buffer. These two operations are done at once using vst4.
        "mov r0, %[global_accumulators]\n"
        "vst4.32 {d0,d2,d4,d6}, [r0]!\n"
        "vst4.32 {d1,d3,d5,d7}, [r0]!\n"
        "vst4.32 {d8,d10,d12,d14}, [r0]!\n"
        "vst4.32 {d9,d11,d13,d15}, [r0]!\n"
        "vst4.32 {d16,d18,d20,d22}, [r0]!\n"
        "vst4.32 {d17,d19,d21,d23}, [r0]!\n"

        /* Main loop */

        GEMMLOWP_LOOP_NEON_32_KERNEL_12X4_DEPTH2_ASSUMING_12BIT_PRODUCTS
        ":\n"

// Overview of register layout:
//
// Registers q4--q16 are the local 16-bit accumulators.
// However, each entry in the result matrix is represented
// by *two* local 16-bit accumulators: one for even levels
// of depth and one for odd levels of depth. These correspond
// to the scalars at even and odd indices within each q-register.
// Thus we effectively use 32 bits of register space for each
// entry in the result matrix. The accumulators register layout
// is the same as was described above for the global 32-bit
// accumulators (3 cells of size 4x4 in diagonal-major order)
// with the only difference that instead of 32bit values we have
// pairs of 16bit values.
//
// A 2x4 cell of Rhs is stored in 8bit in d0.
// A 12x2 block of 3 4x2 cells Lhs is stored in 8bit in d1--d3.
//
//                      +--------+--------+--------+--------+
//                      |d0[0]   |d0[2]   |d0[4]   |d0[6]   |
//                 Rhs  +--------+--------+--------+--------+
//                      |d0[1]   |d0[3]   |d0[5]   |d0[7]   |
//                      +--------+--------+--------+--------+
//
//                      |        |        |        |        |
//
//    Lhs               |        |        |        |        |
//
//  +-----+-----+ - - - +--------+--------+--------+--------+
//  |d1[0]|d1[1]|       |q4[0,1] |q5[0,1] |q6[0,1] |q7[0,1] |
//  |d1[2]|d1[3]|       |q7[2,3] |q4[2,3] |q5[2,3] |q6[2,3] |
//  |d1[4]|d1[5]|       |q6[4,5] |q7[4,5] |q4[4,5] |q5[4,5] |
//  |d1[6]|d1[7]|       |q5[6,7] |q6[6,7] |q7[6,7] |q4[6,7] |
//  +-----+-----+ - - - +--------+--------+--------+--------+
//  |d2[0]|d2[1]|       |q8[0,1] |q8[0,1] |q8[0,1] |q8[0,1] |
//  |d2[2]|d2[3]|       |q9[2,3] |q9[2,3] |q9[2,3] |q9[2,3] |
//  |d2[4]|d2[5]|       |q10[4,5]|q10[4,5]|q10[4,5]|q10[4,5]|
//  |d2[6]|d2[7]|       |q11[6,7]|q11[6,7]|q11[6,7]|q11[6,7]|
//  +-----+-----+ - - - +--------+--------+--------+--------+
//  |d3[0]|d3[1]|       |q12[0,1]|q12[0,1]|q12[0,1]|q12[0,1]|
//  |d3[2]|d3[3]|       |q13[2,3]|q13[2,3]|q13[2,3]|q13[2,3]|
//  |d3[4]|d3[5]|       |q14[4,5]|q14[4,5]|q14[4,5]|q14[4,5]|
//  |d3[6]|d3[7]|       |q15[6,7]|q15[6,7]|q15[6,7]|q15[6,7]|
//  +-----+-----+ - - - +--------+--------+--------+--------+
//
//                            Local 16-bit accumulators
//                         Note: 2 scalars per matrix entry

#define GEMMLOWP_ACCUMULATE_2_LEVELS_OF_DEPTH \
  /* Load 3 Lhs cells of size 4x2 */          \
  "vld1.8 {d1,d2,d3}, [%[lhs_ptr]:64]!\n"     \
                                              \
  /* Load 1 Rhs cell of size 2x4 */           \
  "vld1.8 {d0}, [%[rhs_ptr]:64]!\n"           \
                                              \
  /* Multiply-accumulate */                   \
  "vmlal.u8 q4, d1, d0\n"                     \
  "vmlal.u8 q8, d2, d0\n"                     \
  "vmlal.u8 q12, d3, d0\n"                    \
  "vext.8 d0, d0, d0, #2\n"                   \
  "vmlal.u8 q5, d1, d0\n"                     \
  "vmlal.u8 q9, d2, d0\n"                     \
  "vmlal.u8 q13, d3, d0\n"                    \
  "vext.8 d0, d0, d0, #2\n"                   \
  "vmlal.u8 q6, d1, d0\n"                     \
  "vmlal.u8 q10, d2, d0\n"                    \
  "vmlal.u8 q14, d3, d0\n"                    \
  "vext.8 d0, d0, d0, #2\n"                   \
  "vmlal.u8 q7, d1, d0\n"                     \
  "vmlal.u8 q11, d2, d0\n"                    \
  "vmlal.u8 q15, d3, d0\n"                    \
                                              \
  "sub %[run_depth], #2\n"

#define GEMMLOWP_ACCUMULATE_8_LEVELS_OF_DEPTH \
  GEMMLOWP_ACCUMULATE_2_LEVELS_OF_DEPTH       \
  GEMMLOWP_ACCUMULATE_2_LEVELS_OF_DEPTH       \
  GEMMLOWP_ACCUMULATE_2_LEVELS_OF_DEPTH       \
  GEMMLOWP_ACCUMULATE_2_LEVELS_OF_DEPTH

        // Clear local 16-bit accumulators
        "vmov.s32 q4, #0\n"
        "vmov.s32 q5, q4\n"
        "vmov.s32 q6, q4\n"
        "vmov.s32 q7, q4\n"
        "vmov.s32 q8, q4\n"
        "vmov.s32 q9, q4\n"
        "vmov.s32 q10, q4\n"
        "vmov.s32 q11, q4\n"
        "vmov.s32 q12, q4\n"
        "vmov.s32 q13, q4\n"
        "vmov.s32 q14, q4\n"
        "vmov.s32 q15, q4\n"

        // Select a suitable number of depth levels
        // to process at this iteration. TODO (benoitjacob) I guess that
        // someone who really knows asm should make this a jump table.
        "cmp %[run_depth], #32\n"
        "bge " GEMMLOWP_LABEL_32
        "f\n"
        "cmp %[run_depth], #24\n"
        "bge " GEMMLOWP_LABEL_24
        "f\n"
        "cmp %[run_depth], #16\n"
        "bge " GEMMLOWP_LABEL_16
        "f\n"
        "cmp %[run_depth], #8\n"
        "bge " GEMMLOWP_LABEL_8
        "f\n"
        "b " GEMMLOWP_LABEL_2 "f\n"

        GEMMLOWP_LABEL_32
        ":\n" GEMMLOWP_ACCUMULATE_8_LEVELS_OF_DEPTH GEMMLOWP_LABEL_24
        ":\n" GEMMLOWP_ACCUMULATE_8_LEVELS_OF_DEPTH GEMMLOWP_LABEL_16
        ":\n" GEMMLOWP_ACCUMULATE_8_LEVELS_OF_DEPTH GEMMLOWP_LABEL_8
        ":\n" GEMMLOWP_ACCUMULATE_2_LEVELS_OF_DEPTH
            GEMMLOWP_ACCUMULATE_2_LEVELS_OF_DEPTH
                GEMMLOWP_ACCUMULATE_2_LEVELS_OF_DEPTH GEMMLOWP_LABEL_2
        ":\n" GEMMLOWP_ACCUMULATE_2_LEVELS_OF_DEPTH

        // Accumulate the local accumulators into the global accumulators.
        // This is about summing adjacent pairs of 16-bit scalars into
        // single 32-bit scalars, so we use pairwise long addition (vpadal).
        "mov r0, %[global_accumulators]\n"
        "mov r1, %[global_accumulators]\n"
        "vld1.32 {d0,d1,d2,d3}, [r0]!\n"
        "vld1.32 {d4,d5,d6,d7}, [r0]!\n"
        "vpadal.u16 q0, q4\n"
        "vpadal.u16 q1, q5\n"
        "vpadal.u16 q2, q6\n"
        "vpadal.u16 q3, q7\n"
        "vst1.32 {d0,d1,d2,d3}, [r1]!\n"
        "vst1.32 {d4,d5,d6,d7}, [r1]!\n"
        "vld1.32 {d0,d1,d2,d3}, [r0]!\n"
        "vld1.32 {d4,d5,d6,d7}, [r0]!\n"
        "vpadal.u16 q0, q8\n"
        "vpadal.u16 q1, q9\n"
        "vpadal.u16 q2, q10\n"
        "vpadal.u16 q3, q11\n"
        "vst1.32 {d0,d1,d2,d3}, [r1]!\n"
        "vst1.32 {d4,d5,d6,d7}, [r1]!\n"
        "vld1.32 {d0,d1,d2,d3}, [r0]!\n"
        "vld1.32 {d4,d5,d6,d7}, [r0]!\n"
        "vpadal.u16 q0, q12\n"
        "vpadal.u16 q1, q13\n"
        "vpadal.u16 q2, q14\n"
        "vpadal.u16 q3, q15\n"
        "vst1.32 {d0,d1,d2,d3}, [r1]!\n"
        "vst1.32 {d4,d5,d6,d7}, [r1]!\n"

        // Loop.
        "cmp %[run_depth], #0\n"
        "bne " GEMMLOWP_LOOP_NEON_32_KERNEL_12X4_DEPTH2_ASSUMING_12BIT_PRODUCTS
        "b\n"

#undef GEMMLOWP_CLEAR_LOCAL_ACCUMULATORS
#undef GEMMLOWP_ACCUMULATE_8_LEVELS_OF_DEPTH
#undef GEMMLOWP_ACCUMULATE_2_LEVELS_OF_DEPTH
#undef GEMMLOWP_ADD_TO_GLOBAL_ACCUMULATORS

        /* end of main loop */

        // Store the global accumulators to the destination matrix
        // (column-major)
        // This is the reverse of the steps that we followed at the beginning
        // when we load the global accumulators from the destination matrix.
        // The problem is the same: how to convert 4x4 blocks
        // between column-major and diagonal-major orders.
        // Like above, we do this by rotating rows, and we achieve that by
        // tranposing, rotating columns, and transposing again.
        //
        // Load and transpose 4x4 blocks of global accumulators
        // These two steps are done at once by the vld4 instruction.
        "mov r0, %[global_accumulators]\n"
        "vld4.32 {d0,d2,d4,d6}, [r0]!\n"
        "vld4.32 {d1,d3,d5,d7}, [r0]!\n"
        "vld4.32 {d8,d10,d12,d14}, [r0]!\n"
        "vld4.32 {d9,d11,d13,d15}, [r0]!\n"
        "vld4.32 {d16,d18,d20,d22}, [r0]!\n"
        "vld4.32 {d17,d19,d21,d23}, [r0]!\n"
        // Rotate the rows of each 4x4 block
        "vext.32 q1, q1, q1, #3\n"
        "vext.32 q2, q2, q2, #2\n"
        "vext.32 q3, q3, q3, #1\n"
        "vext.32 q5, q5, q5, #3\n"
        "vext.32 q6, q6, q6, #2\n"
        "vext.32 q7, q7, q7, #1\n"
        "vext.32 q9, q9, q9, #3\n"
        "vext.32 q10, q10, q10, #2\n"
        "vext.32 q11, q11, q11, #1\n"
        // Transpose again each 4x4 block
        "vtrn.32 q0, q1\n"
        "vtrn.32 q2, q3\n"
        "vswp d1, d4\n"
        "vswp d3, d6\n"
        "vtrn.32 q4, q5\n"
        "vtrn.32 q6, q7\n"
        "vswp d9, d12\n"
        "vswp d11, d14\n"
        "vtrn.32 q8, q9\n"
        "vtrn.32 q10, q11\n"
        "vswp d17, d20\n"
        "vswp d19, d22\n"
        // Store into the column-major destination matrix
        "mov r1, %[dst_ptr]\n"
        "mov r0, %[dst_col_stride]\n"
        "sub r0, #32\n"
        "vst1.32 {d0,d1}, [r1]!\n"
        "vst1.32 {d8,d9}, [r1]!\n"
        "vst1.32 {d16,d17}, [r1], r0\n"
        "vst1.32 {d2,d3}, [r1]!\n"
        "vst1.32 {d10,d11}, [r1]!\n"
        "vst1.32 {d18,d19}, [r1], r0\n"
        "vst1.32 {d4,d5}, [r1]!\n"
        "vst1.32 {d12,d13}, [r1]!\n"
        "vst1.32 {d20,d21}, [r1], r0\n"
        "vst1.32 {d6,d7}, [r1]!\n"
        "vst1.32 {d14,d15}, [r1]!\n"
        "vst1.32 {d22,d23}, [r1], r0\n"
        :  // outputs
        [lhs_ptr] "+r"(lhs_ptr), [rhs_ptr] "+r"(rhs_ptr),
        [dst_ptr] "+r"(dst_ptr),
        [run_depth] "+r"(run_depth)
        :  // inputs
        [start_depth] "r"(start_depth), [dst_col_stride] "r"(dst_col_stride),
        [global_accumulators] "r"(&global_accumulators[0])
        :  // clobbers
        "cc", "memory", "r0", "r1",
        // note: someone on internet says that quad registers are
        // unsupported in the clobber list!
        "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10",
        "d11", "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19", "d20",
        "d21", "d22", "d23", "d24", "d25", "d26", "d27", "d28", "d29", "d30",
        "d31");
#undef GEMMLOWP_LOOP_NEON_32_KERNEL_12X4_DEPTH2_ASSUMING_12BIT_PRODUCTS
#undef GEMMLOWP_LOAD_GLOBAL_ACCUMULATORS_NEON_32_KERNEL_12X4_DEPTH2_12BIT
#undef GEMMLOWP_LABEL_32
#undef GEMMLOWP_LABEL_24
#undef GEMMLOWP_LABEL_16
#undef GEMMLOWP_LABEL_8
#undef GEMMLOWP_LABEL_2
  }
};

#endif  // GEMMLOWP_NEON_32

// The kernels here are specifically arm 64bit assembly, not arm 32bit.
#ifdef GEMMLOWP_NEON_64

// Our main GEMM kernel.
struct NEON_64_Kernel12x8Depth2 : KernelBase {
  typedef KernelFormat<KernelSideFormat<CellFormat<4, 2>, 3>,
                       KernelSideFormat<CellFormat<4, 2>, 2> >
      Format;

  const char* Name() const override { return "NEON, 12x8, depth 2"; }

  // TODO(benoitjacob): reorder function arguments so dst comes last
  void Run(std::int32_t* dst_ptr, std::size_t dst_row_stride,
           std::size_t dst_col_stride, const std::uint8_t* lhs_ptr,
           const std::uint8_t* rhs_ptr, std::size_t start_depth,
           std::size_t run_depth) const override {
    ScopedProfilingLabel label("optimized kernel (NEON 12x8)");
// See comments above for why we need local numerical labels in our asm.
#define GEMMLOWP_LOOP_NEON_64_KERNEL_12X8_DEPTH2 "1"
#define GEMMLOWP_STORE_RESULT_NEON_64_KERNEL_12x8_DEPTH2 "2"

    assert(dst_row_stride == 1);
    asm volatile(
        // Clear accumulator registers (see layout below)
        "dup v8.4s, wzr\n"
        "dup v9.4s, wzr\n"
        "dup v10.4s, wzr\n"
        "dup v11.4s, wzr\n"
        "dup v12.4s, wzr\n"
        "dup v13.4s, wzr\n"
        "dup v14.4s, wzr\n"
        "dup v15.4s, wzr\n"
        "dup v16.4s, wzr\n"
        "dup v17.4s, wzr\n"
        "dup v18.4s, wzr\n"
        "dup v19.4s, wzr\n"
        "dup v20.4s, wzr\n"
        "dup v21.4s, wzr\n"
        "dup v22.4s, wzr\n"
        "dup v23.4s, wzr\n"
        "dup v24.4s, wzr\n"
        "dup v25.4s, wzr\n"
        "dup v26.4s, wzr\n"
        "dup v27.4s, wzr\n"
        "dup v28.4s, wzr\n"
        "dup v29.4s, wzr\n"
        "dup v30.4s, wzr\n"
        "dup v31.4s, wzr\n"

        /* Main loop */

        GEMMLOWP_LOOP_NEON_64_KERNEL_12X8_DEPTH2
        ":\n"

        // Overview of register layout:
        //
        // A 2x8 block of 2 2x4 cells of Rhs is stored in 16bit in v0--v1.
        // A 12x2 block of 3 4x2 cells Lhs is stored in 16bit in v2--v4.
        // A 12x8 block of accumulators is stored in 32bit in v8--v31.
        //
        //                         +--------+--------+-----+--------+--------+
        //                         |v0.h[0] |v0.h[1] | ... |v1.h[2] |v1.h[3] |
        //                    Rhs  +--------+--------+-----+--------+--------+
        //                         |v0.h[4] |v0.h[5] | ... |v1.h[6] |v1.h[7] |
        //                         +--------+--------+-----+--------+--------+
        //
        //                         |        |        |     |        |        |
        //
        //    Lhs                  |        |        |     |        |        |
        //
        //  +-------+-------+ - -  +--------+--------+-----+--------+--------+
        //  |v2.h[0]|v2.h[4]|      |v8.s[0] |v9.s[0] | ... |v14.s[0]|v15.s[0]|
        //  |v2.h[1]|v2.h[5]|      |v8.s[1] |v9.s[1] | ... |v14.s[1]|v15.s[1]|
        //  |v2.h[2]|v2.h[6]|      |v8.s[2] |v9.s[2] | ... |v14.s[2]|v15.s[2]|
        //  |v2.h[3]|v2.h[7]|      |v8.s[3] |v9.s[3] | ... |v14.s[3]|v15.s[3]|
        //  +-------+-------+ - -  +--------+--------+-----+--------+--------+
        //  |v3.h[0]|v3.h[4]|      |v16.s[0]|v17.s[0]| ... |v22.s[0]|v23.s[0]|
        //  |v3.h[1]|v3.h[5]|      |v16.s[1]|v17.s[1]| ... |v22.s[1]|v23.s[1]|
        //  |v3.h[2]|v3.h[6]|      |v16.s[2]|v17.s[2]| ... |v22.s[2]|v23.s[2]|
        //  |v3.h[3]|v3.h[7]|      |v16.s[3]|v17.s[3]| ... |v22.s[3]|v23.s[3]|
        //  +-------+-------+ - -  +--------+--------+-----+--------+--------+
        //  |v4.h[0]|v4.h[4]|      |v24.s[0]|v25.s[0]| ... |v30.s[0]|v31.s[0]|
        //  |v4.h[1]|v4.h[5]|      |v24.s[1]|v25.s[1]| ... |v30.s[1]|v31.s[1]|
        //  |v4.h[2]|v4.h[6]|      |v24.s[2]|v25.s[2]| ... |v30.s[2]|v31.s[2]|
        //  |v4.h[3]|v4.h[7]|      |v24.s[3]|v25.s[3]| ... |v30.s[3]|v31.s[3]|
        //  +-------+-------+ - -  +--------+--------+-----+--------+--------+
        //
        //                            Accumulator

        // Load 1 Rhs cell of size 2x8
        "ld1 {v0.8b}, [%[rhs_ptr]], #8\n"
        "ld1 {v1.8b}, [%[rhs_ptr]], #8\n"

        // Load 3 Lhs cells of size 4x2 each
        "ld1 {v2.8b}, [%[lhs_ptr]], #8\n"
        "ld1 {v3.8b}, [%[lhs_ptr]], #8\n"
        "ld1 {v4.8b}, [%[lhs_ptr]], #8\n"

        // Expand Lhs/Rhs cells to 16 bit.
        "uxtl v0.8h, v0.8b\n"
        "uxtl v1.8h, v1.8b\n"
        "uxtl v2.8h, v2.8b\n"
        "uxtl v3.8h, v3.8b\n"
        "uxtl v4.8h, v4.8b\n"

        // Multiply-accumulate, level of depth 0
        "umlal v8.4s, v2.4h, v0.h[0]\n"
        "umlal v9.4s, v2.4h, v0.h[1]\n"
        "umlal v10.4s, v2.4h, v0.h[2]\n"
        "umlal v11.4s, v2.4h, v0.h[3]\n"
        "umlal v12.4s, v2.4h, v1.h[0]\n"
        "umlal v13.4s, v2.4h, v1.h[1]\n"
        "umlal v14.4s, v2.4h, v1.h[2]\n"
        "umlal v15.4s, v2.4h, v1.h[3]\n"
        "umlal v16.4s, v3.4h, v0.h[0]\n"
        "umlal v17.4s, v3.4h, v0.h[1]\n"
        "umlal v18.4s, v3.4h, v0.h[2]\n"
        "umlal v19.4s, v3.4h, v0.h[3]\n"
        "umlal v20.4s, v3.4h, v1.h[0]\n"
        "umlal v21.4s, v3.4h, v1.h[1]\n"
        "umlal v22.4s, v3.4h, v1.h[2]\n"
        "umlal v23.4s, v3.4h, v1.h[3]\n"
        "umlal v24.4s, v4.4h, v0.h[0]\n"
        "umlal v25.4s, v4.4h, v0.h[1]\n"
        "umlal v26.4s, v4.4h, v0.h[2]\n"
        "umlal v27.4s, v4.4h, v0.h[3]\n"
        "umlal v28.4s, v4.4h, v1.h[0]\n"
        "umlal v29.4s, v4.4h, v1.h[1]\n"
        "umlal v30.4s, v4.4h, v1.h[2]\n"
        "umlal v31.4s, v4.4h, v1.h[3]\n"

        // Multiply-accumulate, level of depth 1
        "umlal2 v8.4s, v2.8h, v0.h[4]\n"
        "umlal2 v9.4s, v2.8h, v0.h[5]\n"
        "umlal2 v10.4s, v2.8h, v0.h[6]\n"
        "umlal2 v11.4s, v2.8h, v0.h[7]\n"
        "umlal2 v12.4s, v2.8h, v1.h[4]\n"
        "umlal2 v13.4s, v2.8h, v1.h[5]\n"
        "umlal2 v14.4s, v2.8h, v1.h[6]\n"
        "umlal2 v15.4s, v2.8h, v1.h[7]\n"
        "umlal2 v16.4s, v3.8h, v0.h[4]\n"
        "umlal2 v17.4s, v3.8h, v0.h[5]\n"
        "umlal2 v18.4s, v3.8h, v0.h[6]\n"
        "umlal2 v19.4s, v3.8h, v0.h[7]\n"
        "umlal2 v20.4s, v3.8h, v1.h[4]\n"
        "umlal2 v21.4s, v3.8h, v1.h[5]\n"
        "umlal2 v22.4s, v3.8h, v1.h[6]\n"
        "umlal2 v23.4s, v3.8h, v1.h[7]\n"
        "umlal2 v24.4s, v4.8h, v0.h[4]\n"
        "umlal2 v25.4s, v4.8h, v0.h[5]\n"
        "umlal2 v26.4s, v4.8h, v0.h[6]\n"
        "umlal2 v27.4s, v4.8h, v0.h[7]\n"
        "umlal2 v28.4s, v4.8h, v1.h[4]\n"
        "umlal2 v29.4s, v4.8h, v1.h[5]\n"
        "umlal2 v30.4s, v4.8h, v1.h[6]\n"
        "umlal2 v31.4s, v4.8h, v1.h[7]\n"

        // Loop. Decrement loop index (depth) by 2, since we just handled 2
        // levels of depth (Kernel::kDepth=2).
        "subs %[run_depth], %[run_depth], #2\n"
        "bne " GEMMLOWP_LOOP_NEON_64_KERNEL_12X8_DEPTH2
        "b\n"

        /* end of main loop */

        /* Accumulate our local accumulator registers into the destination block
           */

        // Compute stride between consecutive columns, in bytes
        "mov x0, #4\n"  // multiply by 4 = sizeof(int32)
        "mul %[dst_col_stride], %[dst_col_stride], x0\n"

        // If start_depth == 0, then there is no preexisting accumulator
        // to accumulate, so we can simply store our result.
        "cmp %[start_depth], #0\n"
        "beq " GEMMLOWP_STORE_RESULT_NEON_64_KERNEL_12x8_DEPTH2
        "f\n"

        "mov x0, %[dst_ptr]\n"

        // Load a column
        "mov x1, x0\n"
        "ld1 {v0.4s}, [x1], #16\n"
        "ld1 {v1.4s}, [x1], #16\n"
        "ld1 {v2.4s}, [x1], #16\n"
        // Accumulate a column
        "add v8.4s, v8.4s, v0.4s\n"
        "add v16.4s, v16.4s, v1.4s\n"
        "add v24.4s, v24.4s, v2.4s\n"

        "add x0, x0, %[dst_col_stride]\n"
        // Load a column
        "mov x1, x0\n"
        "ld1 {v0.4s}, [x1], #16\n"
        "ld1 {v1.4s}, [x1], #16\n"
        "ld1 {v2.4s}, [x1], #16\n"
        // Accumulate a column
        "add v9.4s, v9.4s, v0.4s\n"
        "add v17.4s, v17.4s, v1.4s\n"
        "add v25.4s, v25.4s, v2.4s\n"

        "add x0, x0, %[dst_col_stride]\n"
        // Load a column
        "mov x1, x0\n"
        "ld1 {v0.4s}, [x1], #16\n"
        "ld1 {v1.4s}, [x1], #16\n"
        "ld1 {v2.4s}, [x1], #16\n"
        // Accumulate a column
        "add v10.4s, v10.4s, v0.4s\n"
        "add v18.4s, v18.4s, v1.4s\n"
        "add v26.4s, v26.4s, v2.4s\n"

        "add x0, x0, %[dst_col_stride]\n"
        // Load a column
        "mov x1, x0\n"
        "ld1 {v0.4s}, [x1], #16\n"
        "ld1 {v1.4s}, [x1], #16\n"
        "ld1 {v2.4s}, [x1], #16\n"
        // Accumulate a column
        "add v11.4s, v11.4s, v0.4s\n"
        "add v19.4s, v19.4s, v1.4s\n"
        "add v27.4s, v27.4s, v2.4s\n"

        "add x0, x0, %[dst_col_stride]\n"
        // Load a column
        "mov x1, x0\n"
        "ld1 {v0.4s}, [x1], #16\n"
        "ld1 {v1.4s}, [x1], #16\n"
        "ld1 {v2.4s}, [x1], #16\n"
        // Accumulate a column
        "add v12.4s, v12.4s, v0.4s\n"
        "add v20.4s, v20.4s, v1.4s\n"
        "add v28.4s, v28.4s, v2.4s\n"

        "add x0, x0, %[dst_col_stride]\n"
        // Load a column
        "mov x1, x0\n"
        "ld1 {v0.4s}, [x1], #16\n"
        "ld1 {v1.4s}, [x1], #16\n"
        "ld1 {v2.4s}, [x1], #16\n"
        // Accumulate a column
        "add v13.4s, v13.4s, v0.4s\n"
        "add v21.4s, v21.4s, v1.4s\n"
        "add v29.4s, v29.4s, v2.4s\n"

        "add x0, x0, %[dst_col_stride]\n"
        // Load a column
        "mov x1, x0\n"
        "ld1 {v0.4s}, [x1], #16\n"
        "ld1 {v1.4s}, [x1], #16\n"
        "ld1 {v2.4s}, [x1], #16\n"
        // Accumulate a column
        "add v14.4s, v14.4s, v0.4s\n"
        "add v22.4s, v22.4s, v1.4s\n"
        "add v30.4s, v30.4s, v2.4s\n"

        "add x0, x0, %[dst_col_stride]\n"
        // Load a column
        "mov x1, x0\n"
        "ld1 {v0.4s}, [x1], #16\n"
        "ld1 {v1.4s}, [x1], #16\n"
        "ld1 {v2.4s}, [x1], #16\n"
        // Accumulate a column
        "add v15.4s, v15.4s, v0.4s\n"
        "add v23.4s, v23.4s, v1.4s\n"
        "add v31.4s, v31.4s, v2.4s\n"

        GEMMLOWP_STORE_RESULT_NEON_64_KERNEL_12x8_DEPTH2
        ":\n"

        "mov x0, %[dst_ptr]\n"
        // Store a column
        "mov x1, x0\n"
        "st1 {v8.4s}, [x1], #16\n"
        "st1 {v16.4s}, [x1], #16\n"
        "st1 {v24.4s}, [x1], #16\n"
        // Store a column
        "add x0, x0, %[dst_col_stride]\n"
        "mov x1, x0\n"
        "st1 {v9.4s}, [x1], #16\n"
        "st1 {v17.4s}, [x1], #16\n"
        "st1 {v25.4s}, [x1], #16\n"
        // Store a column
        "add x0, x0, %[dst_col_stride]\n"
        "mov x1, x0\n"
        "st1 {v10.4s}, [x1], #16\n"
        "st1 {v18.4s}, [x1], #16\n"
        "st1 {v26.4s}, [x1], #16\n"
        // Store a column
        "add x0, x0, %[dst_col_stride]\n"
        "mov x1, x0\n"
        "st1 {v11.4s}, [x1], #16\n"
        "st1 {v19.4s}, [x1], #16\n"
        "st1 {v27.4s}, [x1], #16\n"
        // Store a column
        "add x0, x0, %[dst_col_stride]\n"
        "mov x1, x0\n"
        "st1 {v12.4s}, [x1], #16\n"
        "st1 {v20.4s}, [x1], #16\n"
        "st1 {v28.4s}, [x1], #16\n"
        // Store a column
        "add x0, x0, %[dst_col_stride]\n"
        "mov x1, x0\n"
        "st1 {v13.4s}, [x1], #16\n"
        "st1 {v21.4s}, [x1], #16\n"
        "st1 {v29.4s}, [x1], #16\n"
        // Store a column
        "add x0, x0, %[dst_col_stride]\n"
        "mov x1, x0\n"
        "st1 {v14.4s}, [x1], #16\n"
        "st1 {v22.4s}, [x1], #16\n"
        "st1 {v30.4s}, [x1], #16\n"
        // Store a column
        "add x0, x0, %[dst_col_stride]\n"
        "mov x1, x0\n"
        "st1 {v15.4s}, [x1], #16\n"
        "st1 {v23.4s}, [x1], #16\n"
        "st1 {v31.4s}, [x1], #16\n"
        :  // outputs
        [lhs_ptr] "+r"(lhs_ptr), [rhs_ptr] "+r"(rhs_ptr),
        [dst_ptr] "+r"(dst_ptr),
        [run_depth] "+r"(run_depth)
        :  // inputs
        [start_depth] "r"(start_depth),
        [dst_col_stride] "r"(dst_col_stride)
        :  // clobbers
        "cc", "memory", "x0", "x1", "v0", "v1", "v2", "v3", "v4", "v5", "v6",
        "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16",
        "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26",
        "v27", "v28", "v29", "v30", "v31");
#undef GEMMLOWP_LOOP_NEON_64_KERNEL_12X8_DEPTH2
#undef GEMMLOWP_STORE_RESULT_NEON_64_KERNEL_12x8_DEPTH2
  }
};

#endif  // GEMMLOWP_NEON_64

// Our main GEMV kernel.
// Because our GEMV performance is low and not dominated by the kernel
// at the moment, it's not worth optimizing too hard yet.
// Using intrinsics allows us to write one implementation for both 32bit and
// 64bit ARM, and should also perform OK here because the register pressure
// is not so high in this GEMV kernel.
// When/if we get serious about GEMV performance, we will want to
// implement it to bypass packing altogether, and use source data in-place
// with different GEMV kernels for row-major and column-major LHS.
template <int Cells>
struct NEONKernel4Nx1Depth2 : KernelBase {
  typedef KernelFormat<KernelSideFormat<CellFormat<4, 2>, Cells>,
                       KernelSideFormat<CellFormat<1, 2>, 1> >
      Format;

  const char* Name() const override { return "NEON intrinsics, 4Nx1, depth 2"; }

  void Run(std::int32_t* dst_ptr, std::size_t dst_row_stride,
           std::size_t dst_col_stride, const std::uint8_t* lhs_ptr,
           const std::uint8_t* rhs_ptr, std::size_t start_depth,
           std::size_t run_depth) const override {
    ScopedProfilingLabel label("optimized kernel (NEON 4Nx1)");

    assert(dst_row_stride == 1);

    // Clear accumulators
    uint32x4_t acc[Cells];
    for (int cell = 0; cell < Cells; cell++) {
      acc[cell] = vdupq_n_u32(0);
    }
    // Main loop
    for (std::size_t d = 0; d < run_depth; d += 2) {
      // Load LHS cells
      uint16x8_t lhs[Cells];
      for (int cell = 0; cell < Cells; cell++) {
        lhs[cell] = vmovl_u8(vld1_u8(lhs_ptr));
        lhs_ptr += 8;
      }
      // Load RHS cell
      uint16_t rhs0 = rhs_ptr[0];
      uint16_t rhs1 = rhs_ptr[1];
      rhs_ptr += 2;
      // Multiply-accumulate, level of depth 0
      for (int cell = 0; cell < Cells; cell++) {
        acc[cell] = vmlal_n_u16(acc[cell], vget_low_u16(lhs[cell]), rhs0);
      }
      // Multiply-accumulate, level of depth 1
      for (int cell = 0; cell < Cells; cell++) {
        acc[cell] = vmlal_n_u16(acc[cell], vget_high_u16(lhs[cell]), rhs1);
      }
    }
    // If start_depth is nonzero, accumulate with the existing accumulator
    if (start_depth) {
      for (int cell = 0; cell < Cells; cell++) {
        acc[cell] = vaddq_u32(
            acc[cell], vreinterpretq_u32_s32(vld1q_s32(dst_ptr + 4 * cell)));
      }
    }
    // Store the accumulators
    for (int cell = 0; cell < Cells; cell++) {
      vst1q_s32(dst_ptr + 4 * cell, vreinterpretq_s32_u32(acc[cell]));
    }
  }
};

}  // namespace gemmlowp

#endif  // GEMMLOWP_INTERNAL_KERNEL_NEON_H_
