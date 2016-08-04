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

#ifndef GEMMLOWP_META_SINGLE_THREAD_GEMM_ARM32_H_
#define GEMMLOWP_META_SINGLE_THREAD_GEMM_ARM32_H_

#ifdef GEMMLOWP_NEON_32

#include <cassert>

namespace gemmlowp {
namespace meta {
namespace internal {

void zip_1x8_aligned(const std::uint8_t* source, std::int32_t count,
                     std::int32_t stride, std::uint8_t* destination,
                     std::int32_t multiplicative_offset,
                     std::int32_t additive_offset) {
  asm volatile(
      "vmov.i16 q4, #0\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store.
      "vld1.8 {d0}, [%[source]:64]!\n"
      "vaddw.u8 q4, q4, d0\n"
      "vst1.8 {d0}, [%[destination]:64]!\n"

      "bne 1b\n"

      // Aggregator Reduction.
      "vmov.32 d1[0], %[multiplicative_offset]\n"
      "vdup.32 d2, %[additive_offset]\n"
      "vpaddl.u16 q4, q4\n"
      "vpadd.u32 d8, d8, d9\n"
      "vpadd.u32 d3, d8, d8\n"
      "vmul.i32 d3, d3, d1[0]\n"
      "vadd.i32 d3, d3, d2\n"
      "vst1.32 {d3}, [%[destination]:64]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "d0", "d1", "d2", "d3", "d8", "d9", "cc", "memory");
}

void zip_1x8_1_aligned(const std::uint8_t* source, std::int32_t count,
                       std::int32_t stride, std::uint8_t* destination,
                       std::int32_t multiplicative_offset,
                       std::int32_t additive_offset) {
  asm volatile(
      "sub %[count], %[count], #1\n"
      "vmov.i16 q4, #0\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store.
      "vld1.8 {d0}, [%[source]:64]!\n"
      "vaddw.u8 q4, q4, d0\n"
      "vst1.8 {d0}, [%[destination]:64]!\n"

      "bne 1b\n"

      // Leftover Load Aggregate Store.
      "vmov.i8 d0, #0\n"
      "vld1.8 {d0[0]}, [%[source]]!\n"
      "vaddw.u8 q4, q4, d0\n"
      "vst1.8 {d0}, [%[destination]:64]!\n"

      // Aggregator Reduction.
      "vmov.32 d1[0], %[multiplicative_offset]\n"
      "vdup.32 d2, %[additive_offset]\n"
      "vpaddl.u16 q4, q4\n"
      "vpadd.u32 d8, d8, d9\n"
      "vpadd.u32 d3, d8, d8\n"
      "vmul.i32 d3, d3, d1[0]\n"
      "vadd.i32 d3, d3, d2\n"
      "vst1.32 {d3}, [%[destination]:64]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "d0", "d1", "d2", "d3", "d8", "d9", "cc", "memory");
}

void zip_1x8_2_aligned(const std::uint8_t* source, std::int32_t count,
                       std::int32_t stride, std::uint8_t* destination,
                       std::int32_t multiplicative_offset,
                       std::int32_t additive_offset) {
  asm volatile(
      "sub %[count], %[count], #2\n"
      "vmov.i16 q4, #0\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store.
      "vld1.8 {d0}, [%[source]:64]!\n"
      "vaddw.u8 q4, q4, d0\n"
      "vst1.8 {d0}, [%[destination]:64]!\n"

      "bne 1b\n"

      // Leftover Load Aggregate Store.
      "vmov.i8 d0, #0\n"
      "vld1.16 {d0[0]}, [%[source]]!\n"
      "vaddw.u8 q4, q4, d0\n"
      "vst1.8 {d0}, [%[destination]:64]!\n"

      // Aggregator Reduction.
      "vmov.32 d1[0], %[multiplicative_offset]\n"
      "vdup.32 d2, %[additive_offset]\n"
      "vpaddl.u16 q4, q4\n"
      "vpadd.u32 d8, d8, d9\n"
      "vpadd.u32 d3, d8, d8\n"
      "vmul.i32 d3, d3, d1[0]\n"
      "vadd.i32 d3, d3, d2\n"
      "vst1.32 {d3}, [%[destination]:64]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "d0", "d1", "d2", "d3", "d8", "d9", "cc", "memory");
}

void zip_1x8_3_aligned(const std::uint8_t* source, std::int32_t count,
                       std::int32_t stride, std::uint8_t* destination,
                       std::int32_t multiplicative_offset,
                       std::int32_t additive_offset) {
  asm volatile(
      "sub %[count], %[count], #3\n"
      "vmov.i16 q4, #0\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store.
      "vld1.8 {d0}, [%[source]:64]!\n"
      "vaddw.u8 q4, q4, d0\n"
      "vst1.8 {d0}, [%[destination]:64]!\n"

      "bne 1b\n"

      // Leftover Load Aggregate Store.
      "vmov.i8 d0, #0\n"
      "vld1.16 {d0[0]}, [%[source]]!\n"
      "vld1.8 {d0[2]}, [%[source]]!\n"
      "vaddw.u8 q4, q4, d0\n"
      "vst1.8 {d0}, [%[destination]:64]!\n"

      // Aggregator Reduction.
      "vmov.32 d1[0], %[multiplicative_offset]\n"
      "vdup.32 d2, %[additive_offset]\n"
      "vpaddl.u16 q4, q4\n"
      "vpadd.u32 d8, d8, d9\n"
      "vpadd.u32 d3, d8, d8\n"
      "vmul.i32 d3, d3, d1[0]\n"
      "vadd.i32 d3, d3, d2\n"
      "vst1.32 {d3}, [%[destination]:64]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "d0", "d1", "d2", "d3", "d8", "d9", "cc", "memory");
}

void zip_1x8_4_aligned(const std::uint8_t* source, std::int32_t count,
                       std::int32_t stride, std::uint8_t* destination,
                       std::int32_t multiplicative_offset,
                       std::int32_t additive_offset) {
  asm volatile(
      "sub %[count], %[count], #4\n"
      "vmov.i16 q4, #0\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store.
      "vld1.8 {d0}, [%[source]:64]!\n"
      "vaddw.u8 q4, q4, d0\n"
      "vst1.8 {d0}, [%[destination]:64]!\n"

      "bne 1b\n"

      // Leftover Load Aggregate Store.
      "vmov.i8 d0, #0\n"
      "vld1.32 {d0[0]}, [%[source]]!\n"
      "vaddw.u8 q4, q4, d0\n"
      "vst1.8 {d0}, [%[destination]:64]!\n"

      // Aggregator Reduction.
      "vmov.32 d1[0], %[multiplicative_offset]\n"
      "vdup.32 d2, %[additive_offset]\n"
      "vpaddl.u16 q4, q4\n"
      "vpadd.u32 d8, d8, d9\n"
      "vpadd.u32 d3, d8, d8\n"
      "vmul.i32 d3, d3, d1[0]\n"
      "vadd.i32 d3, d3, d2\n"
      "vst1.32 {d3}, [%[destination]:64]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "d0", "d1", "d2", "d3", "d8", "d9", "cc", "memory");
}

void zip_1x8_5_aligned(const std::uint8_t* source, std::int32_t count,
                       std::int32_t stride, std::uint8_t* destination,
                       std::int32_t multiplicative_offset,
                       std::int32_t additive_offset) {
  asm volatile(
      "sub %[count], %[count], #5\n"
      "vmov.i16 q4, #0\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store.
      "vld1.8 {d0}, [%[source]:64]!\n"
      "vaddw.u8 q4, q4, d0\n"
      "vst1.8 {d0}, [%[destination]:64]!\n"

      "bne 1b\n"

      // Leftover Load Aggregate Store.
      "vmov.i8 d0, #0\n"
      "vld1.32 {d0[0]}, [%[source]]!\n"
      "vld1.8 {d0[4]}, [%[source]]!\n"
      "vaddw.u8 q4, q4, d0\n"
      "vst1.8 {d0}, [%[destination]:64]!\n"

      // Aggregator Reduction.
      "vmov.32 d1[0], %[multiplicative_offset]\n"
      "vdup.32 d2, %[additive_offset]\n"
      "vpaddl.u16 q4, q4\n"
      "vpadd.u32 d8, d8, d9\n"
      "vpadd.u32 d3, d8, d8\n"
      "vmul.i32 d3, d3, d1[0]\n"
      "vadd.i32 d3, d3, d2\n"
      "vst1.32 {d3}, [%[destination]:64]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "d0", "d1", "d2", "d3", "d8", "d9", "cc", "memory");
}

void zip_1x8_6_aligned(const std::uint8_t* source, std::int32_t count,
                       std::int32_t stride, std::uint8_t* destination,
                       std::int32_t multiplicative_offset,
                       std::int32_t additive_offset) {
  asm volatile(
      "sub %[count], %[count], #6\n"
      "vmov.i16 q4, #0\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store.
      "vld1.8 {d0}, [%[source]:64]!\n"
      "vaddw.u8 q4, q4, d0\n"
      "vst1.8 {d0}, [%[destination]:64]!\n"

      "bne 1b\n"

      // Leftover Load Aggregate Store.
      "vmov.i8 d0, #0\n"
      "vld1.32 {d0[0]}, [%[source]]!\n"
      "vld1.16 {d0[2]}, [%[source]]!\n"
      "vaddw.u8 q4, q4, d0\n"
      "vst1.8 {d0}, [%[destination]:64]!\n"

      // Aggregator Reduction.
      "vmov.32 d1[0], %[multiplicative_offset]\n"
      "vdup.32 d2, %[additive_offset]\n"
      "vpaddl.u16 q4, q4\n"
      "vpadd.u32 d8, d8, d9\n"
      "vpadd.u32 d3, d8, d8\n"
      "vmul.i32 d3, d3, d1[0]\n"
      "vadd.i32 d3, d3, d2\n"
      "vst1.32 {d3}, [%[destination]:64]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "d0", "d1", "d2", "d3", "d8", "d9", "cc", "memory");
}

void zip_1x8_7_aligned(const std::uint8_t* source, std::int32_t count,
                       std::int32_t stride, std::uint8_t* destination,
                       std::int32_t multiplicative_offset,
                       std::int32_t additive_offset) {
  asm volatile(
      "sub %[count], %[count], #7\n"
      "vmov.i16 q4, #0\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store.
      "vld1.8 {d0}, [%[source]:64]!\n"
      "vaddw.u8 q4, q4, d0\n"
      "vst1.8 {d0}, [%[destination]:64]!\n"

      "bne 1b\n"

      // Leftover Load Aggregate Store.
      "vmov.i8 d0, #0\n"
      "vld1.32 {d0[0]}, [%[source]]!\n"
      "vld1.16 {d0[2]}, [%[source]]!\n"
      "vld1.8 {d0[6]}, [%[source]]!\n"
      "vaddw.u8 q4, q4, d0\n"
      "vst1.8 {d0}, [%[destination]:64]!\n"

      // Aggregator Reduction.
      "vmov.32 d1[0], %[multiplicative_offset]\n"
      "vdup.32 d2, %[additive_offset]\n"
      "vpaddl.u16 q4, q4\n"
      "vpadd.u32 d8, d8, d9\n"
      "vpadd.u32 d3, d8, d8\n"
      "vmul.i32 d3, d3, d1[0]\n"
      "vadd.i32 d3, d3, d2\n"
      "vst1.32 {d3}, [%[destination]:64]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "d0", "d1", "d2", "d3", "d8", "d9", "cc", "memory");
}

void zip_2x8_aligned(const std::uint8_t* source, std::int32_t count,
                     std::int32_t stride, std::uint8_t* destination,
                     std::int32_t multiplicative_offset,
                     std::int32_t additive_offset) {
  asm volatile(
      "add r0, %[source], %[stride]\n"
      "vmov.i16 q4, #0\n"
      "vmov.i16 q5, #0\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store.
      "vld1.8 {d0}, [%[source]:64]!\n"
      "vld1.8 {d1}, [r0:64]!\n"
      "vaddw.u8 q4, q4, d0\n"
      "vaddw.u8 q5, q5, d1\n"
      "vst1.8 {d0, d1}, [%[destination]:64]!\n"

      "bne 1b\n"

      // Aggregator Reduction.
      "vmov.32 d2[0], %[multiplicative_offset]\n"
      "vdup.32 d3, %[additive_offset]\n"
      "vpaddl.u16 q4, q4\n"
      "vpaddl.u16 q5, q5\n"
      "vpadd.u32 d8, d8, d9\n"
      "vpadd.u32 d10, d10, d11\n"
      "vpadd.u32 d4, d8, d10\n"
      "vmul.i32 d4, d4, d2[0]\n"
      "vadd.i32 d4, d4, d3\n"
      "vst1.32 {d4}, [%[destination]:64]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "r0", "d0", "d1", "d2", "d3", "d4", "d8", "d9", "d10", "d11", "cc",
        "memory");
}

void zip_2x8_1_aligned(const std::uint8_t* source, std::int32_t count,
                       std::int32_t stride, std::uint8_t* destination,
                       std::int32_t multiplicative_offset,
                       std::int32_t additive_offset) {
  asm volatile(
      "add r0, %[source], %[stride]\n"
      "sub %[count], %[count], #1\n"
      "vmov.i16 q4, #0\n"
      "vmov.i16 q5, #0\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store.
      "vld1.8 {d0}, [%[source]:64]!\n"
      "vld1.8 {d1}, [r0:64]!\n"
      "vaddw.u8 q4, q4, d0\n"
      "vaddw.u8 q5, q5, d1\n"
      "vst1.8 {d0, d1}, [%[destination]:64]!\n"

      "bne 1b\n"

      // Leftover Load Aggregate Store.
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vld1.8 {d0[0]}, [%[source]]!\n"
      "vld1.8 {d1[0]}, [r0]!\n"
      "vaddw.u8 q4, q4, d0\n"
      "vaddw.u8 q5, q5, d1\n"
      "vst1.8 {d0, d1}, [%[destination]:64]!\n"

      // Aggregator Reduction.
      "vmov.32 d2[0], %[multiplicative_offset]\n"
      "vdup.32 d3, %[additive_offset]\n"
      "vpaddl.u16 q4, q4\n"
      "vpaddl.u16 q5, q5\n"
      "vpadd.u32 d8, d8, d9\n"
      "vpadd.u32 d10, d10, d11\n"
      "vpadd.u32 d4, d8, d10\n"
      "vmul.i32 d4, d4, d2[0]\n"
      "vadd.i32 d4, d4, d3\n"
      "vst1.32 {d4}, [%[destination]:64]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "r0", "d0", "d1", "d2", "d3", "d4", "d8", "d9", "d10", "d11", "cc",
        "memory");
}

void zip_2x8_2_aligned(const std::uint8_t* source, std::int32_t count,
                       std::int32_t stride, std::uint8_t* destination,
                       std::int32_t multiplicative_offset,
                       std::int32_t additive_offset) {
  asm volatile(
      "add r0, %[source], %[stride]\n"
      "sub %[count], %[count], #2\n"
      "vmov.i16 q4, #0\n"
      "vmov.i16 q5, #0\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store.
      "vld1.8 {d0}, [%[source]:64]!\n"
      "vld1.8 {d1}, [r0:64]!\n"
      "vaddw.u8 q4, q4, d0\n"
      "vaddw.u8 q5, q5, d1\n"
      "vst1.8 {d0, d1}, [%[destination]:64]!\n"

      "bne 1b\n"

      // Leftover Load Aggregate Store.
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vld1.16 {d0[0]}, [%[source]]!\n"
      "vld1.16 {d1[0]}, [r0]!\n"
      "vaddw.u8 q4, q4, d0\n"
      "vaddw.u8 q5, q5, d1\n"
      "vst1.8 {d0, d1}, [%[destination]:64]!\n"

      // Aggregator Reduction.
      "vmov.32 d2[0], %[multiplicative_offset]\n"
      "vdup.32 d3, %[additive_offset]\n"
      "vpaddl.u16 q4, q4\n"
      "vpaddl.u16 q5, q5\n"
      "vpadd.u32 d8, d8, d9\n"
      "vpadd.u32 d10, d10, d11\n"
      "vpadd.u32 d4, d8, d10\n"
      "vmul.i32 d4, d4, d2[0]\n"
      "vadd.i32 d4, d4, d3\n"
      "vst1.32 {d4}, [%[destination]:64]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "r0", "d0", "d1", "d2", "d3", "d4", "d8", "d9", "d10", "d11", "cc",
        "memory");
}

void zip_2x8_3_aligned(const std::uint8_t* source, std::int32_t count,
                       std::int32_t stride, std::uint8_t* destination,
                       std::int32_t multiplicative_offset,
                       std::int32_t additive_offset) {
  asm volatile(
      "add r0, %[source], %[stride]\n"
      "sub %[count], %[count], #3\n"
      "vmov.i16 q4, #0\n"
      "vmov.i16 q5, #0\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store.
      "vld1.8 {d0}, [%[source]:64]!\n"
      "vld1.8 {d1}, [r0:64]!\n"
      "vaddw.u8 q4, q4, d0\n"
      "vaddw.u8 q5, q5, d1\n"
      "vst1.8 {d0, d1}, [%[destination]:64]!\n"

      "bne 1b\n"

      // Leftover Load Aggregate Store.
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vld1.16 {d0[0]}, [%[source]]!\n"
      "vld1.8 {d0[2]}, [%[source]]!\n"
      "vld1.16 {d1[0]}, [r0]!\n"
      "vld1.8 {d1[2]}, [r0]!\n"
      "vaddw.u8 q4, q4, d0\n"
      "vaddw.u8 q5, q5, d1\n"
      "vst1.8 {d0, d1}, [%[destination]:64]!\n"

      // Aggregator Reduction.
      "vmov.32 d2[0], %[multiplicative_offset]\n"
      "vdup.32 d3, %[additive_offset]\n"
      "vpaddl.u16 q4, q4\n"
      "vpaddl.u16 q5, q5\n"
      "vpadd.u32 d8, d8, d9\n"
      "vpadd.u32 d10, d10, d11\n"
      "vpadd.u32 d4, d8, d10\n"
      "vmul.i32 d4, d4, d2[0]\n"
      "vadd.i32 d4, d4, d3\n"
      "vst1.32 {d4}, [%[destination]:64]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "r0", "d0", "d1", "d2", "d3", "d4", "d8", "d9", "d10", "d11", "cc",
        "memory");
}

void zip_2x8_4_aligned(const std::uint8_t* source, std::int32_t count,
                       std::int32_t stride, std::uint8_t* destination,
                       std::int32_t multiplicative_offset,
                       std::int32_t additive_offset) {
  asm volatile(
      "add r0, %[source], %[stride]\n"
      "sub %[count], %[count], #4\n"
      "vmov.i16 q4, #0\n"
      "vmov.i16 q5, #0\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store.
      "vld1.8 {d0}, [%[source]:64]!\n"
      "vld1.8 {d1}, [r0:64]!\n"
      "vaddw.u8 q4, q4, d0\n"
      "vaddw.u8 q5, q5, d1\n"
      "vst1.8 {d0, d1}, [%[destination]:64]!\n"

      "bne 1b\n"

      // Leftover Load Aggregate Store.
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vld1.32 {d0[0]}, [%[source]]!\n"
      "vld1.32 {d1[0]}, [r0]!\n"
      "vaddw.u8 q4, q4, d0\n"
      "vaddw.u8 q5, q5, d1\n"
      "vst1.8 {d0, d1}, [%[destination]:64]!\n"

      // Aggregator Reduction.
      "vmov.32 d2[0], %[multiplicative_offset]\n"
      "vdup.32 d3, %[additive_offset]\n"
      "vpaddl.u16 q4, q4\n"
      "vpaddl.u16 q5, q5\n"
      "vpadd.u32 d8, d8, d9\n"
      "vpadd.u32 d10, d10, d11\n"
      "vpadd.u32 d4, d8, d10\n"
      "vmul.i32 d4, d4, d2[0]\n"
      "vadd.i32 d4, d4, d3\n"
      "vst1.32 {d4}, [%[destination]:64]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "r0", "d0", "d1", "d2", "d3", "d4", "d8", "d9", "d10", "d11", "cc",
        "memory");
}

void zip_2x8_5_aligned(const std::uint8_t* source, std::int32_t count,
                       std::int32_t stride, std::uint8_t* destination,
                       std::int32_t multiplicative_offset,
                       std::int32_t additive_offset) {
  asm volatile(
      "add r0, %[source], %[stride]\n"
      "sub %[count], %[count], #5\n"
      "vmov.i16 q4, #0\n"
      "vmov.i16 q5, #0\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store.
      "vld1.8 {d0}, [%[source]:64]!\n"
      "vld1.8 {d1}, [r0:64]!\n"
      "vaddw.u8 q4, q4, d0\n"
      "vaddw.u8 q5, q5, d1\n"
      "vst1.8 {d0, d1}, [%[destination]:64]!\n"

      "bne 1b\n"

      // Leftover Load Aggregate Store.
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vld1.32 {d0[0]}, [%[source]]!\n"
      "vld1.8 {d0[4]}, [%[source]]!\n"
      "vld1.32 {d1[0]}, [r0]!\n"
      "vld1.8 {d1[4]}, [r0]!\n"
      "vaddw.u8 q4, q4, d0\n"
      "vaddw.u8 q5, q5, d1\n"
      "vst1.8 {d0, d1}, [%[destination]:64]!\n"

      // Aggregator Reduction.
      "vmov.32 d2[0], %[multiplicative_offset]\n"
      "vdup.32 d3, %[additive_offset]\n"
      "vpaddl.u16 q4, q4\n"
      "vpaddl.u16 q5, q5\n"
      "vpadd.u32 d8, d8, d9\n"
      "vpadd.u32 d10, d10, d11\n"
      "vpadd.u32 d4, d8, d10\n"
      "vmul.i32 d4, d4, d2[0]\n"
      "vadd.i32 d4, d4, d3\n"
      "vst1.32 {d4}, [%[destination]:64]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "r0", "d0", "d1", "d2", "d3", "d4", "d8", "d9", "d10", "d11", "cc",
        "memory");
}

void zip_2x8_6_aligned(const std::uint8_t* source, std::int32_t count,
                       std::int32_t stride, std::uint8_t* destination,
                       std::int32_t multiplicative_offset,
                       std::int32_t additive_offset) {
  asm volatile(
      "add r0, %[source], %[stride]\n"
      "sub %[count], %[count], #6\n"
      "vmov.i16 q4, #0\n"
      "vmov.i16 q5, #0\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store.
      "vld1.8 {d0}, [%[source]:64]!\n"
      "vld1.8 {d1}, [r0:64]!\n"
      "vaddw.u8 q4, q4, d0\n"
      "vaddw.u8 q5, q5, d1\n"
      "vst1.8 {d0, d1}, [%[destination]:64]!\n"

      "bne 1b\n"

      // Leftover Load Aggregate Store.
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vld1.32 {d0[0]}, [%[source]]!\n"
      "vld1.16 {d0[2]}, [%[source]]!\n"
      "vld1.32 {d1[0]}, [r0]!\n"
      "vld1.16 {d1[2]}, [r0]!\n"
      "vaddw.u8 q4, q4, d0\n"
      "vaddw.u8 q5, q5, d1\n"
      "vst1.8 {d0, d1}, [%[destination]:64]!\n"

      // Aggregator Reduction.
      "vmov.32 d2[0], %[multiplicative_offset]\n"
      "vdup.32 d3, %[additive_offset]\n"
      "vpaddl.u16 q4, q4\n"
      "vpaddl.u16 q5, q5\n"
      "vpadd.u32 d8, d8, d9\n"
      "vpadd.u32 d10, d10, d11\n"
      "vpadd.u32 d4, d8, d10\n"
      "vmul.i32 d4, d4, d2[0]\n"
      "vadd.i32 d4, d4, d3\n"
      "vst1.32 {d4}, [%[destination]:64]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "r0", "d0", "d1", "d2", "d3", "d4", "d8", "d9", "d10", "d11", "cc",
        "memory");
}

void zip_2x8_7_aligned(const std::uint8_t* source, std::int32_t count,
                       std::int32_t stride, std::uint8_t* destination,
                       std::int32_t multiplicative_offset,
                       std::int32_t additive_offset) {
  asm volatile(
      "add r0, %[source], %[stride]\n"
      "sub %[count], %[count], #7\n"
      "vmov.i16 q4, #0\n"
      "vmov.i16 q5, #0\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store.
      "vld1.8 {d0}, [%[source]:64]!\n"
      "vld1.8 {d1}, [r0:64]!\n"
      "vaddw.u8 q4, q4, d0\n"
      "vaddw.u8 q5, q5, d1\n"
      "vst1.8 {d0, d1}, [%[destination]:64]!\n"

      "bne 1b\n"

      // Leftover Load Aggregate Store.
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vld1.32 {d0[0]}, [%[source]]!\n"
      "vld1.16 {d0[2]}, [%[source]]!\n"
      "vld1.8 {d0[6]}, [%[source]]!\n"
      "vld1.32 {d1[0]}, [r0]!\n"
      "vld1.16 {d1[2]}, [r0]!\n"
      "vld1.8 {d1[6]}, [r0]!\n"
      "vaddw.u8 q4, q4, d0\n"
      "vaddw.u8 q5, q5, d1\n"
      "vst1.8 {d0, d1}, [%[destination]:64]!\n"

      // Aggregator Reduction.
      "vmov.32 d2[0], %[multiplicative_offset]\n"
      "vdup.32 d3, %[additive_offset]\n"
      "vpaddl.u16 q4, q4\n"
      "vpaddl.u16 q5, q5\n"
      "vpadd.u32 d8, d8, d9\n"
      "vpadd.u32 d10, d10, d11\n"
      "vpadd.u32 d4, d8, d10\n"
      "vmul.i32 d4, d4, d2[0]\n"
      "vadd.i32 d4, d4, d3\n"
      "vst1.32 {d4}, [%[destination]:64]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "r0", "d0", "d1", "d2", "d3", "d4", "d8", "d9", "d10", "d11", "cc",
        "memory");
}

void zip_3x8_aligned(const std::uint8_t* source, std::int32_t count,
                     std::int32_t stride, std::uint8_t* destination,
                     std::int32_t multiplicative_offset,
                     std::int32_t additive_offset) {
  asm volatile(
      "add r0, %[source], %[stride]\n"
      "add r1, r0, %[stride]\n"
      "vmov.i16 q4, #0\n"
      "vmov.i16 q5, #0\n"
      "vmov.i16 q6, #0\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store.
      "vld1.8 {d0}, [%[source]:64]!\n"
      "vld1.8 {d1}, [r0:64]!\n"
      "vld1.8 {d2}, [r1:64]!\n"
      "vaddw.u8 q4, q4, d0\n"
      "vaddw.u8 q5, q5, d1\n"
      "vaddw.u8 q6, q6, d2\n"
      "vst1.8 {d0, d1, d2}, [%[destination]:64]!\n"

      "bne 1b\n"

      // Aggregator Reduction.
      "vmov.32 d3[0], %[multiplicative_offset]\n"
      "vdup.32 q2, %[additive_offset]\n"
      "vpaddl.u16 q4, q4\n"
      "vpaddl.u16 q5, q5\n"
      "vpaddl.u16 q6, q6\n"
      "vpadd.u32 d8, d8, d9\n"
      "vpadd.u32 d10, d10, d11\n"
      "vpadd.u32 d12, d12, d13\n"
      "vpadd.u32 d6, d8, d10\n"
      "vpadd.u32 d7, d12, d12\n"
      "vmul.i32 q3, q3, d3[0]\n"
      "vadd.i32 q3, q3, q2\n"
      "vst1.32 {d6, d7}, [%[destination]:64]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "r0", "r1", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9",
        "d10", "d11", "d12", "d13", "cc", "memory");
}

void zip_3x8_1_aligned(const std::uint8_t* source, std::int32_t count,
                       std::int32_t stride, std::uint8_t* destination,
                       std::int32_t multiplicative_offset,
                       std::int32_t additive_offset) {
  asm volatile(
      "add r0, %[source], %[stride]\n"
      "add r1, r0, %[stride]\n"
      "sub %[count], %[count], #1\n"
      "vmov.i16 q4, #0\n"
      "vmov.i16 q5, #0\n"
      "vmov.i16 q6, #0\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store.
      "vld1.8 {d0}, [%[source]:64]!\n"
      "vld1.8 {d1}, [r0:64]!\n"
      "vld1.8 {d2}, [r1:64]!\n"
      "vaddw.u8 q4, q4, d0\n"
      "vaddw.u8 q5, q5, d1\n"
      "vaddw.u8 q6, q6, d2\n"
      "vst1.8 {d0, d1, d2}, [%[destination]:64]!\n"

      "bne 1b\n"

      // Leftover Load Aggregate Store.
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vmov.i8 d2, #0\n"
      "vld1.8 {d0[0]}, [%[source]]!\n"
      "vld1.8 {d1[0]}, [r0]!\n"
      "vld1.8 {d2[0]}, [r1]!\n"
      "vaddw.u8 q4, q4, d0\n"
      "vaddw.u8 q5, q5, d1\n"
      "vaddw.u8 q6, q6, d2\n"
      "vst1.8 {d0, d1, d2}, [%[destination]:64]!\n"

      // Aggregator Reduction.
      "vmov.32 d3[0], %[multiplicative_offset]\n"
      "vdup.32 q2, %[additive_offset]\n"
      "vpaddl.u16 q4, q4\n"
      "vpaddl.u16 q5, q5\n"
      "vpaddl.u16 q6, q6\n"
      "vpadd.u32 d8, d8, d9\n"
      "vpadd.u32 d10, d10, d11\n"
      "vpadd.u32 d12, d12, d13\n"
      "vpadd.u32 d6, d8, d10\n"
      "vpadd.u32 d7, d12, d12\n"
      "vmul.i32 q3, q3, d3[0]\n"
      "vadd.i32 q3, q3, q2\n"
      "vst1.32 {d6, d7}, [%[destination]:64]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "r0", "r1", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9",
        "d10", "d11", "d12", "d13", "cc", "memory");
}

void zip_3x8_2_aligned(const std::uint8_t* source, std::int32_t count,
                       std::int32_t stride, std::uint8_t* destination,
                       std::int32_t multiplicative_offset,
                       std::int32_t additive_offset) {
  asm volatile(
      "add r0, %[source], %[stride]\n"
      "add r1, r0, %[stride]\n"
      "sub %[count], %[count], #2\n"
      "vmov.i16 q4, #0\n"
      "vmov.i16 q5, #0\n"
      "vmov.i16 q6, #0\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store.
      "vld1.8 {d0}, [%[source]:64]!\n"
      "vld1.8 {d1}, [r0:64]!\n"
      "vld1.8 {d2}, [r1:64]!\n"
      "vaddw.u8 q4, q4, d0\n"
      "vaddw.u8 q5, q5, d1\n"
      "vaddw.u8 q6, q6, d2\n"
      "vst1.8 {d0, d1, d2}, [%[destination]:64]!\n"

      "bne 1b\n"

      // Leftover Load Aggregate Store.
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vmov.i8 d2, #0\n"
      "vld1.16 {d0[0]}, [%[source]]!\n"
      "vld1.16 {d1[0]}, [r0]!\n"
      "vld1.16 {d2[0]}, [r1]!\n"
      "vaddw.u8 q4, q4, d0\n"
      "vaddw.u8 q5, q5, d1\n"
      "vaddw.u8 q6, q6, d2\n"
      "vst1.8 {d0, d1, d2}, [%[destination]:64]!\n"

      // Aggregator Reduction.
      "vmov.32 d3[0], %[multiplicative_offset]\n"
      "vdup.32 q2, %[additive_offset]\n"
      "vpaddl.u16 q4, q4\n"
      "vpaddl.u16 q5, q5\n"
      "vpaddl.u16 q6, q6\n"
      "vpadd.u32 d8, d8, d9\n"
      "vpadd.u32 d10, d10, d11\n"
      "vpadd.u32 d12, d12, d13\n"
      "vpadd.u32 d6, d8, d10\n"
      "vpadd.u32 d7, d12, d12\n"
      "vmul.i32 q3, q3, d3[0]\n"
      "vadd.i32 q3, q3, q2\n"
      "vst1.32 {d6, d7}, [%[destination]:64]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "r0", "r1", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9",
        "d10", "d11", "d12", "d13", "cc", "memory");
}

void zip_3x8_3_aligned(const std::uint8_t* source, std::int32_t count,
                       std::int32_t stride, std::uint8_t* destination,
                       std::int32_t multiplicative_offset,
                       std::int32_t additive_offset) {
  asm volatile(
      "add r0, %[source], %[stride]\n"
      "add r1, r0, %[stride]\n"
      "sub %[count], %[count], #3\n"
      "vmov.i16 q4, #0\n"
      "vmov.i16 q5, #0\n"
      "vmov.i16 q6, #0\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store.
      "vld1.8 {d0}, [%[source]:64]!\n"
      "vld1.8 {d1}, [r0:64]!\n"
      "vld1.8 {d2}, [r1:64]!\n"
      "vaddw.u8 q4, q4, d0\n"
      "vaddw.u8 q5, q5, d1\n"
      "vaddw.u8 q6, q6, d2\n"
      "vst1.8 {d0, d1, d2}, [%[destination]:64]!\n"

      "bne 1b\n"

      // Leftover Load Aggregate Store.
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vmov.i8 d2, #0\n"
      "vld1.16 {d0[0]}, [%[source]]!\n"
      "vld1.8 {d0[2]}, [%[source]]!\n"
      "vld1.16 {d1[0]}, [r0]!\n"
      "vld1.8 {d1[2]}, [r0]!\n"
      "vld1.16 {d2[0]}, [r1]!\n"
      "vld1.8 {d2[2]}, [r1]!\n"
      "vaddw.u8 q4, q4, d0\n"
      "vaddw.u8 q5, q5, d1\n"
      "vaddw.u8 q6, q6, d2\n"
      "vst1.8 {d0, d1, d2}, [%[destination]:64]!\n"

      // Aggregator Reduction.
      "vmov.32 d3[0], %[multiplicative_offset]\n"
      "vdup.32 q2, %[additive_offset]\n"
      "vpaddl.u16 q4, q4\n"
      "vpaddl.u16 q5, q5\n"
      "vpaddl.u16 q6, q6\n"
      "vpadd.u32 d8, d8, d9\n"
      "vpadd.u32 d10, d10, d11\n"
      "vpadd.u32 d12, d12, d13\n"
      "vpadd.u32 d6, d8, d10\n"
      "vpadd.u32 d7, d12, d12\n"
      "vmul.i32 q3, q3, d3[0]\n"
      "vadd.i32 q3, q3, q2\n"
      "vst1.32 {d6, d7}, [%[destination]:64]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "r0", "r1", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9",
        "d10", "d11", "d12", "d13", "cc", "memory");
}

void zip_3x8_4_aligned(const std::uint8_t* source, std::int32_t count,
                       std::int32_t stride, std::uint8_t* destination,
                       std::int32_t multiplicative_offset,
                       std::int32_t additive_offset) {
  asm volatile(
      "add r0, %[source], %[stride]\n"
      "add r1, r0, %[stride]\n"
      "sub %[count], %[count], #4\n"
      "vmov.i16 q4, #0\n"
      "vmov.i16 q5, #0\n"
      "vmov.i16 q6, #0\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store.
      "vld1.8 {d0}, [%[source]:64]!\n"
      "vld1.8 {d1}, [r0:64]!\n"
      "vld1.8 {d2}, [r1:64]!\n"
      "vaddw.u8 q4, q4, d0\n"
      "vaddw.u8 q5, q5, d1\n"
      "vaddw.u8 q6, q6, d2\n"
      "vst1.8 {d0, d1, d2}, [%[destination]:64]!\n"

      "bne 1b\n"

      // Leftover Load Aggregate Store.
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vmov.i8 d2, #0\n"
      "vld1.32 {d0[0]}, [%[source]]!\n"
      "vld1.32 {d1[0]}, [r0]!\n"
      "vld1.32 {d2[0]}, [r1]!\n"
      "vaddw.u8 q4, q4, d0\n"
      "vaddw.u8 q5, q5, d1\n"
      "vaddw.u8 q6, q6, d2\n"
      "vst1.8 {d0, d1, d2}, [%[destination]:64]!\n"

      // Aggregator Reduction.
      "vmov.32 d3[0], %[multiplicative_offset]\n"
      "vdup.32 q2, %[additive_offset]\n"
      "vpaddl.u16 q4, q4\n"
      "vpaddl.u16 q5, q5\n"
      "vpaddl.u16 q6, q6\n"
      "vpadd.u32 d8, d8, d9\n"
      "vpadd.u32 d10, d10, d11\n"
      "vpadd.u32 d12, d12, d13\n"
      "vpadd.u32 d6, d8, d10\n"
      "vpadd.u32 d7, d12, d12\n"
      "vmul.i32 q3, q3, d3[0]\n"
      "vadd.i32 q3, q3, q2\n"
      "vst1.32 {d6, d7}, [%[destination]:64]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "r0", "r1", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9",
        "d10", "d11", "d12", "d13", "cc", "memory");
}

void zip_3x8_5_aligned(const std::uint8_t* source, std::int32_t count,
                       std::int32_t stride, std::uint8_t* destination,
                       std::int32_t multiplicative_offset,
                       std::int32_t additive_offset) {
  asm volatile(
      "add r0, %[source], %[stride]\n"
      "add r1, r0, %[stride]\n"
      "sub %[count], %[count], #5\n"
      "vmov.i16 q4, #0\n"
      "vmov.i16 q5, #0\n"
      "vmov.i16 q6, #0\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store.
      "vld1.8 {d0}, [%[source]:64]!\n"
      "vld1.8 {d1}, [r0:64]!\n"
      "vld1.8 {d2}, [r1:64]!\n"
      "vaddw.u8 q4, q4, d0\n"
      "vaddw.u8 q5, q5, d1\n"
      "vaddw.u8 q6, q6, d2\n"
      "vst1.8 {d0, d1, d2}, [%[destination]:64]!\n"

      "bne 1b\n"

      // Leftover Load Aggregate Store.
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vmov.i8 d2, #0\n"
      "vld1.32 {d0[0]}, [%[source]]!\n"
      "vld1.8 {d0[4]}, [%[source]]!\n"
      "vld1.32 {d1[0]}, [r0]!\n"
      "vld1.8 {d1[4]}, [r0]!\n"
      "vld1.32 {d2[0]}, [r1]!\n"
      "vld1.8 {d2[4]}, [r1]!\n"
      "vaddw.u8 q4, q4, d0\n"
      "vaddw.u8 q5, q5, d1\n"
      "vaddw.u8 q6, q6, d2\n"
      "vst1.8 {d0, d1, d2}, [%[destination]:64]!\n"

      // Aggregator Reduction.
      "vmov.32 d3[0], %[multiplicative_offset]\n"
      "vdup.32 q2, %[additive_offset]\n"
      "vpaddl.u16 q4, q4\n"
      "vpaddl.u16 q5, q5\n"
      "vpaddl.u16 q6, q6\n"
      "vpadd.u32 d8, d8, d9\n"
      "vpadd.u32 d10, d10, d11\n"
      "vpadd.u32 d12, d12, d13\n"
      "vpadd.u32 d6, d8, d10\n"
      "vpadd.u32 d7, d12, d12\n"
      "vmul.i32 q3, q3, d3[0]\n"
      "vadd.i32 q3, q3, q2\n"
      "vst1.32 {d6, d7}, [%[destination]:64]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "r0", "r1", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9",
        "d10", "d11", "d12", "d13", "cc", "memory");
}

void zip_3x8_6_aligned(const std::uint8_t* source, std::int32_t count,
                       std::int32_t stride, std::uint8_t* destination,
                       std::int32_t multiplicative_offset,
                       std::int32_t additive_offset) {
  asm volatile(
      "add r0, %[source], %[stride]\n"
      "add r1, r0, %[stride]\n"
      "sub %[count], %[count], #6\n"
      "vmov.i16 q4, #0\n"
      "vmov.i16 q5, #0\n"
      "vmov.i16 q6, #0\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store.
      "vld1.8 {d0}, [%[source]:64]!\n"
      "vld1.8 {d1}, [r0:64]!\n"
      "vld1.8 {d2}, [r1:64]!\n"
      "vaddw.u8 q4, q4, d0\n"
      "vaddw.u8 q5, q5, d1\n"
      "vaddw.u8 q6, q6, d2\n"
      "vst1.8 {d0, d1, d2}, [%[destination]:64]!\n"

      "bne 1b\n"

      // Leftover Load Aggregate Store.
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vmov.i8 d2, #0\n"
      "vld1.32 {d0[0]}, [%[source]]!\n"
      "vld1.16 {d0[2]}, [%[source]]!\n"
      "vld1.32 {d1[0]}, [r0]!\n"
      "vld1.16 {d1[2]}, [r0]!\n"
      "vld1.32 {d2[0]}, [r1]!\n"
      "vld1.16 {d2[2]}, [r1]!\n"
      "vaddw.u8 q4, q4, d0\n"
      "vaddw.u8 q5, q5, d1\n"
      "vaddw.u8 q6, q6, d2\n"
      "vst1.8 {d0, d1, d2}, [%[destination]:64]!\n"

      // Aggregator Reduction.
      "vmov.32 d3[0], %[multiplicative_offset]\n"
      "vdup.32 q2, %[additive_offset]\n"
      "vpaddl.u16 q4, q4\n"
      "vpaddl.u16 q5, q5\n"
      "vpaddl.u16 q6, q6\n"
      "vpadd.u32 d8, d8, d9\n"
      "vpadd.u32 d10, d10, d11\n"
      "vpadd.u32 d12, d12, d13\n"
      "vpadd.u32 d6, d8, d10\n"
      "vpadd.u32 d7, d12, d12\n"
      "vmul.i32 q3, q3, d3[0]\n"
      "vadd.i32 q3, q3, q2\n"
      "vst1.32 {d6, d7}, [%[destination]:64]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "r0", "r1", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9",
        "d10", "d11", "d12", "d13", "cc", "memory");
}

void zip_3x8_7_aligned(const std::uint8_t* source, std::int32_t count,
                       std::int32_t stride, std::uint8_t* destination,
                       std::int32_t multiplicative_offset,
                       std::int32_t additive_offset) {
  asm volatile(
      "add r0, %[source], %[stride]\n"
      "add r1, r0, %[stride]\n"
      "sub %[count], %[count], #7\n"
      "vmov.i16 q4, #0\n"
      "vmov.i16 q5, #0\n"
      "vmov.i16 q6, #0\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store.
      "vld1.8 {d0}, [%[source]:64]!\n"
      "vld1.8 {d1}, [r0:64]!\n"
      "vld1.8 {d2}, [r1:64]!\n"
      "vaddw.u8 q4, q4, d0\n"
      "vaddw.u8 q5, q5, d1\n"
      "vaddw.u8 q6, q6, d2\n"
      "vst1.8 {d0, d1, d2}, [%[destination]:64]!\n"

      "bne 1b\n"

      // Leftover Load Aggregate Store.
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vmov.i8 d2, #0\n"
      "vld1.32 {d0[0]}, [%[source]]!\n"
      "vld1.16 {d0[2]}, [%[source]]!\n"
      "vld1.8 {d0[6]}, [%[source]]!\n"
      "vld1.32 {d1[0]}, [r0]!\n"
      "vld1.16 {d1[2]}, [r0]!\n"
      "vld1.8 {d1[6]}, [r0]!\n"
      "vld1.32 {d2[0]}, [r1]!\n"
      "vld1.16 {d2[2]}, [r1]!\n"
      "vld1.8 {d2[6]}, [r1]!\n"
      "vaddw.u8 q4, q4, d0\n"
      "vaddw.u8 q5, q5, d1\n"
      "vaddw.u8 q6, q6, d2\n"
      "vst1.8 {d0, d1, d2}, [%[destination]:64]!\n"

      // Aggregator Reduction.
      "vmov.32 d3[0], %[multiplicative_offset]\n"
      "vdup.32 q2, %[additive_offset]\n"
      "vpaddl.u16 q4, q4\n"
      "vpaddl.u16 q5, q5\n"
      "vpaddl.u16 q6, q6\n"
      "vpadd.u32 d8, d8, d9\n"
      "vpadd.u32 d10, d10, d11\n"
      "vpadd.u32 d12, d12, d13\n"
      "vpadd.u32 d6, d8, d10\n"
      "vpadd.u32 d7, d12, d12\n"
      "vmul.i32 q3, q3, d3[0]\n"
      "vadd.i32 q3, q3, q2\n"
      "vst1.32 {d6, d7}, [%[destination]:64]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "r0", "r1", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9",
        "d10", "d11", "d12", "d13", "cc", "memory");
}

void zip_4x8_aligned(const std::uint8_t* source, std::int32_t count,
                     std::int32_t stride, std::uint8_t* destination,
                     std::int32_t multiplicative_offset,
                     std::int32_t additive_offset) {
  asm volatile(
      "add r0, %[source], %[stride]\n"
      "add r1, r0, %[stride]\n"
      "add r2, r1, %[stride]\n"
      "vmov.i16 q4, #0\n"
      "vmov.i16 q5, #0\n"
      "vmov.i16 q6, #0\n"
      "vmov.i16 q7, #0\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store.
      "vld1.8 {d0}, [%[source]:64]!\n"
      "vld1.8 {d1}, [r0:64]!\n"
      "vld1.8 {d2}, [r1:64]!\n"
      "vld1.8 {d3}, [r2:64]!\n"
      "vaddw.u8 q4, q4, d0\n"
      "vaddw.u8 q5, q5, d1\n"
      "vaddw.u8 q6, q6, d2\n"
      "vaddw.u8 q7, q7, d3\n"
      "vst1.8 {d0, d1, d2, d3}, [%[destination]:64]!\n"

      "bne 1b\n"

      // Aggregator Reduction.
      "vmov.32 d4[0], %[multiplicative_offset]\n"
      "vdup.32 q3, %[additive_offset]\n"
      "vpaddl.u16 q4, q4\n"
      "vpaddl.u16 q5, q5\n"
      "vpaddl.u16 q6, q6\n"
      "vpaddl.u16 q7, q7\n"
      "vpadd.u32 d8, d8, d9\n"
      "vpadd.u32 d10, d10, d11\n"
      "vpadd.u32 d12, d12, d13\n"
      "vpadd.u32 d14, d14, d15\n"
      "vpadd.u32 d16, d8, d10\n"
      "vpadd.u32 d17, d12, d14\n"
      "vmul.i32 q8, q8, d4[0]\n"
      "vadd.i32 q8, q8, q3\n"
      "vst1.32 {d16, d17}, [%[destination]:64]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "r0", "r1", "r2", "d0", "d1", "d2", "d3", "d4", "d6", "d7", "d8", "d9",
        "d10", "d11", "d12", "d13", "d14", "d15", "d16", "d17", "cc", "memory");
}

void zip_4x8_1_aligned(const std::uint8_t* source, std::int32_t count,
                       std::int32_t stride, std::uint8_t* destination,
                       std::int32_t multiplicative_offset,
                       std::int32_t additive_offset) {
  asm volatile(
      "add r0, %[source], %[stride]\n"
      "add r1, r0, %[stride]\n"
      "add r2, r1, %[stride]\n"
      "sub %[count], %[count], #1\n"
      "vmov.i16 q4, #0\n"
      "vmov.i16 q5, #0\n"
      "vmov.i16 q6, #0\n"
      "vmov.i16 q7, #0\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store.
      "vld1.8 {d0}, [%[source]:64]!\n"
      "vld1.8 {d1}, [r0:64]!\n"
      "vld1.8 {d2}, [r1:64]!\n"
      "vld1.8 {d3}, [r2:64]!\n"
      "vaddw.u8 q4, q4, d0\n"
      "vaddw.u8 q5, q5, d1\n"
      "vaddw.u8 q6, q6, d2\n"
      "vaddw.u8 q7, q7, d3\n"
      "vst1.8 {d0, d1, d2, d3}, [%[destination]:64]!\n"

      "bne 1b\n"

      // Leftover Load Aggregate Store.
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vmov.i8 d2, #0\n"
      "vmov.i8 d3, #0\n"
      "vld1.8 {d0[0]}, [%[source]]!\n"
      "vld1.8 {d1[0]}, [r0]!\n"
      "vld1.8 {d2[0]}, [r1]!\n"
      "vld1.8 {d3[0]}, [r2]!\n"
      "vaddw.u8 q4, q4, d0\n"
      "vaddw.u8 q5, q5, d1\n"
      "vaddw.u8 q6, q6, d2\n"
      "vaddw.u8 q7, q7, d3\n"
      "vst1.8 {d0, d1, d2, d3}, [%[destination]:64]!\n"

      // Aggregator Reduction.
      "vmov.32 d4[0], %[multiplicative_offset]\n"
      "vdup.32 q3, %[additive_offset]\n"
      "vpaddl.u16 q4, q4\n"
      "vpaddl.u16 q5, q5\n"
      "vpaddl.u16 q6, q6\n"
      "vpaddl.u16 q7, q7\n"
      "vpadd.u32 d8, d8, d9\n"
      "vpadd.u32 d10, d10, d11\n"
      "vpadd.u32 d12, d12, d13\n"
      "vpadd.u32 d14, d14, d15\n"
      "vpadd.u32 d16, d8, d10\n"
      "vpadd.u32 d17, d12, d14\n"
      "vmul.i32 q8, q8, d4[0]\n"
      "vadd.i32 q8, q8, q3\n"
      "vst1.32 {d16, d17}, [%[destination]:64]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "r0", "r1", "r2", "d0", "d1", "d2", "d3", "d4", "d6", "d7", "d8", "d9",
        "d10", "d11", "d12", "d13", "d14", "d15", "d16", "d17", "cc", "memory");
}

void zip_4x8_2_aligned(const std::uint8_t* source, std::int32_t count,
                       std::int32_t stride, std::uint8_t* destination,
                       std::int32_t multiplicative_offset,
                       std::int32_t additive_offset) {
  asm volatile(
      "add r0, %[source], %[stride]\n"
      "add r1, r0, %[stride]\n"
      "add r2, r1, %[stride]\n"
      "sub %[count], %[count], #2\n"
      "vmov.i16 q4, #0\n"
      "vmov.i16 q5, #0\n"
      "vmov.i16 q6, #0\n"
      "vmov.i16 q7, #0\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store.
      "vld1.8 {d0}, [%[source]:64]!\n"
      "vld1.8 {d1}, [r0:64]!\n"
      "vld1.8 {d2}, [r1:64]!\n"
      "vld1.8 {d3}, [r2:64]!\n"
      "vaddw.u8 q4, q4, d0\n"
      "vaddw.u8 q5, q5, d1\n"
      "vaddw.u8 q6, q6, d2\n"
      "vaddw.u8 q7, q7, d3\n"
      "vst1.8 {d0, d1, d2, d3}, [%[destination]:64]!\n"

      "bne 1b\n"

      // Leftover Load Aggregate Store.
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vmov.i8 d2, #0\n"
      "vmov.i8 d3, #0\n"
      "vld1.16 {d0[0]}, [%[source]]!\n"
      "vld1.16 {d1[0]}, [r0]!\n"
      "vld1.16 {d2[0]}, [r1]!\n"
      "vld1.16 {d3[0]}, [r2]!\n"
      "vaddw.u8 q4, q4, d0\n"
      "vaddw.u8 q5, q5, d1\n"
      "vaddw.u8 q6, q6, d2\n"
      "vaddw.u8 q7, q7, d3\n"
      "vst1.8 {d0, d1, d2, d3}, [%[destination]:64]!\n"

      // Aggregator Reduction.
      "vmov.32 d4[0], %[multiplicative_offset]\n"
      "vdup.32 q3, %[additive_offset]\n"
      "vpaddl.u16 q4, q4\n"
      "vpaddl.u16 q5, q5\n"
      "vpaddl.u16 q6, q6\n"
      "vpaddl.u16 q7, q7\n"
      "vpadd.u32 d8, d8, d9\n"
      "vpadd.u32 d10, d10, d11\n"
      "vpadd.u32 d12, d12, d13\n"
      "vpadd.u32 d14, d14, d15\n"
      "vpadd.u32 d16, d8, d10\n"
      "vpadd.u32 d17, d12, d14\n"
      "vmul.i32 q8, q8, d4[0]\n"
      "vadd.i32 q8, q8, q3\n"
      "vst1.32 {d16, d17}, [%[destination]:64]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "r0", "r1", "r2", "d0", "d1", "d2", "d3", "d4", "d6", "d7", "d8", "d9",
        "d10", "d11", "d12", "d13", "d14", "d15", "d16", "d17", "cc", "memory");
}

void zip_4x8_3_aligned(const std::uint8_t* source, std::int32_t count,
                       std::int32_t stride, std::uint8_t* destination,
                       std::int32_t multiplicative_offset,
                       std::int32_t additive_offset) {
  asm volatile(
      "add r0, %[source], %[stride]\n"
      "add r1, r0, %[stride]\n"
      "add r2, r1, %[stride]\n"
      "sub %[count], %[count], #3\n"
      "vmov.i16 q4, #0\n"
      "vmov.i16 q5, #0\n"
      "vmov.i16 q6, #0\n"
      "vmov.i16 q7, #0\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store.
      "vld1.8 {d0}, [%[source]:64]!\n"
      "vld1.8 {d1}, [r0:64]!\n"
      "vld1.8 {d2}, [r1:64]!\n"
      "vld1.8 {d3}, [r2:64]!\n"
      "vaddw.u8 q4, q4, d0\n"
      "vaddw.u8 q5, q5, d1\n"
      "vaddw.u8 q6, q6, d2\n"
      "vaddw.u8 q7, q7, d3\n"
      "vst1.8 {d0, d1, d2, d3}, [%[destination]:64]!\n"

      "bne 1b\n"

      // Leftover Load Aggregate Store.
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vmov.i8 d2, #0\n"
      "vmov.i8 d3, #0\n"
      "vld1.16 {d0[0]}, [%[source]]!\n"
      "vld1.8 {d0[2]}, [%[source]]!\n"
      "vld1.16 {d1[0]}, [r0]!\n"
      "vld1.8 {d1[2]}, [r0]!\n"
      "vld1.16 {d2[0]}, [r1]!\n"
      "vld1.8 {d2[2]}, [r1]!\n"
      "vld1.16 {d3[0]}, [r2]!\n"
      "vld1.8 {d3[2]}, [r2]!\n"
      "vaddw.u8 q4, q4, d0\n"
      "vaddw.u8 q5, q5, d1\n"
      "vaddw.u8 q6, q6, d2\n"
      "vaddw.u8 q7, q7, d3\n"
      "vst1.8 {d0, d1, d2, d3}, [%[destination]:64]!\n"

      // Aggregator Reduction.
      "vmov.32 d4[0], %[multiplicative_offset]\n"
      "vdup.32 q3, %[additive_offset]\n"
      "vpaddl.u16 q4, q4\n"
      "vpaddl.u16 q5, q5\n"
      "vpaddl.u16 q6, q6\n"
      "vpaddl.u16 q7, q7\n"
      "vpadd.u32 d8, d8, d9\n"
      "vpadd.u32 d10, d10, d11\n"
      "vpadd.u32 d12, d12, d13\n"
      "vpadd.u32 d14, d14, d15\n"
      "vpadd.u32 d16, d8, d10\n"
      "vpadd.u32 d17, d12, d14\n"
      "vmul.i32 q8, q8, d4[0]\n"
      "vadd.i32 q8, q8, q3\n"
      "vst1.32 {d16, d17}, [%[destination]:64]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "r0", "r1", "r2", "d0", "d1", "d2", "d3", "d4", "d6", "d7", "d8", "d9",
        "d10", "d11", "d12", "d13", "d14", "d15", "d16", "d17", "cc", "memory");
}

void zip_4x8_4_aligned(const std::uint8_t* source, std::int32_t count,
                       std::int32_t stride, std::uint8_t* destination,
                       std::int32_t multiplicative_offset,
                       std::int32_t additive_offset) {
  asm volatile(
      "add r0, %[source], %[stride]\n"
      "add r1, r0, %[stride]\n"
      "add r2, r1, %[stride]\n"
      "sub %[count], %[count], #4\n"
      "vmov.i16 q4, #0\n"
      "vmov.i16 q5, #0\n"
      "vmov.i16 q6, #0\n"
      "vmov.i16 q7, #0\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store.
      "vld1.8 {d0}, [%[source]:64]!\n"
      "vld1.8 {d1}, [r0:64]!\n"
      "vld1.8 {d2}, [r1:64]!\n"
      "vld1.8 {d3}, [r2:64]!\n"
      "vaddw.u8 q4, q4, d0\n"
      "vaddw.u8 q5, q5, d1\n"
      "vaddw.u8 q6, q6, d2\n"
      "vaddw.u8 q7, q7, d3\n"
      "vst1.8 {d0, d1, d2, d3}, [%[destination]:64]!\n"

      "bne 1b\n"

      // Leftover Load Aggregate Store.
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vmov.i8 d2, #0\n"
      "vmov.i8 d3, #0\n"
      "vld1.32 {d0[0]}, [%[source]]!\n"
      "vld1.32 {d1[0]}, [r0]!\n"
      "vld1.32 {d2[0]}, [r1]!\n"
      "vld1.32 {d3[0]}, [r2]!\n"
      "vaddw.u8 q4, q4, d0\n"
      "vaddw.u8 q5, q5, d1\n"
      "vaddw.u8 q6, q6, d2\n"
      "vaddw.u8 q7, q7, d3\n"
      "vst1.8 {d0, d1, d2, d3}, [%[destination]:64]!\n"

      // Aggregator Reduction.
      "vmov.32 d4[0], %[multiplicative_offset]\n"
      "vdup.32 q3, %[additive_offset]\n"
      "vpaddl.u16 q4, q4\n"
      "vpaddl.u16 q5, q5\n"
      "vpaddl.u16 q6, q6\n"
      "vpaddl.u16 q7, q7\n"
      "vpadd.u32 d8, d8, d9\n"
      "vpadd.u32 d10, d10, d11\n"
      "vpadd.u32 d12, d12, d13\n"
      "vpadd.u32 d14, d14, d15\n"
      "vpadd.u32 d16, d8, d10\n"
      "vpadd.u32 d17, d12, d14\n"
      "vmul.i32 q8, q8, d4[0]\n"
      "vadd.i32 q8, q8, q3\n"
      "vst1.32 {d16, d17}, [%[destination]:64]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "r0", "r1", "r2", "d0", "d1", "d2", "d3", "d4", "d6", "d7", "d8", "d9",
        "d10", "d11", "d12", "d13", "d14", "d15", "d16", "d17", "cc", "memory");
}

void zip_4x8_5_aligned(const std::uint8_t* source, std::int32_t count,
                       std::int32_t stride, std::uint8_t* destination,
                       std::int32_t multiplicative_offset,
                       std::int32_t additive_offset) {
  asm volatile(
      "add r0, %[source], %[stride]\n"
      "add r1, r0, %[stride]\n"
      "add r2, r1, %[stride]\n"
      "sub %[count], %[count], #5\n"
      "vmov.i16 q4, #0\n"
      "vmov.i16 q5, #0\n"
      "vmov.i16 q6, #0\n"
      "vmov.i16 q7, #0\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store.
      "vld1.8 {d0}, [%[source]:64]!\n"
      "vld1.8 {d1}, [r0:64]!\n"
      "vld1.8 {d2}, [r1:64]!\n"
      "vld1.8 {d3}, [r2:64]!\n"
      "vaddw.u8 q4, q4, d0\n"
      "vaddw.u8 q5, q5, d1\n"
      "vaddw.u8 q6, q6, d2\n"
      "vaddw.u8 q7, q7, d3\n"
      "vst1.8 {d0, d1, d2, d3}, [%[destination]:64]!\n"

      "bne 1b\n"

      // Leftover Load Aggregate Store.
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vmov.i8 d2, #0\n"
      "vmov.i8 d3, #0\n"
      "vld1.32 {d0[0]}, [%[source]]!\n"
      "vld1.8 {d0[4]}, [%[source]]!\n"
      "vld1.32 {d1[0]}, [r0]!\n"
      "vld1.8 {d1[4]}, [r0]!\n"
      "vld1.32 {d2[0]}, [r1]!\n"
      "vld1.8 {d2[4]}, [r1]!\n"
      "vld1.32 {d3[0]}, [r2]!\n"
      "vld1.8 {d3[4]}, [r2]!\n"
      "vaddw.u8 q4, q4, d0\n"
      "vaddw.u8 q5, q5, d1\n"
      "vaddw.u8 q6, q6, d2\n"
      "vaddw.u8 q7, q7, d3\n"
      "vst1.8 {d0, d1, d2, d3}, [%[destination]:64]!\n"

      // Aggregator Reduction.
      "vmov.32 d4[0], %[multiplicative_offset]\n"
      "vdup.32 q3, %[additive_offset]\n"
      "vpaddl.u16 q4, q4\n"
      "vpaddl.u16 q5, q5\n"
      "vpaddl.u16 q6, q6\n"
      "vpaddl.u16 q7, q7\n"
      "vpadd.u32 d8, d8, d9\n"
      "vpadd.u32 d10, d10, d11\n"
      "vpadd.u32 d12, d12, d13\n"
      "vpadd.u32 d14, d14, d15\n"
      "vpadd.u32 d16, d8, d10\n"
      "vpadd.u32 d17, d12, d14\n"
      "vmul.i32 q8, q8, d4[0]\n"
      "vadd.i32 q8, q8, q3\n"
      "vst1.32 {d16, d17}, [%[destination]:64]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "r0", "r1", "r2", "d0", "d1", "d2", "d3", "d4", "d6", "d7", "d8", "d9",
        "d10", "d11", "d12", "d13", "d14", "d15", "d16", "d17", "cc", "memory");
}

void zip_4x8_6_aligned(const std::uint8_t* source, std::int32_t count,
                       std::int32_t stride, std::uint8_t* destination,
                       std::int32_t multiplicative_offset,
                       std::int32_t additive_offset) {
  asm volatile(
      "add r0, %[source], %[stride]\n"
      "add r1, r0, %[stride]\n"
      "add r2, r1, %[stride]\n"
      "sub %[count], %[count], #6\n"
      "vmov.i16 q4, #0\n"
      "vmov.i16 q5, #0\n"
      "vmov.i16 q6, #0\n"
      "vmov.i16 q7, #0\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store.
      "vld1.8 {d0}, [%[source]:64]!\n"
      "vld1.8 {d1}, [r0:64]!\n"
      "vld1.8 {d2}, [r1:64]!\n"
      "vld1.8 {d3}, [r2:64]!\n"
      "vaddw.u8 q4, q4, d0\n"
      "vaddw.u8 q5, q5, d1\n"
      "vaddw.u8 q6, q6, d2\n"
      "vaddw.u8 q7, q7, d3\n"
      "vst1.8 {d0, d1, d2, d3}, [%[destination]:64]!\n"

      "bne 1b\n"

      // Leftover Load Aggregate Store.
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vmov.i8 d2, #0\n"
      "vmov.i8 d3, #0\n"
      "vld1.32 {d0[0]}, [%[source]]!\n"
      "vld1.16 {d0[2]}, [%[source]]!\n"
      "vld1.32 {d1[0]}, [r0]!\n"
      "vld1.16 {d1[2]}, [r0]!\n"
      "vld1.32 {d2[0]}, [r1]!\n"
      "vld1.16 {d2[2]}, [r1]!\n"
      "vld1.32 {d3[0]}, [r2]!\n"
      "vld1.16 {d3[2]}, [r2]!\n"
      "vaddw.u8 q4, q4, d0\n"
      "vaddw.u8 q5, q5, d1\n"
      "vaddw.u8 q6, q6, d2\n"
      "vaddw.u8 q7, q7, d3\n"
      "vst1.8 {d0, d1, d2, d3}, [%[destination]:64]!\n"

      // Aggregator Reduction.
      "vmov.32 d4[0], %[multiplicative_offset]\n"
      "vdup.32 q3, %[additive_offset]\n"
      "vpaddl.u16 q4, q4\n"
      "vpaddl.u16 q5, q5\n"
      "vpaddl.u16 q6, q6\n"
      "vpaddl.u16 q7, q7\n"
      "vpadd.u32 d8, d8, d9\n"
      "vpadd.u32 d10, d10, d11\n"
      "vpadd.u32 d12, d12, d13\n"
      "vpadd.u32 d14, d14, d15\n"
      "vpadd.u32 d16, d8, d10\n"
      "vpadd.u32 d17, d12, d14\n"
      "vmul.i32 q8, q8, d4[0]\n"
      "vadd.i32 q8, q8, q3\n"
      "vst1.32 {d16, d17}, [%[destination]:64]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "r0", "r1", "r2", "d0", "d1", "d2", "d3", "d4", "d6", "d7", "d8", "d9",
        "d10", "d11", "d12", "d13", "d14", "d15", "d16", "d17", "cc", "memory");
}

void zip_4x8_7_aligned(const std::uint8_t* source, std::int32_t count,
                       std::int32_t stride, std::uint8_t* destination,
                       std::int32_t multiplicative_offset,
                       std::int32_t additive_offset) {
  asm volatile(
      "add r0, %[source], %[stride]\n"
      "add r1, r0, %[stride]\n"
      "add r2, r1, %[stride]\n"
      "sub %[count], %[count], #7\n"
      "vmov.i16 q4, #0\n"
      "vmov.i16 q5, #0\n"
      "vmov.i16 q6, #0\n"
      "vmov.i16 q7, #0\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store.
      "vld1.8 {d0}, [%[source]:64]!\n"
      "vld1.8 {d1}, [r0:64]!\n"
      "vld1.8 {d2}, [r1:64]!\n"
      "vld1.8 {d3}, [r2:64]!\n"
      "vaddw.u8 q4, q4, d0\n"
      "vaddw.u8 q5, q5, d1\n"
      "vaddw.u8 q6, q6, d2\n"
      "vaddw.u8 q7, q7, d3\n"
      "vst1.8 {d0, d1, d2, d3}, [%[destination]:64]!\n"

      "bne 1b\n"

      // Leftover Load Aggregate Store.
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vmov.i8 d2, #0\n"
      "vmov.i8 d3, #0\n"
      "vld1.32 {d0[0]}, [%[source]]!\n"
      "vld1.16 {d0[2]}, [%[source]]!\n"
      "vld1.8 {d0[6]}, [%[source]]!\n"
      "vld1.32 {d1[0]}, [r0]!\n"
      "vld1.16 {d1[2]}, [r0]!\n"
      "vld1.8 {d1[6]}, [r0]!\n"
      "vld1.32 {d2[0]}, [r1]!\n"
      "vld1.16 {d2[2]}, [r1]!\n"
      "vld1.8 {d2[6]}, [r1]!\n"
      "vld1.32 {d3[0]}, [r2]!\n"
      "vld1.16 {d3[2]}, [r2]!\n"
      "vld1.8 {d3[6]}, [r2]!\n"
      "vaddw.u8 q4, q4, d0\n"
      "vaddw.u8 q5, q5, d1\n"
      "vaddw.u8 q6, q6, d2\n"
      "vaddw.u8 q7, q7, d3\n"
      "vst1.8 {d0, d1, d2, d3}, [%[destination]:64]!\n"

      // Aggregator Reduction.
      "vmov.32 d4[0], %[multiplicative_offset]\n"
      "vdup.32 q3, %[additive_offset]\n"
      "vpaddl.u16 q4, q4\n"
      "vpaddl.u16 q5, q5\n"
      "vpaddl.u16 q6, q6\n"
      "vpaddl.u16 q7, q7\n"
      "vpadd.u32 d8, d8, d9\n"
      "vpadd.u32 d10, d10, d11\n"
      "vpadd.u32 d12, d12, d13\n"
      "vpadd.u32 d14, d14, d15\n"
      "vpadd.u32 d16, d8, d10\n"
      "vpadd.u32 d17, d12, d14\n"
      "vmul.i32 q8, q8, d4[0]\n"
      "vadd.i32 q8, q8, q3\n"
      "vst1.32 {d16, d17}, [%[destination]:64]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "r0", "r1", "r2", "d0", "d1", "d2", "d3", "d4", "d6", "d7", "d8", "d9",
        "d10", "d11", "d12", "d13", "d14", "d15", "d16", "d17", "cc", "memory");
}

void zip_1x8(const std::uint8_t* source, std::int32_t count,
             std::int32_t stride, std::uint8_t* destination,
             std::int32_t multiplicative_offset, std::int32_t additive_offset) {
  asm volatile(
      "vmov.i16 q4, #0\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store.
      "vld1.8 {d0}, [%[source]]!\n"
      "vaddw.u8 q4, q4, d0\n"
      "vst1.8 {d0}, [%[destination]:64]!\n"

      "bne 1b\n"

      // Aggregator Reduction.
      "vmov.32 d1[0], %[multiplicative_offset]\n"
      "vdup.32 d2, %[additive_offset]\n"
      "vpaddl.u16 q4, q4\n"
      "vpadd.u32 d8, d8, d9\n"
      "vpadd.u32 d3, d8, d8\n"
      "vmul.i32 d3, d3, d1[0]\n"
      "vadd.i32 d3, d3, d2\n"
      "vst1.32 {d3}, [%[destination]:64]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "d0", "d1", "d2", "d3", "d8", "d9", "cc", "memory");
}

void zip_1x8_1(const std::uint8_t* source, std::int32_t count,
               std::int32_t stride, std::uint8_t* destination,
               std::int32_t multiplicative_offset,
               std::int32_t additive_offset) {
  asm volatile(
      "sub %[count], %[count], #1\n"
      "vmov.i16 q4, #0\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store.
      "vld1.8 {d0}, [%[source]]!\n"
      "vaddw.u8 q4, q4, d0\n"
      "vst1.8 {d0}, [%[destination]:64]!\n"

      "bne 1b\n"

      // Leftover Load Aggregate Store.
      "vmov.i8 d0, #0\n"
      "vld1.8 {d0[0]}, [%[source]]!\n"
      "vaddw.u8 q4, q4, d0\n"
      "vst1.8 {d0}, [%[destination]:64]!\n"

      // Aggregator Reduction.
      "vmov.32 d1[0], %[multiplicative_offset]\n"
      "vdup.32 d2, %[additive_offset]\n"
      "vpaddl.u16 q4, q4\n"
      "vpadd.u32 d8, d8, d9\n"
      "vpadd.u32 d3, d8, d8\n"
      "vmul.i32 d3, d3, d1[0]\n"
      "vadd.i32 d3, d3, d2\n"
      "vst1.32 {d3}, [%[destination]:64]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "d0", "d1", "d2", "d3", "d8", "d9", "cc", "memory");
}

void zip_1x8_2(const std::uint8_t* source, std::int32_t count,
               std::int32_t stride, std::uint8_t* destination,
               std::int32_t multiplicative_offset,
               std::int32_t additive_offset) {
  asm volatile(
      "sub %[count], %[count], #2\n"
      "vmov.i16 q4, #0\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store.
      "vld1.8 {d0}, [%[source]]!\n"
      "vaddw.u8 q4, q4, d0\n"
      "vst1.8 {d0}, [%[destination]:64]!\n"

      "bne 1b\n"

      // Leftover Load Aggregate Store.
      "vmov.i8 d0, #0\n"
      "vld1.16 {d0[0]}, [%[source]]!\n"
      "vaddw.u8 q4, q4, d0\n"
      "vst1.8 {d0}, [%[destination]:64]!\n"

      // Aggregator Reduction.
      "vmov.32 d1[0], %[multiplicative_offset]\n"
      "vdup.32 d2, %[additive_offset]\n"
      "vpaddl.u16 q4, q4\n"
      "vpadd.u32 d8, d8, d9\n"
      "vpadd.u32 d3, d8, d8\n"
      "vmul.i32 d3, d3, d1[0]\n"
      "vadd.i32 d3, d3, d2\n"
      "vst1.32 {d3}, [%[destination]:64]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "d0", "d1", "d2", "d3", "d8", "d9", "cc", "memory");
}

void zip_1x8_3(const std::uint8_t* source, std::int32_t count,
               std::int32_t stride, std::uint8_t* destination,
               std::int32_t multiplicative_offset,
               std::int32_t additive_offset) {
  asm volatile(
      "sub %[count], %[count], #3\n"
      "vmov.i16 q4, #0\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store.
      "vld1.8 {d0}, [%[source]]!\n"
      "vaddw.u8 q4, q4, d0\n"
      "vst1.8 {d0}, [%[destination]:64]!\n"

      "bne 1b\n"

      // Leftover Load Aggregate Store.
      "vmov.i8 d0, #0\n"
      "vld1.16 {d0[0]}, [%[source]]!\n"
      "vld1.8 {d0[2]}, [%[source]]!\n"
      "vaddw.u8 q4, q4, d0\n"
      "vst1.8 {d0}, [%[destination]:64]!\n"

      // Aggregator Reduction.
      "vmov.32 d1[0], %[multiplicative_offset]\n"
      "vdup.32 d2, %[additive_offset]\n"
      "vpaddl.u16 q4, q4\n"
      "vpadd.u32 d8, d8, d9\n"
      "vpadd.u32 d3, d8, d8\n"
      "vmul.i32 d3, d3, d1[0]\n"
      "vadd.i32 d3, d3, d2\n"
      "vst1.32 {d3}, [%[destination]:64]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "d0", "d1", "d2", "d3", "d8", "d9", "cc", "memory");
}

void zip_1x8_4(const std::uint8_t* source, std::int32_t count,
               std::int32_t stride, std::uint8_t* destination,
               std::int32_t multiplicative_offset,
               std::int32_t additive_offset) {
  asm volatile(
      "sub %[count], %[count], #4\n"
      "vmov.i16 q4, #0\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store.
      "vld1.8 {d0}, [%[source]]!\n"
      "vaddw.u8 q4, q4, d0\n"
      "vst1.8 {d0}, [%[destination]:64]!\n"

      "bne 1b\n"

      // Leftover Load Aggregate Store.
      "vmov.i8 d0, #0\n"
      "vld1.32 {d0[0]}, [%[source]]!\n"
      "vaddw.u8 q4, q4, d0\n"
      "vst1.8 {d0}, [%[destination]:64]!\n"

      // Aggregator Reduction.
      "vmov.32 d1[0], %[multiplicative_offset]\n"
      "vdup.32 d2, %[additive_offset]\n"
      "vpaddl.u16 q4, q4\n"
      "vpadd.u32 d8, d8, d9\n"
      "vpadd.u32 d3, d8, d8\n"
      "vmul.i32 d3, d3, d1[0]\n"
      "vadd.i32 d3, d3, d2\n"
      "vst1.32 {d3}, [%[destination]:64]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "d0", "d1", "d2", "d3", "d8", "d9", "cc", "memory");
}

void zip_1x8_5(const std::uint8_t* source, std::int32_t count,
               std::int32_t stride, std::uint8_t* destination,
               std::int32_t multiplicative_offset,
               std::int32_t additive_offset) {
  asm volatile(
      "sub %[count], %[count], #5\n"
      "vmov.i16 q4, #0\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store.
      "vld1.8 {d0}, [%[source]]!\n"
      "vaddw.u8 q4, q4, d0\n"
      "vst1.8 {d0}, [%[destination]:64]!\n"

      "bne 1b\n"

      // Leftover Load Aggregate Store.
      "vmov.i8 d0, #0\n"
      "vld1.32 {d0[0]}, [%[source]]!\n"
      "vld1.8 {d0[4]}, [%[source]]!\n"
      "vaddw.u8 q4, q4, d0\n"
      "vst1.8 {d0}, [%[destination]:64]!\n"

      // Aggregator Reduction.
      "vmov.32 d1[0], %[multiplicative_offset]\n"
      "vdup.32 d2, %[additive_offset]\n"
      "vpaddl.u16 q4, q4\n"
      "vpadd.u32 d8, d8, d9\n"
      "vpadd.u32 d3, d8, d8\n"
      "vmul.i32 d3, d3, d1[0]\n"
      "vadd.i32 d3, d3, d2\n"
      "vst1.32 {d3}, [%[destination]:64]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "d0", "d1", "d2", "d3", "d8", "d9", "cc", "memory");
}

void zip_1x8_6(const std::uint8_t* source, std::int32_t count,
               std::int32_t stride, std::uint8_t* destination,
               std::int32_t multiplicative_offset,
               std::int32_t additive_offset) {
  asm volatile(
      "sub %[count], %[count], #6\n"
      "vmov.i16 q4, #0\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store.
      "vld1.8 {d0}, [%[source]]!\n"
      "vaddw.u8 q4, q4, d0\n"
      "vst1.8 {d0}, [%[destination]:64]!\n"

      "bne 1b\n"

      // Leftover Load Aggregate Store.
      "vmov.i8 d0, #0\n"
      "vld1.32 {d0[0]}, [%[source]]!\n"
      "vld1.16 {d0[2]}, [%[source]]!\n"
      "vaddw.u8 q4, q4, d0\n"
      "vst1.8 {d0}, [%[destination]:64]!\n"

      // Aggregator Reduction.
      "vmov.32 d1[0], %[multiplicative_offset]\n"
      "vdup.32 d2, %[additive_offset]\n"
      "vpaddl.u16 q4, q4\n"
      "vpadd.u32 d8, d8, d9\n"
      "vpadd.u32 d3, d8, d8\n"
      "vmul.i32 d3, d3, d1[0]\n"
      "vadd.i32 d3, d3, d2\n"
      "vst1.32 {d3}, [%[destination]:64]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "d0", "d1", "d2", "d3", "d8", "d9", "cc", "memory");
}

void zip_1x8_7(const std::uint8_t* source, std::int32_t count,
               std::int32_t stride, std::uint8_t* destination,
               std::int32_t multiplicative_offset,
               std::int32_t additive_offset) {
  asm volatile(
      "sub %[count], %[count], #7\n"
      "vmov.i16 q4, #0\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store.
      "vld1.8 {d0}, [%[source]]!\n"
      "vaddw.u8 q4, q4, d0\n"
      "vst1.8 {d0}, [%[destination]:64]!\n"

      "bne 1b\n"

      // Leftover Load Aggregate Store.
      "vmov.i8 d0, #0\n"
      "vld1.32 {d0[0]}, [%[source]]!\n"
      "vld1.16 {d0[2]}, [%[source]]!\n"
      "vld1.8 {d0[6]}, [%[source]]!\n"
      "vaddw.u8 q4, q4, d0\n"
      "vst1.8 {d0}, [%[destination]:64]!\n"

      // Aggregator Reduction.
      "vmov.32 d1[0], %[multiplicative_offset]\n"
      "vdup.32 d2, %[additive_offset]\n"
      "vpaddl.u16 q4, q4\n"
      "vpadd.u32 d8, d8, d9\n"
      "vpadd.u32 d3, d8, d8\n"
      "vmul.i32 d3, d3, d1[0]\n"
      "vadd.i32 d3, d3, d2\n"
      "vst1.32 {d3}, [%[destination]:64]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "d0", "d1", "d2", "d3", "d8", "d9", "cc", "memory");
}

void zip_2x8(const std::uint8_t* source, std::int32_t count,
             std::int32_t stride, std::uint8_t* destination,
             std::int32_t multiplicative_offset, std::int32_t additive_offset) {
  asm volatile(
      "add r0, %[source], %[stride]\n"
      "vmov.i16 q4, #0\n"
      "vmov.i16 q5, #0\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store.
      "vld1.8 {d0}, [%[source]]!\n"
      "vld1.8 {d1}, [r0]!\n"
      "vaddw.u8 q4, q4, d0\n"
      "vaddw.u8 q5, q5, d1\n"
      "vst1.8 {d0, d1}, [%[destination]:64]!\n"

      "bne 1b\n"

      // Aggregator Reduction.
      "vmov.32 d2[0], %[multiplicative_offset]\n"
      "vdup.32 d3, %[additive_offset]\n"
      "vpaddl.u16 q4, q4\n"
      "vpaddl.u16 q5, q5\n"
      "vpadd.u32 d8, d8, d9\n"
      "vpadd.u32 d10, d10, d11\n"
      "vpadd.u32 d4, d8, d10\n"
      "vmul.i32 d4, d4, d2[0]\n"
      "vadd.i32 d4, d4, d3\n"
      "vst1.32 {d4}, [%[destination]:64]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "r0", "d0", "d1", "d2", "d3", "d4", "d8", "d9", "d10", "d11", "cc",
        "memory");
}

void zip_2x8_1(const std::uint8_t* source, std::int32_t count,
               std::int32_t stride, std::uint8_t* destination,
               std::int32_t multiplicative_offset,
               std::int32_t additive_offset) {
  asm volatile(
      "add r0, %[source], %[stride]\n"
      "sub %[count], %[count], #1\n"
      "vmov.i16 q4, #0\n"
      "vmov.i16 q5, #0\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store.
      "vld1.8 {d0}, [%[source]]!\n"
      "vld1.8 {d1}, [r0]!\n"
      "vaddw.u8 q4, q4, d0\n"
      "vaddw.u8 q5, q5, d1\n"
      "vst1.8 {d0, d1}, [%[destination]:64]!\n"

      "bne 1b\n"

      // Leftover Load Aggregate Store.
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vld1.8 {d0[0]}, [%[source]]!\n"
      "vld1.8 {d1[0]}, [r0]!\n"
      "vaddw.u8 q4, q4, d0\n"
      "vaddw.u8 q5, q5, d1\n"
      "vst1.8 {d0, d1}, [%[destination]:64]!\n"

      // Aggregator Reduction.
      "vmov.32 d2[0], %[multiplicative_offset]\n"
      "vdup.32 d3, %[additive_offset]\n"
      "vpaddl.u16 q4, q4\n"
      "vpaddl.u16 q5, q5\n"
      "vpadd.u32 d8, d8, d9\n"
      "vpadd.u32 d10, d10, d11\n"
      "vpadd.u32 d4, d8, d10\n"
      "vmul.i32 d4, d4, d2[0]\n"
      "vadd.i32 d4, d4, d3\n"
      "vst1.32 {d4}, [%[destination]:64]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "r0", "d0", "d1", "d2", "d3", "d4", "d8", "d9", "d10", "d11", "cc",
        "memory");
}

void zip_2x8_2(const std::uint8_t* source, std::int32_t count,
               std::int32_t stride, std::uint8_t* destination,
               std::int32_t multiplicative_offset,
               std::int32_t additive_offset) {
  asm volatile(
      "add r0, %[source], %[stride]\n"
      "sub %[count], %[count], #2\n"
      "vmov.i16 q4, #0\n"
      "vmov.i16 q5, #0\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store.
      "vld1.8 {d0}, [%[source]]!\n"
      "vld1.8 {d1}, [r0]!\n"
      "vaddw.u8 q4, q4, d0\n"
      "vaddw.u8 q5, q5, d1\n"
      "vst1.8 {d0, d1}, [%[destination]:64]!\n"

      "bne 1b\n"

      // Leftover Load Aggregate Store.
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vld1.16 {d0[0]}, [%[source]]!\n"
      "vld1.16 {d1[0]}, [r0]!\n"
      "vaddw.u8 q4, q4, d0\n"
      "vaddw.u8 q5, q5, d1\n"
      "vst1.8 {d0, d1}, [%[destination]:64]!\n"

      // Aggregator Reduction.
      "vmov.32 d2[0], %[multiplicative_offset]\n"
      "vdup.32 d3, %[additive_offset]\n"
      "vpaddl.u16 q4, q4\n"
      "vpaddl.u16 q5, q5\n"
      "vpadd.u32 d8, d8, d9\n"
      "vpadd.u32 d10, d10, d11\n"
      "vpadd.u32 d4, d8, d10\n"
      "vmul.i32 d4, d4, d2[0]\n"
      "vadd.i32 d4, d4, d3\n"
      "vst1.32 {d4}, [%[destination]:64]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "r0", "d0", "d1", "d2", "d3", "d4", "d8", "d9", "d10", "d11", "cc",
        "memory");
}

void zip_2x8_3(const std::uint8_t* source, std::int32_t count,
               std::int32_t stride, std::uint8_t* destination,
               std::int32_t multiplicative_offset,
               std::int32_t additive_offset) {
  asm volatile(
      "add r0, %[source], %[stride]\n"
      "sub %[count], %[count], #3\n"
      "vmov.i16 q4, #0\n"
      "vmov.i16 q5, #0\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store.
      "vld1.8 {d0}, [%[source]]!\n"
      "vld1.8 {d1}, [r0]!\n"
      "vaddw.u8 q4, q4, d0\n"
      "vaddw.u8 q5, q5, d1\n"
      "vst1.8 {d0, d1}, [%[destination]:64]!\n"

      "bne 1b\n"

      // Leftover Load Aggregate Store.
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vld1.16 {d0[0]}, [%[source]]!\n"
      "vld1.8 {d0[2]}, [%[source]]!\n"
      "vld1.16 {d1[0]}, [r0]!\n"
      "vld1.8 {d1[2]}, [r0]!\n"
      "vaddw.u8 q4, q4, d0\n"
      "vaddw.u8 q5, q5, d1\n"
      "vst1.8 {d0, d1}, [%[destination]:64]!\n"

      // Aggregator Reduction.
      "vmov.32 d2[0], %[multiplicative_offset]\n"
      "vdup.32 d3, %[additive_offset]\n"
      "vpaddl.u16 q4, q4\n"
      "vpaddl.u16 q5, q5\n"
      "vpadd.u32 d8, d8, d9\n"
      "vpadd.u32 d10, d10, d11\n"
      "vpadd.u32 d4, d8, d10\n"
      "vmul.i32 d4, d4, d2[0]\n"
      "vadd.i32 d4, d4, d3\n"
      "vst1.32 {d4}, [%[destination]:64]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "r0", "d0", "d1", "d2", "d3", "d4", "d8", "d9", "d10", "d11", "cc",
        "memory");
}

void zip_2x8_4(const std::uint8_t* source, std::int32_t count,
               std::int32_t stride, std::uint8_t* destination,
               std::int32_t multiplicative_offset,
               std::int32_t additive_offset) {
  asm volatile(
      "add r0, %[source], %[stride]\n"
      "sub %[count], %[count], #4\n"
      "vmov.i16 q4, #0\n"
      "vmov.i16 q5, #0\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store.
      "vld1.8 {d0}, [%[source]]!\n"
      "vld1.8 {d1}, [r0]!\n"
      "vaddw.u8 q4, q4, d0\n"
      "vaddw.u8 q5, q5, d1\n"
      "vst1.8 {d0, d1}, [%[destination]:64]!\n"

      "bne 1b\n"

      // Leftover Load Aggregate Store.
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vld1.32 {d0[0]}, [%[source]]!\n"
      "vld1.32 {d1[0]}, [r0]!\n"
      "vaddw.u8 q4, q4, d0\n"
      "vaddw.u8 q5, q5, d1\n"
      "vst1.8 {d0, d1}, [%[destination]:64]!\n"

      // Aggregator Reduction.
      "vmov.32 d2[0], %[multiplicative_offset]\n"
      "vdup.32 d3, %[additive_offset]\n"
      "vpaddl.u16 q4, q4\n"
      "vpaddl.u16 q5, q5\n"
      "vpadd.u32 d8, d8, d9\n"
      "vpadd.u32 d10, d10, d11\n"
      "vpadd.u32 d4, d8, d10\n"
      "vmul.i32 d4, d4, d2[0]\n"
      "vadd.i32 d4, d4, d3\n"
      "vst1.32 {d4}, [%[destination]:64]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "r0", "d0", "d1", "d2", "d3", "d4", "d8", "d9", "d10", "d11", "cc",
        "memory");
}

void zip_2x8_5(const std::uint8_t* source, std::int32_t count,
               std::int32_t stride, std::uint8_t* destination,
               std::int32_t multiplicative_offset,
               std::int32_t additive_offset) {
  asm volatile(
      "add r0, %[source], %[stride]\n"
      "sub %[count], %[count], #5\n"
      "vmov.i16 q4, #0\n"
      "vmov.i16 q5, #0\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store.
      "vld1.8 {d0}, [%[source]]!\n"
      "vld1.8 {d1}, [r0]!\n"
      "vaddw.u8 q4, q4, d0\n"
      "vaddw.u8 q5, q5, d1\n"
      "vst1.8 {d0, d1}, [%[destination]:64]!\n"

      "bne 1b\n"

      // Leftover Load Aggregate Store.
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vld1.32 {d0[0]}, [%[source]]!\n"
      "vld1.8 {d0[4]}, [%[source]]!\n"
      "vld1.32 {d1[0]}, [r0]!\n"
      "vld1.8 {d1[4]}, [r0]!\n"
      "vaddw.u8 q4, q4, d0\n"
      "vaddw.u8 q5, q5, d1\n"
      "vst1.8 {d0, d1}, [%[destination]:64]!\n"

      // Aggregator Reduction.
      "vmov.32 d2[0], %[multiplicative_offset]\n"
      "vdup.32 d3, %[additive_offset]\n"
      "vpaddl.u16 q4, q4\n"
      "vpaddl.u16 q5, q5\n"
      "vpadd.u32 d8, d8, d9\n"
      "vpadd.u32 d10, d10, d11\n"
      "vpadd.u32 d4, d8, d10\n"
      "vmul.i32 d4, d4, d2[0]\n"
      "vadd.i32 d4, d4, d3\n"
      "vst1.32 {d4}, [%[destination]:64]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "r0", "d0", "d1", "d2", "d3", "d4", "d8", "d9", "d10", "d11", "cc",
        "memory");
}

void zip_2x8_6(const std::uint8_t* source, std::int32_t count,
               std::int32_t stride, std::uint8_t* destination,
               std::int32_t multiplicative_offset,
               std::int32_t additive_offset) {
  asm volatile(
      "add r0, %[source], %[stride]\n"
      "sub %[count], %[count], #6\n"
      "vmov.i16 q4, #0\n"
      "vmov.i16 q5, #0\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store.
      "vld1.8 {d0}, [%[source]]!\n"
      "vld1.8 {d1}, [r0]!\n"
      "vaddw.u8 q4, q4, d0\n"
      "vaddw.u8 q5, q5, d1\n"
      "vst1.8 {d0, d1}, [%[destination]:64]!\n"

      "bne 1b\n"

      // Leftover Load Aggregate Store.
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vld1.32 {d0[0]}, [%[source]]!\n"
      "vld1.16 {d0[2]}, [%[source]]!\n"
      "vld1.32 {d1[0]}, [r0]!\n"
      "vld1.16 {d1[2]}, [r0]!\n"
      "vaddw.u8 q4, q4, d0\n"
      "vaddw.u8 q5, q5, d1\n"
      "vst1.8 {d0, d1}, [%[destination]:64]!\n"

      // Aggregator Reduction.
      "vmov.32 d2[0], %[multiplicative_offset]\n"
      "vdup.32 d3, %[additive_offset]\n"
      "vpaddl.u16 q4, q4\n"
      "vpaddl.u16 q5, q5\n"
      "vpadd.u32 d8, d8, d9\n"
      "vpadd.u32 d10, d10, d11\n"
      "vpadd.u32 d4, d8, d10\n"
      "vmul.i32 d4, d4, d2[0]\n"
      "vadd.i32 d4, d4, d3\n"
      "vst1.32 {d4}, [%[destination]:64]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "r0", "d0", "d1", "d2", "d3", "d4", "d8", "d9", "d10", "d11", "cc",
        "memory");
}

void zip_2x8_7(const std::uint8_t* source, std::int32_t count,
               std::int32_t stride, std::uint8_t* destination,
               std::int32_t multiplicative_offset,
               std::int32_t additive_offset) {
  asm volatile(
      "add r0, %[source], %[stride]\n"
      "sub %[count], %[count], #7\n"
      "vmov.i16 q4, #0\n"
      "vmov.i16 q5, #0\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store.
      "vld1.8 {d0}, [%[source]]!\n"
      "vld1.8 {d1}, [r0]!\n"
      "vaddw.u8 q4, q4, d0\n"
      "vaddw.u8 q5, q5, d1\n"
      "vst1.8 {d0, d1}, [%[destination]:64]!\n"

      "bne 1b\n"

      // Leftover Load Aggregate Store.
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vld1.32 {d0[0]}, [%[source]]!\n"
      "vld1.16 {d0[2]}, [%[source]]!\n"
      "vld1.8 {d0[6]}, [%[source]]!\n"
      "vld1.32 {d1[0]}, [r0]!\n"
      "vld1.16 {d1[2]}, [r0]!\n"
      "vld1.8 {d1[6]}, [r0]!\n"
      "vaddw.u8 q4, q4, d0\n"
      "vaddw.u8 q5, q5, d1\n"
      "vst1.8 {d0, d1}, [%[destination]:64]!\n"

      // Aggregator Reduction.
      "vmov.32 d2[0], %[multiplicative_offset]\n"
      "vdup.32 d3, %[additive_offset]\n"
      "vpaddl.u16 q4, q4\n"
      "vpaddl.u16 q5, q5\n"
      "vpadd.u32 d8, d8, d9\n"
      "vpadd.u32 d10, d10, d11\n"
      "vpadd.u32 d4, d8, d10\n"
      "vmul.i32 d4, d4, d2[0]\n"
      "vadd.i32 d4, d4, d3\n"
      "vst1.32 {d4}, [%[destination]:64]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "r0", "d0", "d1", "d2", "d3", "d4", "d8", "d9", "d10", "d11", "cc",
        "memory");
}

void zip_3x8(const std::uint8_t* source, std::int32_t count,
             std::int32_t stride, std::uint8_t* destination,
             std::int32_t multiplicative_offset, std::int32_t additive_offset) {
  asm volatile(
      "add r0, %[source], %[stride]\n"
      "add r1, r0, %[stride]\n"
      "vmov.i16 q4, #0\n"
      "vmov.i16 q5, #0\n"
      "vmov.i16 q6, #0\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store.
      "vld1.8 {d0}, [%[source]]!\n"
      "vld1.8 {d1}, [r0]!\n"
      "vld1.8 {d2}, [r1]!\n"
      "vaddw.u8 q4, q4, d0\n"
      "vaddw.u8 q5, q5, d1\n"
      "vaddw.u8 q6, q6, d2\n"
      "vst1.8 {d0, d1, d2}, [%[destination]:64]!\n"

      "bne 1b\n"

      // Aggregator Reduction.
      "vmov.32 d3[0], %[multiplicative_offset]\n"
      "vdup.32 q2, %[additive_offset]\n"
      "vpaddl.u16 q4, q4\n"
      "vpaddl.u16 q5, q5\n"
      "vpaddl.u16 q6, q6\n"
      "vpadd.u32 d8, d8, d9\n"
      "vpadd.u32 d10, d10, d11\n"
      "vpadd.u32 d12, d12, d13\n"
      "vpadd.u32 d6, d8, d10\n"
      "vpadd.u32 d7, d12, d12\n"
      "vmul.i32 q3, q3, d3[0]\n"
      "vadd.i32 q3, q3, q2\n"
      "vst1.32 {d6, d7}, [%[destination]:64]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "r0", "r1", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9",
        "d10", "d11", "d12", "d13", "cc", "memory");
}

void zip_3x8_1(const std::uint8_t* source, std::int32_t count,
               std::int32_t stride, std::uint8_t* destination,
               std::int32_t multiplicative_offset,
               std::int32_t additive_offset) {
  asm volatile(
      "add r0, %[source], %[stride]\n"
      "add r1, r0, %[stride]\n"
      "sub %[count], %[count], #1\n"
      "vmov.i16 q4, #0\n"
      "vmov.i16 q5, #0\n"
      "vmov.i16 q6, #0\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store.
      "vld1.8 {d0}, [%[source]]!\n"
      "vld1.8 {d1}, [r0]!\n"
      "vld1.8 {d2}, [r1]!\n"
      "vaddw.u8 q4, q4, d0\n"
      "vaddw.u8 q5, q5, d1\n"
      "vaddw.u8 q6, q6, d2\n"
      "vst1.8 {d0, d1, d2}, [%[destination]:64]!\n"

      "bne 1b\n"

      // Leftover Load Aggregate Store.
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vmov.i8 d2, #0\n"
      "vld1.8 {d0[0]}, [%[source]]!\n"
      "vld1.8 {d1[0]}, [r0]!\n"
      "vld1.8 {d2[0]}, [r1]!\n"
      "vaddw.u8 q4, q4, d0\n"
      "vaddw.u8 q5, q5, d1\n"
      "vaddw.u8 q6, q6, d2\n"
      "vst1.8 {d0, d1, d2}, [%[destination]:64]!\n"

      // Aggregator Reduction.
      "vmov.32 d3[0], %[multiplicative_offset]\n"
      "vdup.32 q2, %[additive_offset]\n"
      "vpaddl.u16 q4, q4\n"
      "vpaddl.u16 q5, q5\n"
      "vpaddl.u16 q6, q6\n"
      "vpadd.u32 d8, d8, d9\n"
      "vpadd.u32 d10, d10, d11\n"
      "vpadd.u32 d12, d12, d13\n"
      "vpadd.u32 d6, d8, d10\n"
      "vpadd.u32 d7, d12, d12\n"
      "vmul.i32 q3, q3, d3[0]\n"
      "vadd.i32 q3, q3, q2\n"
      "vst1.32 {d6, d7}, [%[destination]:64]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "r0", "r1", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9",
        "d10", "d11", "d12", "d13", "cc", "memory");
}

void zip_3x8_2(const std::uint8_t* source, std::int32_t count,
               std::int32_t stride, std::uint8_t* destination,
               std::int32_t multiplicative_offset,
               std::int32_t additive_offset) {
  asm volatile(
      "add r0, %[source], %[stride]\n"
      "add r1, r0, %[stride]\n"
      "sub %[count], %[count], #2\n"
      "vmov.i16 q4, #0\n"
      "vmov.i16 q5, #0\n"
      "vmov.i16 q6, #0\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store.
      "vld1.8 {d0}, [%[source]]!\n"
      "vld1.8 {d1}, [r0]!\n"
      "vld1.8 {d2}, [r1]!\n"
      "vaddw.u8 q4, q4, d0\n"
      "vaddw.u8 q5, q5, d1\n"
      "vaddw.u8 q6, q6, d2\n"
      "vst1.8 {d0, d1, d2}, [%[destination]:64]!\n"

      "bne 1b\n"

      // Leftover Load Aggregate Store.
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vmov.i8 d2, #0\n"
      "vld1.16 {d0[0]}, [%[source]]!\n"
      "vld1.16 {d1[0]}, [r0]!\n"
      "vld1.16 {d2[0]}, [r1]!\n"
      "vaddw.u8 q4, q4, d0\n"
      "vaddw.u8 q5, q5, d1\n"
      "vaddw.u8 q6, q6, d2\n"
      "vst1.8 {d0, d1, d2}, [%[destination]:64]!\n"

      // Aggregator Reduction.
      "vmov.32 d3[0], %[multiplicative_offset]\n"
      "vdup.32 q2, %[additive_offset]\n"
      "vpaddl.u16 q4, q4\n"
      "vpaddl.u16 q5, q5\n"
      "vpaddl.u16 q6, q6\n"
      "vpadd.u32 d8, d8, d9\n"
      "vpadd.u32 d10, d10, d11\n"
      "vpadd.u32 d12, d12, d13\n"
      "vpadd.u32 d6, d8, d10\n"
      "vpadd.u32 d7, d12, d12\n"
      "vmul.i32 q3, q3, d3[0]\n"
      "vadd.i32 q3, q3, q2\n"
      "vst1.32 {d6, d7}, [%[destination]:64]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "r0", "r1", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9",
        "d10", "d11", "d12", "d13", "cc", "memory");
}

void zip_3x8_3(const std::uint8_t* source, std::int32_t count,
               std::int32_t stride, std::uint8_t* destination,
               std::int32_t multiplicative_offset,
               std::int32_t additive_offset) {
  asm volatile(
      "add r0, %[source], %[stride]\n"
      "add r1, r0, %[stride]\n"
      "sub %[count], %[count], #3\n"
      "vmov.i16 q4, #0\n"
      "vmov.i16 q5, #0\n"
      "vmov.i16 q6, #0\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store.
      "vld1.8 {d0}, [%[source]]!\n"
      "vld1.8 {d1}, [r0]!\n"
      "vld1.8 {d2}, [r1]!\n"
      "vaddw.u8 q4, q4, d0\n"
      "vaddw.u8 q5, q5, d1\n"
      "vaddw.u8 q6, q6, d2\n"
      "vst1.8 {d0, d1, d2}, [%[destination]:64]!\n"

      "bne 1b\n"

      // Leftover Load Aggregate Store.
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vmov.i8 d2, #0\n"
      "vld1.16 {d0[0]}, [%[source]]!\n"
      "vld1.8 {d0[2]}, [%[source]]!\n"
      "vld1.16 {d1[0]}, [r0]!\n"
      "vld1.8 {d1[2]}, [r0]!\n"
      "vld1.16 {d2[0]}, [r1]!\n"
      "vld1.8 {d2[2]}, [r1]!\n"
      "vaddw.u8 q4, q4, d0\n"
      "vaddw.u8 q5, q5, d1\n"
      "vaddw.u8 q6, q6, d2\n"
      "vst1.8 {d0, d1, d2}, [%[destination]:64]!\n"

      // Aggregator Reduction.
      "vmov.32 d3[0], %[multiplicative_offset]\n"
      "vdup.32 q2, %[additive_offset]\n"
      "vpaddl.u16 q4, q4\n"
      "vpaddl.u16 q5, q5\n"
      "vpaddl.u16 q6, q6\n"
      "vpadd.u32 d8, d8, d9\n"
      "vpadd.u32 d10, d10, d11\n"
      "vpadd.u32 d12, d12, d13\n"
      "vpadd.u32 d6, d8, d10\n"
      "vpadd.u32 d7, d12, d12\n"
      "vmul.i32 q3, q3, d3[0]\n"
      "vadd.i32 q3, q3, q2\n"
      "vst1.32 {d6, d7}, [%[destination]:64]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "r0", "r1", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9",
        "d10", "d11", "d12", "d13", "cc", "memory");
}

void zip_3x8_4(const std::uint8_t* source, std::int32_t count,
               std::int32_t stride, std::uint8_t* destination,
               std::int32_t multiplicative_offset,
               std::int32_t additive_offset) {
  asm volatile(
      "add r0, %[source], %[stride]\n"
      "add r1, r0, %[stride]\n"
      "sub %[count], %[count], #4\n"
      "vmov.i16 q4, #0\n"
      "vmov.i16 q5, #0\n"
      "vmov.i16 q6, #0\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store.
      "vld1.8 {d0}, [%[source]]!\n"
      "vld1.8 {d1}, [r0]!\n"
      "vld1.8 {d2}, [r1]!\n"
      "vaddw.u8 q4, q4, d0\n"
      "vaddw.u8 q5, q5, d1\n"
      "vaddw.u8 q6, q6, d2\n"
      "vst1.8 {d0, d1, d2}, [%[destination]:64]!\n"

      "bne 1b\n"

      // Leftover Load Aggregate Store.
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vmov.i8 d2, #0\n"
      "vld1.32 {d0[0]}, [%[source]]!\n"
      "vld1.32 {d1[0]}, [r0]!\n"
      "vld1.32 {d2[0]}, [r1]!\n"
      "vaddw.u8 q4, q4, d0\n"
      "vaddw.u8 q5, q5, d1\n"
      "vaddw.u8 q6, q6, d2\n"
      "vst1.8 {d0, d1, d2}, [%[destination]:64]!\n"

      // Aggregator Reduction.
      "vmov.32 d3[0], %[multiplicative_offset]\n"
      "vdup.32 q2, %[additive_offset]\n"
      "vpaddl.u16 q4, q4\n"
      "vpaddl.u16 q5, q5\n"
      "vpaddl.u16 q6, q6\n"
      "vpadd.u32 d8, d8, d9\n"
      "vpadd.u32 d10, d10, d11\n"
      "vpadd.u32 d12, d12, d13\n"
      "vpadd.u32 d6, d8, d10\n"
      "vpadd.u32 d7, d12, d12\n"
      "vmul.i32 q3, q3, d3[0]\n"
      "vadd.i32 q3, q3, q2\n"
      "vst1.32 {d6, d7}, [%[destination]:64]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "r0", "r1", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9",
        "d10", "d11", "d12", "d13", "cc", "memory");
}

void zip_3x8_5(const std::uint8_t* source, std::int32_t count,
               std::int32_t stride, std::uint8_t* destination,
               std::int32_t multiplicative_offset,
               std::int32_t additive_offset) {
  asm volatile(
      "add r0, %[source], %[stride]\n"
      "add r1, r0, %[stride]\n"
      "sub %[count], %[count], #5\n"
      "vmov.i16 q4, #0\n"
      "vmov.i16 q5, #0\n"
      "vmov.i16 q6, #0\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store.
      "vld1.8 {d0}, [%[source]]!\n"
      "vld1.8 {d1}, [r0]!\n"
      "vld1.8 {d2}, [r1]!\n"
      "vaddw.u8 q4, q4, d0\n"
      "vaddw.u8 q5, q5, d1\n"
      "vaddw.u8 q6, q6, d2\n"
      "vst1.8 {d0, d1, d2}, [%[destination]:64]!\n"

      "bne 1b\n"

      // Leftover Load Aggregate Store.
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vmov.i8 d2, #0\n"
      "vld1.32 {d0[0]}, [%[source]]!\n"
      "vld1.8 {d0[4]}, [%[source]]!\n"
      "vld1.32 {d1[0]}, [r0]!\n"
      "vld1.8 {d1[4]}, [r0]!\n"
      "vld1.32 {d2[0]}, [r1]!\n"
      "vld1.8 {d2[4]}, [r1]!\n"
      "vaddw.u8 q4, q4, d0\n"
      "vaddw.u8 q5, q5, d1\n"
      "vaddw.u8 q6, q6, d2\n"
      "vst1.8 {d0, d1, d2}, [%[destination]:64]!\n"

      // Aggregator Reduction.
      "vmov.32 d3[0], %[multiplicative_offset]\n"
      "vdup.32 q2, %[additive_offset]\n"
      "vpaddl.u16 q4, q4\n"
      "vpaddl.u16 q5, q5\n"
      "vpaddl.u16 q6, q6\n"
      "vpadd.u32 d8, d8, d9\n"
      "vpadd.u32 d10, d10, d11\n"
      "vpadd.u32 d12, d12, d13\n"
      "vpadd.u32 d6, d8, d10\n"
      "vpadd.u32 d7, d12, d12\n"
      "vmul.i32 q3, q3, d3[0]\n"
      "vadd.i32 q3, q3, q2\n"
      "vst1.32 {d6, d7}, [%[destination]:64]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "r0", "r1", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9",
        "d10", "d11", "d12", "d13", "cc", "memory");
}

void zip_3x8_6(const std::uint8_t* source, std::int32_t count,
               std::int32_t stride, std::uint8_t* destination,
               std::int32_t multiplicative_offset,
               std::int32_t additive_offset) {
  asm volatile(
      "add r0, %[source], %[stride]\n"
      "add r1, r0, %[stride]\n"
      "sub %[count], %[count], #6\n"
      "vmov.i16 q4, #0\n"
      "vmov.i16 q5, #0\n"
      "vmov.i16 q6, #0\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store.
      "vld1.8 {d0}, [%[source]]!\n"
      "vld1.8 {d1}, [r0]!\n"
      "vld1.8 {d2}, [r1]!\n"
      "vaddw.u8 q4, q4, d0\n"
      "vaddw.u8 q5, q5, d1\n"
      "vaddw.u8 q6, q6, d2\n"
      "vst1.8 {d0, d1, d2}, [%[destination]:64]!\n"

      "bne 1b\n"

      // Leftover Load Aggregate Store.
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vmov.i8 d2, #0\n"
      "vld1.32 {d0[0]}, [%[source]]!\n"
      "vld1.16 {d0[2]}, [%[source]]!\n"
      "vld1.32 {d1[0]}, [r0]!\n"
      "vld1.16 {d1[2]}, [r0]!\n"
      "vld1.32 {d2[0]}, [r1]!\n"
      "vld1.16 {d2[2]}, [r1]!\n"
      "vaddw.u8 q4, q4, d0\n"
      "vaddw.u8 q5, q5, d1\n"
      "vaddw.u8 q6, q6, d2\n"
      "vst1.8 {d0, d1, d2}, [%[destination]:64]!\n"

      // Aggregator Reduction.
      "vmov.32 d3[0], %[multiplicative_offset]\n"
      "vdup.32 q2, %[additive_offset]\n"
      "vpaddl.u16 q4, q4\n"
      "vpaddl.u16 q5, q5\n"
      "vpaddl.u16 q6, q6\n"
      "vpadd.u32 d8, d8, d9\n"
      "vpadd.u32 d10, d10, d11\n"
      "vpadd.u32 d12, d12, d13\n"
      "vpadd.u32 d6, d8, d10\n"
      "vpadd.u32 d7, d12, d12\n"
      "vmul.i32 q3, q3, d3[0]\n"
      "vadd.i32 q3, q3, q2\n"
      "vst1.32 {d6, d7}, [%[destination]:64]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "r0", "r1", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9",
        "d10", "d11", "d12", "d13", "cc", "memory");
}

void zip_3x8_7(const std::uint8_t* source, std::int32_t count,
               std::int32_t stride, std::uint8_t* destination,
               std::int32_t multiplicative_offset,
               std::int32_t additive_offset) {
  asm volatile(
      "add r0, %[source], %[stride]\n"
      "add r1, r0, %[stride]\n"
      "sub %[count], %[count], #7\n"
      "vmov.i16 q4, #0\n"
      "vmov.i16 q5, #0\n"
      "vmov.i16 q6, #0\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store.
      "vld1.8 {d0}, [%[source]]!\n"
      "vld1.8 {d1}, [r0]!\n"
      "vld1.8 {d2}, [r1]!\n"
      "vaddw.u8 q4, q4, d0\n"
      "vaddw.u8 q5, q5, d1\n"
      "vaddw.u8 q6, q6, d2\n"
      "vst1.8 {d0, d1, d2}, [%[destination]:64]!\n"

      "bne 1b\n"

      // Leftover Load Aggregate Store.
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vmov.i8 d2, #0\n"
      "vld1.32 {d0[0]}, [%[source]]!\n"
      "vld1.16 {d0[2]}, [%[source]]!\n"
      "vld1.8 {d0[6]}, [%[source]]!\n"
      "vld1.32 {d1[0]}, [r0]!\n"
      "vld1.16 {d1[2]}, [r0]!\n"
      "vld1.8 {d1[6]}, [r0]!\n"
      "vld1.32 {d2[0]}, [r1]!\n"
      "vld1.16 {d2[2]}, [r1]!\n"
      "vld1.8 {d2[6]}, [r1]!\n"
      "vaddw.u8 q4, q4, d0\n"
      "vaddw.u8 q5, q5, d1\n"
      "vaddw.u8 q6, q6, d2\n"
      "vst1.8 {d0, d1, d2}, [%[destination]:64]!\n"

      // Aggregator Reduction.
      "vmov.32 d3[0], %[multiplicative_offset]\n"
      "vdup.32 q2, %[additive_offset]\n"
      "vpaddl.u16 q4, q4\n"
      "vpaddl.u16 q5, q5\n"
      "vpaddl.u16 q6, q6\n"
      "vpadd.u32 d8, d8, d9\n"
      "vpadd.u32 d10, d10, d11\n"
      "vpadd.u32 d12, d12, d13\n"
      "vpadd.u32 d6, d8, d10\n"
      "vpadd.u32 d7, d12, d12\n"
      "vmul.i32 q3, q3, d3[0]\n"
      "vadd.i32 q3, q3, q2\n"
      "vst1.32 {d6, d7}, [%[destination]:64]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "r0", "r1", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9",
        "d10", "d11", "d12", "d13", "cc", "memory");
}

void zip_4x8(const std::uint8_t* source, std::int32_t count,
             std::int32_t stride, std::uint8_t* destination,
             std::int32_t multiplicative_offset, std::int32_t additive_offset) {
  asm volatile(
      "add r0, %[source], %[stride]\n"
      "add r1, r0, %[stride]\n"
      "add r2, r1, %[stride]\n"
      "vmov.i16 q4, #0\n"
      "vmov.i16 q5, #0\n"
      "vmov.i16 q6, #0\n"
      "vmov.i16 q7, #0\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store.
      "vld1.8 {d0}, [%[source]]!\n"
      "vld1.8 {d1}, [r0]!\n"
      "vld1.8 {d2}, [r1]!\n"
      "vld1.8 {d3}, [r2]!\n"
      "vaddw.u8 q4, q4, d0\n"
      "vaddw.u8 q5, q5, d1\n"
      "vaddw.u8 q6, q6, d2\n"
      "vaddw.u8 q7, q7, d3\n"
      "vst1.8 {d0, d1, d2, d3}, [%[destination]:64]!\n"

      "bne 1b\n"

      // Aggregator Reduction.
      "vmov.32 d4[0], %[multiplicative_offset]\n"
      "vdup.32 q3, %[additive_offset]\n"
      "vpaddl.u16 q4, q4\n"
      "vpaddl.u16 q5, q5\n"
      "vpaddl.u16 q6, q6\n"
      "vpaddl.u16 q7, q7\n"
      "vpadd.u32 d8, d8, d9\n"
      "vpadd.u32 d10, d10, d11\n"
      "vpadd.u32 d12, d12, d13\n"
      "vpadd.u32 d14, d14, d15\n"
      "vpadd.u32 d16, d8, d10\n"
      "vpadd.u32 d17, d12, d14\n"
      "vmul.i32 q8, q8, d4[0]\n"
      "vadd.i32 q8, q8, q3\n"
      "vst1.32 {d16, d17}, [%[destination]:64]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "r0", "r1", "r2", "d0", "d1", "d2", "d3", "d4", "d6", "d7", "d8", "d9",
        "d10", "d11", "d12", "d13", "d14", "d15", "d16", "d17", "cc", "memory");
}

void zip_4x8_1(const std::uint8_t* source, std::int32_t count,
               std::int32_t stride, std::uint8_t* destination,
               std::int32_t multiplicative_offset,
               std::int32_t additive_offset) {
  asm volatile(
      "add r0, %[source], %[stride]\n"
      "add r1, r0, %[stride]\n"
      "add r2, r1, %[stride]\n"
      "sub %[count], %[count], #1\n"
      "vmov.i16 q4, #0\n"
      "vmov.i16 q5, #0\n"
      "vmov.i16 q6, #0\n"
      "vmov.i16 q7, #0\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store.
      "vld1.8 {d0}, [%[source]]!\n"
      "vld1.8 {d1}, [r0]!\n"
      "vld1.8 {d2}, [r1]!\n"
      "vld1.8 {d3}, [r2]!\n"
      "vaddw.u8 q4, q4, d0\n"
      "vaddw.u8 q5, q5, d1\n"
      "vaddw.u8 q6, q6, d2\n"
      "vaddw.u8 q7, q7, d3\n"
      "vst1.8 {d0, d1, d2, d3}, [%[destination]:64]!\n"

      "bne 1b\n"

      // Leftover Load Aggregate Store.
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vmov.i8 d2, #0\n"
      "vmov.i8 d3, #0\n"
      "vld1.8 {d0[0]}, [%[source]]!\n"
      "vld1.8 {d1[0]}, [r0]!\n"
      "vld1.8 {d2[0]}, [r1]!\n"
      "vld1.8 {d3[0]}, [r2]!\n"
      "vaddw.u8 q4, q4, d0\n"
      "vaddw.u8 q5, q5, d1\n"
      "vaddw.u8 q6, q6, d2\n"
      "vaddw.u8 q7, q7, d3\n"
      "vst1.8 {d0, d1, d2, d3}, [%[destination]:64]!\n"

      // Aggregator Reduction.
      "vmov.32 d4[0], %[multiplicative_offset]\n"
      "vdup.32 q3, %[additive_offset]\n"
      "vpaddl.u16 q4, q4\n"
      "vpaddl.u16 q5, q5\n"
      "vpaddl.u16 q6, q6\n"
      "vpaddl.u16 q7, q7\n"
      "vpadd.u32 d8, d8, d9\n"
      "vpadd.u32 d10, d10, d11\n"
      "vpadd.u32 d12, d12, d13\n"
      "vpadd.u32 d14, d14, d15\n"
      "vpadd.u32 d16, d8, d10\n"
      "vpadd.u32 d17, d12, d14\n"
      "vmul.i32 q8, q8, d4[0]\n"
      "vadd.i32 q8, q8, q3\n"
      "vst1.32 {d16, d17}, [%[destination]:64]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "r0", "r1", "r2", "d0", "d1", "d2", "d3", "d4", "d6", "d7", "d8", "d9",
        "d10", "d11", "d12", "d13", "d14", "d15", "d16", "d17", "cc", "memory");
}

void zip_4x8_2(const std::uint8_t* source, std::int32_t count,
               std::int32_t stride, std::uint8_t* destination,
               std::int32_t multiplicative_offset,
               std::int32_t additive_offset) {
  asm volatile(
      "add r0, %[source], %[stride]\n"
      "add r1, r0, %[stride]\n"
      "add r2, r1, %[stride]\n"
      "sub %[count], %[count], #2\n"
      "vmov.i16 q4, #0\n"
      "vmov.i16 q5, #0\n"
      "vmov.i16 q6, #0\n"
      "vmov.i16 q7, #0\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store.
      "vld1.8 {d0}, [%[source]]!\n"
      "vld1.8 {d1}, [r0]!\n"
      "vld1.8 {d2}, [r1]!\n"
      "vld1.8 {d3}, [r2]!\n"
      "vaddw.u8 q4, q4, d0\n"
      "vaddw.u8 q5, q5, d1\n"
      "vaddw.u8 q6, q6, d2\n"
      "vaddw.u8 q7, q7, d3\n"
      "vst1.8 {d0, d1, d2, d3}, [%[destination]:64]!\n"

      "bne 1b\n"

      // Leftover Load Aggregate Store.
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vmov.i8 d2, #0\n"
      "vmov.i8 d3, #0\n"
      "vld1.16 {d0[0]}, [%[source]]!\n"
      "vld1.16 {d1[0]}, [r0]!\n"
      "vld1.16 {d2[0]}, [r1]!\n"
      "vld1.16 {d3[0]}, [r2]!\n"
      "vaddw.u8 q4, q4, d0\n"
      "vaddw.u8 q5, q5, d1\n"
      "vaddw.u8 q6, q6, d2\n"
      "vaddw.u8 q7, q7, d3\n"
      "vst1.8 {d0, d1, d2, d3}, [%[destination]:64]!\n"

      // Aggregator Reduction.
      "vmov.32 d4[0], %[multiplicative_offset]\n"
      "vdup.32 q3, %[additive_offset]\n"
      "vpaddl.u16 q4, q4\n"
      "vpaddl.u16 q5, q5\n"
      "vpaddl.u16 q6, q6\n"
      "vpaddl.u16 q7, q7\n"
      "vpadd.u32 d8, d8, d9\n"
      "vpadd.u32 d10, d10, d11\n"
      "vpadd.u32 d12, d12, d13\n"
      "vpadd.u32 d14, d14, d15\n"
      "vpadd.u32 d16, d8, d10\n"
      "vpadd.u32 d17, d12, d14\n"
      "vmul.i32 q8, q8, d4[0]\n"
      "vadd.i32 q8, q8, q3\n"
      "vst1.32 {d16, d17}, [%[destination]:64]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "r0", "r1", "r2", "d0", "d1", "d2", "d3", "d4", "d6", "d7", "d8", "d9",
        "d10", "d11", "d12", "d13", "d14", "d15", "d16", "d17", "cc", "memory");
}

void zip_4x8_3(const std::uint8_t* source, std::int32_t count,
               std::int32_t stride, std::uint8_t* destination,
               std::int32_t multiplicative_offset,
               std::int32_t additive_offset) {
  asm volatile(
      "add r0, %[source], %[stride]\n"
      "add r1, r0, %[stride]\n"
      "add r2, r1, %[stride]\n"
      "sub %[count], %[count], #3\n"
      "vmov.i16 q4, #0\n"
      "vmov.i16 q5, #0\n"
      "vmov.i16 q6, #0\n"
      "vmov.i16 q7, #0\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store.
      "vld1.8 {d0}, [%[source]]!\n"
      "vld1.8 {d1}, [r0]!\n"
      "vld1.8 {d2}, [r1]!\n"
      "vld1.8 {d3}, [r2]!\n"
      "vaddw.u8 q4, q4, d0\n"
      "vaddw.u8 q5, q5, d1\n"
      "vaddw.u8 q6, q6, d2\n"
      "vaddw.u8 q7, q7, d3\n"
      "vst1.8 {d0, d1, d2, d3}, [%[destination]:64]!\n"

      "bne 1b\n"

      // Leftover Load Aggregate Store.
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vmov.i8 d2, #0\n"
      "vmov.i8 d3, #0\n"
      "vld1.16 {d0[0]}, [%[source]]!\n"
      "vld1.8 {d0[2]}, [%[source]]!\n"
      "vld1.16 {d1[0]}, [r0]!\n"
      "vld1.8 {d1[2]}, [r0]!\n"
      "vld1.16 {d2[0]}, [r1]!\n"
      "vld1.8 {d2[2]}, [r1]!\n"
      "vld1.16 {d3[0]}, [r2]!\n"
      "vld1.8 {d3[2]}, [r2]!\n"
      "vaddw.u8 q4, q4, d0\n"
      "vaddw.u8 q5, q5, d1\n"
      "vaddw.u8 q6, q6, d2\n"
      "vaddw.u8 q7, q7, d3\n"
      "vst1.8 {d0, d1, d2, d3}, [%[destination]:64]!\n"

      // Aggregator Reduction.
      "vmov.32 d4[0], %[multiplicative_offset]\n"
      "vdup.32 q3, %[additive_offset]\n"
      "vpaddl.u16 q4, q4\n"
      "vpaddl.u16 q5, q5\n"
      "vpaddl.u16 q6, q6\n"
      "vpaddl.u16 q7, q7\n"
      "vpadd.u32 d8, d8, d9\n"
      "vpadd.u32 d10, d10, d11\n"
      "vpadd.u32 d12, d12, d13\n"
      "vpadd.u32 d14, d14, d15\n"
      "vpadd.u32 d16, d8, d10\n"
      "vpadd.u32 d17, d12, d14\n"
      "vmul.i32 q8, q8, d4[0]\n"
      "vadd.i32 q8, q8, q3\n"
      "vst1.32 {d16, d17}, [%[destination]:64]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "r0", "r1", "r2", "d0", "d1", "d2", "d3", "d4", "d6", "d7", "d8", "d9",
        "d10", "d11", "d12", "d13", "d14", "d15", "d16", "d17", "cc", "memory");
}

void zip_4x8_4(const std::uint8_t* source, std::int32_t count,
               std::int32_t stride, std::uint8_t* destination,
               std::int32_t multiplicative_offset,
               std::int32_t additive_offset) {
  asm volatile(
      "add r0, %[source], %[stride]\n"
      "add r1, r0, %[stride]\n"
      "add r2, r1, %[stride]\n"
      "sub %[count], %[count], #4\n"
      "vmov.i16 q4, #0\n"
      "vmov.i16 q5, #0\n"
      "vmov.i16 q6, #0\n"
      "vmov.i16 q7, #0\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store.
      "vld1.8 {d0}, [%[source]]!\n"
      "vld1.8 {d1}, [r0]!\n"
      "vld1.8 {d2}, [r1]!\n"
      "vld1.8 {d3}, [r2]!\n"
      "vaddw.u8 q4, q4, d0\n"
      "vaddw.u8 q5, q5, d1\n"
      "vaddw.u8 q6, q6, d2\n"
      "vaddw.u8 q7, q7, d3\n"
      "vst1.8 {d0, d1, d2, d3}, [%[destination]:64]!\n"

      "bne 1b\n"

      // Leftover Load Aggregate Store.
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vmov.i8 d2, #0\n"
      "vmov.i8 d3, #0\n"
      "vld1.32 {d0[0]}, [%[source]]!\n"
      "vld1.32 {d1[0]}, [r0]!\n"
      "vld1.32 {d2[0]}, [r1]!\n"
      "vld1.32 {d3[0]}, [r2]!\n"
      "vaddw.u8 q4, q4, d0\n"
      "vaddw.u8 q5, q5, d1\n"
      "vaddw.u8 q6, q6, d2\n"
      "vaddw.u8 q7, q7, d3\n"
      "vst1.8 {d0, d1, d2, d3}, [%[destination]:64]!\n"

      // Aggregator Reduction.
      "vmov.32 d4[0], %[multiplicative_offset]\n"
      "vdup.32 q3, %[additive_offset]\n"
      "vpaddl.u16 q4, q4\n"
      "vpaddl.u16 q5, q5\n"
      "vpaddl.u16 q6, q6\n"
      "vpaddl.u16 q7, q7\n"
      "vpadd.u32 d8, d8, d9\n"
      "vpadd.u32 d10, d10, d11\n"
      "vpadd.u32 d12, d12, d13\n"
      "vpadd.u32 d14, d14, d15\n"
      "vpadd.u32 d16, d8, d10\n"
      "vpadd.u32 d17, d12, d14\n"
      "vmul.i32 q8, q8, d4[0]\n"
      "vadd.i32 q8, q8, q3\n"
      "vst1.32 {d16, d17}, [%[destination]:64]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "r0", "r1", "r2", "d0", "d1", "d2", "d3", "d4", "d6", "d7", "d8", "d9",
        "d10", "d11", "d12", "d13", "d14", "d15", "d16", "d17", "cc", "memory");
}

void zip_4x8_5(const std::uint8_t* source, std::int32_t count,
               std::int32_t stride, std::uint8_t* destination,
               std::int32_t multiplicative_offset,
               std::int32_t additive_offset) {
  asm volatile(
      "add r0, %[source], %[stride]\n"
      "add r1, r0, %[stride]\n"
      "add r2, r1, %[stride]\n"
      "sub %[count], %[count], #5\n"
      "vmov.i16 q4, #0\n"
      "vmov.i16 q5, #0\n"
      "vmov.i16 q6, #0\n"
      "vmov.i16 q7, #0\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store.
      "vld1.8 {d0}, [%[source]]!\n"
      "vld1.8 {d1}, [r0]!\n"
      "vld1.8 {d2}, [r1]!\n"
      "vld1.8 {d3}, [r2]!\n"
      "vaddw.u8 q4, q4, d0\n"
      "vaddw.u8 q5, q5, d1\n"
      "vaddw.u8 q6, q6, d2\n"
      "vaddw.u8 q7, q7, d3\n"
      "vst1.8 {d0, d1, d2, d3}, [%[destination]:64]!\n"

      "bne 1b\n"

      // Leftover Load Aggregate Store.
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vmov.i8 d2, #0\n"
      "vmov.i8 d3, #0\n"
      "vld1.32 {d0[0]}, [%[source]]!\n"
      "vld1.8 {d0[4]}, [%[source]]!\n"
      "vld1.32 {d1[0]}, [r0]!\n"
      "vld1.8 {d1[4]}, [r0]!\n"
      "vld1.32 {d2[0]}, [r1]!\n"
      "vld1.8 {d2[4]}, [r1]!\n"
      "vld1.32 {d3[0]}, [r2]!\n"
      "vld1.8 {d3[4]}, [r2]!\n"
      "vaddw.u8 q4, q4, d0\n"
      "vaddw.u8 q5, q5, d1\n"
      "vaddw.u8 q6, q6, d2\n"
      "vaddw.u8 q7, q7, d3\n"
      "vst1.8 {d0, d1, d2, d3}, [%[destination]:64]!\n"

      // Aggregator Reduction.
      "vmov.32 d4[0], %[multiplicative_offset]\n"
      "vdup.32 q3, %[additive_offset]\n"
      "vpaddl.u16 q4, q4\n"
      "vpaddl.u16 q5, q5\n"
      "vpaddl.u16 q6, q6\n"
      "vpaddl.u16 q7, q7\n"
      "vpadd.u32 d8, d8, d9\n"
      "vpadd.u32 d10, d10, d11\n"
      "vpadd.u32 d12, d12, d13\n"
      "vpadd.u32 d14, d14, d15\n"
      "vpadd.u32 d16, d8, d10\n"
      "vpadd.u32 d17, d12, d14\n"
      "vmul.i32 q8, q8, d4[0]\n"
      "vadd.i32 q8, q8, q3\n"
      "vst1.32 {d16, d17}, [%[destination]:64]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "r0", "r1", "r2", "d0", "d1", "d2", "d3", "d4", "d6", "d7", "d8", "d9",
        "d10", "d11", "d12", "d13", "d14", "d15", "d16", "d17", "cc", "memory");
}

void zip_4x8_6(const std::uint8_t* source, std::int32_t count,
               std::int32_t stride, std::uint8_t* destination,
               std::int32_t multiplicative_offset,
               std::int32_t additive_offset) {
  asm volatile(
      "add r0, %[source], %[stride]\n"
      "add r1, r0, %[stride]\n"
      "add r2, r1, %[stride]\n"
      "sub %[count], %[count], #6\n"
      "vmov.i16 q4, #0\n"
      "vmov.i16 q5, #0\n"
      "vmov.i16 q6, #0\n"
      "vmov.i16 q7, #0\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store.
      "vld1.8 {d0}, [%[source]]!\n"
      "vld1.8 {d1}, [r0]!\n"
      "vld1.8 {d2}, [r1]!\n"
      "vld1.8 {d3}, [r2]!\n"
      "vaddw.u8 q4, q4, d0\n"
      "vaddw.u8 q5, q5, d1\n"
      "vaddw.u8 q6, q6, d2\n"
      "vaddw.u8 q7, q7, d3\n"
      "vst1.8 {d0, d1, d2, d3}, [%[destination]:64]!\n"

      "bne 1b\n"

      // Leftover Load Aggregate Store.
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vmov.i8 d2, #0\n"
      "vmov.i8 d3, #0\n"
      "vld1.32 {d0[0]}, [%[source]]!\n"
      "vld1.16 {d0[2]}, [%[source]]!\n"
      "vld1.32 {d1[0]}, [r0]!\n"
      "vld1.16 {d1[2]}, [r0]!\n"
      "vld1.32 {d2[0]}, [r1]!\n"
      "vld1.16 {d2[2]}, [r1]!\n"
      "vld1.32 {d3[0]}, [r2]!\n"
      "vld1.16 {d3[2]}, [r2]!\n"
      "vaddw.u8 q4, q4, d0\n"
      "vaddw.u8 q5, q5, d1\n"
      "vaddw.u8 q6, q6, d2\n"
      "vaddw.u8 q7, q7, d3\n"
      "vst1.8 {d0, d1, d2, d3}, [%[destination]:64]!\n"

      // Aggregator Reduction.
      "vmov.32 d4[0], %[multiplicative_offset]\n"
      "vdup.32 q3, %[additive_offset]\n"
      "vpaddl.u16 q4, q4\n"
      "vpaddl.u16 q5, q5\n"
      "vpaddl.u16 q6, q6\n"
      "vpaddl.u16 q7, q7\n"
      "vpadd.u32 d8, d8, d9\n"
      "vpadd.u32 d10, d10, d11\n"
      "vpadd.u32 d12, d12, d13\n"
      "vpadd.u32 d14, d14, d15\n"
      "vpadd.u32 d16, d8, d10\n"
      "vpadd.u32 d17, d12, d14\n"
      "vmul.i32 q8, q8, d4[0]\n"
      "vadd.i32 q8, q8, q3\n"
      "vst1.32 {d16, d17}, [%[destination]:64]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "r0", "r1", "r2", "d0", "d1", "d2", "d3", "d4", "d6", "d7", "d8", "d9",
        "d10", "d11", "d12", "d13", "d14", "d15", "d16", "d17", "cc", "memory");
}

void zip_4x8_7(const std::uint8_t* source, std::int32_t count,
               std::int32_t stride, std::uint8_t* destination,
               std::int32_t multiplicative_offset,
               std::int32_t additive_offset) {
  asm volatile(
      "add r0, %[source], %[stride]\n"
      "add r1, r0, %[stride]\n"
      "add r2, r1, %[stride]\n"
      "sub %[count], %[count], #7\n"
      "vmov.i16 q4, #0\n"
      "vmov.i16 q5, #0\n"
      "vmov.i16 q6, #0\n"
      "vmov.i16 q7, #0\n"

      "1:"
      "subs %[count], %[count], #8\n"

      // Load Aggregate Store.
      "vld1.8 {d0}, [%[source]]!\n"
      "vld1.8 {d1}, [r0]!\n"
      "vld1.8 {d2}, [r1]!\n"
      "vld1.8 {d3}, [r2]!\n"
      "vaddw.u8 q4, q4, d0\n"
      "vaddw.u8 q5, q5, d1\n"
      "vaddw.u8 q6, q6, d2\n"
      "vaddw.u8 q7, q7, d3\n"
      "vst1.8 {d0, d1, d2, d3}, [%[destination]:64]!\n"

      "bne 1b\n"

      // Leftover Load Aggregate Store.
      "vmov.i8 d0, #0\n"
      "vmov.i8 d1, #0\n"
      "vmov.i8 d2, #0\n"
      "vmov.i8 d3, #0\n"
      "vld1.32 {d0[0]}, [%[source]]!\n"
      "vld1.16 {d0[2]}, [%[source]]!\n"
      "vld1.8 {d0[6]}, [%[source]]!\n"
      "vld1.32 {d1[0]}, [r0]!\n"
      "vld1.16 {d1[2]}, [r0]!\n"
      "vld1.8 {d1[6]}, [r0]!\n"
      "vld1.32 {d2[0]}, [r1]!\n"
      "vld1.16 {d2[2]}, [r1]!\n"
      "vld1.8 {d2[6]}, [r1]!\n"
      "vld1.32 {d3[0]}, [r2]!\n"
      "vld1.16 {d3[2]}, [r2]!\n"
      "vld1.8 {d3[6]}, [r2]!\n"
      "vaddw.u8 q4, q4, d0\n"
      "vaddw.u8 q5, q5, d1\n"
      "vaddw.u8 q6, q6, d2\n"
      "vaddw.u8 q7, q7, d3\n"
      "vst1.8 {d0, d1, d2, d3}, [%[destination]:64]!\n"

      // Aggregator Reduction.
      "vmov.32 d4[0], %[multiplicative_offset]\n"
      "vdup.32 q3, %[additive_offset]\n"
      "vpaddl.u16 q4, q4\n"
      "vpaddl.u16 q5, q5\n"
      "vpaddl.u16 q6, q6\n"
      "vpaddl.u16 q7, q7\n"
      "vpadd.u32 d8, d8, d9\n"
      "vpadd.u32 d10, d10, d11\n"
      "vpadd.u32 d12, d12, d13\n"
      "vpadd.u32 d14, d14, d15\n"
      "vpadd.u32 d16, d8, d10\n"
      "vpadd.u32 d17, d12, d14\n"
      "vmul.i32 q8, q8, d4[0]\n"
      "vadd.i32 q8, q8, q3\n"
      "vst1.32 {d16, d17}, [%[destination]:64]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "r0", "r1", "r2", "d0", "d1", "d2", "d3", "d4", "d6", "d7", "d8", "d9",
        "d10", "d11", "d12", "d13", "d14", "d15", "d16", "d17", "cc", "memory");
}

inline void mul_1x8_1x8_int32_rhsadd(const std::uint8_t* lhs,
                                     const std::uint8_t* rhs,
                                     std::int32_t count, std::int32_t* result,
                                     std::int32_t result_stride) {
  asm volatile(
      "pld [%[lhs]]\n"
      "pld [%[rhs]]\n"
      // Clear aggregators.
      "vmov.i32 q0, #0\n"

      // General NxM lanes loop.
      "1:"

      // Subtract counter.
      "subs %[count], %[count], #8\n"

      "vld1.8 {d2}, [%[lhs]:64]!\n"
      "vld1.8 {d3}, [%[rhs]:64]!\n"
      "pld [%[lhs], #64]\n"
      "pld [%[rhs], #64]\n"
      "vmull.u8 q2, d3, d2\n"
      "vpadal.u16 q0, q2\n"

      // Loop break.
      "bne 1b\n"

      "vld1.32 {d8}, [%[rhs]:64]\n"

      // Reduce rows.
      "vpadd.u32 d0, d0, d1\n"
      "vpadd.u32 d2, d0, d0\n"

      // Add rhs offset to aggregated rows.
      "vadd.s32 d2, d2, d8\n"

      // Store reduced rows.
      "vst1.32 {d2[0]}, [%[result]], %[result_stride]\n"
      : [count] "+r"(count), [result_stride] "+r"(result_stride),
        [rhs] "+r"(rhs), [lhs] "+r"(lhs), [result] "+r"(result)
      :
      : "d0", "d1", "d2", "d3", "d4", "d5", "d8", "cc", "memory");
}

inline void mul_1x8_2x8_int32_rhsadd(const std::uint8_t* lhs,
                                     const std::uint8_t* rhs,
                                     std::int32_t count, std::int32_t* result,
                                     std::int32_t result_stride) {
  asm volatile(
      "pld [%[lhs]]\n"
      "pld [%[rhs]]\n"
      // Clear aggregators.
      "vmov.i32 q0, #0\n"
      "vmov.i32 q1, #0\n"

      // General NxM lanes loop.
      "1:"

      // Subtract counter.
      "subs %[count], %[count], #8\n"

      "vld1.8 {d4}, [%[lhs]:64]!\n"
      "vld1.8 {d5, d6}, [%[rhs]:64]!\n"
      "pld [%[lhs], #64]\n"
      "pld [%[rhs], #64]\n"
      "vmull.u8 q4, d5, d4\n"
      "vmull.u8 q5, d6, d4\n"
      "vpadal.u16 q0, q4\n"
      "vpadal.u16 q1, q5\n"

      // Loop break.
      "bne 1b\n"

      "vld1.32 {d8}, [%[rhs]:64]\n"

      // Reduce rows.
      "vpadd.u32 d0, d0, d1\n"
      "vpadd.u32 d2, d2, d3\n"
      "vpadd.u32 d4, d0, d2\n"

      // Add rhs offset to aggregated rows.
      "vadd.s32 d4, d4, d8\n"

      // Store reduced rows.
      "vst1.32 {d4}, [%[result]], %[result_stride]\n"
      : [count] "+r"(count), [result_stride] "+r"(result_stride),
        [rhs] "+r"(rhs), [lhs] "+r"(lhs), [result] "+r"(result)
      :
      : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d8", "d9", "d10", "d11",
        "cc", "memory");
}

inline void mul_1x8_3x8_int32_rhsadd(const std::uint8_t* lhs,
                                     const std::uint8_t* rhs,
                                     std::int32_t count, std::int32_t* result,
                                     std::int32_t result_stride) {
  asm volatile(
      "pld [%[lhs]]\n"
      "pld [%[rhs]]\n"
      // Clear aggregators.
      "vmov.i32 q0, #0\n"
      "vmov.i32 q1, #0\n"
      "vmov.i32 q2, #0\n"

      // General NxM lanes loop.
      "1:"

      // Subtract counter.
      "subs %[count], %[count], #8\n"

      "vld1.8 {d6}, [%[lhs]:64]!\n"
      "vld1.8 {d7, d8, d9}, [%[rhs]:64]!\n"
      "pld [%[lhs], #64]\n"
      "pld [%[rhs], #64]\n"
      "vmull.u8 q5, d7, d6\n"
      "vmull.u8 q6, d8, d6\n"
      "vmull.u8 q7, d9, d6\n"
      "vpadal.u16 q0, q5\n"
      "vpadal.u16 q1, q6\n"
      "vpadal.u16 q2, q7\n"

      // Loop break.
      "bne 1b\n"

      "vld1.32 {d8, d9}, [%[rhs]:64]\n"

      // Reduce rows.
      "vpadd.u32 d0, d0, d1\n"
      "vpadd.u32 d2, d2, d3\n"
      "vpadd.u32 d4, d4, d5\n"
      "vpadd.u32 d0, d0, d2\n"
      "vpadd.u32 d1, d4, d4\n"

      // Add rhs offset to aggregated rows.
      "vadd.s32 q0, q0, q4\n"

      // Store reduced rows.
      "vst1.32 {d0}, [%[result]]!\n"
      "vst1.32 {d1[0]}, [%[result]], %[result_stride]\n"
      "sub %[result], %[result], #8\n"
      : [count] "+r"(count), [result_stride] "+r"(result_stride),
        [rhs] "+r"(rhs), [lhs] "+r"(lhs), [result] "+r"(result)
      :
      : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10",
        "d11", "d12", "d13", "d14", "d15", "cc", "memory");
}

inline void mul_2x8_1x8_int32_rhsadd(const std::uint8_t* lhs,
                                     const std::uint8_t* rhs,
                                     std::int32_t count, std::int32_t* result,
                                     std::int32_t result_stride) {
  asm volatile(
      "pld [%[lhs]]\n"
      "pld [%[rhs]]\n"
      // Clear aggregators.
      "vmov.i32 q0, #0\n"
      "vmov.i32 q1, #0\n"

      // General NxM lanes loop.
      "1:"

      // Subtract counter.
      "subs %[count], %[count], #8\n"

      "vld1.8 {d4, d5}, [%[lhs]:64]!\n"
      "vld1.8 {d6}, [%[rhs]:64]!\n"
      "pld [%[lhs], #64]\n"
      "pld [%[rhs], #64]\n"
      "vmull.u8 q4, d6, d4\n"
      "vmull.u8 q5, d6, d5\n"
      "vpadal.u16 q0, q4\n"
      "vpadal.u16 q1, q5\n"

      // Loop break.
      "bne 1b\n"

      "vld1.32 {d8}, [%[rhs]:64]\n"

      // Reduce rows.
      "vpadd.u32 d0, d0, d1\n"
      "vpadd.u32 d4, d0, d0\n"
      "vpadd.u32 d2, d2, d3\n"
      "vpadd.u32 d5, d2, d2\n"

      // Add rhs offset to aggregated rows.
      "vadd.s32 d4, d4, d8\n"
      "vadd.s32 d5, d5, d8\n"

      // Store reduced rows.
      "vst1.32 {d4[0]}, [%[result]], %[result_stride]\n"
      "vst1.32 {d5[0]}, [%[result]], %[result_stride]\n"
      : [count] "+r"(count), [result_stride] "+r"(result_stride),
        [rhs] "+r"(rhs), [lhs] "+r"(lhs), [result] "+r"(result)
      :
      : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d8", "d9", "d10", "d11",
        "cc", "memory");
}

inline void mul_2x8_2x8_int32_rhsadd(const std::uint8_t* lhs,
                                     const std::uint8_t* rhs,
                                     std::int32_t count, std::int32_t* result,
                                     std::int32_t result_stride) {
  asm volatile(
      "pld [%[lhs]]\n"
      "pld [%[rhs]]\n"
      // Clear aggregators.
      "vmov.i32 q0, #0\n"
      "vmov.i32 q1, #0\n"
      "vmov.i32 q2, #0\n"
      "vmov.i32 q3, q0\n"

      // General NxM lanes loop.
      "1:"

      // Subtract counter.
      "subs %[count], %[count], #8\n"

      "vld1.8 {d8, d9}, [%[lhs]:64]!\n"
      "vld1.8 {d10, d11}, [%[rhs]:64]!\n"
      "pld [%[lhs], #64]\n"
      "pld [%[rhs], #64]\n"
      "vmull.u8 q6, d10, d8\n"
      "vmull.u8 q7, d11, d8\n"
      "vmull.u8 q8, d10, d9\n"
      "vmull.u8 q9, d11, d9\n"
      "vpadal.u16 q0, q6\n"
      "vpadal.u16 q1, q7\n"
      "vpadal.u16 q2, q8\n"
      "vpadal.u16 q3, q9\n"

      // Loop break.
      "bne 1b\n"

      "vld1.32 {d8}, [%[rhs]:64]\n"

      // Reduce rows.
      "vpadd.u32 d0, d0, d1\n"
      "vpadd.u32 d2, d2, d3\n"
      "vpadd.u32 d9, d0, d2\n"
      "vpadd.u32 d4, d4, d5\n"
      "vpadd.u32 d6, d6, d7\n"
      "vpadd.u32 d10, d4, d6\n"

      // Add rhs offset to aggregated rows.
      "vadd.s32 d9, d9, d8\n"
      "vadd.s32 d10, d10, d8\n"

      // Store reduced rows.
      "vst1.32 {d9}, [%[result]], %[result_stride]\n"
      "vst1.32 {d10}, [%[result]], %[result_stride]\n"
      : [count] "+r"(count), [result_stride] "+r"(result_stride),
        [rhs] "+r"(rhs), [lhs] "+r"(lhs), [result] "+r"(result)
      :
      : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10",
        "d11", "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19", "cc",
        "memory");
}

inline void mul_2x8_3x8_int32_rhsadd(const std::uint8_t* lhs,
                                     const std::uint8_t* rhs,
                                     std::int32_t count, std::int32_t* result,
                                     std::int32_t result_stride) {
  asm volatile(
      "pld [%[lhs]]\n"
      "pld [%[rhs]]\n"
      // Clear aggregators.
      "vmov.i32 q0, #0\n"
      "vmov.i32 q1, #0\n"
      "vmov.i32 q2, #0\n"
      "vmov.i32 q3, q0\n"
      "vmov.i32 q4, q1\n"
      "vmov.i32 q5, q2\n"

      // General NxM lanes loop.
      "1:"

      // Subtract counter.
      "subs %[count], %[count], #8\n"

      "vld1.8 {d12, d13}, [%[lhs]:64]!\n"
      "vld1.8 {d14, d15, d16}, [%[rhs]:64]!\n"
      "pld [%[lhs], #64]\n"
      "pld [%[rhs], #64]\n"
      "vmull.u8 q9, d14, d12\n"
      "vmull.u8 q10, d15, d12\n"
      "vmull.u8 q11, d16, d12\n"
      "vmull.u8 q12, d14, d13\n"
      "vmull.u8 q13, d15, d13\n"
      "vmull.u8 q14, d16, d13\n"
      "vpadal.u16 q0, q9\n"
      "vpadal.u16 q1, q10\n"
      "vpadal.u16 q2, q11\n"
      "vpadal.u16 q3, q12\n"
      "vpadal.u16 q4, q13\n"
      "vpadal.u16 q5, q14\n"

      // Loop break.
      "bne 1b\n"

      "vld1.32 {d12, d13}, [%[rhs]:64]\n"

      // Reduce rows.
      "vpadd.u32 d0, d0, d1\n"
      "vpadd.u32 d2, d2, d3\n"
      "vpadd.u32 d4, d4, d5\n"
      "vpadd.u32 d0, d0, d2\n"
      "vpadd.u32 d1, d4, d4\n"
      "vpadd.u32 d6, d6, d7\n"
      "vpadd.u32 d8, d8, d9\n"
      "vpadd.u32 d10, d10, d11\n"
      "vpadd.u32 d6, d6, d8\n"
      "vpadd.u32 d7, d10, d10\n"

      // Add rhs offset to aggregated rows.
      "vadd.s32 q0, q0, q6\n"
      "vadd.s32 q3, q3, q6\n"

      // Store reduced rows.
      "vst1.32 {d0}, [%[result]]!\n"
      "vst1.32 {d1[0]}, [%[result]], %[result_stride]\n"
      "sub %[result], %[result], #8\n"
      "vst1.32 {d6}, [%[result]]!\n"
      "vst1.32 {d7[0]}, [%[result]], %[result_stride]\n"
      "sub %[result], %[result], #8\n"
      : [count] "+r"(count), [result_stride] "+r"(result_stride),
        [rhs] "+r"(rhs), [lhs] "+r"(lhs), [result] "+r"(result)
      :
      : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10",
        "d11", "d12", "d13", "d14", "d15", "d16", "d18", "d19", "d20", "d21",
        "d22", "d23", "d24", "d25", "d26", "d27", "d28", "d29", "cc", "memory");
}

inline void mul_3x8_1x8_int32_rhsadd(const std::uint8_t* lhs,
                                     const std::uint8_t* rhs,
                                     std::int32_t count, std::int32_t* result,
                                     std::int32_t result_stride) {
  asm volatile(
      "pld [%[lhs]]\n"
      "pld [%[rhs]]\n"
      // Clear aggregators.
      "vmov.i32 q0, #0\n"
      "vmov.i32 q1, #0\n"
      "vmov.i32 q2, #0\n"

      // General NxM lanes loop.
      "1:"

      // Subtract counter.
      "subs %[count], %[count], #8\n"

      "vld1.8 {d6, d7, d8}, [%[lhs]:64]!\n"
      "vld1.8 {d9}, [%[rhs]:64]!\n"
      "pld [%[lhs], #64]\n"
      "pld [%[rhs], #64]\n"
      "vmull.u8 q5, d9, d6\n"
      "vmull.u8 q6, d9, d7\n"
      "vmull.u8 q7, d9, d8\n"
      "vpadal.u16 q0, q5\n"
      "vpadal.u16 q1, q6\n"
      "vpadal.u16 q2, q7\n"

      // Loop break.
      "bne 1b\n"

      "vld1.32 {d8}, [%[rhs]:64]\n"

      // Reduce rows.
      "vpadd.u32 d0, d0, d1\n"
      "vpadd.u32 d6, d0, d0\n"
      "vpadd.u32 d2, d2, d3\n"
      "vpadd.u32 d7, d2, d2\n"
      "vpadd.u32 d4, d4, d5\n"
      "vpadd.u32 d9, d4, d4\n"

      // Add rhs offset to aggregated rows.
      "vadd.s32 d6, d6, d8\n"
      "vadd.s32 d7, d7, d8\n"
      "vadd.s32 d9, d9, d8\n"

      // Store reduced rows.
      "vst1.32 {d6[0]}, [%[result]], %[result_stride]\n"
      "vst1.32 {d7[0]}, [%[result]], %[result_stride]\n"
      "vst1.32 {d9[0]}, [%[result]], %[result_stride]\n"
      : [count] "+r"(count), [result_stride] "+r"(result_stride),
        [rhs] "+r"(rhs), [lhs] "+r"(lhs), [result] "+r"(result)
      :
      : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10",
        "d11", "d12", "d13", "d14", "d15", "cc", "memory");
}

inline void mul_3x8_2x8_int32_rhsadd(const std::uint8_t* lhs,
                                     const std::uint8_t* rhs,
                                     std::int32_t count, std::int32_t* result,
                                     std::int32_t result_stride) {
  asm volatile(
      "pld [%[lhs]]\n"
      "pld [%[rhs]]\n"
      // Clear aggregators.
      "vmov.i32 q0, #0\n"
      "vmov.i32 q1, #0\n"
      "vmov.i32 q2, #0\n"
      "vmov.i32 q3, q0\n"
      "vmov.i32 q4, q1\n"
      "vmov.i32 q5, q2\n"

      // General NxM lanes loop.
      "1:"

      // Subtract counter.
      "subs %[count], %[count], #8\n"

      "vld1.8 {d12, d13, d14}, [%[lhs]:64]!\n"
      "vld1.8 {d15, d16}, [%[rhs]:64]!\n"
      "pld [%[lhs], #64]\n"
      "pld [%[rhs], #64]\n"
      "vmull.u8 q9, d15, d12\n"
      "vmull.u8 q10, d16, d12\n"
      "vmull.u8 q11, d15, d13\n"
      "vmull.u8 q12, d16, d13\n"
      "vmull.u8 q13, d15, d14\n"
      "vmull.u8 q14, d16, d14\n"
      "vpadal.u16 q0, q9\n"
      "vpadal.u16 q1, q10\n"
      "vpadal.u16 q2, q11\n"
      "vpadal.u16 q3, q12\n"
      "vpadal.u16 q4, q13\n"
      "vpadal.u16 q5, q14\n"

      // Loop break.
      "bne 1b\n"

      "vld1.32 {d12}, [%[rhs]:64]\n"

      // Reduce rows.
      "vpadd.u32 d0, d0, d1\n"
      "vpadd.u32 d2, d2, d3\n"
      "vpadd.u32 d13, d0, d2\n"
      "vpadd.u32 d4, d4, d5\n"
      "vpadd.u32 d6, d6, d7\n"
      "vpadd.u32 d14, d4, d6\n"
      "vpadd.u32 d8, d8, d9\n"
      "vpadd.u32 d10, d10, d11\n"
      "vpadd.u32 d15, d8, d10\n"

      // Add rhs offset to aggregated rows.
      "vadd.s32 d13, d13, d12\n"
      "vadd.s32 d14, d14, d12\n"
      "vadd.s32 d15, d15, d12\n"

      // Store reduced rows.
      "vst1.32 {d13}, [%[result]], %[result_stride]\n"
      "vst1.32 {d14}, [%[result]], %[result_stride]\n"
      "vst1.32 {d15}, [%[result]], %[result_stride]\n"
      : [count] "+r"(count), [result_stride] "+r"(result_stride),
        [rhs] "+r"(rhs), [lhs] "+r"(lhs), [result] "+r"(result)
      :
      : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10",
        "d11", "d12", "d13", "d14", "d15", "d16", "d18", "d19", "d20", "d21",
        "d22", "d23", "d24", "d25", "d26", "d27", "d28", "d29", "cc", "memory");
}

inline void mul_3x8_3x8_int32_rhsadd(const std::uint8_t* lhs,
                                     const std::uint8_t* rhs,
                                     std::int32_t count, std::int32_t* result,
                                     std::int32_t result_stride) {
  asm volatile(
      "pld [%[lhs]]\n"
      "pld [%[rhs]]\n"
      // Clear aggregators.
      "vmov.i32 q0, #0\n"
      "vmov.i32 q1, #0\n"
      "vmov.i32 q2, #0\n"
      "vmov.i32 q3, q0\n"
      "vmov.i32 q4, q1\n"
      "vmov.i32 q5, q2\n"
      "vmov.i32 q6, q3\n"
      "vmov.i32 q7, q4\n"
      "vmov.i32 q8, q5\n"

      // 3x3 lanes loop.
      "1:"

      "vld1.8 {d21, d22, d23}, [%[rhs]:64]!\n"
      "vld1.8 {d18}, [%[lhs]:64]!\n"
      "vmull.u8 q12, d18, d21\n"
      "vld1.8 {d19}, [%[lhs]:64]!\n"
      "vmull.u8 q13, d18, d22\n"
      "vld1.8 {d20}, [%[lhs]:64]!\n"
      "vmull.u8 q14, d18, d23\n"
      "pld [%[lhs], #64]\n"
      "vmull.u8 q15, d19, d21\n"
      "pld [%[rhs], #64]\n"
      "vpadal.u16 q0, q12\n"
      "vpadal.u16 q1, q13\n"
      "vpadal.u16 q2, q14\n"
      "vpadal.u16 q3, q15\n"
      "vmull.u8 q12, d19, d22\n"
      "vmull.u8 q13, d19, d23\n"
      "vmull.u8 q14, d20, d21\n"
      "vmull.u8 q15, d20, d22\n"

      // Subtract counter.
      "subs %[count], %[count], #8\n"

      "vmull.u8 q9, d20, d23\n"
      "vpadal.u16 q4, q12\n"
      "vpadal.u16 q5, q13\n"
      "vpadal.u16 q6, q14\n"
      "vpadal.u16 q7, q15\n"
      "vpadal.u16 q8, q9\n"

      // Loop break.
      "bne 1b\n"

      "vld1.32 {d18, d19}, [%[rhs]:64]\n"

      // Reduce rows.
      "vpadd.u32 d0, d0, d1\n"
      "vpadd.u32 d2, d2, d3\n"
      "vpadd.u32 d4, d4, d5\n"
      "vpadd.u32 d0, d0, d2\n"
      "vpadd.u32 d1, d4, d4\n"
      "vpadd.u32 d6, d6, d7\n"
      "vpadd.u32 d8, d8, d9\n"
      "vpadd.u32 d10, d10, d11\n"
      "vpadd.u32 d6, d6, d8\n"
      "vpadd.u32 d7, d10, d10\n"
      "vpadd.u32 d12, d12, d13\n"
      "vpadd.u32 d14, d14, d15\n"
      "vpadd.u32 d16, d16, d17\n"
      "vpadd.u32 d12, d12, d14\n"
      "vpadd.u32 d13, d16, d16\n"

      // Add rhs offset to aggregated rows.
      "vadd.s32 q0, q0, q9\n"
      "vadd.s32 q3, q3, q9\n"
      "vadd.s32 q6, q6, q9\n"

      // Store reduced rows.
      "vst1.32 {d0}, [%[result]]!\n"
      "vst1.32 {d1[0]}, [%[result]], %[result_stride]\n"
      "sub %[result], %[result], #8\n"
      "vst1.32 {d6}, [%[result]]!\n"
      "vst1.32 {d7[0]}, [%[result]], %[result_stride]\n"
      "sub %[result], %[result], #8\n"
      "vst1.32 {d12}, [%[result]]!\n"
      "vst1.32 {d13[0]}, [%[result]], %[result_stride]\n"
      "sub %[result], %[result], #8\n"
      : [count] "+r"(count), [result_stride] "+r"(result_stride),
        [rhs] "+r"(rhs), [lhs] "+r"(lhs), [result] "+r"(result)
      :
      : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10",
        "d11", "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19", "d20",
        "d21", "d22", "d23", "d24", "d25", "d26", "d27", "d28", "d29", "d30",
        "d31", "cc", "memory");
}

inline void mul_1x8_4x8_int32_rhsadd(const std::uint8_t* lhs,
                                     const std::uint8_t* rhs,
                                     std::int32_t count, std::int32_t* result,
                                     std::int32_t result_stride) {
  asm volatile(
      "pld [%[lhs]]\n"
      "pld [%[rhs]]\n"
      // Clear aggregators.
      "vmov.i32 q0, #0\n"
      "vmov.i32 q1, #0\n"
      "vmov.i32 q2, #0\n"
      "vmov.i32 q3, q0\n"

      // General NxM lanes loop.
      "1:"

      // Subtract counter.
      "subs %[count], %[count], #8\n"

      "vld1.8 {d8}, [%[lhs]:64]!\n"
      "vld1.8 {d9, d10, d11, d12}, [%[rhs]:64]!\n"
      "pld [%[lhs], #64]\n"
      "pld [%[rhs], #64]\n"
      "vmull.u8 q7, d9, d8\n"
      "vmull.u8 q8, d10, d8\n"
      "vmull.u8 q9, d11, d8\n"
      "vmull.u8 q10, d12, d8\n"
      "vpadal.u16 q0, q7\n"
      "vpadal.u16 q1, q8\n"
      "vpadal.u16 q2, q9\n"
      "vpadal.u16 q3, q10\n"

      // Loop break.
      "bne 1b\n"

      "vld1.32 {d8, d9}, [%[rhs]:64]\n"

      // Reduce rows.
      "vpadd.u32 d0, d0, d1\n"
      "vpadd.u32 d2, d2, d3\n"
      "vpadd.u32 d4, d4, d5\n"
      "vpadd.u32 d6, d6, d7\n"
      "vpadd.u32 d0, d0, d2\n"
      "vpadd.u32 d1, d4, d6\n"

      // Add rhs offset to aggregated rows.
      "vadd.s32 q0, q0, q4\n"

      // Store reduced rows.
      "vst1.32 {d0, d1}, [%[result]], %[result_stride]\n"
      : [count] "+r"(count), [result_stride] "+r"(result_stride),
        [rhs] "+r"(rhs), [lhs] "+r"(lhs), [result] "+r"(result)
      :
      : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10",
        "d11", "d12", "d14", "d15", "d16", "d17", "d18", "d19", "d20", "d21",
        "cc", "memory");
}

inline void mul_1x8_1x8_int32_lhsadd_rhsadd(const std::uint8_t* lhs,
                                            const std::uint8_t* rhs,
                                            std::int32_t count,
                                            std::int32_t* result,
                                            std::int32_t result_stride) {
  asm volatile(
      "pld [%[lhs]]\n"
      "pld [%[rhs]]\n"
      // Clear aggregators.
      "vmov.i32 q0, #0\n"

      // General NxM lanes loop.
      "1:"

      // Subtract counter.
      "subs %[count], %[count], #8\n"

      "vld1.8 {d2}, [%[lhs]:64]!\n"
      "vld1.8 {d3}, [%[rhs]:64]!\n"
      "pld [%[lhs], #64]\n"
      "pld [%[rhs], #64]\n"
      "vmull.u8 q2, d3, d2\n"
      "vpadal.u16 q0, q2\n"

      // Loop break.
      "bne 1b\n"

      "vld1.32 {d8}, [%[lhs]:64]\n"
      "vdup.32 d9, d8[0]\n"
      "vld1.32 {d10}, [%[rhs]:64]\n"

      // Reduce rows.
      "vpadd.u32 d0, d0, d1\n"
      "vpadd.u32 d2, d0, d0\n"

      // Add lhs offsets to aggregated rows.
      "vadd.s32 d2, d2, d9\n"

      // Add rhs offset to aggregated rows.
      "vadd.s32 d2, d2, d10\n"

      // Store reduced rows.
      "vst1.32 {d2[0]}, [%[result]], %[result_stride]\n"
      : [count] "+r"(count), [result_stride] "+r"(result_stride),
        [rhs] "+r"(rhs), [lhs] "+r"(lhs), [result] "+r"(result)
      :
      : "d0", "d1", "d2", "d3", "d4", "d5", "d8", "d9", "d10", "cc", "memory");
}

inline void mul_1x8_2x8_int32_lhsadd_rhsadd(const std::uint8_t* lhs,
                                            const std::uint8_t* rhs,
                                            std::int32_t count,
                                            std::int32_t* result,
                                            std::int32_t result_stride) {
  asm volatile(
      "pld [%[lhs]]\n"
      "pld [%[rhs]]\n"
      // Clear aggregators.
      "vmov.i32 q0, #0\n"
      "vmov.i32 q1, #0\n"

      // General NxM lanes loop.
      "1:"

      // Subtract counter.
      "subs %[count], %[count], #8\n"

      "vld1.8 {d4}, [%[lhs]:64]!\n"
      "vld1.8 {d5, d6}, [%[rhs]:64]!\n"
      "pld [%[lhs], #64]\n"
      "pld [%[rhs], #64]\n"
      "vmull.u8 q4, d5, d4\n"
      "vmull.u8 q5, d6, d4\n"
      "vpadal.u16 q0, q4\n"
      "vpadal.u16 q1, q5\n"

      // Loop break.
      "bne 1b\n"

      "vld1.32 {d8}, [%[lhs]:64]\n"
      "vdup.32 d9, d8[0]\n"
      "vld1.32 {d10}, [%[rhs]:64]\n"

      // Reduce rows.
      "vpadd.u32 d0, d0, d1\n"
      "vpadd.u32 d2, d2, d3\n"
      "vpadd.u32 d4, d0, d2\n"

      // Add lhs offsets to aggregated rows.
      "vadd.s32 d4, d4, d9\n"

      // Add rhs offset to aggregated rows.
      "vadd.s32 d4, d4, d10\n"

      // Store reduced rows.
      "vst1.32 {d4}, [%[result]], %[result_stride]\n"
      : [count] "+r"(count), [result_stride] "+r"(result_stride),
        [rhs] "+r"(rhs), [lhs] "+r"(lhs), [result] "+r"(result)
      :
      : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d8", "d9", "d10", "d11",
        "cc", "memory");
}

inline void mul_1x8_3x8_int32_lhsadd_rhsadd(const std::uint8_t* lhs,
                                            const std::uint8_t* rhs,
                                            std::int32_t count,
                                            std::int32_t* result,
                                            std::int32_t result_stride) {
  asm volatile(
      "pld [%[lhs]]\n"
      "pld [%[rhs]]\n"
      // Clear aggregators.
      "vmov.i32 q0, #0\n"
      "vmov.i32 q1, #0\n"
      "vmov.i32 q2, #0\n"

      // General NxM lanes loop.
      "1:"

      // Subtract counter.
      "subs %[count], %[count], #8\n"

      "vld1.8 {d6}, [%[lhs]:64]!\n"
      "vld1.8 {d7, d8, d9}, [%[rhs]:64]!\n"
      "pld [%[lhs], #64]\n"
      "pld [%[rhs], #64]\n"
      "vmull.u8 q5, d7, d6\n"
      "vmull.u8 q6, d8, d6\n"
      "vmull.u8 q7, d9, d6\n"
      "vpadal.u16 q0, q5\n"
      "vpadal.u16 q1, q6\n"
      "vpadal.u16 q2, q7\n"

      // Loop break.
      "bne 1b\n"

      "vld1.32 {d8}, [%[lhs]:64]\n"
      "vdup.32 q5, d8[0]\n"
      "vld1.32 {d12, d13}, [%[rhs]:64]\n"

      // Reduce rows.
      "vpadd.u32 d0, d0, d1\n"
      "vpadd.u32 d2, d2, d3\n"
      "vpadd.u32 d4, d4, d5\n"
      "vpadd.u32 d0, d0, d2\n"
      "vpadd.u32 d1, d4, d4\n"

      // Add lhs offsets to aggregated rows.
      "vadd.s32 q0, q0, q5\n"

      // Add rhs offset to aggregated rows.
      "vadd.s32 q0, q0, q6\n"

      // Store reduced rows.
      "vst1.32 {d0}, [%[result]]!\n"
      "vst1.32 {d1[0]}, [%[result]], %[result_stride]\n"
      "sub %[result], %[result], #8\n"
      : [count] "+r"(count), [result_stride] "+r"(result_stride),
        [rhs] "+r"(rhs), [lhs] "+r"(lhs), [result] "+r"(result)
      :
      : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10",
        "d11", "d12", "d13", "d14", "d15", "cc", "memory");
}

inline void mul_2x8_1x8_int32_lhsadd_rhsadd(const std::uint8_t* lhs,
                                            const std::uint8_t* rhs,
                                            std::int32_t count,
                                            std::int32_t* result,
                                            std::int32_t result_stride) {
  asm volatile(
      "pld [%[lhs]]\n"
      "pld [%[rhs]]\n"
      // Clear aggregators.
      "vmov.i32 q0, #0\n"
      "vmov.i32 q1, #0\n"

      // General NxM lanes loop.
      "1:"

      // Subtract counter.
      "subs %[count], %[count], #8\n"

      "vld1.8 {d4, d5}, [%[lhs]:64]!\n"
      "vld1.8 {d6}, [%[rhs]:64]!\n"
      "pld [%[lhs], #64]\n"
      "pld [%[rhs], #64]\n"
      "vmull.u8 q4, d6, d4\n"
      "vmull.u8 q5, d6, d5\n"
      "vpadal.u16 q0, q4\n"
      "vpadal.u16 q1, q5\n"

      // Loop break.
      "bne 1b\n"

      "vld1.32 {d8}, [%[lhs]:64]\n"
      "vdup.32 d9, d8[0]\n"
      "vdup.32 d10, d8[1]\n"
      "vld1.32 {d11}, [%[rhs]:64]\n"

      // Reduce rows.
      "vpadd.u32 d0, d0, d1\n"
      "vpadd.u32 d4, d0, d0\n"
      "vpadd.u32 d2, d2, d3\n"
      "vpadd.u32 d5, d2, d2\n"

      // Add lhs offsets to aggregated rows.
      "vadd.s32 d4, d4, d9\n"
      "vadd.s32 d5, d5, d10\n"

      // Add rhs offset to aggregated rows.
      "vadd.s32 d4, d4, d11\n"
      "vadd.s32 d5, d5, d11\n"

      // Store reduced rows.
      "vst1.32 {d4[0]}, [%[result]], %[result_stride]\n"
      "vst1.32 {d5[0]}, [%[result]], %[result_stride]\n"
      : [count] "+r"(count), [result_stride] "+r"(result_stride),
        [rhs] "+r"(rhs), [lhs] "+r"(lhs), [result] "+r"(result)
      :
      : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d8", "d9", "d10", "d11",
        "cc", "memory");
}

inline void mul_2x8_2x8_int32_lhsadd_rhsadd(const std::uint8_t* lhs,
                                            const std::uint8_t* rhs,
                                            std::int32_t count,
                                            std::int32_t* result,
                                            std::int32_t result_stride) {
  asm volatile(
      "pld [%[lhs]]\n"
      "pld [%[rhs]]\n"
      // Clear aggregators.
      "vmov.i32 q0, #0\n"
      "vmov.i32 q1, #0\n"
      "vmov.i32 q2, #0\n"
      "vmov.i32 q3, q0\n"

      // General NxM lanes loop.
      "1:"

      // Subtract counter.
      "subs %[count], %[count], #8\n"

      "vld1.8 {d8, d9}, [%[lhs]:64]!\n"
      "vld1.8 {d10, d11}, [%[rhs]:64]!\n"
      "pld [%[lhs], #64]\n"
      "pld [%[rhs], #64]\n"
      "vmull.u8 q6, d10, d8\n"
      "vmull.u8 q7, d11, d8\n"
      "vmull.u8 q8, d10, d9\n"
      "vmull.u8 q9, d11, d9\n"
      "vpadal.u16 q0, q6\n"
      "vpadal.u16 q1, q7\n"
      "vpadal.u16 q2, q8\n"
      "vpadal.u16 q3, q9\n"

      // Loop break.
      "bne 1b\n"

      "vld1.32 {d8}, [%[lhs]:64]\n"
      "vdup.32 d9, d8[0]\n"
      "vdup.32 d10, d8[1]\n"
      "vld1.32 {d11}, [%[rhs]:64]\n"

      // Reduce rows.
      "vpadd.u32 d0, d0, d1\n"
      "vpadd.u32 d2, d2, d3\n"
      "vpadd.u32 d12, d0, d2\n"
      "vpadd.u32 d4, d4, d5\n"
      "vpadd.u32 d6, d6, d7\n"
      "vpadd.u32 d13, d4, d6\n"

      // Add lhs offsets to aggregated rows.
      "vadd.s32 d12, d12, d9\n"
      "vadd.s32 d13, d13, d10\n"

      // Add rhs offset to aggregated rows.
      "vadd.s32 d12, d12, d11\n"
      "vadd.s32 d13, d13, d11\n"

      // Store reduced rows.
      "vst1.32 {d12}, [%[result]], %[result_stride]\n"
      "vst1.32 {d13}, [%[result]], %[result_stride]\n"
      : [count] "+r"(count), [result_stride] "+r"(result_stride),
        [rhs] "+r"(rhs), [lhs] "+r"(lhs), [result] "+r"(result)
      :
      : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10",
        "d11", "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19", "cc",
        "memory");
}

inline void mul_2x8_3x8_int32_lhsadd_rhsadd(const std::uint8_t* lhs,
                                            const std::uint8_t* rhs,
                                            std::int32_t count,
                                            std::int32_t* result,
                                            std::int32_t result_stride) {
  asm volatile(
      "pld [%[lhs]]\n"
      "pld [%[rhs]]\n"
      // Clear aggregators.
      "vmov.i32 q0, #0\n"
      "vmov.i32 q1, #0\n"
      "vmov.i32 q2, #0\n"
      "vmov.i32 q3, q0\n"
      "vmov.i32 q4, q1\n"
      "vmov.i32 q5, q2\n"

      // General NxM lanes loop.
      "1:"

      // Subtract counter.
      "subs %[count], %[count], #8\n"

      "vld1.8 {d12, d13}, [%[lhs]:64]!\n"
      "vld1.8 {d14, d15, d16}, [%[rhs]:64]!\n"
      "pld [%[lhs], #64]\n"
      "pld [%[rhs], #64]\n"
      "vmull.u8 q9, d14, d12\n"
      "vmull.u8 q10, d15, d12\n"
      "vmull.u8 q11, d16, d12\n"
      "vmull.u8 q12, d14, d13\n"
      "vmull.u8 q13, d15, d13\n"
      "vmull.u8 q14, d16, d13\n"
      "vpadal.u16 q0, q9\n"
      "vpadal.u16 q1, q10\n"
      "vpadal.u16 q2, q11\n"
      "vpadal.u16 q3, q12\n"
      "vpadal.u16 q4, q13\n"
      "vpadal.u16 q5, q14\n"

      // Loop break.
      "bne 1b\n"

      "vld1.32 {d12}, [%[lhs]:64]\n"
      "vdup.32 q7, d12[0]\n"
      "vdup.32 q8, d12[1]\n"
      "vld1.32 {d18, d19}, [%[rhs]:64]\n"

      // Reduce rows.
      "vpadd.u32 d0, d0, d1\n"
      "vpadd.u32 d2, d2, d3\n"
      "vpadd.u32 d4, d4, d5\n"
      "vpadd.u32 d0, d0, d2\n"
      "vpadd.u32 d1, d4, d4\n"
      "vpadd.u32 d6, d6, d7\n"
      "vpadd.u32 d8, d8, d9\n"
      "vpadd.u32 d10, d10, d11\n"
      "vpadd.u32 d6, d6, d8\n"
      "vpadd.u32 d7, d10, d10\n"

      // Add lhs offsets to aggregated rows.
      "vadd.s32 q0, q0, q7\n"
      "vadd.s32 q3, q3, q8\n"

      // Add rhs offset to aggregated rows.
      "vadd.s32 q0, q0, q9\n"
      "vadd.s32 q3, q3, q9\n"

      // Store reduced rows.
      "vst1.32 {d0}, [%[result]]!\n"
      "vst1.32 {d1[0]}, [%[result]], %[result_stride]\n"
      "sub %[result], %[result], #8\n"
      "vst1.32 {d6}, [%[result]]!\n"
      "vst1.32 {d7[0]}, [%[result]], %[result_stride]\n"
      "sub %[result], %[result], #8\n"
      : [count] "+r"(count), [result_stride] "+r"(result_stride),
        [rhs] "+r"(rhs), [lhs] "+r"(lhs), [result] "+r"(result)
      :
      : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10",
        "d11", "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19", "d20",
        "d21", "d22", "d23", "d24", "d25", "d26", "d27", "d28", "d29", "cc",
        "memory");
}

inline void mul_3x8_1x8_int32_lhsadd_rhsadd(const std::uint8_t* lhs,
                                            const std::uint8_t* rhs,
                                            std::int32_t count,
                                            std::int32_t* result,
                                            std::int32_t result_stride) {
  asm volatile(
      "pld [%[lhs]]\n"
      "pld [%[rhs]]\n"
      // Clear aggregators.
      "vmov.i32 q0, #0\n"
      "vmov.i32 q1, #0\n"
      "vmov.i32 q2, #0\n"

      // General NxM lanes loop.
      "1:"

      // Subtract counter.
      "subs %[count], %[count], #8\n"

      "vld1.8 {d6, d7, d8}, [%[lhs]:64]!\n"
      "vld1.8 {d9}, [%[rhs]:64]!\n"
      "pld [%[lhs], #64]\n"
      "pld [%[rhs], #64]\n"
      "vmull.u8 q5, d9, d6\n"
      "vmull.u8 q6, d9, d7\n"
      "vmull.u8 q7, d9, d8\n"
      "vpadal.u16 q0, q5\n"
      "vpadal.u16 q1, q6\n"
      "vpadal.u16 q2, q7\n"

      // Loop break.
      "bne 1b\n"

      "vld1.32 {d8, d9}, [%[lhs]:64]\n"
      "vdup.32 d10, d8[0]\n"
      "vdup.32 d11, d8[1]\n"
      "vdup.32 d12, d9[0]\n"
      "vld1.32 {d13}, [%[rhs]:64]\n"

      // Reduce rows.
      "vpadd.u32 d0, d0, d1\n"
      "vpadd.u32 d6, d0, d0\n"
      "vpadd.u32 d2, d2, d3\n"
      "vpadd.u32 d7, d2, d2\n"
      "vpadd.u32 d4, d4, d5\n"
      "vpadd.u32 d14, d4, d4\n"

      // Add lhs offsets to aggregated rows.
      "vadd.s32 d6, d6, d10\n"
      "vadd.s32 d7, d7, d11\n"
      "vadd.s32 d14, d14, d12\n"

      // Add rhs offset to aggregated rows.
      "vadd.s32 d6, d6, d13\n"
      "vadd.s32 d7, d7, d13\n"
      "vadd.s32 d14, d14, d13\n"

      // Store reduced rows.
      "vst1.32 {d6[0]}, [%[result]], %[result_stride]\n"
      "vst1.32 {d7[0]}, [%[result]], %[result_stride]\n"
      "vst1.32 {d14[0]}, [%[result]], %[result_stride]\n"
      : [count] "+r"(count), [result_stride] "+r"(result_stride),
        [rhs] "+r"(rhs), [lhs] "+r"(lhs), [result] "+r"(result)
      :
      : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10",
        "d11", "d12", "d13", "d14", "d15", "cc", "memory");
}

inline void mul_3x8_2x8_int32_lhsadd_rhsadd(const std::uint8_t* lhs,
                                            const std::uint8_t* rhs,
                                            std::int32_t count,
                                            std::int32_t* result,
                                            std::int32_t result_stride) {
  asm volatile(
      "pld [%[lhs]]\n"
      "pld [%[rhs]]\n"
      // Clear aggregators.
      "vmov.i32 q0, #0\n"
      "vmov.i32 q1, #0\n"
      "vmov.i32 q2, #0\n"
      "vmov.i32 q3, q0\n"
      "vmov.i32 q4, q1\n"
      "vmov.i32 q5, q2\n"

      // General NxM lanes loop.
      "1:"

      // Subtract counter.
      "subs %[count], %[count], #8\n"

      "vld1.8 {d12, d13, d14}, [%[lhs]:64]!\n"
      "vld1.8 {d15, d16}, [%[rhs]:64]!\n"
      "pld [%[lhs], #64]\n"
      "pld [%[rhs], #64]\n"
      "vmull.u8 q9, d15, d12\n"
      "vmull.u8 q10, d16, d12\n"
      "vmull.u8 q11, d15, d13\n"
      "vmull.u8 q12, d16, d13\n"
      "vmull.u8 q13, d15, d14\n"
      "vmull.u8 q14, d16, d14\n"
      "vpadal.u16 q0, q9\n"
      "vpadal.u16 q1, q10\n"
      "vpadal.u16 q2, q11\n"
      "vpadal.u16 q3, q12\n"
      "vpadal.u16 q4, q13\n"
      "vpadal.u16 q5, q14\n"

      // Loop break.
      "bne 1b\n"

      "vld1.32 {d12, d13}, [%[lhs]:64]\n"
      "vdup.32 d14, d12[0]\n"
      "vdup.32 d15, d12[1]\n"
      "vdup.32 d16, d13[0]\n"
      "vld1.32 {d17}, [%[rhs]:64]\n"

      // Reduce rows.
      "vpadd.u32 d0, d0, d1\n"
      "vpadd.u32 d2, d2, d3\n"
      "vpadd.u32 d18, d0, d2\n"
      "vpadd.u32 d4, d4, d5\n"
      "vpadd.u32 d6, d6, d7\n"
      "vpadd.u32 d19, d4, d6\n"
      "vpadd.u32 d8, d8, d9\n"
      "vpadd.u32 d10, d10, d11\n"
      "vpadd.u32 d20, d8, d10\n"

      // Add lhs offsets to aggregated rows.
      "vadd.s32 d18, d18, d14\n"
      "vadd.s32 d19, d19, d15\n"
      "vadd.s32 d20, d20, d16\n"

      // Add rhs offset to aggregated rows.
      "vadd.s32 d18, d18, d17\n"
      "vadd.s32 d19, d19, d17\n"
      "vadd.s32 d20, d20, d17\n"

      // Store reduced rows.
      "vst1.32 {d18}, [%[result]], %[result_stride]\n"
      "vst1.32 {d19}, [%[result]], %[result_stride]\n"
      "vst1.32 {d20}, [%[result]], %[result_stride]\n"
      : [count] "+r"(count), [result_stride] "+r"(result_stride),
        [rhs] "+r"(rhs), [lhs] "+r"(lhs), [result] "+r"(result)
      :
      : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10",
        "d11", "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19", "d20",
        "d21", "d22", "d23", "d24", "d25", "d26", "d27", "d28", "d29", "cc",
        "memory");
}

inline void mul_3x8_3x8_int32_lhsadd_rhsadd(const std::uint8_t* lhs,
                                            const std::uint8_t* rhs,
                                            std::int32_t count,
                                            std::int32_t* result,
                                            std::int32_t result_stride) {
  asm volatile(
      "pld [%[lhs]]\n"
      "pld [%[rhs]]\n"
      // Clear aggregators.
      "vmov.i32 q0, #0\n"
      "vmov.i32 q1, #0\n"
      "vmov.i32 q2, #0\n"
      "vmov.i32 q3, q0\n"
      "vmov.i32 q4, q1\n"
      "vmov.i32 q5, q2\n"
      "vmov.i32 q6, q3\n"
      "vmov.i32 q7, q4\n"
      "vmov.i32 q8, q5\n"

      // 3x3 lanes loop.
      "1:"

      "vld1.8 {d21, d22, d23}, [%[rhs]:64]!\n"
      "vld1.8 {d18}, [%[lhs]:64]!\n"
      "vmull.u8 q12, d18, d21\n"
      "vld1.8 {d19}, [%[lhs]:64]!\n"
      "vmull.u8 q13, d18, d22\n"
      "vld1.8 {d20}, [%[lhs]:64]!\n"
      "vmull.u8 q14, d18, d23\n"
      "pld [%[lhs], #64]\n"
      "vmull.u8 q15, d19, d21\n"
      "pld [%[rhs], #64]\n"
      "vpadal.u16 q0, q12\n"
      "vpadal.u16 q1, q13\n"
      "vpadal.u16 q2, q14\n"
      "vpadal.u16 q3, q15\n"
      "vmull.u8 q12, d19, d22\n"
      "vmull.u8 q13, d19, d23\n"
      "vmull.u8 q14, d20, d21\n"
      "vmull.u8 q15, d20, d22\n"

      // Subtract counter.
      "subs %[count], %[count], #8\n"

      "vmull.u8 q9, d20, d23\n"
      "vpadal.u16 q4, q12\n"
      "vpadal.u16 q5, q13\n"
      "vpadal.u16 q6, q14\n"
      "vpadal.u16 q7, q15\n"
      "vpadal.u16 q8, q9\n"

      // Loop break.
      "bne 1b\n"

      "vld1.32 {d18, d19}, [%[lhs]:64]\n"
      "vdup.32 q10, d18[0]\n"
      "vdup.32 q11, d18[1]\n"
      "vdup.32 q12, d19[0]\n"
      "vld1.32 {d26, d27}, [%[rhs]:64]\n"

      // Reduce rows.
      "vpadd.u32 d0, d0, d1\n"
      "vpadd.u32 d2, d2, d3\n"
      "vpadd.u32 d4, d4, d5\n"
      "vpadd.u32 d0, d0, d2\n"
      "vpadd.u32 d1, d4, d4\n"
      "vpadd.u32 d6, d6, d7\n"
      "vpadd.u32 d8, d8, d9\n"
      "vpadd.u32 d10, d10, d11\n"
      "vpadd.u32 d6, d6, d8\n"
      "vpadd.u32 d7, d10, d10\n"
      "vpadd.u32 d12, d12, d13\n"
      "vpadd.u32 d14, d14, d15\n"
      "vpadd.u32 d16, d16, d17\n"
      "vpadd.u32 d12, d12, d14\n"
      "vpadd.u32 d13, d16, d16\n"

      // Add lhs offsets to aggregated rows.
      "vadd.s32 q0, q0, q10\n"
      "vadd.s32 q3, q3, q11\n"
      "vadd.s32 q6, q6, q12\n"

      // Add rhs offset to aggregated rows.
      "vadd.s32 q0, q0, q13\n"
      "vadd.s32 q3, q3, q13\n"
      "vadd.s32 q6, q6, q13\n"

      // Store reduced rows.
      "vst1.32 {d0}, [%[result]]!\n"
      "vst1.32 {d1[0]}, [%[result]], %[result_stride]\n"
      "sub %[result], %[result], #8\n"
      "vst1.32 {d6}, [%[result]]!\n"
      "vst1.32 {d7[0]}, [%[result]], %[result_stride]\n"
      "sub %[result], %[result], #8\n"
      "vst1.32 {d12}, [%[result]]!\n"
      "vst1.32 {d13[0]}, [%[result]], %[result_stride]\n"
      "sub %[result], %[result], #8\n"
      : [count] "+r"(count), [result_stride] "+r"(result_stride),
        [rhs] "+r"(rhs), [lhs] "+r"(lhs), [result] "+r"(result)
      :
      : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10",
        "d11", "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19", "d20",
        "d21", "d22", "d23", "d24", "d25", "d26", "d27", "d28", "d29", "d30",
        "d31", "cc", "memory");
}

inline void mul_1x8_4x8_int32_lhsadd_rhsadd(const std::uint8_t* lhs,
                                            const std::uint8_t* rhs,
                                            std::int32_t count,
                                            std::int32_t* result,
                                            std::int32_t result_stride) {
  asm volatile(
      "pld [%[lhs]]\n"
      "pld [%[rhs]]\n"
      // Clear aggregators.
      "vmov.i32 q0, #0\n"
      "vmov.i32 q1, #0\n"
      "vmov.i32 q2, #0\n"
      "vmov.i32 q3, q0\n"

      // General NxM lanes loop.
      "1:"

      // Subtract counter.
      "subs %[count], %[count], #8\n"

      "vld1.8 {d8}, [%[lhs]:64]!\n"
      "vld1.8 {d9, d10, d11, d12}, [%[rhs]:64]!\n"
      "pld [%[lhs], #64]\n"
      "pld [%[rhs], #64]\n"
      "vmull.u8 q7, d9, d8\n"
      "vmull.u8 q8, d10, d8\n"
      "vmull.u8 q9, d11, d8\n"
      "vmull.u8 q10, d12, d8\n"
      "vpadal.u16 q0, q7\n"
      "vpadal.u16 q1, q8\n"
      "vpadal.u16 q2, q9\n"
      "vpadal.u16 q3, q10\n"

      // Loop break.
      "bne 1b\n"

      "vld1.32 {d8}, [%[lhs]:64]\n"
      "vdup.32 q5, d8[0]\n"
      "vld1.32 {d12, d13}, [%[rhs]:64]\n"

      // Reduce rows.
      "vpadd.u32 d0, d0, d1\n"
      "vpadd.u32 d2, d2, d3\n"
      "vpadd.u32 d4, d4, d5\n"
      "vpadd.u32 d6, d6, d7\n"
      "vpadd.u32 d0, d0, d2\n"
      "vpadd.u32 d1, d4, d6\n"

      // Add lhs offsets to aggregated rows.
      "vadd.s32 q0, q0, q5\n"

      // Add rhs offset to aggregated rows.
      "vadd.s32 q0, q0, q6\n"

      // Store reduced rows.
      "vst1.32 {d0, d1}, [%[result]], %[result_stride]\n"
      : [count] "+r"(count), [result_stride] "+r"(result_stride),
        [rhs] "+r"(rhs), [lhs] "+r"(lhs), [result] "+r"(result)
      :
      : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10",
        "d11", "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19", "d20",
        "d21", "cc", "memory");
}

inline void mul_1x8_1x8_float_lhsadd_rhsadd(const std::uint8_t* lhs,
                                            const std::uint8_t* rhs,
                                            std::int32_t count, float* result,
                                            std::int32_t result_stride,
                                            float result_scale) {
  asm volatile(
      "pld [%[lhs]]\n"
      "pld [%[rhs]]\n"
      // Clear aggregators.
      "vmov.i32 q0, #0\n"

      // General NxM lanes loop.
      "1:"

      // Subtract counter.
      "subs %[count], %[count], #8\n"

      "vld1.8 {d2}, [%[lhs]:64]!\n"
      "vld1.8 {d3}, [%[rhs]:64]!\n"
      "pld [%[lhs], #64]\n"
      "pld [%[rhs], #64]\n"
      "vmull.u8 q2, d3, d2\n"
      "vpadal.u16 q0, q2\n"

      // Loop break.
      "bne 1b\n"

      "vld1.32 {d8}, [%[lhs]:64]\n"
      "vdup.32 d9, d8[0]\n"
      "vld1.32 {d10}, [%[rhs]:64]\n"
      "vdup.32 d11, %[result_scale]\n"

      // Reduce rows.
      "vpadd.u32 d0, d0, d1\n"
      "vpadd.u32 d2, d0, d0\n"

      // Add lhs offsets to aggregated rows.
      "vadd.s32 d2, d2, d9\n"

      // Add rhs offset to aggregated rows.
      "vadd.s32 d2, d2, d10\n"

      // Convert to float. Multiply by result scale.
      "vcvt.f32.s32 d2, d2\n"
      "vmul.f32 d2, d2, d11\n"

      // Store reduced rows.
      "vst1.32 {d2[0]}, [%[result]], %[result_stride]\n"
      : [count] "+r"(count), [result_scale] "+r"(result_scale),
        [result_stride] "+r"(result_stride), [rhs] "+r"(rhs), [lhs] "+r"(lhs),
        [result] "+r"(result)
      :
      : "d0", "d1", "d2", "d3", "d4", "d5", "d8", "d9", "d10", "d11", "cc",
        "memory");
}

inline void mul_1x8_2x8_float_lhsadd_rhsadd(const std::uint8_t* lhs,
                                            const std::uint8_t* rhs,
                                            std::int32_t count, float* result,
                                            std::int32_t result_stride,
                                            float result_scale) {
  asm volatile(
      "pld [%[lhs]]\n"
      "pld [%[rhs]]\n"
      // Clear aggregators.
      "vmov.i32 q0, #0\n"
      "vmov.i32 q1, #0\n"

      // General NxM lanes loop.
      "1:"

      // Subtract counter.
      "subs %[count], %[count], #8\n"

      "vld1.8 {d4}, [%[lhs]:64]!\n"
      "vld1.8 {d5, d6}, [%[rhs]:64]!\n"
      "pld [%[lhs], #64]\n"
      "pld [%[rhs], #64]\n"
      "vmull.u8 q4, d5, d4\n"
      "vmull.u8 q5, d6, d4\n"
      "vpadal.u16 q0, q4\n"
      "vpadal.u16 q1, q5\n"

      // Loop break.
      "bne 1b\n"

      "vld1.32 {d8}, [%[lhs]:64]\n"
      "vdup.32 d9, d8[0]\n"
      "vld1.32 {d10}, [%[rhs]:64]\n"
      "vdup.32 d11, %[result_scale]\n"

      // Reduce rows.
      "vpadd.u32 d0, d0, d1\n"
      "vpadd.u32 d2, d2, d3\n"
      "vpadd.u32 d4, d0, d2\n"

      // Add lhs offsets to aggregated rows.
      "vadd.s32 d4, d4, d9\n"

      // Add rhs offset to aggregated rows.
      "vadd.s32 d4, d4, d10\n"

      // Convert to float. Multiply by result scale.
      "vcvt.f32.s32 d4, d4\n"
      "vmul.f32 d4, d4, d11\n"

      // Store reduced rows.
      "vst1.32 {d4}, [%[result]], %[result_stride]\n"
      : [count] "+r"(count), [result_scale] "+r"(result_scale),
        [result_stride] "+r"(result_stride), [rhs] "+r"(rhs), [lhs] "+r"(lhs),
        [result] "+r"(result)
      :
      : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d8", "d9", "d10", "d11",
        "cc", "memory");
}

inline void mul_1x8_3x8_float_lhsadd_rhsadd(const std::uint8_t* lhs,
                                            const std::uint8_t* rhs,
                                            std::int32_t count, float* result,
                                            std::int32_t result_stride,
                                            float result_scale) {
  asm volatile(
      "pld [%[lhs]]\n"
      "pld [%[rhs]]\n"
      // Clear aggregators.
      "vmov.i32 q0, #0\n"
      "vmov.i32 q1, #0\n"
      "vmov.i32 q2, #0\n"

      // General NxM lanes loop.
      "1:"

      // Subtract counter.
      "subs %[count], %[count], #8\n"

      "vld1.8 {d6}, [%[lhs]:64]!\n"
      "vld1.8 {d7, d8, d9}, [%[rhs]:64]!\n"
      "pld [%[lhs], #64]\n"
      "pld [%[rhs], #64]\n"
      "vmull.u8 q5, d7, d6\n"
      "vmull.u8 q6, d8, d6\n"
      "vmull.u8 q7, d9, d6\n"
      "vpadal.u16 q0, q5\n"
      "vpadal.u16 q1, q6\n"
      "vpadal.u16 q2, q7\n"

      // Loop break.
      "bne 1b\n"

      "vld1.32 {d8}, [%[lhs]:64]\n"
      "vdup.32 q5, d8[0]\n"
      "vld1.32 {d12, d13}, [%[rhs]:64]\n"
      "vdup.32 q7, %[result_scale]\n"

      // Reduce rows.
      "vpadd.u32 d0, d0, d1\n"
      "vpadd.u32 d2, d2, d3\n"
      "vpadd.u32 d4, d4, d5\n"
      "vpadd.u32 d0, d0, d2\n"
      "vpadd.u32 d1, d4, d4\n"

      // Add lhs offsets to aggregated rows.
      "vadd.s32 q0, q0, q5\n"

      // Add rhs offset to aggregated rows.
      "vadd.s32 q0, q0, q6\n"

      // Convert to float. Multiply by result scale.
      "vcvt.f32.s32 q0, q0\n"
      "vmul.f32 q0, q0, q7\n"

      // Store reduced rows.
      "vst1.32 {d0}, [%[result]]!\n"
      "vst1.32 {d1[0]}, [%[result]], %[result_stride]\n"
      "sub %[result], %[result], #8\n"
      : [count] "+r"(count), [result_scale] "+r"(result_scale),
        [result_stride] "+r"(result_stride), [rhs] "+r"(rhs), [lhs] "+r"(lhs),
        [result] "+r"(result)
      :
      : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10",
        "d11", "d12", "d13", "d14", "d15", "cc", "memory");
}

inline void mul_2x8_1x8_float_lhsadd_rhsadd(const std::uint8_t* lhs,
                                            const std::uint8_t* rhs,
                                            std::int32_t count, float* result,
                                            std::int32_t result_stride,
                                            float result_scale) {
  asm volatile(
      "pld [%[lhs]]\n"
      "pld [%[rhs]]\n"
      // Clear aggregators.
      "vmov.i32 q0, #0\n"
      "vmov.i32 q1, #0\n"

      // General NxM lanes loop.
      "1:"

      // Subtract counter.
      "subs %[count], %[count], #8\n"

      "vld1.8 {d4, d5}, [%[lhs]:64]!\n"
      "vld1.8 {d6}, [%[rhs]:64]!\n"
      "pld [%[lhs], #64]\n"
      "pld [%[rhs], #64]\n"
      "vmull.u8 q4, d6, d4\n"
      "vmull.u8 q5, d6, d5\n"
      "vpadal.u16 q0, q4\n"
      "vpadal.u16 q1, q5\n"

      // Loop break.
      "bne 1b\n"

      "vld1.32 {d8}, [%[lhs]:64]\n"
      "vdup.32 d9, d8[0]\n"
      "vdup.32 d10, d8[1]\n"
      "vld1.32 {d11}, [%[rhs]:64]\n"
      "vdup.32 d12, %[result_scale]\n"

      // Reduce rows.
      "vpadd.u32 d0, d0, d1\n"
      "vpadd.u32 d4, d0, d0\n"
      "vpadd.u32 d2, d2, d3\n"
      "vpadd.u32 d5, d2, d2\n"

      // Add lhs offsets to aggregated rows.
      "vadd.s32 d4, d4, d9\n"
      "vadd.s32 d5, d5, d10\n"

      // Add rhs offset to aggregated rows.
      "vadd.s32 d4, d4, d11\n"
      "vadd.s32 d5, d5, d11\n"

      // Convert to float. Multiply by result scale.
      "vcvt.f32.s32 d4, d4\n"
      "vcvt.f32.s32 d5, d5\n"
      "vmul.f32 d4, d4, d12\n"
      "vmul.f32 d5, d5, d12\n"

      // Store reduced rows.
      "vst1.32 {d4[0]}, [%[result]], %[result_stride]\n"
      "vst1.32 {d5[0]}, [%[result]], %[result_stride]\n"
      : [count] "+r"(count), [result_scale] "+r"(result_scale),
        [result_stride] "+r"(result_stride), [rhs] "+r"(rhs), [lhs] "+r"(lhs),
        [result] "+r"(result)
      :
      : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d8", "d9", "d10", "d11",
        "d12", "cc", "memory");
}

inline void mul_2x8_2x8_float_lhsadd_rhsadd(const std::uint8_t* lhs,
                                            const std::uint8_t* rhs,
                                            std::int32_t count, float* result,
                                            std::int32_t result_stride,
                                            float result_scale) {
  asm volatile(
      "pld [%[lhs]]\n"
      "pld [%[rhs]]\n"
      // Clear aggregators.
      "vmov.i32 q0, #0\n"
      "vmov.i32 q1, #0\n"
      "vmov.i32 q2, #0\n"
      "vmov.i32 q3, q0\n"

      // General NxM lanes loop.
      "1:"

      // Subtract counter.
      "subs %[count], %[count], #8\n"

      "vld1.8 {d8, d9}, [%[lhs]:64]!\n"
      "vld1.8 {d10, d11}, [%[rhs]:64]!\n"
      "pld [%[lhs], #64]\n"
      "pld [%[rhs], #64]\n"
      "vmull.u8 q6, d10, d8\n"
      "vmull.u8 q7, d11, d8\n"
      "vmull.u8 q8, d10, d9\n"
      "vmull.u8 q9, d11, d9\n"
      "vpadal.u16 q0, q6\n"
      "vpadal.u16 q1, q7\n"
      "vpadal.u16 q2, q8\n"
      "vpadal.u16 q3, q9\n"

      // Loop break.
      "bne 1b\n"

      "vld1.32 {d8}, [%[lhs]:64]\n"
      "vdup.32 d9, d8[0]\n"
      "vdup.32 d10, d8[1]\n"
      "vld1.32 {d11}, [%[rhs]:64]\n"
      "vdup.32 d12, %[result_scale]\n"

      // Reduce rows.
      "vpadd.u32 d0, d0, d1\n"
      "vpadd.u32 d2, d2, d3\n"
      "vpadd.u32 d13, d0, d2\n"
      "vpadd.u32 d4, d4, d5\n"
      "vpadd.u32 d6, d6, d7\n"
      "vpadd.u32 d14, d4, d6\n"

      // Add lhs offsets to aggregated rows.
      "vadd.s32 d13, d13, d9\n"
      "vadd.s32 d14, d14, d10\n"

      // Add rhs offset to aggregated rows.
      "vadd.s32 d13, d13, d11\n"
      "vadd.s32 d14, d14, d11\n"

      // Convert to float. Multiply by result scale.
      "vcvt.f32.s32 d13, d13\n"
      "vcvt.f32.s32 d14, d14\n"
      "vmul.f32 d13, d13, d12\n"
      "vmul.f32 d14, d14, d12\n"

      // Store reduced rows.
      "vst1.32 {d13}, [%[result]], %[result_stride]\n"
      "vst1.32 {d14}, [%[result]], %[result_stride]\n"
      : [count] "+r"(count), [result_scale] "+r"(result_scale),
        [result_stride] "+r"(result_stride), [rhs] "+r"(rhs), [lhs] "+r"(lhs),
        [result] "+r"(result)
      :
      : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10",
        "d11", "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19", "cc",
        "memory");
}

inline void mul_2x8_3x8_float_lhsadd_rhsadd(const std::uint8_t* lhs,
                                            const std::uint8_t* rhs,
                                            std::int32_t count, float* result,
                                            std::int32_t result_stride,
                                            float result_scale) {
  asm volatile(
      "pld [%[lhs]]\n"
      "pld [%[rhs]]\n"
      // Clear aggregators.
      "vmov.i32 q0, #0\n"
      "vmov.i32 q1, #0\n"
      "vmov.i32 q2, #0\n"
      "vmov.i32 q3, q0\n"
      "vmov.i32 q4, q1\n"
      "vmov.i32 q5, q2\n"

      // General NxM lanes loop.
      "1:"

      // Subtract counter.
      "subs %[count], %[count], #8\n"

      "vld1.8 {d12, d13}, [%[lhs]:64]!\n"
      "vld1.8 {d14, d15, d16}, [%[rhs]:64]!\n"
      "pld [%[lhs], #64]\n"
      "pld [%[rhs], #64]\n"
      "vmull.u8 q9, d14, d12\n"
      "vmull.u8 q10, d15, d12\n"
      "vmull.u8 q11, d16, d12\n"
      "vmull.u8 q12, d14, d13\n"
      "vmull.u8 q13, d15, d13\n"
      "vmull.u8 q14, d16, d13\n"
      "vpadal.u16 q0, q9\n"
      "vpadal.u16 q1, q10\n"
      "vpadal.u16 q2, q11\n"
      "vpadal.u16 q3, q12\n"
      "vpadal.u16 q4, q13\n"
      "vpadal.u16 q5, q14\n"

      // Loop break.
      "bne 1b\n"

      "vld1.32 {d12}, [%[lhs]:64]\n"
      "vdup.32 q7, d12[0]\n"
      "vdup.32 q8, d12[1]\n"
      "vld1.32 {d18, d19}, [%[rhs]:64]\n"
      "vdup.32 q10, %[result_scale]\n"

      // Reduce rows.
      "vpadd.u32 d0, d0, d1\n"
      "vpadd.u32 d2, d2, d3\n"
      "vpadd.u32 d4, d4, d5\n"
      "vpadd.u32 d0, d0, d2\n"
      "vpadd.u32 d1, d4, d4\n"
      "vpadd.u32 d6, d6, d7\n"
      "vpadd.u32 d8, d8, d9\n"
      "vpadd.u32 d10, d10, d11\n"
      "vpadd.u32 d6, d6, d8\n"
      "vpadd.u32 d7, d10, d10\n"

      // Add lhs offsets to aggregated rows.
      "vadd.s32 q0, q0, q7\n"
      "vadd.s32 q3, q3, q8\n"

      // Add rhs offset to aggregated rows.
      "vadd.s32 q0, q0, q9\n"
      "vadd.s32 q3, q3, q9\n"

      // Convert to float. Multiply by result scale.
      "vcvt.f32.s32 q0, q0\n"
      "vcvt.f32.s32 q3, q3\n"
      "vmul.f32 q0, q0, q10\n"
      "vmul.f32 q3, q3, q10\n"

      // Store reduced rows.
      "vst1.32 {d0}, [%[result]]!\n"
      "vst1.32 {d1[0]}, [%[result]], %[result_stride]\n"
      "sub %[result], %[result], #8\n"
      "vst1.32 {d6}, [%[result]]!\n"
      "vst1.32 {d7[0]}, [%[result]], %[result_stride]\n"
      "sub %[result], %[result], #8\n"
      : [count] "+r"(count), [result_scale] "+r"(result_scale),
        [result_stride] "+r"(result_stride), [rhs] "+r"(rhs), [lhs] "+r"(lhs),
        [result] "+r"(result)
      :
      : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10",
        "d11", "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19", "d20",
        "d21", "d22", "d23", "d24", "d25", "d26", "d27", "d28", "d29", "cc",
        "memory");
}

inline void mul_3x8_1x8_float_lhsadd_rhsadd(const std::uint8_t* lhs,
                                            const std::uint8_t* rhs,
                                            std::int32_t count, float* result,
                                            std::int32_t result_stride,
                                            float result_scale) {
  asm volatile(
      "pld [%[lhs]]\n"
      "pld [%[rhs]]\n"
      // Clear aggregators.
      "vmov.i32 q0, #0\n"
      "vmov.i32 q1, #0\n"
      "vmov.i32 q2, #0\n"

      // General NxM lanes loop.
      "1:"

      // Subtract counter.
      "subs %[count], %[count], #8\n"

      "vld1.8 {d6, d7, d8}, [%[lhs]:64]!\n"
      "vld1.8 {d9}, [%[rhs]:64]!\n"
      "pld [%[lhs], #64]\n"
      "pld [%[rhs], #64]\n"
      "vmull.u8 q5, d9, d6\n"
      "vmull.u8 q6, d9, d7\n"
      "vmull.u8 q7, d9, d8\n"
      "vpadal.u16 q0, q5\n"
      "vpadal.u16 q1, q6\n"
      "vpadal.u16 q2, q7\n"

      // Loop break.
      "bne 1b\n"

      "vld1.32 {d8, d9}, [%[lhs]:64]\n"
      "vdup.32 d10, d8[0]\n"
      "vdup.32 d11, d8[1]\n"
      "vdup.32 d12, d9[0]\n"
      "vld1.32 {d13}, [%[rhs]:64]\n"
      "vdup.32 d14, %[result_scale]\n"

      // Reduce rows.
      "vpadd.u32 d0, d0, d1\n"
      "vpadd.u32 d6, d0, d0\n"
      "vpadd.u32 d2, d2, d3\n"
      "vpadd.u32 d7, d2, d2\n"
      "vpadd.u32 d4, d4, d5\n"
      "vpadd.u32 d15, d4, d4\n"

      // Add lhs offsets to aggregated rows.
      "vadd.s32 d6, d6, d10\n"
      "vadd.s32 d7, d7, d11\n"
      "vadd.s32 d15, d15, d12\n"

      // Add rhs offset to aggregated rows.
      "vadd.s32 d6, d6, d13\n"
      "vadd.s32 d7, d7, d13\n"
      "vadd.s32 d15, d15, d13\n"

      // Convert to float. Multiply by result scale.
      "vcvt.f32.s32 d6, d6\n"
      "vcvt.f32.s32 d7, d7\n"
      "vcvt.f32.s32 d15, d15\n"
      "vmul.f32 d6, d6, d14\n"
      "vmul.f32 d7, d7, d14\n"
      "vmul.f32 d15, d15, d14\n"

      // Store reduced rows.
      "vst1.32 {d6[0]}, [%[result]], %[result_stride]\n"
      "vst1.32 {d7[0]}, [%[result]], %[result_stride]\n"
      "vst1.32 {d15[0]}, [%[result]], %[result_stride]\n"
      : [count] "+r"(count), [result_scale] "+r"(result_scale),
        [result_stride] "+r"(result_stride), [rhs] "+r"(rhs), [lhs] "+r"(lhs),
        [result] "+r"(result)
      :
      : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10",
        "d11", "d12", "d13", "d14", "d15", "cc", "memory");
}

inline void mul_3x8_2x8_float_lhsadd_rhsadd(const std::uint8_t* lhs,
                                            const std::uint8_t* rhs,
                                            std::int32_t count, float* result,
                                            std::int32_t result_stride,
                                            float result_scale) {
  asm volatile(
      "pld [%[lhs]]\n"
      "pld [%[rhs]]\n"
      // Clear aggregators.
      "vmov.i32 q0, #0\n"
      "vmov.i32 q1, #0\n"
      "vmov.i32 q2, #0\n"
      "vmov.i32 q3, q0\n"
      "vmov.i32 q4, q1\n"
      "vmov.i32 q5, q2\n"

      // General NxM lanes loop.
      "1:"

      // Subtract counter.
      "subs %[count], %[count], #8\n"

      "vld1.8 {d12, d13, d14}, [%[lhs]:64]!\n"
      "vld1.8 {d15, d16}, [%[rhs]:64]!\n"
      "pld [%[lhs], #64]\n"
      "pld [%[rhs], #64]\n"
      "vmull.u8 q9, d15, d12\n"
      "vmull.u8 q10, d16, d12\n"
      "vmull.u8 q11, d15, d13\n"
      "vmull.u8 q12, d16, d13\n"
      "vmull.u8 q13, d15, d14\n"
      "vmull.u8 q14, d16, d14\n"
      "vpadal.u16 q0, q9\n"
      "vpadal.u16 q1, q10\n"
      "vpadal.u16 q2, q11\n"
      "vpadal.u16 q3, q12\n"
      "vpadal.u16 q4, q13\n"
      "vpadal.u16 q5, q14\n"

      // Loop break.
      "bne 1b\n"

      "vld1.32 {d12, d13}, [%[lhs]:64]\n"
      "vdup.32 d14, d12[0]\n"
      "vdup.32 d15, d12[1]\n"
      "vdup.32 d16, d13[0]\n"
      "vld1.32 {d17}, [%[rhs]:64]\n"
      "vdup.32 d18, %[result_scale]\n"

      // Reduce rows.
      "vpadd.u32 d0, d0, d1\n"
      "vpadd.u32 d2, d2, d3\n"
      "vpadd.u32 d19, d0, d2\n"
      "vpadd.u32 d4, d4, d5\n"
      "vpadd.u32 d6, d6, d7\n"
      "vpadd.u32 d20, d4, d6\n"
      "vpadd.u32 d8, d8, d9\n"
      "vpadd.u32 d10, d10, d11\n"
      "vpadd.u32 d21, d8, d10\n"

      // Add lhs offsets to aggregated rows.
      "vadd.s32 d19, d19, d14\n"
      "vadd.s32 d20, d20, d15\n"
      "vadd.s32 d21, d21, d16\n"

      // Add rhs offset to aggregated rows.
      "vadd.s32 d19, d19, d17\n"
      "vadd.s32 d20, d20, d17\n"
      "vadd.s32 d21, d21, d17\n"

      // Convert to float. Multiply by result scale.
      "vcvt.f32.s32 d19, d19\n"
      "vcvt.f32.s32 d20, d20\n"
      "vcvt.f32.s32 d21, d21\n"
      "vmul.f32 d19, d19, d18\n"
      "vmul.f32 d20, d20, d18\n"
      "vmul.f32 d21, d21, d18\n"

      // Store reduced rows.
      "vst1.32 {d19}, [%[result]], %[result_stride]\n"
      "vst1.32 {d20}, [%[result]], %[result_stride]\n"
      "vst1.32 {d21}, [%[result]], %[result_stride]\n"
      : [count] "+r"(count), [result_scale] "+r"(result_scale),
        [result_stride] "+r"(result_stride), [rhs] "+r"(rhs), [lhs] "+r"(lhs),
        [result] "+r"(result)
      :
      : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10",
        "d11", "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19", "d20",
        "d21", "d22", "d23", "d24", "d25", "d26", "d27", "d28", "d29", "cc",
        "memory");
}

inline void mul_3x8_3x8_float_lhsadd_rhsadd(const std::uint8_t* lhs,
                                            const std::uint8_t* rhs,
                                            std::int32_t count, float* result,
                                            std::int32_t result_stride,
                                            float result_scale) {
  asm volatile(
      "pld [%[lhs]]\n"
      "pld [%[rhs]]\n"
      // Clear aggregators.
      "vmov.i32 q0, #0\n"
      "vmov.i32 q1, #0\n"
      "vmov.i32 q2, #0\n"
      "vmov.i32 q3, q0\n"
      "vmov.i32 q4, q1\n"
      "vmov.i32 q5, q2\n"
      "vmov.i32 q6, q3\n"
      "vmov.i32 q7, q4\n"
      "vmov.i32 q8, q5\n"

      // 3x3 lanes loop.
      "1:"

      "vld1.8 {d21, d22, d23}, [%[rhs]:64]!\n"
      "vld1.8 {d18}, [%[lhs]:64]!\n"
      "vmull.u8 q12, d18, d21\n"
      "vld1.8 {d19}, [%[lhs]:64]!\n"
      "vmull.u8 q13, d18, d22\n"
      "vld1.8 {d20}, [%[lhs]:64]!\n"
      "vmull.u8 q14, d18, d23\n"
      "pld [%[lhs], #64]\n"
      "vmull.u8 q15, d19, d21\n"
      "pld [%[rhs], #64]\n"
      "vpadal.u16 q0, q12\n"
      "vpadal.u16 q1, q13\n"
      "vpadal.u16 q2, q14\n"
      "vpadal.u16 q3, q15\n"
      "vmull.u8 q12, d19, d22\n"
      "vmull.u8 q13, d19, d23\n"
      "vmull.u8 q14, d20, d21\n"
      "vmull.u8 q15, d20, d22\n"

      // Subtract counter.
      "subs %[count], %[count], #8\n"

      "vmull.u8 q9, d20, d23\n"
      "vpadal.u16 q4, q12\n"
      "vpadal.u16 q5, q13\n"
      "vpadal.u16 q6, q14\n"
      "vpadal.u16 q7, q15\n"
      "vpadal.u16 q8, q9\n"

      // Loop break.
      "bne 1b\n"

      "vld1.32 {d18, d19}, [%[lhs]:64]\n"
      "vdup.32 q10, d18[0]\n"
      "vdup.32 q11, d18[1]\n"
      "vdup.32 q12, d19[0]\n"
      "vld1.32 {d26, d27}, [%[rhs]:64]\n"
      "vdup.32 q14, %[result_scale]\n"

      // Reduce rows.
      "vpadd.u32 d0, d0, d1\n"
      "vpadd.u32 d2, d2, d3\n"
      "vpadd.u32 d4, d4, d5\n"
      "vpadd.u32 d0, d0, d2\n"
      "vpadd.u32 d1, d4, d4\n"
      "vpadd.u32 d6, d6, d7\n"
      "vpadd.u32 d8, d8, d9\n"
      "vpadd.u32 d10, d10, d11\n"
      "vpadd.u32 d6, d6, d8\n"
      "vpadd.u32 d7, d10, d10\n"
      "vpadd.u32 d12, d12, d13\n"
      "vpadd.u32 d14, d14, d15\n"
      "vpadd.u32 d16, d16, d17\n"
      "vpadd.u32 d12, d12, d14\n"
      "vpadd.u32 d13, d16, d16\n"

      // Add lhs offsets to aggregated rows.
      "vadd.s32 q0, q0, q10\n"
      "vadd.s32 q3, q3, q11\n"
      "vadd.s32 q6, q6, q12\n"

      // Add rhs offset to aggregated rows.
      "vadd.s32 q0, q0, q13\n"
      "vadd.s32 q3, q3, q13\n"
      "vadd.s32 q6, q6, q13\n"

      // Convert to float. Multiply by result scale.
      "vcvt.f32.s32 q0, q0\n"
      "vcvt.f32.s32 q3, q3\n"
      "vcvt.f32.s32 q6, q6\n"
      "vmul.f32 q0, q0, q14\n"
      "vmul.f32 q3, q3, q14\n"
      "vmul.f32 q6, q6, q14\n"

      // Store reduced rows.
      "vst1.32 {d0}, [%[result]]!\n"
      "vst1.32 {d1[0]}, [%[result]], %[result_stride]\n"
      "sub %[result], %[result], #8\n"
      "vst1.32 {d6}, [%[result]]!\n"
      "vst1.32 {d7[0]}, [%[result]], %[result_stride]\n"
      "sub %[result], %[result], #8\n"
      "vst1.32 {d12}, [%[result]]!\n"
      "vst1.32 {d13[0]}, [%[result]], %[result_stride]\n"
      "sub %[result], %[result], #8\n"
      : [count] "+r"(count), [result_scale] "+r"(result_scale),
        [result_stride] "+r"(result_stride), [rhs] "+r"(rhs), [lhs] "+r"(lhs),
        [result] "+r"(result)
      :
      : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10",
        "d11", "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19", "d20",
        "d21", "d22", "d23", "d24", "d25", "d26", "d27", "d28", "d29", "d30",
        "d31", "cc", "memory");
}

inline void mul_1x8_4x8_float_lhsadd_rhsadd(const std::uint8_t* lhs,
                                            const std::uint8_t* rhs,
                                            std::int32_t count, float* result,
                                            std::int32_t result_stride,
                                            float result_scale) {
  asm volatile(
      "pld [%[lhs]]\n"
      "pld [%[rhs]]\n"
      // Clear aggregators.
      "vmov.i32 q0, #0\n"
      "vmov.i32 q1, #0\n"
      "vmov.i32 q2, #0\n"
      "vmov.i32 q3, q0\n"

      // General NxM lanes loop.
      "1:"

      // Subtract counter.
      "subs %[count], %[count], #8\n"

      "vld1.8 {d8}, [%[lhs]:64]!\n"
      "vld1.8 {d9, d10, d11, d12}, [%[rhs]:64]!\n"
      "pld [%[lhs], #64]\n"
      "pld [%[rhs], #64]\n"
      "vmull.u8 q7, d9, d8\n"
      "vmull.u8 q8, d10, d8\n"
      "vmull.u8 q9, d11, d8\n"
      "vmull.u8 q10, d12, d8\n"
      "vpadal.u16 q0, q7\n"
      "vpadal.u16 q1, q8\n"
      "vpadal.u16 q2, q9\n"
      "vpadal.u16 q3, q10\n"

      // Loop break.
      "bne 1b\n"

      "vld1.32 {d8}, [%[lhs]:64]\n"
      "vdup.32 q5, d8[0]\n"
      "vld1.32 {d12, d13}, [%[rhs]:64]\n"
      "vdup.32 q7, %[result_scale]\n"

      // Reduce rows.
      "vpadd.u32 d0, d0, d1\n"
      "vpadd.u32 d2, d2, d3\n"
      "vpadd.u32 d4, d4, d5\n"
      "vpadd.u32 d6, d6, d7\n"
      "vpadd.u32 d0, d0, d2\n"
      "vpadd.u32 d1, d4, d6\n"

      // Add lhs offsets to aggregated rows.
      "vadd.s32 q0, q0, q5\n"

      // Add rhs offset to aggregated rows.
      "vadd.s32 q0, q0, q6\n"

      // Convert to float. Multiply by result scale.
      "vcvt.f32.s32 q0, q0\n"
      "vmul.f32 q0, q0, q7\n"

      // Store reduced rows.
      "vst1.32 {d0, d1}, [%[result]], %[result_stride]\n"
      : [count] "+r"(count), [result_scale] "+r"(result_scale),
        [result_stride] "+r"(result_stride), [rhs] "+r"(rhs), [lhs] "+r"(lhs),
        [result] "+r"(result)
      :
      : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10",
        "d11", "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19", "d20",
        "d21", "cc", "memory");
}

inline void mul_1x8_5x8_int32_rhsadd(const std::uint8_t* lhs,
                                     const std::uint8_t* rhs_1,
                                     const std::uint8_t* rhs_2,
                                     std::int32_t count, std::int32_t* result) {
  asm volatile(
      "pld [%[lhs]]\n"
      "pld [%[rhs_1]]\n"
      "pld [%[rhs_2]]\n"

      // Clear aggregators.
      "vmov.i32 q0, #0\n"
      "vmov.i32 q1, #0\n"
      "vmov.i32 q2, #0\n"
      "vmov.i32 q3, q0\n"
      "vmov.i32 q4, q1\n"

      // General 1xM lanes loop.
      "1:"

      // Subtract counter.
      "subs %[count], %[count], #8\n"

      "vld1.8 {d14}, [%[lhs]:64]!\n"
      "vld1.8 {d10, d11, d12, d13}, [%[rhs_1]:64]!\n"
      "pld [%[lhs], #64]\n"
      "pld [%[rhs_1], #128]\n"
      "vmull.u8 q8, d10, d14\n"
      "vmull.u8 q9, d11, d14\n"
      "vmull.u8 q10, d12, d14\n"
      "vmull.u8 q11, d13, d14\n"
      "vld1.8 {d10}, [%[rhs_2]:64]!\n"
      "pld [%[rhs_2], #32]\n"
      "vpadal.u16 q0, q8\n"
      "vpadal.u16 q1, q9\n"
      "vpadal.u16 q2, q10\n"
      "vpadal.u16 q3, q11\n"
      "vmull.u8 q8, d10, d14\n"
      "vpadal.u16 q4, q8\n"

      // Loop break.
      "bne 1b\n"

      "vld1.32 {d14, d15}, [%[rhs_1]:64]\n"
      "vld1.32 {d13}, [%[rhs_2]:64]\n"

      // Horizontal reduce aggregators.
      "vpadd.u32 d0, d0, d1\n"
      "vpadd.u32 d2, d2, d3\n"
      "vpadd.u32 d4, d4, d5\n"
      "vpadd.u32 d6, d6, d7\n"
      "vpadd.u32 d8, d8, d9\n"
      "vpadd.u32 d10, d0, d2\n"
      "vpadd.u32 d11, d4, d6\n"
      "vpadd.u32 d12, d8, d8\n"

      // Add rhs offset to aggregated rows.
      "vadd.s32 q5, q5, q7\n"
      "vadd.s32 d12, d12, d13\n"

      // Store results.
      "vst1.32 {d10, d11}, [%[result]]!\n"
      "vst1.32 {d12[0]}, [%[result]]!\n"
      : [count] "+r"(count), [rhs_1] "+r"(rhs_1), [rhs_2] "+r"(rhs_2),
        [lhs] "+r"(lhs), [result] "+r"(result)
      :
      : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10",
        "d11", "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19", "d20",
        "d21", "d22", "d23", "cc", "memory");
}

inline void mul_1x8_6x8_int32_rhsadd(const std::uint8_t* lhs,
                                     const std::uint8_t* rhs_1,
                                     const std::uint8_t* rhs_2,
                                     std::int32_t count, std::int32_t* result) {
  asm volatile(
      "pld [%[lhs]]\n"
      "pld [%[rhs_1]]\n"
      "pld [%[rhs_2]]\n"

      // Clear aggregators.
      "vmov.i32 q0, #0\n"
      "vmov.i32 q1, #0\n"
      "vmov.i32 q2, #0\n"
      "vmov.i32 q3, q0\n"
      "vmov.i32 q4, q1\n"
      "vmov.i32 q5, q2\n"

      // General 1xM lanes loop.
      "1:"

      // Subtract counter.
      "subs %[count], %[count], #8\n"

      "vld1.8 {d16}, [%[lhs]:64]!\n"
      "vld1.8 {d12, d13, d14, d15}, [%[rhs_1]:64]!\n"
      "pld [%[lhs], #64]\n"
      "pld [%[rhs_1], #128]\n"
      "vmull.u8 q9, d12, d16\n"
      "vmull.u8 q10, d13, d16\n"
      "vmull.u8 q11, d14, d16\n"
      "vmull.u8 q12, d15, d16\n"
      "vld1.8 {d12, d13}, [%[rhs_2]:64]!\n"
      "pld [%[rhs_2], #64]\n"
      "vpadal.u16 q0, q9\n"
      "vpadal.u16 q1, q10\n"
      "vpadal.u16 q2, q11\n"
      "vpadal.u16 q3, q12\n"
      "vmull.u8 q9, d12, d16\n"
      "vmull.u8 q10, d13, d16\n"
      "vpadal.u16 q4, q9\n"
      "vpadal.u16 q5, q10\n"

      // Loop break.
      "bne 1b\n"

      "vld1.32 {d16, d17}, [%[rhs_1]:64]\n"
      "vld1.32 {d15}, [%[rhs_2]:64]\n"

      // Horizontal reduce aggregators.
      "vpadd.u32 d0, d0, d1\n"
      "vpadd.u32 d2, d2, d3\n"
      "vpadd.u32 d4, d4, d5\n"
      "vpadd.u32 d6, d6, d7\n"
      "vpadd.u32 d8, d8, d9\n"
      "vpadd.u32 d10, d10, d11\n"
      "vpadd.u32 d12, d0, d2\n"
      "vpadd.u32 d13, d4, d6\n"
      "vpadd.u32 d14, d8, d10\n"

      // Add rhs offset to aggregated rows.
      "vadd.s32 q6, q6, q8\n"
      "vadd.s32 d14, d14, d15\n"

      // Store results.
      "vst1.32 {d12, d13}, [%[result]]!\n"
      "vst1.32 {d14}, [%[result]]!\n"
      : [count] "+r"(count), [rhs_1] "+r"(rhs_1), [rhs_2] "+r"(rhs_2),
        [lhs] "+r"(lhs), [result] "+r"(result)
      :
      : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10",
        "d11", "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19", "d20",
        "d21", "d22", "d23", "d24", "d25", "cc", "memory");
}

inline void mul_1x8_7x8_int32_rhsadd(const std::uint8_t* lhs,
                                     const std::uint8_t* rhs_1,
                                     const std::uint8_t* rhs_2,
                                     std::int32_t count, std::int32_t* result) {
  asm volatile(
      "pld [%[lhs]]\n"
      "pld [%[rhs_1]]\n"
      "pld [%[rhs_2]]\n"

      // Clear aggregators.
      "vmov.i32 q0, #0\n"
      "vmov.i32 q1, #0\n"
      "vmov.i32 q2, #0\n"
      "vmov.i32 q3, q0\n"
      "vmov.i32 q4, q1\n"
      "vmov.i32 q5, q2\n"
      "vmov.i32 q6, q3\n"

      // General 1xM lanes loop.
      "1:"

      // Subtract counter.
      "subs %[count], %[count], #8\n"

      "vld1.8 {d18}, [%[lhs]:64]!\n"
      "vld1.8 {d14, d15, d16, d17}, [%[rhs_1]:64]!\n"
      "pld [%[lhs], #64]\n"
      "pld [%[rhs_1], #128]\n"
      "vmull.u8 q10, d14, d18\n"
      "vmull.u8 q11, d15, d18\n"
      "vmull.u8 q12, d16, d18\n"
      "vmull.u8 q13, d17, d18\n"
      "vld1.8 {d14, d15, d16}, [%[rhs_2]:64]!\n"
      "pld [%[rhs_2], #96]\n"
      "vpadal.u16 q0, q10\n"
      "vpadal.u16 q1, q11\n"
      "vpadal.u16 q2, q12\n"
      "vpadal.u16 q3, q13\n"
      "vmull.u8 q10, d14, d18\n"
      "vmull.u8 q11, d15, d18\n"
      "vmull.u8 q12, d16, d18\n"
      "vpadal.u16 q4, q10\n"
      "vpadal.u16 q5, q11\n"
      "vpadal.u16 q6, q12\n"

      // Loop break.
      "bne 1b\n"

      "vld1.32 {d18, d19}, [%[rhs_1]:64]\n"
      "vld1.32 {d20, d21}, [%[rhs_2]:64]\n"

      // Horizontal reduce aggregators.
      "vpadd.u32 d0, d0, d1\n"
      "vpadd.u32 d2, d2, d3\n"
      "vpadd.u32 d4, d4, d5\n"
      "vpadd.u32 d6, d6, d7\n"
      "vpadd.u32 d8, d8, d9\n"
      "vpadd.u32 d10, d10, d11\n"
      "vpadd.u32 d12, d12, d13\n"
      "vpadd.u32 d14, d0, d2\n"
      "vpadd.u32 d15, d4, d6\n"
      "vpadd.u32 d16, d8, d10\n"
      "vpadd.u32 d17, d12, d12\n"

      // Add rhs offset to aggregated rows.
      "vadd.s32 q7, q7, q9\n"
      "vadd.s32 q8, q8, q10\n"

      // Store results.
      "vst1.32 {d14, d15}, [%[result]]!\n"
      "vst1.32 {d16}, [%[result]]!\n"
      "vst1.32 {d17[0]}, [%[result]]!\n"
      : [count] "+r"(count), [rhs_1] "+r"(rhs_1), [rhs_2] "+r"(rhs_2),
        [lhs] "+r"(lhs), [result] "+r"(result)
      :
      : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10",
        "d11", "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19", "d20",
        "d21", "d22", "d23", "d24", "d25", "d26", "d27", "cc", "memory");
}

inline void mul_1x8_8x8_int32_rhsadd(const std::uint8_t* lhs,
                                     const std::uint8_t* rhs_1,
                                     const std::uint8_t* rhs_2,
                                     std::int32_t count, std::int32_t* result) {
  asm volatile(
      "pld [%[lhs]]\n"
      "pld [%[rhs_1]]\n"
      "pld [%[rhs_2]]\n"

      // Clear aggregators.
      "vmov.i32 q0, #0\n"
      "vmov.i32 q1, #0\n"
      "vmov.i32 q2, #0\n"
      "vmov.i32 q3, q0\n"
      "vmov.i32 q4, q1\n"
      "vmov.i32 q5, q2\n"
      "vmov.i32 q6, q3\n"
      "vmov.i32 q7, q4\n"

      // General 1xM lanes loop.
      "1:"

      // Subtract counter.
      "subs %[count], %[count], #8\n"

      "vld1.8 {d20}, [%[lhs]:64]!\n"
      "vld1.8 {d16, d17, d18, d19}, [%[rhs_1]:64]!\n"
      "pld [%[lhs], #64]\n"
      "pld [%[rhs_1], #128]\n"
      "vmull.u8 q11, d16, d20\n"
      "vmull.u8 q12, d17, d20\n"
      "vmull.u8 q13, d18, d20\n"
      "vmull.u8 q14, d19, d20\n"
      "vld1.8 {d16, d17, d18, d19}, [%[rhs_2]:64]!\n"
      "pld [%[rhs_2], #128]\n"
      "vpadal.u16 q0, q11\n"
      "vpadal.u16 q1, q12\n"
      "vpadal.u16 q2, q13\n"
      "vpadal.u16 q3, q14\n"
      "vmull.u8 q11, d16, d20\n"
      "vmull.u8 q12, d17, d20\n"
      "vmull.u8 q13, d18, d20\n"
      "vmull.u8 q14, d19, d20\n"
      "vpadal.u16 q4, q11\n"
      "vpadal.u16 q5, q12\n"
      "vpadal.u16 q6, q13\n"
      "vpadal.u16 q7, q14\n"

      // Loop break.
      "bne 1b\n"

      "vld1.32 {d20, d21}, [%[rhs_1]:64]\n"
      "vld1.32 {d22, d23}, [%[rhs_2]:64]\n"

      // Horizontal reduce aggregators.
      "vpadd.u32 d0, d0, d1\n"
      "vpadd.u32 d2, d2, d3\n"
      "vpadd.u32 d4, d4, d5\n"
      "vpadd.u32 d6, d6, d7\n"
      "vpadd.u32 d8, d8, d9\n"
      "vpadd.u32 d10, d10, d11\n"
      "vpadd.u32 d12, d12, d13\n"
      "vpadd.u32 d14, d14, d15\n"
      "vpadd.u32 d16, d0, d2\n"
      "vpadd.u32 d17, d4, d6\n"
      "vpadd.u32 d18, d8, d10\n"
      "vpadd.u32 d19, d12, d14\n"

      // Add rhs offset to aggregated rows.
      "vadd.s32 q8, q8, q10\n"
      "vadd.s32 q9, q9, q11\n"

      // Store results.
      "vst1.32 {d16, d17}, [%[result]]!\n"
      "vst1.32 {d18, d19}, [%[result]]!\n"
      : [count] "+r"(count), [rhs_1] "+r"(rhs_1), [rhs_2] "+r"(rhs_2),
        [lhs] "+r"(lhs), [result] "+r"(result)
      :
      : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10",
        "d11", "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19", "d20",
        "d21", "d22", "d23", "d24", "d25", "d26", "d27", "d28", "d29", "cc",
        "memory");
}

inline void mul_1x8_5x8_int32_lhsadd_rhsadd(const std::uint8_t* lhs,
                                            const std::uint8_t* rhs_1,
                                            const std::uint8_t* rhs_2,
                                            std::int32_t count,
                                            std::int32_t* result) {
  asm volatile(
      "pld [%[lhs]]\n"
      "pld [%[rhs_1]]\n"
      "pld [%[rhs_2]]\n"

      // Clear aggregators.
      "vmov.i32 q0, #0\n"
      "vmov.i32 q1, #0\n"
      "vmov.i32 q2, #0\n"
      "vmov.i32 q3, q0\n"
      "vmov.i32 q4, q1\n"

      // General 1xM lanes loop.
      "1:"

      // Subtract counter.
      "subs %[count], %[count], #8\n"

      "vld1.8 {d14}, [%[lhs]:64]!\n"
      "vld1.8 {d10, d11, d12, d13}, [%[rhs_1]:64]!\n"
      "pld [%[lhs], #64]\n"
      "pld [%[rhs_1], #128]\n"
      "vmull.u8 q8, d10, d14\n"
      "vmull.u8 q9, d11, d14\n"
      "vmull.u8 q10, d12, d14\n"
      "vmull.u8 q11, d13, d14\n"
      "vld1.8 {d10}, [%[rhs_2]:64]!\n"
      "pld [%[rhs_2], #32]\n"
      "vpadal.u16 q0, q8\n"
      "vpadal.u16 q1, q9\n"
      "vpadal.u16 q2, q10\n"
      "vpadal.u16 q3, q11\n"
      "vmull.u8 q8, d10, d14\n"
      "vpadal.u16 q4, q8\n"

      // Loop break.
      "bne 1b\n"

      "vld1.32 {d14[], d15[]}, [%[lhs]]\n"
      "vld1.32 {d16, d17}, [%[rhs_1]:64]\n"
      "vld1.32 {d13}, [%[rhs_2]:64]\n"

      // Horizontal reduce aggregators.
      "vpadd.u32 d0, d0, d1\n"
      "vpadd.u32 d2, d2, d3\n"
      "vpadd.u32 d4, d4, d5\n"
      "vpadd.u32 d6, d6, d7\n"
      "vpadd.u32 d8, d8, d9\n"
      "vpadd.u32 d10, d0, d2\n"
      "vpadd.u32 d11, d4, d6\n"
      "vpadd.u32 d12, d8, d8\n"

      // Add lhs offsets to aggregated rows.
      "vadd.s32 q5, q5, q7\n"
      "vadd.s32 d12, d12, d14\n"

      // Add rhs offset to aggregated rows.
      "vadd.s32 q5, q5, q8\n"
      "vadd.s32 d12, d12, d13\n"

      // Store results.
      "vst1.32 {d10, d11}, [%[result]]!\n"
      "vst1.32 {d12[0]}, [%[result]]!\n"
      : [count] "+r"(count), [rhs_1] "+r"(rhs_1), [rhs_2] "+r"(rhs_2),
        [lhs] "+r"(lhs), [result] "+r"(result)
      :
      : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10",
        "d11", "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19", "d20",
        "d21", "d22", "d23", "cc", "memory");
}

inline void mul_1x8_6x8_int32_lhsadd_rhsadd(const std::uint8_t* lhs,
                                            const std::uint8_t* rhs_1,
                                            const std::uint8_t* rhs_2,
                                            std::int32_t count,
                                            std::int32_t* result) {
  asm volatile(
      "pld [%[lhs]]\n"
      "pld [%[rhs_1]]\n"
      "pld [%[rhs_2]]\n"

      // Clear aggregators.
      "vmov.i32 q0, #0\n"
      "vmov.i32 q1, #0\n"
      "vmov.i32 q2, #0\n"
      "vmov.i32 q3, q0\n"
      "vmov.i32 q4, q1\n"
      "vmov.i32 q5, q2\n"

      // General 1xM lanes loop.
      "1:"

      // Subtract counter.
      "subs %[count], %[count], #8\n"

      "vld1.8 {d16}, [%[lhs]:64]!\n"
      "vld1.8 {d12, d13, d14, d15}, [%[rhs_1]:64]!\n"
      "pld [%[lhs], #64]\n"
      "pld [%[rhs_1], #128]\n"
      "vmull.u8 q9, d12, d16\n"
      "vmull.u8 q10, d13, d16\n"
      "vmull.u8 q11, d14, d16\n"
      "vmull.u8 q12, d15, d16\n"
      "vld1.8 {d12, d13}, [%[rhs_2]:64]!\n"
      "pld [%[rhs_2], #64]\n"
      "vpadal.u16 q0, q9\n"
      "vpadal.u16 q1, q10\n"
      "vpadal.u16 q2, q11\n"
      "vpadal.u16 q3, q12\n"
      "vmull.u8 q9, d12, d16\n"
      "vmull.u8 q10, d13, d16\n"
      "vpadal.u16 q4, q9\n"
      "vpadal.u16 q5, q10\n"

      // Loop break.
      "bne 1b\n"

      "vld1.32 {d16[], d17[]}, [%[lhs]]\n"
      "vld1.32 {d18, d19}, [%[rhs_1]:64]\n"
      "vld1.32 {d15}, [%[rhs_2]:64]\n"

      // Horizontal reduce aggregators.
      "vpadd.u32 d0, d0, d1\n"
      "vpadd.u32 d2, d2, d3\n"
      "vpadd.u32 d4, d4, d5\n"
      "vpadd.u32 d6, d6, d7\n"
      "vpadd.u32 d8, d8, d9\n"
      "vpadd.u32 d10, d10, d11\n"
      "vpadd.u32 d12, d0, d2\n"
      "vpadd.u32 d13, d4, d6\n"
      "vpadd.u32 d14, d8, d10\n"

      // Add lhs offsets to aggregated rows.
      "vadd.s32 q6, q6, q8\n"
      "vadd.s32 d14, d14, d16\n"

      // Add rhs offset to aggregated rows.
      "vadd.s32 q6, q6, q9\n"
      "vadd.s32 d14, d14, d15\n"

      // Store results.
      "vst1.32 {d12, d13}, [%[result]]!\n"
      "vst1.32 {d14}, [%[result]]!\n"
      : [count] "+r"(count), [rhs_1] "+r"(rhs_1), [rhs_2] "+r"(rhs_2),
        [lhs] "+r"(lhs), [result] "+r"(result)
      :
      : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10",
        "d11", "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19", "d20",
        "d21", "d22", "d23", "d24", "d25", "cc", "memory");
}

inline void mul_1x8_7x8_int32_lhsadd_rhsadd(const std::uint8_t* lhs,
                                            const std::uint8_t* rhs_1,
                                            const std::uint8_t* rhs_2,
                                            std::int32_t count,
                                            std::int32_t* result) {
  asm volatile(
      "pld [%[lhs]]\n"
      "pld [%[rhs_1]]\n"
      "pld [%[rhs_2]]\n"

      // Clear aggregators.
      "vmov.i32 q0, #0\n"
      "vmov.i32 q1, #0\n"
      "vmov.i32 q2, #0\n"
      "vmov.i32 q3, q0\n"
      "vmov.i32 q4, q1\n"
      "vmov.i32 q5, q2\n"
      "vmov.i32 q6, q3\n"

      // General 1xM lanes loop.
      "1:"

      // Subtract counter.
      "subs %[count], %[count], #8\n"

      "vld1.8 {d18}, [%[lhs]:64]!\n"
      "vld1.8 {d14, d15, d16, d17}, [%[rhs_1]:64]!\n"
      "pld [%[lhs], #64]\n"
      "pld [%[rhs_1], #128]\n"
      "vmull.u8 q10, d14, d18\n"
      "vmull.u8 q11, d15, d18\n"
      "vmull.u8 q12, d16, d18\n"
      "vmull.u8 q13, d17, d18\n"
      "vld1.8 {d14, d15, d16}, [%[rhs_2]:64]!\n"
      "pld [%[rhs_2], #96]\n"
      "vpadal.u16 q0, q10\n"
      "vpadal.u16 q1, q11\n"
      "vpadal.u16 q2, q12\n"
      "vpadal.u16 q3, q13\n"
      "vmull.u8 q10, d14, d18\n"
      "vmull.u8 q11, d15, d18\n"
      "vmull.u8 q12, d16, d18\n"
      "vpadal.u16 q4, q10\n"
      "vpadal.u16 q5, q11\n"
      "vpadal.u16 q6, q12\n"

      // Loop break.
      "bne 1b\n"

      "vld1.32 {d18[], d19[]}, [%[lhs]]\n"
      "vld1.32 {d20, d21}, [%[rhs_1]:64]\n"
      "vld1.32 {d22, d23}, [%[rhs_2]:64]\n"

      // Horizontal reduce aggregators.
      "vpadd.u32 d0, d0, d1\n"
      "vpadd.u32 d2, d2, d3\n"
      "vpadd.u32 d4, d4, d5\n"
      "vpadd.u32 d6, d6, d7\n"
      "vpadd.u32 d8, d8, d9\n"
      "vpadd.u32 d10, d10, d11\n"
      "vpadd.u32 d12, d12, d13\n"
      "vpadd.u32 d14, d0, d2\n"
      "vpadd.u32 d15, d4, d6\n"
      "vpadd.u32 d16, d8, d10\n"
      "vpadd.u32 d17, d12, d12\n"

      // Add lhs offsets to aggregated rows.
      "vadd.s32 q7, q7, q9\n"
      "vadd.s32 q8, q8, q9\n"

      // Add rhs offset to aggregated rows.
      "vadd.s32 q7, q7, q10\n"
      "vadd.s32 q8, q8, q11\n"

      // Store results.
      "vst1.32 {d14, d15}, [%[result]]!\n"
      "vst1.32 {d16}, [%[result]]!\n"
      "vst1.32 {d17[0]}, [%[result]]!\n"
      : [count] "+r"(count), [rhs_1] "+r"(rhs_1), [rhs_2] "+r"(rhs_2),
        [lhs] "+r"(lhs), [result] "+r"(result)
      :
      : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10",
        "d11", "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19", "d20",
        "d21", "d22", "d23", "d24", "d25", "d26", "d27", "cc", "memory");
}

inline void mul_1x8_8x8_int32_lhsadd_rhsadd(const std::uint8_t* lhs,
                                            const std::uint8_t* rhs_1,
                                            const std::uint8_t* rhs_2,
                                            std::int32_t count,
                                            std::int32_t* result) {
  asm volatile(
      "pld [%[lhs]]\n"
      "pld [%[rhs_1]]\n"
      "pld [%[rhs_2]]\n"

      // Clear aggregators.
      "vmov.i32 q0, #0\n"
      "vmov.i32 q1, #0\n"
      "vmov.i32 q2, #0\n"
      "vmov.i32 q3, q0\n"
      "vmov.i32 q4, q1\n"
      "vmov.i32 q5, q2\n"
      "vmov.i32 q6, q3\n"
      "vmov.i32 q7, q4\n"

      // General 1xM lanes loop.
      "1:"

      // Subtract counter.
      "subs %[count], %[count], #8\n"

      "vld1.8 {d20}, [%[lhs]:64]!\n"
      "vld1.8 {d16, d17, d18, d19}, [%[rhs_1]:64]!\n"
      "pld [%[lhs], #64]\n"
      "pld [%[rhs_1], #128]\n"
      "vmull.u8 q11, d16, d20\n"
      "vmull.u8 q12, d17, d20\n"
      "vmull.u8 q13, d18, d20\n"
      "vmull.u8 q14, d19, d20\n"
      "vld1.8 {d16, d17, d18, d19}, [%[rhs_2]:64]!\n"
      "pld [%[rhs_2], #128]\n"
      "vpadal.u16 q0, q11\n"
      "vpadal.u16 q1, q12\n"
      "vpadal.u16 q2, q13\n"
      "vpadal.u16 q3, q14\n"
      "vmull.u8 q11, d16, d20\n"
      "vmull.u8 q12, d17, d20\n"
      "vmull.u8 q13, d18, d20\n"
      "vmull.u8 q14, d19, d20\n"
      "vpadal.u16 q4, q11\n"
      "vpadal.u16 q5, q12\n"
      "vpadal.u16 q6, q13\n"
      "vpadal.u16 q7, q14\n"

      // Loop break.
      "bne 1b\n"

      "vld1.32 {d20[], d21[]}, [%[lhs]]\n"
      "vld1.32 {d22, d23}, [%[rhs_1]:64]\n"
      "vld1.32 {d24, d25}, [%[rhs_2]:64]\n"

      // Horizontal reduce aggregators.
      "vpadd.u32 d0, d0, d1\n"
      "vpadd.u32 d2, d2, d3\n"
      "vpadd.u32 d4, d4, d5\n"
      "vpadd.u32 d6, d6, d7\n"
      "vpadd.u32 d8, d8, d9\n"
      "vpadd.u32 d10, d10, d11\n"
      "vpadd.u32 d12, d12, d13\n"
      "vpadd.u32 d14, d14, d15\n"
      "vpadd.u32 d16, d0, d2\n"
      "vpadd.u32 d17, d4, d6\n"
      "vpadd.u32 d18, d8, d10\n"
      "vpadd.u32 d19, d12, d14\n"

      // Add lhs offsets to aggregated rows.
      "vadd.s32 q8, q8, q10\n"
      "vadd.s32 q9, q9, q10\n"

      // Add rhs offset to aggregated rows.
      "vadd.s32 q8, q8, q11\n"
      "vadd.s32 q9, q9, q12\n"

      // Store results.
      "vst1.32 {d16, d17}, [%[result]]!\n"
      "vst1.32 {d18, d19}, [%[result]]!\n"
      : [count] "+r"(count), [rhs_1] "+r"(rhs_1), [rhs_2] "+r"(rhs_2),
        [lhs] "+r"(lhs), [result] "+r"(result)
      :
      : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10",
        "d11", "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19", "d20",
        "d21", "d22", "d23", "d24", "d25", "d26", "d27", "d28", "d29", "cc",
        "memory");
}

inline void mul_1x8_5x8_float_lhsadd_rhsadd(const std::uint8_t* lhs,
                                            const std::uint8_t* rhs_1,
                                            const std::uint8_t* rhs_2,
                                            std::int32_t count, float* result,
                                            float result_scale) {
  asm volatile(
      "pld [%[lhs]]\n"
      "pld [%[rhs_1]]\n"
      "pld [%[rhs_2]]\n"

      // Clear aggregators.
      "vmov.i32 q0, #0\n"
      "vmov.i32 q1, #0\n"
      "vmov.i32 q2, #0\n"
      "vmov.i32 q3, q0\n"
      "vmov.i32 q4, q1\n"

      // General 1xM lanes loop.
      "1:"

      // Subtract counter.
      "subs %[count], %[count], #8\n"

      "vld1.8 {d14}, [%[lhs]:64]!\n"
      "vld1.8 {d10, d11, d12, d13}, [%[rhs_1]:64]!\n"
      "pld [%[lhs], #64]\n"
      "pld [%[rhs_1], #128]\n"
      "vmull.u8 q8, d10, d14\n"
      "vmull.u8 q9, d11, d14\n"
      "vmull.u8 q10, d12, d14\n"
      "vmull.u8 q11, d13, d14\n"
      "vld1.8 {d10}, [%[rhs_2]:64]!\n"
      "pld [%[rhs_2], #32]\n"
      "vpadal.u16 q0, q8\n"
      "vpadal.u16 q1, q9\n"
      "vpadal.u16 q2, q10\n"
      "vpadal.u16 q3, q11\n"
      "vmull.u8 q8, d10, d14\n"
      "vpadal.u16 q4, q8\n"

      // Loop break.
      "bne 1b\n"

      "vld1.32 {d14[], d15[]}, [%[lhs]]\n"
      "vld1.32 {d16, d17}, [%[rhs_1]:64]\n"
      "vld1.32 {d13}, [%[rhs_2]:64]\n"
      "vdup.32 q9, %[result_scale]\n"

      // Horizontal reduce aggregators.
      "vpadd.u32 d0, d0, d1\n"
      "vpadd.u32 d2, d2, d3\n"
      "vpadd.u32 d4, d4, d5\n"
      "vpadd.u32 d6, d6, d7\n"
      "vpadd.u32 d8, d8, d9\n"
      "vpadd.u32 d10, d0, d2\n"
      "vpadd.u32 d11, d4, d6\n"
      "vpadd.u32 d12, d8, d8\n"

      // Add lhs offsets to aggregated rows.
      "vadd.s32 q5, q5, q7\n"
      "vadd.s32 d12, d12, d14\n"

      // Add rhs offset to aggregated rows.
      "vadd.s32 q5, q5, q8\n"
      "vadd.s32 d12, d12, d13\n"

      // Convert to float and scale.
      "vcvt.f32.s32 q5, q5\n"
      "vcvt.f32.s32 d12, d12\n"
      "vmul.f32 q5, q5, q9\n"
      "vmul.f32 d12, d12, d18\n"

      // Store results.
      "vst1.32 {d10, d11}, [%[result]]!\n"
      "vst1.32 {d12[0]}, [%[result]]!\n"
      : [count] "+r"(count), [result_scale] "+r"(result_scale), [lhs] "+r"(lhs),
        [result] "+r"(result), [rhs_1] "+r"(rhs_1), [rhs_2] "+r"(rhs_2)
      :
      : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10",
        "d11", "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19", "d20",
        "d21", "d22", "d23", "cc", "memory");
}

inline void mul_1x8_6x8_float_lhsadd_rhsadd(const std::uint8_t* lhs,
                                            const std::uint8_t* rhs_1,
                                            const std::uint8_t* rhs_2,
                                            std::int32_t count, float* result,
                                            float result_scale) {
  asm volatile(
      "pld [%[lhs]]\n"
      "pld [%[rhs_1]]\n"
      "pld [%[rhs_2]]\n"

      // Clear aggregators.
      "vmov.i32 q0, #0\n"
      "vmov.i32 q1, #0\n"
      "vmov.i32 q2, #0\n"
      "vmov.i32 q3, q0\n"
      "vmov.i32 q4, q1\n"
      "vmov.i32 q5, q2\n"

      // General 1xM lanes loop.
      "1:"

      // Subtract counter.
      "subs %[count], %[count], #8\n"

      "vld1.8 {d16}, [%[lhs]:64]!\n"
      "vld1.8 {d12, d13, d14, d15}, [%[rhs_1]:64]!\n"
      "pld [%[lhs], #64]\n"
      "pld [%[rhs_1], #128]\n"
      "vmull.u8 q9, d12, d16\n"
      "vmull.u8 q10, d13, d16\n"
      "vmull.u8 q11, d14, d16\n"
      "vmull.u8 q12, d15, d16\n"
      "vld1.8 {d12, d13}, [%[rhs_2]:64]!\n"
      "pld [%[rhs_2], #64]\n"
      "vpadal.u16 q0, q9\n"
      "vpadal.u16 q1, q10\n"
      "vpadal.u16 q2, q11\n"
      "vpadal.u16 q3, q12\n"
      "vmull.u8 q9, d12, d16\n"
      "vmull.u8 q10, d13, d16\n"
      "vpadal.u16 q4, q9\n"
      "vpadal.u16 q5, q10\n"

      // Loop break.
      "bne 1b\n"

      "vld1.32 {d16[], d17[]}, [%[lhs]]\n"
      "vld1.32 {d18, d19}, [%[rhs_1]:64]\n"
      "vld1.32 {d15}, [%[rhs_2]:64]\n"
      "vdup.32 q10, %[result_scale]\n"

      // Horizontal reduce aggregators.
      "vpadd.u32 d0, d0, d1\n"
      "vpadd.u32 d2, d2, d3\n"
      "vpadd.u32 d4, d4, d5\n"
      "vpadd.u32 d6, d6, d7\n"
      "vpadd.u32 d8, d8, d9\n"
      "vpadd.u32 d10, d10, d11\n"
      "vpadd.u32 d12, d0, d2\n"
      "vpadd.u32 d13, d4, d6\n"
      "vpadd.u32 d14, d8, d10\n"

      // Add lhs offsets to aggregated rows.
      "vadd.s32 q6, q6, q8\n"
      "vadd.s32 d14, d14, d16\n"

      // Add rhs offset to aggregated rows.
      "vadd.s32 q6, q6, q9\n"
      "vadd.s32 d14, d14, d15\n"

      // Convert to float and scale.
      "vcvt.f32.s32 q6, q6\n"
      "vcvt.f32.s32 d14, d14\n"
      "vmul.f32 q6, q6, q10\n"
      "vmul.f32 d14, d14, d20\n"

      // Store results.
      "vst1.32 {d12, d13}, [%[result]]!\n"
      "vst1.32 {d14}, [%[result]]!\n"
      : [count] "+r"(count), [result_scale] "+r"(result_scale), [lhs] "+r"(lhs),
        [result] "+r"(result), [rhs_1] "+r"(rhs_1), [rhs_2] "+r"(rhs_2)
      :
      : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10",
        "d11", "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19", "d20",
        "d21", "d22", "d23", "d24", "d25", "cc", "memory");
}

inline void mul_1x8_7x8_float_lhsadd_rhsadd(const std::uint8_t* lhs,
                                            const std::uint8_t* rhs_1,
                                            const std::uint8_t* rhs_2,
                                            std::int32_t count, float* result,
                                            float result_scale) {
  asm volatile(
      "pld [%[lhs]]\n"
      "pld [%[rhs_1]]\n"
      "pld [%[rhs_2]]\n"

      // Clear aggregators.
      "vmov.i32 q0, #0\n"
      "vmov.i32 q1, #0\n"
      "vmov.i32 q2, #0\n"
      "vmov.i32 q3, q0\n"
      "vmov.i32 q4, q1\n"
      "vmov.i32 q5, q2\n"
      "vmov.i32 q6, q3\n"

      // General 1xM lanes loop.
      "1:"

      // Subtract counter.
      "subs %[count], %[count], #8\n"

      "vld1.8 {d18}, [%[lhs]:64]!\n"
      "vld1.8 {d14, d15, d16, d17}, [%[rhs_1]:64]!\n"
      "pld [%[lhs], #64]\n"
      "pld [%[rhs_1], #128]\n"
      "vmull.u8 q10, d14, d18\n"
      "vmull.u8 q11, d15, d18\n"
      "vmull.u8 q12, d16, d18\n"
      "vmull.u8 q13, d17, d18\n"
      "vld1.8 {d14, d15, d16}, [%[rhs_2]:64]!\n"
      "pld [%[rhs_2], #96]\n"
      "vpadal.u16 q0, q10\n"
      "vpadal.u16 q1, q11\n"
      "vpadal.u16 q2, q12\n"
      "vpadal.u16 q3, q13\n"
      "vmull.u8 q10, d14, d18\n"
      "vmull.u8 q11, d15, d18\n"
      "vmull.u8 q12, d16, d18\n"
      "vpadal.u16 q4, q10\n"
      "vpadal.u16 q5, q11\n"
      "vpadal.u16 q6, q12\n"

      // Loop break.
      "bne 1b\n"

      "vld1.32 {d18[], d19[]}, [%[lhs]]\n"
      "vld1.32 {d20, d21}, [%[rhs_1]:64]\n"
      "vld1.32 {d22, d23}, [%[rhs_2]:64]\n"
      "vdup.32 q12, %[result_scale]\n"

      // Horizontal reduce aggregators.
      "vpadd.u32 d0, d0, d1\n"
      "vpadd.u32 d2, d2, d3\n"
      "vpadd.u32 d4, d4, d5\n"
      "vpadd.u32 d6, d6, d7\n"
      "vpadd.u32 d8, d8, d9\n"
      "vpadd.u32 d10, d10, d11\n"
      "vpadd.u32 d12, d12, d13\n"
      "vpadd.u32 d14, d0, d2\n"
      "vpadd.u32 d15, d4, d6\n"
      "vpadd.u32 d16, d8, d10\n"
      "vpadd.u32 d17, d12, d12\n"

      // Add lhs offsets to aggregated rows.
      "vadd.s32 q7, q7, q9\n"
      "vadd.s32 q8, q8, q9\n"

      // Add rhs offset to aggregated rows.
      "vadd.s32 q7, q7, q10\n"
      "vadd.s32 q8, q8, q11\n"

      // Convert to float and scale.
      "vcvt.f32.s32 q7, q7\n"
      "vcvt.f32.s32 q8, q8\n"
      "vmul.f32 q7, q7, q12\n"
      "vmul.f32 q8, q8, q12\n"

      // Store results.
      "vst1.32 {d14, d15}, [%[result]]!\n"
      "vst1.32 {d16}, [%[result]]!\n"
      "vst1.32 {d17[0]}, [%[result]]!\n"
      : [count] "+r"(count), [result_scale] "+r"(result_scale), [lhs] "+r"(lhs),
        [result] "+r"(result), [rhs_1] "+r"(rhs_1), [rhs_2] "+r"(rhs_2)
      :
      : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10",
        "d11", "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19", "d20",
        "d21", "d22", "d23", "d24", "d25", "d26", "d27", "cc", "memory");
}

inline void mul_1x8_8x8_float_lhsadd_rhsadd(const std::uint8_t* lhs,
                                            const std::uint8_t* rhs_1,
                                            const std::uint8_t* rhs_2,
                                            std::int32_t count, float* result,
                                            float result_scale) {
  asm volatile(
      "pld [%[lhs]]\n"
      "pld [%[rhs_1]]\n"
      "pld [%[rhs_2]]\n"

      // Clear aggregators.
      "vmov.i32 q0, #0\n"
      "vmov.i32 q1, #0\n"
      "vmov.i32 q2, #0\n"
      "vmov.i32 q3, q0\n"
      "vmov.i32 q4, q1\n"
      "vmov.i32 q5, q2\n"
      "vmov.i32 q6, q3\n"
      "vmov.i32 q7, q4\n"

      // General 1xM lanes loop.
      "1:"

      // Subtract counter.
      "subs %[count], %[count], #8\n"

      "vld1.8 {d20}, [%[lhs]:64]!\n"
      "vld1.8 {d16, d17, d18, d19}, [%[rhs_1]:64]!\n"
      "pld [%[lhs], #64]\n"
      "pld [%[rhs_1], #128]\n"
      "vmull.u8 q11, d16, d20\n"
      "vmull.u8 q12, d17, d20\n"
      "vmull.u8 q13, d18, d20\n"
      "vmull.u8 q14, d19, d20\n"
      "vld1.8 {d16, d17, d18, d19}, [%[rhs_2]:64]!\n"
      "pld [%[rhs_2], #128]\n"
      "vpadal.u16 q0, q11\n"
      "vpadal.u16 q1, q12\n"
      "vpadal.u16 q2, q13\n"
      "vpadal.u16 q3, q14\n"
      "vmull.u8 q11, d16, d20\n"
      "vmull.u8 q12, d17, d20\n"
      "vmull.u8 q13, d18, d20\n"
      "vmull.u8 q14, d19, d20\n"
      "vpadal.u16 q4, q11\n"
      "vpadal.u16 q5, q12\n"
      "vpadal.u16 q6, q13\n"
      "vpadal.u16 q7, q14\n"

      // Loop break.
      "bne 1b\n"

      "vld1.32 {d20[], d21[]}, [%[lhs]]\n"
      "vld1.32 {d22, d23}, [%[rhs_1]:64]\n"
      "vld1.32 {d24, d25}, [%[rhs_2]:64]\n"
      "vdup.32 q13, %[result_scale]\n"

      // Horizontal reduce aggregators.
      "vpadd.u32 d0, d0, d1\n"
      "vpadd.u32 d2, d2, d3\n"
      "vpadd.u32 d4, d4, d5\n"
      "vpadd.u32 d6, d6, d7\n"
      "vpadd.u32 d8, d8, d9\n"
      "vpadd.u32 d10, d10, d11\n"
      "vpadd.u32 d12, d12, d13\n"
      "vpadd.u32 d14, d14, d15\n"
      "vpadd.u32 d16, d0, d2\n"
      "vpadd.u32 d17, d4, d6\n"
      "vpadd.u32 d18, d8, d10\n"
      "vpadd.u32 d19, d12, d14\n"

      // Add lhs offsets to aggregated rows.
      "vadd.s32 q8, q8, q10\n"
      "vadd.s32 q9, q9, q10\n"

      // Add rhs offset to aggregated rows.
      "vadd.s32 q8, q8, q11\n"
      "vadd.s32 q9, q9, q12\n"

      // Convert to float and scale.
      "vcvt.f32.s32 q8, q8\n"
      "vcvt.f32.s32 q9, q9\n"
      "vmul.f32 q8, q8, q13\n"
      "vmul.f32 q9, q9, q13\n"

      // Store results.
      "vst1.32 {d16, d17}, [%[result]]!\n"
      "vst1.32 {d18, d19}, [%[result]]!\n"
      : [count] "+r"(count), [result_scale] "+r"(result_scale), [lhs] "+r"(lhs),
        [result] "+r"(result), [rhs_1] "+r"(rhs_1), [rhs_2] "+r"(rhs_2)
      :
      : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10",
        "d11", "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19", "d20",
        "d21", "d22", "d23", "d24", "d25", "d26", "d27", "d28", "d29", "cc",
        "memory");
}

void qnt_1x8_aligned(const std::int32_t* source, std::int32_t count,
                     std::int32_t stride, const std::int32_t* offsets,
                     std::uint8_t* destination, std::int32_t destination_stride,
                     std::int32_t multiplicative_offset,
                     std::int32_t rounding_offset, std::int32_t shift) {
  asm volatile(
      "vdup.32 q0, %[multiplicative_offset]\n"
      "vdup.32 q1, %[rounding_offset]\n"
      "vdup.32 q2, %[shift]\n"
      "vld1.32 {d6[], d7[]}, [%[offsets]:32]!\n"

      "1:"
      "subs %[count], %[count], #8\n"
      "vld1.32 {d8, d9, d10, d11}, [%[source]:128]!\n"
      "pld [%[source]]\n"
      "vadd.i32 q4, q4, q3\n"
      "vadd.i32 q5, q5, q3\n"
      "vmul.i32 q4, q4, q0\n"
      "vmul.i32 q5, q5, q0\n"
      "vadd.i32 q4, q4, q1\n"
      "vadd.i32 q5, q5, q1\n"
      "vshl.s32 q4, q4, q2\n"
      "vshl.s32 q5, q5, q2\n"
      "vqmovn.s32 d8, q4\n"
      "vqmovn.s32 d9, q5\n"
      "vqmovun.s16 d8, q4\n"
      "vst1.32 {d8}, [%[destination]:64]!\n"

      "bne 1b\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [stride] "+r"(stride), [shift] "+r"(shift),
        [destination] "+r"(destination), [offsets] "+r"(offsets),
        [source] "+r"(source), [destination_stride] "+r"(destination_stride),
        [rounding_offset] "+r"(rounding_offset)
      :
      : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10",
        "d11", "cc", "memory");
}

void qnt_1x8_1_aligned(const std::int32_t* source, std::int32_t count,
                       std::int32_t stride, const std::int32_t* offsets,
                       std::uint8_t* destination,
                       std::int32_t destination_stride,
                       std::int32_t multiplicative_offset,
                       std::int32_t rounding_offset, std::int32_t shift) {
  asm volatile(
      "vdup.32 q0, %[multiplicative_offset]\n"
      "vdup.32 q1, %[rounding_offset]\n"
      "vdup.32 q2, %[shift]\n"
      "vld1.32 {d6[], d7[]}, [%[offsets]:32]!\n"
      "subs %[count], %[count], #1\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"
      "vld1.32 {d8, d9, d10, d11}, [%[source]:128]!\n"
      "pld [%[source]]\n"
      "vadd.i32 q4, q4, q3\n"
      "vadd.i32 q5, q5, q3\n"
      "vmul.i32 q4, q4, q0\n"
      "vmul.i32 q5, q5, q0\n"
      "vadd.i32 q4, q4, q1\n"
      "vadd.i32 q5, q5, q1\n"
      "vshl.s32 q4, q4, q2\n"
      "vshl.s32 q5, q5, q2\n"
      "vqmovn.s32 d8, q4\n"
      "vqmovn.s32 d9, q5\n"
      "vqmovun.s16 d8, q4\n"
      "vst1.32 {d8}, [%[destination]:64]!\n"

      "bne 1b\n"
      "2:"
      "vld1.32 {d8[0]}, [%[source]]!\n"
      "vadd.i32 q4, q4, q3\n"
      "vmul.i32 q4, q4, q0\n"
      "vadd.i32 q4, q4, q1\n"
      "vshl.s32 q4, q4, q2\n"
      "vqmovn.s32 d8, q4\n"
      "vqmovun.s16 d8, q4\n"
      "vst1.8 {d8[0]}, [%[destination]]!\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [stride] "+r"(stride), [shift] "+r"(shift),
        [destination] "+r"(destination), [offsets] "+r"(offsets),
        [source] "+r"(source), [destination_stride] "+r"(destination_stride),
        [rounding_offset] "+r"(rounding_offset)
      :
      : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10",
        "d11", "cc", "memory");
}

void qnt_1x8_2_aligned(const std::int32_t* source, std::int32_t count,
                       std::int32_t stride, const std::int32_t* offsets,
                       std::uint8_t* destination,
                       std::int32_t destination_stride,
                       std::int32_t multiplicative_offset,
                       std::int32_t rounding_offset, std::int32_t shift) {
  asm volatile(
      "vdup.32 q0, %[multiplicative_offset]\n"
      "vdup.32 q1, %[rounding_offset]\n"
      "vdup.32 q2, %[shift]\n"
      "vld1.32 {d6[], d7[]}, [%[offsets]:32]!\n"
      "subs %[count], %[count], #2\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"
      "vld1.32 {d8, d9, d10, d11}, [%[source]:128]!\n"
      "pld [%[source]]\n"
      "vadd.i32 q4, q4, q3\n"
      "vadd.i32 q5, q5, q3\n"
      "vmul.i32 q4, q4, q0\n"
      "vmul.i32 q5, q5, q0\n"
      "vadd.i32 q4, q4, q1\n"
      "vadd.i32 q5, q5, q1\n"
      "vshl.s32 q4, q4, q2\n"
      "vshl.s32 q5, q5, q2\n"
      "vqmovn.s32 d8, q4\n"
      "vqmovn.s32 d9, q5\n"
      "vqmovun.s16 d8, q4\n"
      "vst1.32 {d8}, [%[destination]:64]!\n"

      "bne 1b\n"
      "2:"
      "vld1.32 {d8}, [%[source]:64]!\n"
      "vadd.i32 q4, q4, q3\n"
      "vmul.i32 q4, q4, q0\n"
      "vadd.i32 q4, q4, q1\n"
      "vshl.s32 q4, q4, q2\n"
      "vqmovn.s32 d8, q4\n"
      "vqmovun.s16 d8, q4\n"
      "vst1.16 {d8[0]}, [%[destination]]!\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [stride] "+r"(stride), [shift] "+r"(shift),
        [destination] "+r"(destination), [offsets] "+r"(offsets),
        [source] "+r"(source), [destination_stride] "+r"(destination_stride),
        [rounding_offset] "+r"(rounding_offset)
      :
      : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10",
        "d11", "cc", "memory");
}

void qnt_1x8_3_aligned(const std::int32_t* source, std::int32_t count,
                       std::int32_t stride, const std::int32_t* offsets,
                       std::uint8_t* destination,
                       std::int32_t destination_stride,
                       std::int32_t multiplicative_offset,
                       std::int32_t rounding_offset, std::int32_t shift) {
  asm volatile(
      "vdup.32 q0, %[multiplicative_offset]\n"
      "vdup.32 q1, %[rounding_offset]\n"
      "vdup.32 q2, %[shift]\n"
      "vld1.32 {d6[], d7[]}, [%[offsets]:32]!\n"
      "subs %[count], %[count], #3\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"
      "vld1.32 {d8, d9, d10, d11}, [%[source]:128]!\n"
      "pld [%[source]]\n"
      "vadd.i32 q4, q4, q3\n"
      "vadd.i32 q5, q5, q3\n"
      "vmul.i32 q4, q4, q0\n"
      "vmul.i32 q5, q5, q0\n"
      "vadd.i32 q4, q4, q1\n"
      "vadd.i32 q5, q5, q1\n"
      "vshl.s32 q4, q4, q2\n"
      "vshl.s32 q5, q5, q2\n"
      "vqmovn.s32 d8, q4\n"
      "vqmovn.s32 d9, q5\n"
      "vqmovun.s16 d8, q4\n"
      "vst1.32 {d8}, [%[destination]:64]!\n"

      "bne 1b\n"
      "2:"
      "vld1.32 {d8}, [%[source]:64]!\n"
      "vld1.32 {d9[0]}, [%[source]]!\n"
      "vadd.i32 q4, q4, q3\n"
      "vmul.i32 q4, q4, q0\n"
      "vadd.i32 q4, q4, q1\n"
      "vshl.s32 q4, q4, q2\n"
      "vqmovn.s32 d8, q4\n"
      "vqmovun.s16 d8, q4\n"
      "vst1.16 {d8[0]}, [%[destination]]!\n"
      "vst1.8 {d8[2]}, [%[destination]]!\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [stride] "+r"(stride), [shift] "+r"(shift),
        [destination] "+r"(destination), [offsets] "+r"(offsets),
        [source] "+r"(source), [destination_stride] "+r"(destination_stride),
        [rounding_offset] "+r"(rounding_offset)
      :
      : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10",
        "d11", "cc", "memory");
}

void qnt_1x8_4_aligned(const std::int32_t* source, std::int32_t count,
                       std::int32_t stride, const std::int32_t* offsets,
                       std::uint8_t* destination,
                       std::int32_t destination_stride,
                       std::int32_t multiplicative_offset,
                       std::int32_t rounding_offset, std::int32_t shift) {
  asm volatile(
      "vdup.32 q0, %[multiplicative_offset]\n"
      "vdup.32 q1, %[rounding_offset]\n"
      "vdup.32 q2, %[shift]\n"
      "vld1.32 {d6[], d7[]}, [%[offsets]:32]!\n"
      "subs %[count], %[count], #4\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"
      "vld1.32 {d8, d9, d10, d11}, [%[source]:128]!\n"
      "pld [%[source]]\n"
      "vadd.i32 q4, q4, q3\n"
      "vadd.i32 q5, q5, q3\n"
      "vmul.i32 q4, q4, q0\n"
      "vmul.i32 q5, q5, q0\n"
      "vadd.i32 q4, q4, q1\n"
      "vadd.i32 q5, q5, q1\n"
      "vshl.s32 q4, q4, q2\n"
      "vshl.s32 q5, q5, q2\n"
      "vqmovn.s32 d8, q4\n"
      "vqmovn.s32 d9, q5\n"
      "vqmovun.s16 d8, q4\n"
      "vst1.32 {d8}, [%[destination]:64]!\n"

      "bne 1b\n"
      "2:"
      "vld1.32 {d8, d9}, [%[source]:64]!\n"
      "vadd.i32 q4, q4, q3\n"
      "vmul.i32 q4, q4, q0\n"
      "vadd.i32 q4, q4, q1\n"
      "vshl.s32 q4, q4, q2\n"
      "vqmovn.s32 d8, q4\n"
      "vqmovun.s16 d8, q4\n"
      "vst1.32 {d8[0]}, [%[destination]]!\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [stride] "+r"(stride), [shift] "+r"(shift),
        [destination] "+r"(destination), [offsets] "+r"(offsets),
        [source] "+r"(source), [destination_stride] "+r"(destination_stride),
        [rounding_offset] "+r"(rounding_offset)
      :
      : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10",
        "d11", "cc", "memory");
}

void qnt_1x8_5_aligned(const std::int32_t* source, std::int32_t count,
                       std::int32_t stride, const std::int32_t* offsets,
                       std::uint8_t* destination,
                       std::int32_t destination_stride,
                       std::int32_t multiplicative_offset,
                       std::int32_t rounding_offset, std::int32_t shift) {
  asm volatile(
      "vdup.32 q0, %[multiplicative_offset]\n"
      "vdup.32 q1, %[rounding_offset]\n"
      "vdup.32 q2, %[shift]\n"
      "vld1.32 {d6[], d7[]}, [%[offsets]:32]!\n"
      "subs %[count], %[count], #5\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"
      "vld1.32 {d8, d9, d10, d11}, [%[source]:128]!\n"
      "pld [%[source]]\n"
      "vadd.i32 q4, q4, q3\n"
      "vadd.i32 q5, q5, q3\n"
      "vmul.i32 q4, q4, q0\n"
      "vmul.i32 q5, q5, q0\n"
      "vadd.i32 q4, q4, q1\n"
      "vadd.i32 q5, q5, q1\n"
      "vshl.s32 q4, q4, q2\n"
      "vshl.s32 q5, q5, q2\n"
      "vqmovn.s32 d8, q4\n"
      "vqmovn.s32 d9, q5\n"
      "vqmovun.s16 d8, q4\n"
      "vst1.32 {d8}, [%[destination]:64]!\n"

      "bne 1b\n"
      "2:"
      "vld1.32 {d8, d9}, [%[source]:64]!\n"
      "vld1.32 {d10[0]}, [%[source]]!\n"
      "vadd.i32 q4, q4, q3\n"
      "vadd.i32 q5, q5, q3\n"
      "vmul.i32 q4, q4, q0\n"
      "vmul.i32 q5, q5, q0\n"
      "vadd.i32 q4, q4, q1\n"
      "vadd.i32 q5, q5, q1\n"
      "vshl.s32 q4, q4, q2\n"
      "vshl.s32 q5, q5, q2\n"
      "vqmovn.s32 d8, q4\n"
      "vqmovn.s32 d9, q5\n"
      "vqmovun.s16 d8, q4\n"
      "vst1.32 {d8[0]}, [%[destination]]!\n"
      "vst1.8 {d8[4]}, [%[destination]]!\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [stride] "+r"(stride), [shift] "+r"(shift),
        [destination] "+r"(destination), [offsets] "+r"(offsets),
        [source] "+r"(source), [destination_stride] "+r"(destination_stride),
        [rounding_offset] "+r"(rounding_offset)
      :
      : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10",
        "d11", "cc", "memory");
}

void qnt_1x8_6_aligned(const std::int32_t* source, std::int32_t count,
                       std::int32_t stride, const std::int32_t* offsets,
                       std::uint8_t* destination,
                       std::int32_t destination_stride,
                       std::int32_t multiplicative_offset,
                       std::int32_t rounding_offset, std::int32_t shift) {
  asm volatile(
      "vdup.32 q0, %[multiplicative_offset]\n"
      "vdup.32 q1, %[rounding_offset]\n"
      "vdup.32 q2, %[shift]\n"
      "vld1.32 {d6[], d7[]}, [%[offsets]:32]!\n"
      "subs %[count], %[count], #6\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"
      "vld1.32 {d8, d9, d10, d11}, [%[source]:128]!\n"
      "pld [%[source]]\n"
      "vadd.i32 q4, q4, q3\n"
      "vadd.i32 q5, q5, q3\n"
      "vmul.i32 q4, q4, q0\n"
      "vmul.i32 q5, q5, q0\n"
      "vadd.i32 q4, q4, q1\n"
      "vadd.i32 q5, q5, q1\n"
      "vshl.s32 q4, q4, q2\n"
      "vshl.s32 q5, q5, q2\n"
      "vqmovn.s32 d8, q4\n"
      "vqmovn.s32 d9, q5\n"
      "vqmovun.s16 d8, q4\n"
      "vst1.32 {d8}, [%[destination]:64]!\n"

      "bne 1b\n"
      "2:"
      "vld1.32 {d8, d9, d10}, [%[source]:64]!\n"
      "vadd.i32 q4, q4, q3\n"
      "vadd.i32 q5, q5, q3\n"
      "vmul.i32 q4, q4, q0\n"
      "vmul.i32 q5, q5, q0\n"
      "vadd.i32 q4, q4, q1\n"
      "vadd.i32 q5, q5, q1\n"
      "vshl.s32 q4, q4, q2\n"
      "vshl.s32 q5, q5, q2\n"
      "vqmovn.s32 d8, q4\n"
      "vqmovn.s32 d9, q5\n"
      "vqmovun.s16 d8, q4\n"
      "vst1.32 {d8[0]}, [%[destination]]!\n"
      "vst1.16 {d8[2]}, [%[destination]]!\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [stride] "+r"(stride), [shift] "+r"(shift),
        [destination] "+r"(destination), [offsets] "+r"(offsets),
        [source] "+r"(source), [destination_stride] "+r"(destination_stride),
        [rounding_offset] "+r"(rounding_offset)
      :
      : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10",
        "d11", "cc", "memory");
}

void qnt_1x8_7_aligned(const std::int32_t* source, std::int32_t count,
                       std::int32_t stride, const std::int32_t* offsets,
                       std::uint8_t* destination,
                       std::int32_t destination_stride,
                       std::int32_t multiplicative_offset,
                       std::int32_t rounding_offset, std::int32_t shift) {
  asm volatile(
      "vdup.32 q0, %[multiplicative_offset]\n"
      "vdup.32 q1, %[rounding_offset]\n"
      "vdup.32 q2, %[shift]\n"
      "vld1.32 {d6[], d7[]}, [%[offsets]:32]!\n"
      "subs %[count], %[count], #7\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"
      "vld1.32 {d8, d9, d10, d11}, [%[source]:128]!\n"
      "pld [%[source]]\n"
      "vadd.i32 q4, q4, q3\n"
      "vadd.i32 q5, q5, q3\n"
      "vmul.i32 q4, q4, q0\n"
      "vmul.i32 q5, q5, q0\n"
      "vadd.i32 q4, q4, q1\n"
      "vadd.i32 q5, q5, q1\n"
      "vshl.s32 q4, q4, q2\n"
      "vshl.s32 q5, q5, q2\n"
      "vqmovn.s32 d8, q4\n"
      "vqmovn.s32 d9, q5\n"
      "vqmovun.s16 d8, q4\n"
      "vst1.32 {d8}, [%[destination]:64]!\n"

      "bne 1b\n"
      "2:"
      "vld1.32 {d8, d9, d10}, [%[source]:64]!\n"
      "vld1.32 {d11[0]}, [%[source]]!\n"
      "vadd.i32 q4, q4, q3\n"
      "vadd.i32 q5, q5, q3\n"
      "vmul.i32 q4, q4, q0\n"
      "vmul.i32 q5, q5, q0\n"
      "vadd.i32 q4, q4, q1\n"
      "vadd.i32 q5, q5, q1\n"
      "vshl.s32 q4, q4, q2\n"
      "vshl.s32 q5, q5, q2\n"
      "vqmovn.s32 d8, q4\n"
      "vqmovn.s32 d9, q5\n"
      "vqmovun.s16 d8, q4\n"
      "vst1.32 {d8[0]}, [%[destination]]!\n"
      "vst1.16 {d8[2]}, [%[destination]]!\n"
      "vst1.8 {d8[6]}, [%[destination]]!\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [stride] "+r"(stride), [shift] "+r"(shift),
        [destination] "+r"(destination), [offsets] "+r"(offsets),
        [source] "+r"(source), [destination_stride] "+r"(destination_stride),
        [rounding_offset] "+r"(rounding_offset)
      :
      : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10",
        "d11", "cc", "memory");
}

void qnt_2x8_aligned(const std::int32_t* source, std::int32_t count,
                     std::int32_t stride, const std::int32_t* offsets,
                     std::uint8_t* destination, std::int32_t destination_stride,
                     std::int32_t multiplicative_offset,
                     std::int32_t rounding_offset, std::int32_t shift) {
  asm volatile(
      "vdup.32 q0, %[multiplicative_offset]\n"
      "vdup.32 q1, %[rounding_offset]\n"
      "vdup.32 q2, %[shift]\n"
      "vld1.32 {d6[], d7[]}, [%[offsets]:32]!\n"
      "vld1.32 {d8[], d9[]}, [%[offsets]:32]!\n"
      "add r0, %[source], %[stride]\n"
      "add r1, %[destination], %[destination_stride]\n"

      "1:"
      "subs %[count], %[count], #8\n"
      "vld1.32 {d10, d11, d12, d13}, [%[source]:128]!\n"
      "vld1.32 {d14, d15, d16, d17}, [r0:128]!\n"
      "pld [%[source]]\n"
      "pld [r0]\n"
      "vadd.i32 q5, q5, q3\n"
      "vadd.i32 q6, q6, q3\n"
      "vadd.i32 q7, q7, q4\n"
      "vadd.i32 q8, q8, q4\n"
      "vmul.i32 q5, q5, q0\n"
      "vmul.i32 q6, q6, q0\n"
      "vmul.i32 q7, q7, q0\n"
      "vmul.i32 q8, q8, q0\n"
      "vadd.i32 q5, q5, q1\n"
      "vadd.i32 q6, q6, q1\n"
      "vadd.i32 q7, q7, q1\n"
      "vadd.i32 q8, q8, q1\n"
      "vshl.s32 q5, q5, q2\n"
      "vshl.s32 q6, q6, q2\n"
      "vshl.s32 q7, q7, q2\n"
      "vshl.s32 q8, q8, q2\n"
      "vqmovn.s32 d10, q5\n"
      "vqmovn.s32 d11, q6\n"
      "vqmovn.s32 d14, q7\n"
      "vqmovn.s32 d15, q8\n"
      "vqmovun.s16 d10, q5\n"
      "vqmovun.s16 d14, q7\n"
      "vst1.32 {d10}, [%[destination]:64]!\n"
      "vst1.32 {d14}, [r1:64]!\n"

      "bne 1b\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [stride] "+r"(stride), [shift] "+r"(shift),
        [destination] "+r"(destination), [offsets] "+r"(offsets),
        [source] "+r"(source), [destination_stride] "+r"(destination_stride),
        [rounding_offset] "+r"(rounding_offset)
      :
      : "r0", "r1", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9",
        "d10", "d11", "d12", "d13", "d14", "d15", "d16", "d17", "cc", "memory");
}

void qnt_2x8_1_aligned(const std::int32_t* source, std::int32_t count,
                       std::int32_t stride, const std::int32_t* offsets,
                       std::uint8_t* destination,
                       std::int32_t destination_stride,
                       std::int32_t multiplicative_offset,
                       std::int32_t rounding_offset, std::int32_t shift) {
  asm volatile(
      "vdup.32 q0, %[multiplicative_offset]\n"
      "vdup.32 q1, %[rounding_offset]\n"
      "vdup.32 q2, %[shift]\n"
      "vld1.32 {d6[], d7[]}, [%[offsets]:32]!\n"
      "vld1.32 {d8[], d9[]}, [%[offsets]:32]!\n"
      "add r0, %[source], %[stride]\n"
      "add r1, %[destination], %[destination_stride]\n"
      "subs %[count], %[count], #1\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"
      "vld1.32 {d10, d11, d12, d13}, [%[source]:128]!\n"
      "vld1.32 {d14, d15, d16, d17}, [r0:128]!\n"
      "pld [%[source]]\n"
      "pld [r0]\n"
      "vadd.i32 q5, q5, q3\n"
      "vadd.i32 q6, q6, q3\n"
      "vadd.i32 q7, q7, q4\n"
      "vadd.i32 q8, q8, q4\n"
      "vmul.i32 q5, q5, q0\n"
      "vmul.i32 q6, q6, q0\n"
      "vmul.i32 q7, q7, q0\n"
      "vmul.i32 q8, q8, q0\n"
      "vadd.i32 q5, q5, q1\n"
      "vadd.i32 q6, q6, q1\n"
      "vadd.i32 q7, q7, q1\n"
      "vadd.i32 q8, q8, q1\n"
      "vshl.s32 q5, q5, q2\n"
      "vshl.s32 q6, q6, q2\n"
      "vshl.s32 q7, q7, q2\n"
      "vshl.s32 q8, q8, q2\n"
      "vqmovn.s32 d10, q5\n"
      "vqmovn.s32 d11, q6\n"
      "vqmovn.s32 d14, q7\n"
      "vqmovn.s32 d15, q8\n"
      "vqmovun.s16 d10, q5\n"
      "vqmovun.s16 d14, q7\n"
      "vst1.32 {d10}, [%[destination]:64]!\n"
      "vst1.32 {d14}, [r1:64]!\n"

      "bne 1b\n"
      "2:"
      "vld1.32 {d10[0]}, [%[source]]!\n"
      "vld1.32 {d14[0]}, [r0]!\n"
      "vadd.i32 q5, q5, q3\n"
      "vadd.i32 q7, q7, q4\n"
      "vmul.i32 q5, q5, q0\n"
      "vmul.i32 q7, q7, q0\n"
      "vadd.i32 q5, q5, q1\n"
      "vadd.i32 q7, q7, q1\n"
      "vshl.s32 q5, q5, q2\n"
      "vshl.s32 q7, q7, q2\n"
      "vqmovn.s32 d10, q5\n"
      "vqmovn.s32 d14, q7\n"
      "vqmovun.s16 d10, q5\n"
      "vqmovun.s16 d14, q7\n"
      "vst1.8 {d10[0]}, [%[destination]]!\n"
      "vst1.8 {d14[0]}, [r1]!\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [stride] "+r"(stride), [shift] "+r"(shift),
        [destination] "+r"(destination), [offsets] "+r"(offsets),
        [source] "+r"(source), [destination_stride] "+r"(destination_stride),
        [rounding_offset] "+r"(rounding_offset)
      :
      : "r0", "r1", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9",
        "d10", "d11", "d12", "d13", "d14", "d15", "d16", "d17", "cc", "memory");
}

void qnt_2x8_2_aligned(const std::int32_t* source, std::int32_t count,
                       std::int32_t stride, const std::int32_t* offsets,
                       std::uint8_t* destination,
                       std::int32_t destination_stride,
                       std::int32_t multiplicative_offset,
                       std::int32_t rounding_offset, std::int32_t shift) {
  asm volatile(
      "vdup.32 q0, %[multiplicative_offset]\n"
      "vdup.32 q1, %[rounding_offset]\n"
      "vdup.32 q2, %[shift]\n"
      "vld1.32 {d6[], d7[]}, [%[offsets]:32]!\n"
      "vld1.32 {d8[], d9[]}, [%[offsets]:32]!\n"
      "add r0, %[source], %[stride]\n"
      "add r1, %[destination], %[destination_stride]\n"
      "subs %[count], %[count], #2\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"
      "vld1.32 {d10, d11, d12, d13}, [%[source]:128]!\n"
      "vld1.32 {d14, d15, d16, d17}, [r0:128]!\n"
      "pld [%[source]]\n"
      "pld [r0]\n"
      "vadd.i32 q5, q5, q3\n"
      "vadd.i32 q6, q6, q3\n"
      "vadd.i32 q7, q7, q4\n"
      "vadd.i32 q8, q8, q4\n"
      "vmul.i32 q5, q5, q0\n"
      "vmul.i32 q6, q6, q0\n"
      "vmul.i32 q7, q7, q0\n"
      "vmul.i32 q8, q8, q0\n"
      "vadd.i32 q5, q5, q1\n"
      "vadd.i32 q6, q6, q1\n"
      "vadd.i32 q7, q7, q1\n"
      "vadd.i32 q8, q8, q1\n"
      "vshl.s32 q5, q5, q2\n"
      "vshl.s32 q6, q6, q2\n"
      "vshl.s32 q7, q7, q2\n"
      "vshl.s32 q8, q8, q2\n"
      "vqmovn.s32 d10, q5\n"
      "vqmovn.s32 d11, q6\n"
      "vqmovn.s32 d14, q7\n"
      "vqmovn.s32 d15, q8\n"
      "vqmovun.s16 d10, q5\n"
      "vqmovun.s16 d14, q7\n"
      "vst1.32 {d10}, [%[destination]:64]!\n"
      "vst1.32 {d14}, [r1:64]!\n"

      "bne 1b\n"
      "2:"
      "vld1.32 {d10}, [%[source]:64]!\n"
      "vld1.32 {d14}, [r0:64]!\n"
      "vadd.i32 q5, q5, q3\n"
      "vadd.i32 q7, q7, q4\n"
      "vmul.i32 q5, q5, q0\n"
      "vmul.i32 q7, q7, q0\n"
      "vadd.i32 q5, q5, q1\n"
      "vadd.i32 q7, q7, q1\n"
      "vshl.s32 q5, q5, q2\n"
      "vshl.s32 q7, q7, q2\n"
      "vqmovn.s32 d10, q5\n"
      "vqmovn.s32 d14, q7\n"
      "vqmovun.s16 d10, q5\n"
      "vqmovun.s16 d14, q7\n"
      "vst1.16 {d10[0]}, [%[destination]]!\n"
      "vst1.16 {d14[0]}, [r1]!\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [stride] "+r"(stride), [shift] "+r"(shift),
        [destination] "+r"(destination), [offsets] "+r"(offsets),
        [source] "+r"(source), [destination_stride] "+r"(destination_stride),
        [rounding_offset] "+r"(rounding_offset)
      :
      : "r0", "r1", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9",
        "d10", "d11", "d12", "d13", "d14", "d15", "d16", "d17", "cc", "memory");
}

void qnt_2x8_3_aligned(const std::int32_t* source, std::int32_t count,
                       std::int32_t stride, const std::int32_t* offsets,
                       std::uint8_t* destination,
                       std::int32_t destination_stride,
                       std::int32_t multiplicative_offset,
                       std::int32_t rounding_offset, std::int32_t shift) {
  asm volatile(
      "vdup.32 q0, %[multiplicative_offset]\n"
      "vdup.32 q1, %[rounding_offset]\n"
      "vdup.32 q2, %[shift]\n"
      "vld1.32 {d6[], d7[]}, [%[offsets]:32]!\n"
      "vld1.32 {d8[], d9[]}, [%[offsets]:32]!\n"
      "add r0, %[source], %[stride]\n"
      "add r1, %[destination], %[destination_stride]\n"
      "subs %[count], %[count], #3\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"
      "vld1.32 {d10, d11, d12, d13}, [%[source]:128]!\n"
      "vld1.32 {d14, d15, d16, d17}, [r0:128]!\n"
      "pld [%[source]]\n"
      "pld [r0]\n"
      "vadd.i32 q5, q5, q3\n"
      "vadd.i32 q6, q6, q3\n"
      "vadd.i32 q7, q7, q4\n"
      "vadd.i32 q8, q8, q4\n"
      "vmul.i32 q5, q5, q0\n"
      "vmul.i32 q6, q6, q0\n"
      "vmul.i32 q7, q7, q0\n"
      "vmul.i32 q8, q8, q0\n"
      "vadd.i32 q5, q5, q1\n"
      "vadd.i32 q6, q6, q1\n"
      "vadd.i32 q7, q7, q1\n"
      "vadd.i32 q8, q8, q1\n"
      "vshl.s32 q5, q5, q2\n"
      "vshl.s32 q6, q6, q2\n"
      "vshl.s32 q7, q7, q2\n"
      "vshl.s32 q8, q8, q2\n"
      "vqmovn.s32 d10, q5\n"
      "vqmovn.s32 d11, q6\n"
      "vqmovn.s32 d14, q7\n"
      "vqmovn.s32 d15, q8\n"
      "vqmovun.s16 d10, q5\n"
      "vqmovun.s16 d14, q7\n"
      "vst1.32 {d10}, [%[destination]:64]!\n"
      "vst1.32 {d14}, [r1:64]!\n"

      "bne 1b\n"
      "2:"
      "vld1.32 {d10}, [%[source]:64]!\n"
      "vld1.32 {d11[0]}, [%[source]]!\n"
      "vld1.32 {d14}, [r0:64]!\n"
      "vld1.32 {d15[0]}, [r0]!\n"
      "vadd.i32 q5, q5, q3\n"
      "vadd.i32 q7, q7, q4\n"
      "vmul.i32 q5, q5, q0\n"
      "vmul.i32 q7, q7, q0\n"
      "vadd.i32 q5, q5, q1\n"
      "vadd.i32 q7, q7, q1\n"
      "vshl.s32 q5, q5, q2\n"
      "vshl.s32 q7, q7, q2\n"
      "vqmovn.s32 d10, q5\n"
      "vqmovn.s32 d14, q7\n"
      "vqmovun.s16 d10, q5\n"
      "vqmovun.s16 d14, q7\n"
      "vst1.16 {d10[0]}, [%[destination]]!\n"
      "vst1.8 {d10[2]}, [%[destination]]!\n"
      "vst1.16 {d14[0]}, [r1]!\n"
      "vst1.8 {d14[2]}, [r1]!\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [stride] "+r"(stride), [shift] "+r"(shift),
        [destination] "+r"(destination), [offsets] "+r"(offsets),
        [source] "+r"(source), [destination_stride] "+r"(destination_stride),
        [rounding_offset] "+r"(rounding_offset)
      :
      : "r0", "r1", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9",
        "d10", "d11", "d12", "d13", "d14", "d15", "d16", "d17", "cc", "memory");
}

void qnt_2x8_4_aligned(const std::int32_t* source, std::int32_t count,
                       std::int32_t stride, const std::int32_t* offsets,
                       std::uint8_t* destination,
                       std::int32_t destination_stride,
                       std::int32_t multiplicative_offset,
                       std::int32_t rounding_offset, std::int32_t shift) {
  asm volatile(
      "vdup.32 q0, %[multiplicative_offset]\n"
      "vdup.32 q1, %[rounding_offset]\n"
      "vdup.32 q2, %[shift]\n"
      "vld1.32 {d6[], d7[]}, [%[offsets]:32]!\n"
      "vld1.32 {d8[], d9[]}, [%[offsets]:32]!\n"
      "add r0, %[source], %[stride]\n"
      "add r1, %[destination], %[destination_stride]\n"
      "subs %[count], %[count], #4\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"
      "vld1.32 {d10, d11, d12, d13}, [%[source]:128]!\n"
      "vld1.32 {d14, d15, d16, d17}, [r0:128]!\n"
      "pld [%[source]]\n"
      "pld [r0]\n"
      "vadd.i32 q5, q5, q3\n"
      "vadd.i32 q6, q6, q3\n"
      "vadd.i32 q7, q7, q4\n"
      "vadd.i32 q8, q8, q4\n"
      "vmul.i32 q5, q5, q0\n"
      "vmul.i32 q6, q6, q0\n"
      "vmul.i32 q7, q7, q0\n"
      "vmul.i32 q8, q8, q0\n"
      "vadd.i32 q5, q5, q1\n"
      "vadd.i32 q6, q6, q1\n"
      "vadd.i32 q7, q7, q1\n"
      "vadd.i32 q8, q8, q1\n"
      "vshl.s32 q5, q5, q2\n"
      "vshl.s32 q6, q6, q2\n"
      "vshl.s32 q7, q7, q2\n"
      "vshl.s32 q8, q8, q2\n"
      "vqmovn.s32 d10, q5\n"
      "vqmovn.s32 d11, q6\n"
      "vqmovn.s32 d14, q7\n"
      "vqmovn.s32 d15, q8\n"
      "vqmovun.s16 d10, q5\n"
      "vqmovun.s16 d14, q7\n"
      "vst1.32 {d10}, [%[destination]:64]!\n"
      "vst1.32 {d14}, [r1:64]!\n"

      "bne 1b\n"
      "2:"
      "vld1.32 {d10, d11}, [%[source]:64]!\n"
      "vld1.32 {d14, d15}, [r0:64]!\n"
      "vadd.i32 q5, q5, q3\n"
      "vadd.i32 q7, q7, q4\n"
      "vmul.i32 q5, q5, q0\n"
      "vmul.i32 q7, q7, q0\n"
      "vadd.i32 q5, q5, q1\n"
      "vadd.i32 q7, q7, q1\n"
      "vshl.s32 q5, q5, q2\n"
      "vshl.s32 q7, q7, q2\n"
      "vqmovn.s32 d10, q5\n"
      "vqmovn.s32 d14, q7\n"
      "vqmovun.s16 d10, q5\n"
      "vqmovun.s16 d14, q7\n"
      "vst1.32 {d10[0]}, [%[destination]]!\n"
      "vst1.32 {d14[0]}, [r1]!\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [stride] "+r"(stride), [shift] "+r"(shift),
        [destination] "+r"(destination), [offsets] "+r"(offsets),
        [source] "+r"(source), [destination_stride] "+r"(destination_stride),
        [rounding_offset] "+r"(rounding_offset)
      :
      : "r0", "r1", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9",
        "d10", "d11", "d12", "d13", "d14", "d15", "d16", "d17", "cc", "memory");
}

void qnt_2x8_5_aligned(const std::int32_t* source, std::int32_t count,
                       std::int32_t stride, const std::int32_t* offsets,
                       std::uint8_t* destination,
                       std::int32_t destination_stride,
                       std::int32_t multiplicative_offset,
                       std::int32_t rounding_offset, std::int32_t shift) {
  asm volatile(
      "vdup.32 q0, %[multiplicative_offset]\n"
      "vdup.32 q1, %[rounding_offset]\n"
      "vdup.32 q2, %[shift]\n"
      "vld1.32 {d6[], d7[]}, [%[offsets]:32]!\n"
      "vld1.32 {d8[], d9[]}, [%[offsets]:32]!\n"
      "add r0, %[source], %[stride]\n"
      "add r1, %[destination], %[destination_stride]\n"
      "subs %[count], %[count], #5\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"
      "vld1.32 {d10, d11, d12, d13}, [%[source]:128]!\n"
      "vld1.32 {d14, d15, d16, d17}, [r0:128]!\n"
      "pld [%[source]]\n"
      "pld [r0]\n"
      "vadd.i32 q5, q5, q3\n"
      "vadd.i32 q6, q6, q3\n"
      "vadd.i32 q7, q7, q4\n"
      "vadd.i32 q8, q8, q4\n"
      "vmul.i32 q5, q5, q0\n"
      "vmul.i32 q6, q6, q0\n"
      "vmul.i32 q7, q7, q0\n"
      "vmul.i32 q8, q8, q0\n"
      "vadd.i32 q5, q5, q1\n"
      "vadd.i32 q6, q6, q1\n"
      "vadd.i32 q7, q7, q1\n"
      "vadd.i32 q8, q8, q1\n"
      "vshl.s32 q5, q5, q2\n"
      "vshl.s32 q6, q6, q2\n"
      "vshl.s32 q7, q7, q2\n"
      "vshl.s32 q8, q8, q2\n"
      "vqmovn.s32 d10, q5\n"
      "vqmovn.s32 d11, q6\n"
      "vqmovn.s32 d14, q7\n"
      "vqmovn.s32 d15, q8\n"
      "vqmovun.s16 d10, q5\n"
      "vqmovun.s16 d14, q7\n"
      "vst1.32 {d10}, [%[destination]:64]!\n"
      "vst1.32 {d14}, [r1:64]!\n"

      "bne 1b\n"
      "2:"
      "vld1.32 {d10, d11}, [%[source]:64]!\n"
      "vld1.32 {d12[0]}, [%[source]]!\n"
      "vld1.32 {d14, d15}, [r0:64]!\n"
      "vld1.32 {d16[0]}, [r0]!\n"
      "vadd.i32 q5, q5, q3\n"
      "vadd.i32 q6, q6, q3\n"
      "vadd.i32 q7, q7, q4\n"
      "vadd.i32 q8, q8, q4\n"
      "vmul.i32 q5, q5, q0\n"
      "vmul.i32 q6, q6, q0\n"
      "vmul.i32 q7, q7, q0\n"
      "vmul.i32 q8, q8, q0\n"
      "vadd.i32 q5, q5, q1\n"
      "vadd.i32 q6, q6, q1\n"
      "vadd.i32 q7, q7, q1\n"
      "vadd.i32 q8, q8, q1\n"
      "vshl.s32 q5, q5, q2\n"
      "vshl.s32 q6, q6, q2\n"
      "vshl.s32 q7, q7, q2\n"
      "vshl.s32 q8, q8, q2\n"
      "vqmovn.s32 d10, q5\n"
      "vqmovn.s32 d11, q6\n"
      "vqmovn.s32 d14, q7\n"
      "vqmovn.s32 d15, q8\n"
      "vqmovun.s16 d10, q5\n"
      "vqmovun.s16 d14, q7\n"
      "vst1.32 {d10[0]}, [%[destination]]!\n"
      "vst1.8 {d10[4]}, [%[destination]]!\n"
      "vst1.32 {d14[0]}, [r1]!\n"
      "vst1.8 {d14[4]}, [r1]!\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [stride] "+r"(stride), [shift] "+r"(shift),
        [destination] "+r"(destination), [offsets] "+r"(offsets),
        [source] "+r"(source), [destination_stride] "+r"(destination_stride),
        [rounding_offset] "+r"(rounding_offset)
      :
      : "r0", "r1", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9",
        "d10", "d11", "d12", "d13", "d14", "d15", "d16", "d17", "cc", "memory");
}

void qnt_2x8_6_aligned(const std::int32_t* source, std::int32_t count,
                       std::int32_t stride, const std::int32_t* offsets,
                       std::uint8_t* destination,
                       std::int32_t destination_stride,
                       std::int32_t multiplicative_offset,
                       std::int32_t rounding_offset, std::int32_t shift) {
  asm volatile(
      "vdup.32 q0, %[multiplicative_offset]\n"
      "vdup.32 q1, %[rounding_offset]\n"
      "vdup.32 q2, %[shift]\n"
      "vld1.32 {d6[], d7[]}, [%[offsets]:32]!\n"
      "vld1.32 {d8[], d9[]}, [%[offsets]:32]!\n"
      "add r0, %[source], %[stride]\n"
      "add r1, %[destination], %[destination_stride]\n"
      "subs %[count], %[count], #6\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"
      "vld1.32 {d10, d11, d12, d13}, [%[source]:128]!\n"
      "vld1.32 {d14, d15, d16, d17}, [r0:128]!\n"
      "pld [%[source]]\n"
      "pld [r0]\n"
      "vadd.i32 q5, q5, q3\n"
      "vadd.i32 q6, q6, q3\n"
      "vadd.i32 q7, q7, q4\n"
      "vadd.i32 q8, q8, q4\n"
      "vmul.i32 q5, q5, q0\n"
      "vmul.i32 q6, q6, q0\n"
      "vmul.i32 q7, q7, q0\n"
      "vmul.i32 q8, q8, q0\n"
      "vadd.i32 q5, q5, q1\n"
      "vadd.i32 q6, q6, q1\n"
      "vadd.i32 q7, q7, q1\n"
      "vadd.i32 q8, q8, q1\n"
      "vshl.s32 q5, q5, q2\n"
      "vshl.s32 q6, q6, q2\n"
      "vshl.s32 q7, q7, q2\n"
      "vshl.s32 q8, q8, q2\n"
      "vqmovn.s32 d10, q5\n"
      "vqmovn.s32 d11, q6\n"
      "vqmovn.s32 d14, q7\n"
      "vqmovn.s32 d15, q8\n"
      "vqmovun.s16 d10, q5\n"
      "vqmovun.s16 d14, q7\n"
      "vst1.32 {d10}, [%[destination]:64]!\n"
      "vst1.32 {d14}, [r1:64]!\n"

      "bne 1b\n"
      "2:"
      "vld1.32 {d10, d11, d12}, [%[source]:64]!\n"
      "vld1.32 {d14, d15, d16}, [r0:64]!\n"
      "vadd.i32 q5, q5, q3\n"
      "vadd.i32 q6, q6, q3\n"
      "vadd.i32 q7, q7, q4\n"
      "vadd.i32 q8, q8, q4\n"
      "vmul.i32 q5, q5, q0\n"
      "vmul.i32 q6, q6, q0\n"
      "vmul.i32 q7, q7, q0\n"
      "vmul.i32 q8, q8, q0\n"
      "vadd.i32 q5, q5, q1\n"
      "vadd.i32 q6, q6, q1\n"
      "vadd.i32 q7, q7, q1\n"
      "vadd.i32 q8, q8, q1\n"
      "vshl.s32 q5, q5, q2\n"
      "vshl.s32 q6, q6, q2\n"
      "vshl.s32 q7, q7, q2\n"
      "vshl.s32 q8, q8, q2\n"
      "vqmovn.s32 d10, q5\n"
      "vqmovn.s32 d11, q6\n"
      "vqmovn.s32 d14, q7\n"
      "vqmovn.s32 d15, q8\n"
      "vqmovun.s16 d10, q5\n"
      "vqmovun.s16 d14, q7\n"
      "vst1.32 {d10[0]}, [%[destination]]!\n"
      "vst1.16 {d10[2]}, [%[destination]]!\n"
      "vst1.32 {d14[0]}, [r1]!\n"
      "vst1.16 {d14[2]}, [r1]!\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [stride] "+r"(stride), [shift] "+r"(shift),
        [destination] "+r"(destination), [offsets] "+r"(offsets),
        [source] "+r"(source), [destination_stride] "+r"(destination_stride),
        [rounding_offset] "+r"(rounding_offset)
      :
      : "r0", "r1", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9",
        "d10", "d11", "d12", "d13", "d14", "d15", "d16", "d17", "cc", "memory");
}

void qnt_2x8_7_aligned(const std::int32_t* source, std::int32_t count,
                       std::int32_t stride, const std::int32_t* offsets,
                       std::uint8_t* destination,
                       std::int32_t destination_stride,
                       std::int32_t multiplicative_offset,
                       std::int32_t rounding_offset, std::int32_t shift) {
  asm volatile(
      "vdup.32 q0, %[multiplicative_offset]\n"
      "vdup.32 q1, %[rounding_offset]\n"
      "vdup.32 q2, %[shift]\n"
      "vld1.32 {d6[], d7[]}, [%[offsets]:32]!\n"
      "vld1.32 {d8[], d9[]}, [%[offsets]:32]!\n"
      "add r0, %[source], %[stride]\n"
      "add r1, %[destination], %[destination_stride]\n"
      "subs %[count], %[count], #7\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"
      "vld1.32 {d10, d11, d12, d13}, [%[source]:128]!\n"
      "vld1.32 {d14, d15, d16, d17}, [r0:128]!\n"
      "pld [%[source]]\n"
      "pld [r0]\n"
      "vadd.i32 q5, q5, q3\n"
      "vadd.i32 q6, q6, q3\n"
      "vadd.i32 q7, q7, q4\n"
      "vadd.i32 q8, q8, q4\n"
      "vmul.i32 q5, q5, q0\n"
      "vmul.i32 q6, q6, q0\n"
      "vmul.i32 q7, q7, q0\n"
      "vmul.i32 q8, q8, q0\n"
      "vadd.i32 q5, q5, q1\n"
      "vadd.i32 q6, q6, q1\n"
      "vadd.i32 q7, q7, q1\n"
      "vadd.i32 q8, q8, q1\n"
      "vshl.s32 q5, q5, q2\n"
      "vshl.s32 q6, q6, q2\n"
      "vshl.s32 q7, q7, q2\n"
      "vshl.s32 q8, q8, q2\n"
      "vqmovn.s32 d10, q5\n"
      "vqmovn.s32 d11, q6\n"
      "vqmovn.s32 d14, q7\n"
      "vqmovn.s32 d15, q8\n"
      "vqmovun.s16 d10, q5\n"
      "vqmovun.s16 d14, q7\n"
      "vst1.32 {d10}, [%[destination]:64]!\n"
      "vst1.32 {d14}, [r1:64]!\n"

      "bne 1b\n"
      "2:"
      "vld1.32 {d10, d11, d12}, [%[source]:64]!\n"
      "vld1.32 {d13[0]}, [%[source]]!\n"
      "vld1.32 {d14, d15, d16}, [r0:64]!\n"
      "vld1.32 {d17[0]}, [r0]!\n"
      "vadd.i32 q5, q5, q3\n"
      "vadd.i32 q6, q6, q3\n"
      "vadd.i32 q7, q7, q4\n"
      "vadd.i32 q8, q8, q4\n"
      "vmul.i32 q5, q5, q0\n"
      "vmul.i32 q6, q6, q0\n"
      "vmul.i32 q7, q7, q0\n"
      "vmul.i32 q8, q8, q0\n"
      "vadd.i32 q5, q5, q1\n"
      "vadd.i32 q6, q6, q1\n"
      "vadd.i32 q7, q7, q1\n"
      "vadd.i32 q8, q8, q1\n"
      "vshl.s32 q5, q5, q2\n"
      "vshl.s32 q6, q6, q2\n"
      "vshl.s32 q7, q7, q2\n"
      "vshl.s32 q8, q8, q2\n"
      "vqmovn.s32 d10, q5\n"
      "vqmovn.s32 d11, q6\n"
      "vqmovn.s32 d14, q7\n"
      "vqmovn.s32 d15, q8\n"
      "vqmovun.s16 d10, q5\n"
      "vqmovun.s16 d14, q7\n"
      "vst1.32 {d10[0]}, [%[destination]]!\n"
      "vst1.16 {d10[2]}, [%[destination]]!\n"
      "vst1.8 {d10[6]}, [%[destination]]!\n"
      "vst1.32 {d14[0]}, [r1]!\n"
      "vst1.16 {d14[2]}, [r1]!\n"
      "vst1.8 {d14[6]}, [r1]!\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [stride] "+r"(stride), [shift] "+r"(shift),
        [destination] "+r"(destination), [offsets] "+r"(offsets),
        [source] "+r"(source), [destination_stride] "+r"(destination_stride),
        [rounding_offset] "+r"(rounding_offset)
      :
      : "r0", "r1", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9",
        "d10", "d11", "d12", "d13", "d14", "d15", "d16", "d17", "cc", "memory");
}

void qnt_3x8_aligned(const std::int32_t* source, std::int32_t count,
                     std::int32_t stride, const std::int32_t* offsets,
                     std::uint8_t* destination, std::int32_t destination_stride,
                     std::int32_t multiplicative_offset,
                     std::int32_t rounding_offset, std::int32_t shift) {
  asm volatile(
      "vdup.32 q0, %[multiplicative_offset]\n"
      "vdup.32 q1, %[rounding_offset]\n"
      "vdup.32 q2, %[shift]\n"
      "vld1.32 {d6[], d7[]}, [%[offsets]:32]!\n"
      "vld1.32 {d8[], d9[]}, [%[offsets]:32]!\n"
      "vld1.32 {d10[], d11[]}, [%[offsets]:32]!\n"
      "add r0, %[source], %[stride]\n"
      "add r1, %[destination], %[destination_stride]\n"
      "add r2, r0, %[stride]\n"
      "add r3, r1, %[destination_stride]\n"

      "1:"
      "subs %[count], %[count], #8\n"
      "vld1.32 {d12, d13, d14, d15}, [%[source]:128]!\n"
      "vld1.32 {d16, d17, d18, d19}, [r0:128]!\n"
      "vld1.32 {d20, d21, d22, d23}, [r2:128]!\n"
      "pld [%[source]]\n"
      "pld [r0]\n"
      "pld [r2]\n"
      "vadd.i32 q6, q6, q3\n"
      "vadd.i32 q7, q7, q3\n"
      "vadd.i32 q8, q8, q4\n"
      "vadd.i32 q9, q9, q4\n"
      "vadd.i32 q10, q10, q5\n"
      "vadd.i32 q11, q11, q5\n"
      "vmul.i32 q6, q6, q0\n"
      "vmul.i32 q7, q7, q0\n"
      "vmul.i32 q8, q8, q0\n"
      "vmul.i32 q9, q9, q0\n"
      "vmul.i32 q10, q10, q0\n"
      "vmul.i32 q11, q11, q0\n"
      "vadd.i32 q6, q6, q1\n"
      "vadd.i32 q7, q7, q1\n"
      "vadd.i32 q8, q8, q1\n"
      "vadd.i32 q9, q9, q1\n"
      "vadd.i32 q10, q10, q1\n"
      "vadd.i32 q11, q11, q1\n"
      "vshl.s32 q6, q6, q2\n"
      "vshl.s32 q7, q7, q2\n"
      "vshl.s32 q8, q8, q2\n"
      "vshl.s32 q9, q9, q2\n"
      "vshl.s32 q10, q10, q2\n"
      "vshl.s32 q11, q11, q2\n"
      "vqmovn.s32 d12, q6\n"
      "vqmovn.s32 d13, q7\n"
      "vqmovn.s32 d16, q8\n"
      "vqmovn.s32 d17, q9\n"
      "vqmovn.s32 d20, q10\n"
      "vqmovn.s32 d21, q11\n"
      "vqmovun.s16 d12, q6\n"
      "vqmovun.s16 d16, q8\n"
      "vqmovun.s16 d20, q10\n"
      "vst1.32 {d12}, [%[destination]:64]!\n"
      "vst1.32 {d16}, [r1:64]!\n"
      "vst1.32 {d20}, [r3:64]!\n"

      "bne 1b\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [stride] "+r"(stride), [shift] "+r"(shift),
        [destination] "+r"(destination), [offsets] "+r"(offsets),
        [source] "+r"(source), [destination_stride] "+r"(destination_stride),
        [rounding_offset] "+r"(rounding_offset)
      :
      : "r0", "r1", "r2", "r3", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7",
        "d8", "d9", "d10", "d11", "d12", "d13", "d14", "d15", "d16", "d17",
        "d18", "d19", "d20", "d21", "d22", "d23", "cc", "memory");
}

void qnt_3x8_1_aligned(const std::int32_t* source, std::int32_t count,
                       std::int32_t stride, const std::int32_t* offsets,
                       std::uint8_t* destination,
                       std::int32_t destination_stride,
                       std::int32_t multiplicative_offset,
                       std::int32_t rounding_offset, std::int32_t shift) {
  asm volatile(
      "vdup.32 q0, %[multiplicative_offset]\n"
      "vdup.32 q1, %[rounding_offset]\n"
      "vdup.32 q2, %[shift]\n"
      "vld1.32 {d6[], d7[]}, [%[offsets]:32]!\n"
      "vld1.32 {d8[], d9[]}, [%[offsets]:32]!\n"
      "vld1.32 {d10[], d11[]}, [%[offsets]:32]!\n"
      "add r0, %[source], %[stride]\n"
      "add r1, %[destination], %[destination_stride]\n"
      "add r2, r0, %[stride]\n"
      "add r3, r1, %[destination_stride]\n"
      "subs %[count], %[count], #1\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"
      "vld1.32 {d12, d13, d14, d15}, [%[source]:128]!\n"
      "vld1.32 {d16, d17, d18, d19}, [r0:128]!\n"
      "vld1.32 {d20, d21, d22, d23}, [r2:128]!\n"
      "pld [%[source]]\n"
      "pld [r0]\n"
      "pld [r2]\n"
      "vadd.i32 q6, q6, q3\n"
      "vadd.i32 q7, q7, q3\n"
      "vadd.i32 q8, q8, q4\n"
      "vadd.i32 q9, q9, q4\n"
      "vadd.i32 q10, q10, q5\n"
      "vadd.i32 q11, q11, q5\n"
      "vmul.i32 q6, q6, q0\n"
      "vmul.i32 q7, q7, q0\n"
      "vmul.i32 q8, q8, q0\n"
      "vmul.i32 q9, q9, q0\n"
      "vmul.i32 q10, q10, q0\n"
      "vmul.i32 q11, q11, q0\n"
      "vadd.i32 q6, q6, q1\n"
      "vadd.i32 q7, q7, q1\n"
      "vadd.i32 q8, q8, q1\n"
      "vadd.i32 q9, q9, q1\n"
      "vadd.i32 q10, q10, q1\n"
      "vadd.i32 q11, q11, q1\n"
      "vshl.s32 q6, q6, q2\n"
      "vshl.s32 q7, q7, q2\n"
      "vshl.s32 q8, q8, q2\n"
      "vshl.s32 q9, q9, q2\n"
      "vshl.s32 q10, q10, q2\n"
      "vshl.s32 q11, q11, q2\n"
      "vqmovn.s32 d12, q6\n"
      "vqmovn.s32 d13, q7\n"
      "vqmovn.s32 d16, q8\n"
      "vqmovn.s32 d17, q9\n"
      "vqmovn.s32 d20, q10\n"
      "vqmovn.s32 d21, q11\n"
      "vqmovun.s16 d12, q6\n"
      "vqmovun.s16 d16, q8\n"
      "vqmovun.s16 d20, q10\n"
      "vst1.32 {d12}, [%[destination]:64]!\n"
      "vst1.32 {d16}, [r1:64]!\n"
      "vst1.32 {d20}, [r3:64]!\n"

      "bne 1b\n"
      "2:"
      "vld1.32 {d12[0]}, [%[source]]!\n"
      "vld1.32 {d16[0]}, [r0]!\n"
      "vld1.32 {d20[0]}, [r2]!\n"
      "vadd.i32 q6, q6, q3\n"
      "vadd.i32 q8, q8, q4\n"
      "vadd.i32 q10, q10, q5\n"
      "vmul.i32 q6, q6, q0\n"
      "vmul.i32 q8, q8, q0\n"
      "vmul.i32 q10, q10, q0\n"
      "vadd.i32 q6, q6, q1\n"
      "vadd.i32 q8, q8, q1\n"
      "vadd.i32 q10, q10, q1\n"
      "vshl.s32 q6, q6, q2\n"
      "vshl.s32 q8, q8, q2\n"
      "vshl.s32 q10, q10, q2\n"
      "vqmovn.s32 d12, q6\n"
      "vqmovn.s32 d16, q8\n"
      "vqmovn.s32 d20, q10\n"
      "vqmovun.s16 d12, q6\n"
      "vqmovun.s16 d16, q8\n"
      "vqmovun.s16 d20, q10\n"
      "vst1.8 {d12[0]}, [%[destination]]!\n"
      "vst1.8 {d16[0]}, [r1]!\n"
      "vst1.8 {d20[0]}, [r3]!\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [stride] "+r"(stride), [shift] "+r"(shift),
        [destination] "+r"(destination), [offsets] "+r"(offsets),
        [source] "+r"(source), [destination_stride] "+r"(destination_stride),
        [rounding_offset] "+r"(rounding_offset)
      :
      : "r0", "r1", "r2", "r3", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7",
        "d8", "d9", "d10", "d11", "d12", "d13", "d14", "d15", "d16", "d17",
        "d18", "d19", "d20", "d21", "d22", "d23", "cc", "memory");
}

void qnt_3x8_2_aligned(const std::int32_t* source, std::int32_t count,
                       std::int32_t stride, const std::int32_t* offsets,
                       std::uint8_t* destination,
                       std::int32_t destination_stride,
                       std::int32_t multiplicative_offset,
                       std::int32_t rounding_offset, std::int32_t shift) {
  asm volatile(
      "vdup.32 q0, %[multiplicative_offset]\n"
      "vdup.32 q1, %[rounding_offset]\n"
      "vdup.32 q2, %[shift]\n"
      "vld1.32 {d6[], d7[]}, [%[offsets]:32]!\n"
      "vld1.32 {d8[], d9[]}, [%[offsets]:32]!\n"
      "vld1.32 {d10[], d11[]}, [%[offsets]:32]!\n"
      "add r0, %[source], %[stride]\n"
      "add r1, %[destination], %[destination_stride]\n"
      "add r2, r0, %[stride]\n"
      "add r3, r1, %[destination_stride]\n"
      "subs %[count], %[count], #2\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"
      "vld1.32 {d12, d13, d14, d15}, [%[source]:128]!\n"
      "vld1.32 {d16, d17, d18, d19}, [r0:128]!\n"
      "vld1.32 {d20, d21, d22, d23}, [r2:128]!\n"
      "pld [%[source]]\n"
      "pld [r0]\n"
      "pld [r2]\n"
      "vadd.i32 q6, q6, q3\n"
      "vadd.i32 q7, q7, q3\n"
      "vadd.i32 q8, q8, q4\n"
      "vadd.i32 q9, q9, q4\n"
      "vadd.i32 q10, q10, q5\n"
      "vadd.i32 q11, q11, q5\n"
      "vmul.i32 q6, q6, q0\n"
      "vmul.i32 q7, q7, q0\n"
      "vmul.i32 q8, q8, q0\n"
      "vmul.i32 q9, q9, q0\n"
      "vmul.i32 q10, q10, q0\n"
      "vmul.i32 q11, q11, q0\n"
      "vadd.i32 q6, q6, q1\n"
      "vadd.i32 q7, q7, q1\n"
      "vadd.i32 q8, q8, q1\n"
      "vadd.i32 q9, q9, q1\n"
      "vadd.i32 q10, q10, q1\n"
      "vadd.i32 q11, q11, q1\n"
      "vshl.s32 q6, q6, q2\n"
      "vshl.s32 q7, q7, q2\n"
      "vshl.s32 q8, q8, q2\n"
      "vshl.s32 q9, q9, q2\n"
      "vshl.s32 q10, q10, q2\n"
      "vshl.s32 q11, q11, q2\n"
      "vqmovn.s32 d12, q6\n"
      "vqmovn.s32 d13, q7\n"
      "vqmovn.s32 d16, q8\n"
      "vqmovn.s32 d17, q9\n"
      "vqmovn.s32 d20, q10\n"
      "vqmovn.s32 d21, q11\n"
      "vqmovun.s16 d12, q6\n"
      "vqmovun.s16 d16, q8\n"
      "vqmovun.s16 d20, q10\n"
      "vst1.32 {d12}, [%[destination]:64]!\n"
      "vst1.32 {d16}, [r1:64]!\n"
      "vst1.32 {d20}, [r3:64]!\n"

      "bne 1b\n"
      "2:"
      "vld1.32 {d12}, [%[source]:64]!\n"
      "vld1.32 {d16}, [r0:64]!\n"
      "vld1.32 {d20}, [r2:64]!\n"
      "vadd.i32 q6, q6, q3\n"
      "vadd.i32 q8, q8, q4\n"
      "vadd.i32 q10, q10, q5\n"
      "vmul.i32 q6, q6, q0\n"
      "vmul.i32 q8, q8, q0\n"
      "vmul.i32 q10, q10, q0\n"
      "vadd.i32 q6, q6, q1\n"
      "vadd.i32 q8, q8, q1\n"
      "vadd.i32 q10, q10, q1\n"
      "vshl.s32 q6, q6, q2\n"
      "vshl.s32 q8, q8, q2\n"
      "vshl.s32 q10, q10, q2\n"
      "vqmovn.s32 d12, q6\n"
      "vqmovn.s32 d16, q8\n"
      "vqmovn.s32 d20, q10\n"
      "vqmovun.s16 d12, q6\n"
      "vqmovun.s16 d16, q8\n"
      "vqmovun.s16 d20, q10\n"
      "vst1.16 {d12[0]}, [%[destination]]!\n"
      "vst1.16 {d16[0]}, [r1]!\n"
      "vst1.16 {d20[0]}, [r3]!\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [stride] "+r"(stride), [shift] "+r"(shift),
        [destination] "+r"(destination), [offsets] "+r"(offsets),
        [source] "+r"(source), [destination_stride] "+r"(destination_stride),
        [rounding_offset] "+r"(rounding_offset)
      :
      : "r0", "r1", "r2", "r3", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7",
        "d8", "d9", "d10", "d11", "d12", "d13", "d14", "d15", "d16", "d17",
        "d18", "d19", "d20", "d21", "d22", "d23", "cc", "memory");
}

void qnt_3x8_3_aligned(const std::int32_t* source, std::int32_t count,
                       std::int32_t stride, const std::int32_t* offsets,
                       std::uint8_t* destination,
                       std::int32_t destination_stride,
                       std::int32_t multiplicative_offset,
                       std::int32_t rounding_offset, std::int32_t shift) {
  asm volatile(
      "vdup.32 q0, %[multiplicative_offset]\n"
      "vdup.32 q1, %[rounding_offset]\n"
      "vdup.32 q2, %[shift]\n"
      "vld1.32 {d6[], d7[]}, [%[offsets]:32]!\n"
      "vld1.32 {d8[], d9[]}, [%[offsets]:32]!\n"
      "vld1.32 {d10[], d11[]}, [%[offsets]:32]!\n"
      "add r0, %[source], %[stride]\n"
      "add r1, %[destination], %[destination_stride]\n"
      "add r2, r0, %[stride]\n"
      "add r3, r1, %[destination_stride]\n"
      "subs %[count], %[count], #3\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"
      "vld1.32 {d12, d13, d14, d15}, [%[source]:128]!\n"
      "vld1.32 {d16, d17, d18, d19}, [r0:128]!\n"
      "vld1.32 {d20, d21, d22, d23}, [r2:128]!\n"
      "pld [%[source]]\n"
      "pld [r0]\n"
      "pld [r2]\n"
      "vadd.i32 q6, q6, q3\n"
      "vadd.i32 q7, q7, q3\n"
      "vadd.i32 q8, q8, q4\n"
      "vadd.i32 q9, q9, q4\n"
      "vadd.i32 q10, q10, q5\n"
      "vadd.i32 q11, q11, q5\n"
      "vmul.i32 q6, q6, q0\n"
      "vmul.i32 q7, q7, q0\n"
      "vmul.i32 q8, q8, q0\n"
      "vmul.i32 q9, q9, q0\n"
      "vmul.i32 q10, q10, q0\n"
      "vmul.i32 q11, q11, q0\n"
      "vadd.i32 q6, q6, q1\n"
      "vadd.i32 q7, q7, q1\n"
      "vadd.i32 q8, q8, q1\n"
      "vadd.i32 q9, q9, q1\n"
      "vadd.i32 q10, q10, q1\n"
      "vadd.i32 q11, q11, q1\n"
      "vshl.s32 q6, q6, q2\n"
      "vshl.s32 q7, q7, q2\n"
      "vshl.s32 q8, q8, q2\n"
      "vshl.s32 q9, q9, q2\n"
      "vshl.s32 q10, q10, q2\n"
      "vshl.s32 q11, q11, q2\n"
      "vqmovn.s32 d12, q6\n"
      "vqmovn.s32 d13, q7\n"
      "vqmovn.s32 d16, q8\n"
      "vqmovn.s32 d17, q9\n"
      "vqmovn.s32 d20, q10\n"
      "vqmovn.s32 d21, q11\n"
      "vqmovun.s16 d12, q6\n"
      "vqmovun.s16 d16, q8\n"
      "vqmovun.s16 d20, q10\n"
      "vst1.32 {d12}, [%[destination]:64]!\n"
      "vst1.32 {d16}, [r1:64]!\n"
      "vst1.32 {d20}, [r3:64]!\n"

      "bne 1b\n"
      "2:"
      "vld1.32 {d12}, [%[source]:64]!\n"
      "vld1.32 {d13[0]}, [%[source]]!\n"
      "vld1.32 {d16}, [r0:64]!\n"
      "vld1.32 {d17[0]}, [r0]!\n"
      "vld1.32 {d20}, [r2:64]!\n"
      "vld1.32 {d21[0]}, [r2]!\n"
      "vadd.i32 q6, q6, q3\n"
      "vadd.i32 q8, q8, q4\n"
      "vadd.i32 q10, q10, q5\n"
      "vmul.i32 q6, q6, q0\n"
      "vmul.i32 q8, q8, q0\n"
      "vmul.i32 q10, q10, q0\n"
      "vadd.i32 q6, q6, q1\n"
      "vadd.i32 q8, q8, q1\n"
      "vadd.i32 q10, q10, q1\n"
      "vshl.s32 q6, q6, q2\n"
      "vshl.s32 q8, q8, q2\n"
      "vshl.s32 q10, q10, q2\n"
      "vqmovn.s32 d12, q6\n"
      "vqmovn.s32 d16, q8\n"
      "vqmovn.s32 d20, q10\n"
      "vqmovun.s16 d12, q6\n"
      "vqmovun.s16 d16, q8\n"
      "vqmovun.s16 d20, q10\n"
      "vst1.16 {d12[0]}, [%[destination]]!\n"
      "vst1.8 {d12[2]}, [%[destination]]!\n"
      "vst1.16 {d16[0]}, [r1]!\n"
      "vst1.8 {d16[2]}, [r1]!\n"
      "vst1.16 {d20[0]}, [r3]!\n"
      "vst1.8 {d20[2]}, [r3]!\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [stride] "+r"(stride), [shift] "+r"(shift),
        [destination] "+r"(destination), [offsets] "+r"(offsets),
        [source] "+r"(source), [destination_stride] "+r"(destination_stride),
        [rounding_offset] "+r"(rounding_offset)
      :
      : "r0", "r1", "r2", "r3", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7",
        "d8", "d9", "d10", "d11", "d12", "d13", "d14", "d15", "d16", "d17",
        "d18", "d19", "d20", "d21", "d22", "d23", "cc", "memory");
}

void qnt_3x8_4_aligned(const std::int32_t* source, std::int32_t count,
                       std::int32_t stride, const std::int32_t* offsets,
                       std::uint8_t* destination,
                       std::int32_t destination_stride,
                       std::int32_t multiplicative_offset,
                       std::int32_t rounding_offset, std::int32_t shift) {
  asm volatile(
      "vdup.32 q0, %[multiplicative_offset]\n"
      "vdup.32 q1, %[rounding_offset]\n"
      "vdup.32 q2, %[shift]\n"
      "vld1.32 {d6[], d7[]}, [%[offsets]:32]!\n"
      "vld1.32 {d8[], d9[]}, [%[offsets]:32]!\n"
      "vld1.32 {d10[], d11[]}, [%[offsets]:32]!\n"
      "add r0, %[source], %[stride]\n"
      "add r1, %[destination], %[destination_stride]\n"
      "add r2, r0, %[stride]\n"
      "add r3, r1, %[destination_stride]\n"
      "subs %[count], %[count], #4\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"
      "vld1.32 {d12, d13, d14, d15}, [%[source]:128]!\n"
      "vld1.32 {d16, d17, d18, d19}, [r0:128]!\n"
      "vld1.32 {d20, d21, d22, d23}, [r2:128]!\n"
      "pld [%[source]]\n"
      "pld [r0]\n"
      "pld [r2]\n"
      "vadd.i32 q6, q6, q3\n"
      "vadd.i32 q7, q7, q3\n"
      "vadd.i32 q8, q8, q4\n"
      "vadd.i32 q9, q9, q4\n"
      "vadd.i32 q10, q10, q5\n"
      "vadd.i32 q11, q11, q5\n"
      "vmul.i32 q6, q6, q0\n"
      "vmul.i32 q7, q7, q0\n"
      "vmul.i32 q8, q8, q0\n"
      "vmul.i32 q9, q9, q0\n"
      "vmul.i32 q10, q10, q0\n"
      "vmul.i32 q11, q11, q0\n"
      "vadd.i32 q6, q6, q1\n"
      "vadd.i32 q7, q7, q1\n"
      "vadd.i32 q8, q8, q1\n"
      "vadd.i32 q9, q9, q1\n"
      "vadd.i32 q10, q10, q1\n"
      "vadd.i32 q11, q11, q1\n"
      "vshl.s32 q6, q6, q2\n"
      "vshl.s32 q7, q7, q2\n"
      "vshl.s32 q8, q8, q2\n"
      "vshl.s32 q9, q9, q2\n"
      "vshl.s32 q10, q10, q2\n"
      "vshl.s32 q11, q11, q2\n"
      "vqmovn.s32 d12, q6\n"
      "vqmovn.s32 d13, q7\n"
      "vqmovn.s32 d16, q8\n"
      "vqmovn.s32 d17, q9\n"
      "vqmovn.s32 d20, q10\n"
      "vqmovn.s32 d21, q11\n"
      "vqmovun.s16 d12, q6\n"
      "vqmovun.s16 d16, q8\n"
      "vqmovun.s16 d20, q10\n"
      "vst1.32 {d12}, [%[destination]:64]!\n"
      "vst1.32 {d16}, [r1:64]!\n"
      "vst1.32 {d20}, [r3:64]!\n"

      "bne 1b\n"
      "2:"
      "vld1.32 {d12, d13}, [%[source]:64]!\n"
      "vld1.32 {d16, d17}, [r0:64]!\n"
      "vld1.32 {d20, d21}, [r2:64]!\n"
      "vadd.i32 q6, q6, q3\n"
      "vadd.i32 q8, q8, q4\n"
      "vadd.i32 q10, q10, q5\n"
      "vmul.i32 q6, q6, q0\n"
      "vmul.i32 q8, q8, q0\n"
      "vmul.i32 q10, q10, q0\n"
      "vadd.i32 q6, q6, q1\n"
      "vadd.i32 q8, q8, q1\n"
      "vadd.i32 q10, q10, q1\n"
      "vshl.s32 q6, q6, q2\n"
      "vshl.s32 q8, q8, q2\n"
      "vshl.s32 q10, q10, q2\n"
      "vqmovn.s32 d12, q6\n"
      "vqmovn.s32 d16, q8\n"
      "vqmovn.s32 d20, q10\n"
      "vqmovun.s16 d12, q6\n"
      "vqmovun.s16 d16, q8\n"
      "vqmovun.s16 d20, q10\n"
      "vst1.32 {d12[0]}, [%[destination]]!\n"
      "vst1.32 {d16[0]}, [r1]!\n"
      "vst1.32 {d20[0]}, [r3]!\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [stride] "+r"(stride), [shift] "+r"(shift),
        [destination] "+r"(destination), [offsets] "+r"(offsets),
        [source] "+r"(source), [destination_stride] "+r"(destination_stride),
        [rounding_offset] "+r"(rounding_offset)
      :
      : "r0", "r1", "r2", "r3", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7",
        "d8", "d9", "d10", "d11", "d12", "d13", "d14", "d15", "d16", "d17",
        "d18", "d19", "d20", "d21", "d22", "d23", "cc", "memory");
}

void qnt_3x8_5_aligned(const std::int32_t* source, std::int32_t count,
                       std::int32_t stride, const std::int32_t* offsets,
                       std::uint8_t* destination,
                       std::int32_t destination_stride,
                       std::int32_t multiplicative_offset,
                       std::int32_t rounding_offset, std::int32_t shift) {
  asm volatile(
      "vdup.32 q0, %[multiplicative_offset]\n"
      "vdup.32 q1, %[rounding_offset]\n"
      "vdup.32 q2, %[shift]\n"
      "vld1.32 {d6[], d7[]}, [%[offsets]:32]!\n"
      "vld1.32 {d8[], d9[]}, [%[offsets]:32]!\n"
      "vld1.32 {d10[], d11[]}, [%[offsets]:32]!\n"
      "add r0, %[source], %[stride]\n"
      "add r1, %[destination], %[destination_stride]\n"
      "add r2, r0, %[stride]\n"
      "add r3, r1, %[destination_stride]\n"
      "subs %[count], %[count], #5\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"
      "vld1.32 {d12, d13, d14, d15}, [%[source]:128]!\n"
      "vld1.32 {d16, d17, d18, d19}, [r0:128]!\n"
      "vld1.32 {d20, d21, d22, d23}, [r2:128]!\n"
      "pld [%[source]]\n"
      "pld [r0]\n"
      "pld [r2]\n"
      "vadd.i32 q6, q6, q3\n"
      "vadd.i32 q7, q7, q3\n"
      "vadd.i32 q8, q8, q4\n"
      "vadd.i32 q9, q9, q4\n"
      "vadd.i32 q10, q10, q5\n"
      "vadd.i32 q11, q11, q5\n"
      "vmul.i32 q6, q6, q0\n"
      "vmul.i32 q7, q7, q0\n"
      "vmul.i32 q8, q8, q0\n"
      "vmul.i32 q9, q9, q0\n"
      "vmul.i32 q10, q10, q0\n"
      "vmul.i32 q11, q11, q0\n"
      "vadd.i32 q6, q6, q1\n"
      "vadd.i32 q7, q7, q1\n"
      "vadd.i32 q8, q8, q1\n"
      "vadd.i32 q9, q9, q1\n"
      "vadd.i32 q10, q10, q1\n"
      "vadd.i32 q11, q11, q1\n"
      "vshl.s32 q6, q6, q2\n"
      "vshl.s32 q7, q7, q2\n"
      "vshl.s32 q8, q8, q2\n"
      "vshl.s32 q9, q9, q2\n"
      "vshl.s32 q10, q10, q2\n"
      "vshl.s32 q11, q11, q2\n"
      "vqmovn.s32 d12, q6\n"
      "vqmovn.s32 d13, q7\n"
      "vqmovn.s32 d16, q8\n"
      "vqmovn.s32 d17, q9\n"
      "vqmovn.s32 d20, q10\n"
      "vqmovn.s32 d21, q11\n"
      "vqmovun.s16 d12, q6\n"
      "vqmovun.s16 d16, q8\n"
      "vqmovun.s16 d20, q10\n"
      "vst1.32 {d12}, [%[destination]:64]!\n"
      "vst1.32 {d16}, [r1:64]!\n"
      "vst1.32 {d20}, [r3:64]!\n"

      "bne 1b\n"
      "2:"
      "vld1.32 {d12, d13}, [%[source]:64]!\n"
      "vld1.32 {d14[0]}, [%[source]]!\n"
      "vld1.32 {d16, d17}, [r0:64]!\n"
      "vld1.32 {d18[0]}, [r0]!\n"
      "vld1.32 {d20, d21}, [r2:64]!\n"
      "vld1.32 {d22[0]}, [r2]!\n"
      "vadd.i32 q6, q6, q3\n"
      "vadd.i32 q7, q7, q3\n"
      "vadd.i32 q8, q8, q4\n"
      "vadd.i32 q9, q9, q4\n"
      "vadd.i32 q10, q10, q5\n"
      "vadd.i32 q11, q11, q5\n"
      "vmul.i32 q6, q6, q0\n"
      "vmul.i32 q7, q7, q0\n"
      "vmul.i32 q8, q8, q0\n"
      "vmul.i32 q9, q9, q0\n"
      "vmul.i32 q10, q10, q0\n"
      "vmul.i32 q11, q11, q0\n"
      "vadd.i32 q6, q6, q1\n"
      "vadd.i32 q7, q7, q1\n"
      "vadd.i32 q8, q8, q1\n"
      "vadd.i32 q9, q9, q1\n"
      "vadd.i32 q10, q10, q1\n"
      "vadd.i32 q11, q11, q1\n"
      "vshl.s32 q6, q6, q2\n"
      "vshl.s32 q7, q7, q2\n"
      "vshl.s32 q8, q8, q2\n"
      "vshl.s32 q9, q9, q2\n"
      "vshl.s32 q10, q10, q2\n"
      "vshl.s32 q11, q11, q2\n"
      "vqmovn.s32 d12, q6\n"
      "vqmovn.s32 d13, q7\n"
      "vqmovn.s32 d16, q8\n"
      "vqmovn.s32 d17, q9\n"
      "vqmovn.s32 d20, q10\n"
      "vqmovn.s32 d21, q11\n"
      "vqmovun.s16 d12, q6\n"
      "vqmovun.s16 d16, q8\n"
      "vqmovun.s16 d20, q10\n"
      "vst1.32 {d12[0]}, [%[destination]]!\n"
      "vst1.8 {d12[4]}, [%[destination]]!\n"
      "vst1.32 {d16[0]}, [r1]!\n"
      "vst1.8 {d16[4]}, [r1]!\n"
      "vst1.32 {d20[0]}, [r3]!\n"
      "vst1.8 {d20[4]}, [r3]!\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [stride] "+r"(stride), [shift] "+r"(shift),
        [destination] "+r"(destination), [offsets] "+r"(offsets),
        [source] "+r"(source), [destination_stride] "+r"(destination_stride),
        [rounding_offset] "+r"(rounding_offset)
      :
      : "r0", "r1", "r2", "r3", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7",
        "d8", "d9", "d10", "d11", "d12", "d13", "d14", "d15", "d16", "d17",
        "d18", "d19", "d20", "d21", "d22", "d23", "cc", "memory");
}

void qnt_3x8_6_aligned(const std::int32_t* source, std::int32_t count,
                       std::int32_t stride, const std::int32_t* offsets,
                       std::uint8_t* destination,
                       std::int32_t destination_stride,
                       std::int32_t multiplicative_offset,
                       std::int32_t rounding_offset, std::int32_t shift) {
  asm volatile(
      "vdup.32 q0, %[multiplicative_offset]\n"
      "vdup.32 q1, %[rounding_offset]\n"
      "vdup.32 q2, %[shift]\n"
      "vld1.32 {d6[], d7[]}, [%[offsets]:32]!\n"
      "vld1.32 {d8[], d9[]}, [%[offsets]:32]!\n"
      "vld1.32 {d10[], d11[]}, [%[offsets]:32]!\n"
      "add r0, %[source], %[stride]\n"
      "add r1, %[destination], %[destination_stride]\n"
      "add r2, r0, %[stride]\n"
      "add r3, r1, %[destination_stride]\n"
      "subs %[count], %[count], #6\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"
      "vld1.32 {d12, d13, d14, d15}, [%[source]:128]!\n"
      "vld1.32 {d16, d17, d18, d19}, [r0:128]!\n"
      "vld1.32 {d20, d21, d22, d23}, [r2:128]!\n"
      "pld [%[source]]\n"
      "pld [r0]\n"
      "pld [r2]\n"
      "vadd.i32 q6, q6, q3\n"
      "vadd.i32 q7, q7, q3\n"
      "vadd.i32 q8, q8, q4\n"
      "vadd.i32 q9, q9, q4\n"
      "vadd.i32 q10, q10, q5\n"
      "vadd.i32 q11, q11, q5\n"
      "vmul.i32 q6, q6, q0\n"
      "vmul.i32 q7, q7, q0\n"
      "vmul.i32 q8, q8, q0\n"
      "vmul.i32 q9, q9, q0\n"
      "vmul.i32 q10, q10, q0\n"
      "vmul.i32 q11, q11, q0\n"
      "vadd.i32 q6, q6, q1\n"
      "vadd.i32 q7, q7, q1\n"
      "vadd.i32 q8, q8, q1\n"
      "vadd.i32 q9, q9, q1\n"
      "vadd.i32 q10, q10, q1\n"
      "vadd.i32 q11, q11, q1\n"
      "vshl.s32 q6, q6, q2\n"
      "vshl.s32 q7, q7, q2\n"
      "vshl.s32 q8, q8, q2\n"
      "vshl.s32 q9, q9, q2\n"
      "vshl.s32 q10, q10, q2\n"
      "vshl.s32 q11, q11, q2\n"
      "vqmovn.s32 d12, q6\n"
      "vqmovn.s32 d13, q7\n"
      "vqmovn.s32 d16, q8\n"
      "vqmovn.s32 d17, q9\n"
      "vqmovn.s32 d20, q10\n"
      "vqmovn.s32 d21, q11\n"
      "vqmovun.s16 d12, q6\n"
      "vqmovun.s16 d16, q8\n"
      "vqmovun.s16 d20, q10\n"
      "vst1.32 {d12}, [%[destination]:64]!\n"
      "vst1.32 {d16}, [r1:64]!\n"
      "vst1.32 {d20}, [r3:64]!\n"

      "bne 1b\n"
      "2:"
      "vld1.32 {d12, d13, d14}, [%[source]:64]!\n"
      "vld1.32 {d16, d17, d18}, [r0:64]!\n"
      "vld1.32 {d20, d21, d22}, [r2:64]!\n"
      "vadd.i32 q6, q6, q3\n"
      "vadd.i32 q7, q7, q3\n"
      "vadd.i32 q8, q8, q4\n"
      "vadd.i32 q9, q9, q4\n"
      "vadd.i32 q10, q10, q5\n"
      "vadd.i32 q11, q11, q5\n"
      "vmul.i32 q6, q6, q0\n"
      "vmul.i32 q7, q7, q0\n"
      "vmul.i32 q8, q8, q0\n"
      "vmul.i32 q9, q9, q0\n"
      "vmul.i32 q10, q10, q0\n"
      "vmul.i32 q11, q11, q0\n"
      "vadd.i32 q6, q6, q1\n"
      "vadd.i32 q7, q7, q1\n"
      "vadd.i32 q8, q8, q1\n"
      "vadd.i32 q9, q9, q1\n"
      "vadd.i32 q10, q10, q1\n"
      "vadd.i32 q11, q11, q1\n"
      "vshl.s32 q6, q6, q2\n"
      "vshl.s32 q7, q7, q2\n"
      "vshl.s32 q8, q8, q2\n"
      "vshl.s32 q9, q9, q2\n"
      "vshl.s32 q10, q10, q2\n"
      "vshl.s32 q11, q11, q2\n"
      "vqmovn.s32 d12, q6\n"
      "vqmovn.s32 d13, q7\n"
      "vqmovn.s32 d16, q8\n"
      "vqmovn.s32 d17, q9\n"
      "vqmovn.s32 d20, q10\n"
      "vqmovn.s32 d21, q11\n"
      "vqmovun.s16 d12, q6\n"
      "vqmovun.s16 d16, q8\n"
      "vqmovun.s16 d20, q10\n"
      "vst1.32 {d12[0]}, [%[destination]]!\n"
      "vst1.16 {d12[2]}, [%[destination]]!\n"
      "vst1.32 {d16[0]}, [r1]!\n"
      "vst1.16 {d16[2]}, [r1]!\n"
      "vst1.32 {d20[0]}, [r3]!\n"
      "vst1.16 {d20[2]}, [r3]!\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [stride] "+r"(stride), [shift] "+r"(shift),
        [destination] "+r"(destination), [offsets] "+r"(offsets),
        [source] "+r"(source), [destination_stride] "+r"(destination_stride),
        [rounding_offset] "+r"(rounding_offset)
      :
      : "r0", "r1", "r2", "r3", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7",
        "d8", "d9", "d10", "d11", "d12", "d13", "d14", "d15", "d16", "d17",
        "d18", "d19", "d20", "d21", "d22", "d23", "cc", "memory");
}

void qnt_3x8_7_aligned(const std::int32_t* source, std::int32_t count,
                       std::int32_t stride, const std::int32_t* offsets,
                       std::uint8_t* destination,
                       std::int32_t destination_stride,
                       std::int32_t multiplicative_offset,
                       std::int32_t rounding_offset, std::int32_t shift) {
  asm volatile(
      "vdup.32 q0, %[multiplicative_offset]\n"
      "vdup.32 q1, %[rounding_offset]\n"
      "vdup.32 q2, %[shift]\n"
      "vld1.32 {d6[], d7[]}, [%[offsets]:32]!\n"
      "vld1.32 {d8[], d9[]}, [%[offsets]:32]!\n"
      "vld1.32 {d10[], d11[]}, [%[offsets]:32]!\n"
      "add r0, %[source], %[stride]\n"
      "add r1, %[destination], %[destination_stride]\n"
      "add r2, r0, %[stride]\n"
      "add r3, r1, %[destination_stride]\n"
      "subs %[count], %[count], #7\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"
      "vld1.32 {d12, d13, d14, d15}, [%[source]:128]!\n"
      "vld1.32 {d16, d17, d18, d19}, [r0:128]!\n"
      "vld1.32 {d20, d21, d22, d23}, [r2:128]!\n"
      "pld [%[source]]\n"
      "pld [r0]\n"
      "pld [r2]\n"
      "vadd.i32 q6, q6, q3\n"
      "vadd.i32 q7, q7, q3\n"
      "vadd.i32 q8, q8, q4\n"
      "vadd.i32 q9, q9, q4\n"
      "vadd.i32 q10, q10, q5\n"
      "vadd.i32 q11, q11, q5\n"
      "vmul.i32 q6, q6, q0\n"
      "vmul.i32 q7, q7, q0\n"
      "vmul.i32 q8, q8, q0\n"
      "vmul.i32 q9, q9, q0\n"
      "vmul.i32 q10, q10, q0\n"
      "vmul.i32 q11, q11, q0\n"
      "vadd.i32 q6, q6, q1\n"
      "vadd.i32 q7, q7, q1\n"
      "vadd.i32 q8, q8, q1\n"
      "vadd.i32 q9, q9, q1\n"
      "vadd.i32 q10, q10, q1\n"
      "vadd.i32 q11, q11, q1\n"
      "vshl.s32 q6, q6, q2\n"
      "vshl.s32 q7, q7, q2\n"
      "vshl.s32 q8, q8, q2\n"
      "vshl.s32 q9, q9, q2\n"
      "vshl.s32 q10, q10, q2\n"
      "vshl.s32 q11, q11, q2\n"
      "vqmovn.s32 d12, q6\n"
      "vqmovn.s32 d13, q7\n"
      "vqmovn.s32 d16, q8\n"
      "vqmovn.s32 d17, q9\n"
      "vqmovn.s32 d20, q10\n"
      "vqmovn.s32 d21, q11\n"
      "vqmovun.s16 d12, q6\n"
      "vqmovun.s16 d16, q8\n"
      "vqmovun.s16 d20, q10\n"
      "vst1.32 {d12}, [%[destination]:64]!\n"
      "vst1.32 {d16}, [r1:64]!\n"
      "vst1.32 {d20}, [r3:64]!\n"

      "bne 1b\n"
      "2:"
      "vld1.32 {d12, d13, d14}, [%[source]:64]!\n"
      "vld1.32 {d15[0]}, [%[source]]!\n"
      "vld1.32 {d16, d17, d18}, [r0:64]!\n"
      "vld1.32 {d19[0]}, [r0]!\n"
      "vld1.32 {d20, d21, d22}, [r2:64]!\n"
      "vld1.32 {d23[0]}, [r2]!\n"
      "vadd.i32 q6, q6, q3\n"
      "vadd.i32 q7, q7, q3\n"
      "vadd.i32 q8, q8, q4\n"
      "vadd.i32 q9, q9, q4\n"
      "vadd.i32 q10, q10, q5\n"
      "vadd.i32 q11, q11, q5\n"
      "vmul.i32 q6, q6, q0\n"
      "vmul.i32 q7, q7, q0\n"
      "vmul.i32 q8, q8, q0\n"
      "vmul.i32 q9, q9, q0\n"
      "vmul.i32 q10, q10, q0\n"
      "vmul.i32 q11, q11, q0\n"
      "vadd.i32 q6, q6, q1\n"
      "vadd.i32 q7, q7, q1\n"
      "vadd.i32 q8, q8, q1\n"
      "vadd.i32 q9, q9, q1\n"
      "vadd.i32 q10, q10, q1\n"
      "vadd.i32 q11, q11, q1\n"
      "vshl.s32 q6, q6, q2\n"
      "vshl.s32 q7, q7, q2\n"
      "vshl.s32 q8, q8, q2\n"
      "vshl.s32 q9, q9, q2\n"
      "vshl.s32 q10, q10, q2\n"
      "vshl.s32 q11, q11, q2\n"
      "vqmovn.s32 d12, q6\n"
      "vqmovn.s32 d13, q7\n"
      "vqmovn.s32 d16, q8\n"
      "vqmovn.s32 d17, q9\n"
      "vqmovn.s32 d20, q10\n"
      "vqmovn.s32 d21, q11\n"
      "vqmovun.s16 d12, q6\n"
      "vqmovun.s16 d16, q8\n"
      "vqmovun.s16 d20, q10\n"
      "vst1.32 {d12[0]}, [%[destination]]!\n"
      "vst1.16 {d12[2]}, [%[destination]]!\n"
      "vst1.8 {d12[6]}, [%[destination]]!\n"
      "vst1.32 {d16[0]}, [r1]!\n"
      "vst1.16 {d16[2]}, [r1]!\n"
      "vst1.8 {d16[6]}, [r1]!\n"
      "vst1.32 {d20[0]}, [r3]!\n"
      "vst1.16 {d20[2]}, [r3]!\n"
      "vst1.8 {d20[6]}, [r3]!\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [stride] "+r"(stride), [shift] "+r"(shift),
        [destination] "+r"(destination), [offsets] "+r"(offsets),
        [source] "+r"(source), [destination_stride] "+r"(destination_stride),
        [rounding_offset] "+r"(rounding_offset)
      :
      : "r0", "r1", "r2", "r3", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7",
        "d8", "d9", "d10", "d11", "d12", "d13", "d14", "d15", "d16", "d17",
        "d18", "d19", "d20", "d21", "d22", "d23", "cc", "memory");
}

void qnt_1x8(const std::int32_t* source, std::int32_t count,
             std::int32_t stride, const std::int32_t* offsets,
             std::uint8_t* destination, std::int32_t destination_stride,
             std::int32_t multiplicative_offset, std::int32_t rounding_offset,
             std::int32_t shift) {
  asm volatile(
      "vdup.32 q0, %[multiplicative_offset]\n"
      "vdup.32 q1, %[rounding_offset]\n"
      "vdup.32 q2, %[shift]\n"
      "vld1.32 {d6[], d7[]}, [%[offsets]:32]!\n"

      "1:"
      "subs %[count], %[count], #8\n"
      "vld1.32 {d8, d9, d10, d11}, [%[source]:128]!\n"
      "pld [%[source]]\n"
      "vadd.i32 q4, q4, q3\n"
      "vadd.i32 q5, q5, q3\n"
      "vmul.i32 q4, q4, q0\n"
      "vmul.i32 q5, q5, q0\n"
      "vadd.i32 q4, q4, q1\n"
      "vadd.i32 q5, q5, q1\n"
      "vshl.s32 q4, q4, q2\n"
      "vshl.s32 q5, q5, q2\n"
      "vqmovn.s32 d8, q4\n"
      "vqmovn.s32 d9, q5\n"
      "vqmovun.s16 d8, q4\n"
      "vst1.32 {d8}, [%[destination]]!\n"

      "bne 1b\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [stride] "+r"(stride), [shift] "+r"(shift),
        [destination] "+r"(destination), [offsets] "+r"(offsets),
        [source] "+r"(source), [destination_stride] "+r"(destination_stride),
        [rounding_offset] "+r"(rounding_offset)
      :
      : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10",
        "d11", "cc", "memory");
}

void qnt_1x8_1(const std::int32_t* source, std::int32_t count,
               std::int32_t stride, const std::int32_t* offsets,
               std::uint8_t* destination, std::int32_t destination_stride,
               std::int32_t multiplicative_offset, std::int32_t rounding_offset,
               std::int32_t shift) {
  asm volatile(
      "vdup.32 q0, %[multiplicative_offset]\n"
      "vdup.32 q1, %[rounding_offset]\n"
      "vdup.32 q2, %[shift]\n"
      "vld1.32 {d6[], d7[]}, [%[offsets]:32]!\n"
      "subs %[count], %[count], #1\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"
      "vld1.32 {d8, d9, d10, d11}, [%[source]:128]!\n"
      "pld [%[source]]\n"
      "vadd.i32 q4, q4, q3\n"
      "vadd.i32 q5, q5, q3\n"
      "vmul.i32 q4, q4, q0\n"
      "vmul.i32 q5, q5, q0\n"
      "vadd.i32 q4, q4, q1\n"
      "vadd.i32 q5, q5, q1\n"
      "vshl.s32 q4, q4, q2\n"
      "vshl.s32 q5, q5, q2\n"
      "vqmovn.s32 d8, q4\n"
      "vqmovn.s32 d9, q5\n"
      "vqmovun.s16 d8, q4\n"
      "vst1.32 {d8}, [%[destination]]!\n"

      "bne 1b\n"
      "2:"
      "vld1.32 {d8[0]}, [%[source]]!\n"
      "vadd.i32 q4, q4, q3\n"
      "vmul.i32 q4, q4, q0\n"
      "vadd.i32 q4, q4, q1\n"
      "vshl.s32 q4, q4, q2\n"
      "vqmovn.s32 d8, q4\n"
      "vqmovun.s16 d8, q4\n"
      "vst1.8 {d8[0]}, [%[destination]]!\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [stride] "+r"(stride), [shift] "+r"(shift),
        [destination] "+r"(destination), [offsets] "+r"(offsets),
        [source] "+r"(source), [destination_stride] "+r"(destination_stride),
        [rounding_offset] "+r"(rounding_offset)
      :
      : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10",
        "d11", "cc", "memory");
}

void qnt_1x8_2(const std::int32_t* source, std::int32_t count,
               std::int32_t stride, const std::int32_t* offsets,
               std::uint8_t* destination, std::int32_t destination_stride,
               std::int32_t multiplicative_offset, std::int32_t rounding_offset,
               std::int32_t shift) {
  asm volatile(
      "vdup.32 q0, %[multiplicative_offset]\n"
      "vdup.32 q1, %[rounding_offset]\n"
      "vdup.32 q2, %[shift]\n"
      "vld1.32 {d6[], d7[]}, [%[offsets]:32]!\n"
      "subs %[count], %[count], #2\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"
      "vld1.32 {d8, d9, d10, d11}, [%[source]:128]!\n"
      "pld [%[source]]\n"
      "vadd.i32 q4, q4, q3\n"
      "vadd.i32 q5, q5, q3\n"
      "vmul.i32 q4, q4, q0\n"
      "vmul.i32 q5, q5, q0\n"
      "vadd.i32 q4, q4, q1\n"
      "vadd.i32 q5, q5, q1\n"
      "vshl.s32 q4, q4, q2\n"
      "vshl.s32 q5, q5, q2\n"
      "vqmovn.s32 d8, q4\n"
      "vqmovn.s32 d9, q5\n"
      "vqmovun.s16 d8, q4\n"
      "vst1.32 {d8}, [%[destination]]!\n"

      "bne 1b\n"
      "2:"
      "vld1.32 {d8}, [%[source]:64]!\n"
      "vadd.i32 q4, q4, q3\n"
      "vmul.i32 q4, q4, q0\n"
      "vadd.i32 q4, q4, q1\n"
      "vshl.s32 q4, q4, q2\n"
      "vqmovn.s32 d8, q4\n"
      "vqmovun.s16 d8, q4\n"
      "vst1.16 {d8[0]}, [%[destination]]!\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [stride] "+r"(stride), [shift] "+r"(shift),
        [destination] "+r"(destination), [offsets] "+r"(offsets),
        [source] "+r"(source), [destination_stride] "+r"(destination_stride),
        [rounding_offset] "+r"(rounding_offset)
      :
      : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10",
        "d11", "cc", "memory");
}

void qnt_1x8_3(const std::int32_t* source, std::int32_t count,
               std::int32_t stride, const std::int32_t* offsets,
               std::uint8_t* destination, std::int32_t destination_stride,
               std::int32_t multiplicative_offset, std::int32_t rounding_offset,
               std::int32_t shift) {
  asm volatile(
      "vdup.32 q0, %[multiplicative_offset]\n"
      "vdup.32 q1, %[rounding_offset]\n"
      "vdup.32 q2, %[shift]\n"
      "vld1.32 {d6[], d7[]}, [%[offsets]:32]!\n"
      "subs %[count], %[count], #3\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"
      "vld1.32 {d8, d9, d10, d11}, [%[source]:128]!\n"
      "pld [%[source]]\n"
      "vadd.i32 q4, q4, q3\n"
      "vadd.i32 q5, q5, q3\n"
      "vmul.i32 q4, q4, q0\n"
      "vmul.i32 q5, q5, q0\n"
      "vadd.i32 q4, q4, q1\n"
      "vadd.i32 q5, q5, q1\n"
      "vshl.s32 q4, q4, q2\n"
      "vshl.s32 q5, q5, q2\n"
      "vqmovn.s32 d8, q4\n"
      "vqmovn.s32 d9, q5\n"
      "vqmovun.s16 d8, q4\n"
      "vst1.32 {d8}, [%[destination]]!\n"

      "bne 1b\n"
      "2:"
      "vld1.32 {d8}, [%[source]:64]!\n"
      "vld1.32 {d9[0]}, [%[source]]!\n"
      "vadd.i32 q4, q4, q3\n"
      "vmul.i32 q4, q4, q0\n"
      "vadd.i32 q4, q4, q1\n"
      "vshl.s32 q4, q4, q2\n"
      "vqmovn.s32 d8, q4\n"
      "vqmovun.s16 d8, q4\n"
      "vst1.16 {d8[0]}, [%[destination]]!\n"
      "vst1.8 {d8[2]}, [%[destination]]!\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [stride] "+r"(stride), [shift] "+r"(shift),
        [destination] "+r"(destination), [offsets] "+r"(offsets),
        [source] "+r"(source), [destination_stride] "+r"(destination_stride),
        [rounding_offset] "+r"(rounding_offset)
      :
      : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10",
        "d11", "cc", "memory");
}

void qnt_1x8_4(const std::int32_t* source, std::int32_t count,
               std::int32_t stride, const std::int32_t* offsets,
               std::uint8_t* destination, std::int32_t destination_stride,
               std::int32_t multiplicative_offset, std::int32_t rounding_offset,
               std::int32_t shift) {
  asm volatile(
      "vdup.32 q0, %[multiplicative_offset]\n"
      "vdup.32 q1, %[rounding_offset]\n"
      "vdup.32 q2, %[shift]\n"
      "vld1.32 {d6[], d7[]}, [%[offsets]:32]!\n"
      "subs %[count], %[count], #4\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"
      "vld1.32 {d8, d9, d10, d11}, [%[source]:128]!\n"
      "pld [%[source]]\n"
      "vadd.i32 q4, q4, q3\n"
      "vadd.i32 q5, q5, q3\n"
      "vmul.i32 q4, q4, q0\n"
      "vmul.i32 q5, q5, q0\n"
      "vadd.i32 q4, q4, q1\n"
      "vadd.i32 q5, q5, q1\n"
      "vshl.s32 q4, q4, q2\n"
      "vshl.s32 q5, q5, q2\n"
      "vqmovn.s32 d8, q4\n"
      "vqmovn.s32 d9, q5\n"
      "vqmovun.s16 d8, q4\n"
      "vst1.32 {d8}, [%[destination]]!\n"

      "bne 1b\n"
      "2:"
      "vld1.32 {d8, d9}, [%[source]:64]!\n"
      "vadd.i32 q4, q4, q3\n"
      "vmul.i32 q4, q4, q0\n"
      "vadd.i32 q4, q4, q1\n"
      "vshl.s32 q4, q4, q2\n"
      "vqmovn.s32 d8, q4\n"
      "vqmovun.s16 d8, q4\n"
      "vst1.32 {d8[0]}, [%[destination]]!\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [stride] "+r"(stride), [shift] "+r"(shift),
        [destination] "+r"(destination), [offsets] "+r"(offsets),
        [source] "+r"(source), [destination_stride] "+r"(destination_stride),
        [rounding_offset] "+r"(rounding_offset)
      :
      : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10",
        "d11", "cc", "memory");
}

void qnt_1x8_5(const std::int32_t* source, std::int32_t count,
               std::int32_t stride, const std::int32_t* offsets,
               std::uint8_t* destination, std::int32_t destination_stride,
               std::int32_t multiplicative_offset, std::int32_t rounding_offset,
               std::int32_t shift) {
  asm volatile(
      "vdup.32 q0, %[multiplicative_offset]\n"
      "vdup.32 q1, %[rounding_offset]\n"
      "vdup.32 q2, %[shift]\n"
      "vld1.32 {d6[], d7[]}, [%[offsets]:32]!\n"
      "subs %[count], %[count], #5\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"
      "vld1.32 {d8, d9, d10, d11}, [%[source]:128]!\n"
      "pld [%[source]]\n"
      "vadd.i32 q4, q4, q3\n"
      "vadd.i32 q5, q5, q3\n"
      "vmul.i32 q4, q4, q0\n"
      "vmul.i32 q5, q5, q0\n"
      "vadd.i32 q4, q4, q1\n"
      "vadd.i32 q5, q5, q1\n"
      "vshl.s32 q4, q4, q2\n"
      "vshl.s32 q5, q5, q2\n"
      "vqmovn.s32 d8, q4\n"
      "vqmovn.s32 d9, q5\n"
      "vqmovun.s16 d8, q4\n"
      "vst1.32 {d8}, [%[destination]]!\n"

      "bne 1b\n"
      "2:"
      "vld1.32 {d8, d9}, [%[source]:64]!\n"
      "vld1.32 {d10[0]}, [%[source]]!\n"
      "vadd.i32 q4, q4, q3\n"
      "vadd.i32 q5, q5, q3\n"
      "vmul.i32 q4, q4, q0\n"
      "vmul.i32 q5, q5, q0\n"
      "vadd.i32 q4, q4, q1\n"
      "vadd.i32 q5, q5, q1\n"
      "vshl.s32 q4, q4, q2\n"
      "vshl.s32 q5, q5, q2\n"
      "vqmovn.s32 d8, q4\n"
      "vqmovn.s32 d9, q5\n"
      "vqmovun.s16 d8, q4\n"
      "vst1.32 {d8[0]}, [%[destination]]!\n"
      "vst1.8 {d8[4]}, [%[destination]]!\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [stride] "+r"(stride), [shift] "+r"(shift),
        [destination] "+r"(destination), [offsets] "+r"(offsets),
        [source] "+r"(source), [destination_stride] "+r"(destination_stride),
        [rounding_offset] "+r"(rounding_offset)
      :
      : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10",
        "d11", "cc", "memory");
}

void qnt_1x8_6(const std::int32_t* source, std::int32_t count,
               std::int32_t stride, const std::int32_t* offsets,
               std::uint8_t* destination, std::int32_t destination_stride,
               std::int32_t multiplicative_offset, std::int32_t rounding_offset,
               std::int32_t shift) {
  asm volatile(
      "vdup.32 q0, %[multiplicative_offset]\n"
      "vdup.32 q1, %[rounding_offset]\n"
      "vdup.32 q2, %[shift]\n"
      "vld1.32 {d6[], d7[]}, [%[offsets]:32]!\n"
      "subs %[count], %[count], #6\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"
      "vld1.32 {d8, d9, d10, d11}, [%[source]:128]!\n"
      "pld [%[source]]\n"
      "vadd.i32 q4, q4, q3\n"
      "vadd.i32 q5, q5, q3\n"
      "vmul.i32 q4, q4, q0\n"
      "vmul.i32 q5, q5, q0\n"
      "vadd.i32 q4, q4, q1\n"
      "vadd.i32 q5, q5, q1\n"
      "vshl.s32 q4, q4, q2\n"
      "vshl.s32 q5, q5, q2\n"
      "vqmovn.s32 d8, q4\n"
      "vqmovn.s32 d9, q5\n"
      "vqmovun.s16 d8, q4\n"
      "vst1.32 {d8}, [%[destination]]!\n"

      "bne 1b\n"
      "2:"
      "vld1.32 {d8, d9, d10}, [%[source]:64]!\n"
      "vadd.i32 q4, q4, q3\n"
      "vadd.i32 q5, q5, q3\n"
      "vmul.i32 q4, q4, q0\n"
      "vmul.i32 q5, q5, q0\n"
      "vadd.i32 q4, q4, q1\n"
      "vadd.i32 q5, q5, q1\n"
      "vshl.s32 q4, q4, q2\n"
      "vshl.s32 q5, q5, q2\n"
      "vqmovn.s32 d8, q4\n"
      "vqmovn.s32 d9, q5\n"
      "vqmovun.s16 d8, q4\n"
      "vst1.32 {d8[0]}, [%[destination]]!\n"
      "vst1.16 {d8[2]}, [%[destination]]!\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [stride] "+r"(stride), [shift] "+r"(shift),
        [destination] "+r"(destination), [offsets] "+r"(offsets),
        [source] "+r"(source), [destination_stride] "+r"(destination_stride),
        [rounding_offset] "+r"(rounding_offset)
      :
      : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10",
        "d11", "cc", "memory");
}

void qnt_1x8_7(const std::int32_t* source, std::int32_t count,
               std::int32_t stride, const std::int32_t* offsets,
               std::uint8_t* destination, std::int32_t destination_stride,
               std::int32_t multiplicative_offset, std::int32_t rounding_offset,
               std::int32_t shift) {
  asm volatile(
      "vdup.32 q0, %[multiplicative_offset]\n"
      "vdup.32 q1, %[rounding_offset]\n"
      "vdup.32 q2, %[shift]\n"
      "vld1.32 {d6[], d7[]}, [%[offsets]:32]!\n"
      "subs %[count], %[count], #7\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"
      "vld1.32 {d8, d9, d10, d11}, [%[source]:128]!\n"
      "pld [%[source]]\n"
      "vadd.i32 q4, q4, q3\n"
      "vadd.i32 q5, q5, q3\n"
      "vmul.i32 q4, q4, q0\n"
      "vmul.i32 q5, q5, q0\n"
      "vadd.i32 q4, q4, q1\n"
      "vadd.i32 q5, q5, q1\n"
      "vshl.s32 q4, q4, q2\n"
      "vshl.s32 q5, q5, q2\n"
      "vqmovn.s32 d8, q4\n"
      "vqmovn.s32 d9, q5\n"
      "vqmovun.s16 d8, q4\n"
      "vst1.32 {d8}, [%[destination]]!\n"

      "bne 1b\n"
      "2:"
      "vld1.32 {d8, d9, d10}, [%[source]:64]!\n"
      "vld1.32 {d11[0]}, [%[source]]!\n"
      "vadd.i32 q4, q4, q3\n"
      "vadd.i32 q5, q5, q3\n"
      "vmul.i32 q4, q4, q0\n"
      "vmul.i32 q5, q5, q0\n"
      "vadd.i32 q4, q4, q1\n"
      "vadd.i32 q5, q5, q1\n"
      "vshl.s32 q4, q4, q2\n"
      "vshl.s32 q5, q5, q2\n"
      "vqmovn.s32 d8, q4\n"
      "vqmovn.s32 d9, q5\n"
      "vqmovun.s16 d8, q4\n"
      "vst1.32 {d8[0]}, [%[destination]]!\n"
      "vst1.16 {d8[2]}, [%[destination]]!\n"
      "vst1.8 {d8[6]}, [%[destination]]!\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [stride] "+r"(stride), [shift] "+r"(shift),
        [destination] "+r"(destination), [offsets] "+r"(offsets),
        [source] "+r"(source), [destination_stride] "+r"(destination_stride),
        [rounding_offset] "+r"(rounding_offset)
      :
      : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10",
        "d11", "cc", "memory");
}

void qnt_2x8(const std::int32_t* source, std::int32_t count,
             std::int32_t stride, const std::int32_t* offsets,
             std::uint8_t* destination, std::int32_t destination_stride,
             std::int32_t multiplicative_offset, std::int32_t rounding_offset,
             std::int32_t shift) {
  asm volatile(
      "vdup.32 q0, %[multiplicative_offset]\n"
      "vdup.32 q1, %[rounding_offset]\n"
      "vdup.32 q2, %[shift]\n"
      "vld1.32 {d6[], d7[]}, [%[offsets]:32]!\n"
      "vld1.32 {d8[], d9[]}, [%[offsets]:32]!\n"
      "add r0, %[source], %[stride]\n"
      "add r1, %[destination], %[destination_stride]\n"

      "1:"
      "subs %[count], %[count], #8\n"
      "vld1.32 {d10, d11, d12, d13}, [%[source]:128]!\n"
      "vld1.32 {d14, d15, d16, d17}, [r0:128]!\n"
      "pld [%[source]]\n"
      "pld [r0]\n"
      "vadd.i32 q5, q5, q3\n"
      "vadd.i32 q6, q6, q3\n"
      "vadd.i32 q7, q7, q4\n"
      "vadd.i32 q8, q8, q4\n"
      "vmul.i32 q5, q5, q0\n"
      "vmul.i32 q6, q6, q0\n"
      "vmul.i32 q7, q7, q0\n"
      "vmul.i32 q8, q8, q0\n"
      "vadd.i32 q5, q5, q1\n"
      "vadd.i32 q6, q6, q1\n"
      "vadd.i32 q7, q7, q1\n"
      "vadd.i32 q8, q8, q1\n"
      "vshl.s32 q5, q5, q2\n"
      "vshl.s32 q6, q6, q2\n"
      "vshl.s32 q7, q7, q2\n"
      "vshl.s32 q8, q8, q2\n"
      "vqmovn.s32 d10, q5\n"
      "vqmovn.s32 d11, q6\n"
      "vqmovn.s32 d14, q7\n"
      "vqmovn.s32 d15, q8\n"
      "vqmovun.s16 d10, q5\n"
      "vqmovun.s16 d14, q7\n"
      "vst1.32 {d10}, [%[destination]]!\n"
      "vst1.32 {d14}, [r1]!\n"

      "bne 1b\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [stride] "+r"(stride), [shift] "+r"(shift),
        [destination] "+r"(destination), [offsets] "+r"(offsets),
        [source] "+r"(source), [destination_stride] "+r"(destination_stride),
        [rounding_offset] "+r"(rounding_offset)
      :
      : "r0", "r1", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9",
        "d10", "d11", "d12", "d13", "d14", "d15", "d16", "d17", "cc", "memory");
}

void qnt_2x8_1(const std::int32_t* source, std::int32_t count,
               std::int32_t stride, const std::int32_t* offsets,
               std::uint8_t* destination, std::int32_t destination_stride,
               std::int32_t multiplicative_offset, std::int32_t rounding_offset,
               std::int32_t shift) {
  asm volatile(
      "vdup.32 q0, %[multiplicative_offset]\n"
      "vdup.32 q1, %[rounding_offset]\n"
      "vdup.32 q2, %[shift]\n"
      "vld1.32 {d6[], d7[]}, [%[offsets]:32]!\n"
      "vld1.32 {d8[], d9[]}, [%[offsets]:32]!\n"
      "add r0, %[source], %[stride]\n"
      "add r1, %[destination], %[destination_stride]\n"
      "subs %[count], %[count], #1\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"
      "vld1.32 {d10, d11, d12, d13}, [%[source]:128]!\n"
      "vld1.32 {d14, d15, d16, d17}, [r0:128]!\n"
      "pld [%[source]]\n"
      "pld [r0]\n"
      "vadd.i32 q5, q5, q3\n"
      "vadd.i32 q6, q6, q3\n"
      "vadd.i32 q7, q7, q4\n"
      "vadd.i32 q8, q8, q4\n"
      "vmul.i32 q5, q5, q0\n"
      "vmul.i32 q6, q6, q0\n"
      "vmul.i32 q7, q7, q0\n"
      "vmul.i32 q8, q8, q0\n"
      "vadd.i32 q5, q5, q1\n"
      "vadd.i32 q6, q6, q1\n"
      "vadd.i32 q7, q7, q1\n"
      "vadd.i32 q8, q8, q1\n"
      "vshl.s32 q5, q5, q2\n"
      "vshl.s32 q6, q6, q2\n"
      "vshl.s32 q7, q7, q2\n"
      "vshl.s32 q8, q8, q2\n"
      "vqmovn.s32 d10, q5\n"
      "vqmovn.s32 d11, q6\n"
      "vqmovn.s32 d14, q7\n"
      "vqmovn.s32 d15, q8\n"
      "vqmovun.s16 d10, q5\n"
      "vqmovun.s16 d14, q7\n"
      "vst1.32 {d10}, [%[destination]]!\n"
      "vst1.32 {d14}, [r1]!\n"

      "bne 1b\n"
      "2:"
      "vld1.32 {d10[0]}, [%[source]]!\n"
      "vld1.32 {d14[0]}, [r0]!\n"
      "vadd.i32 q5, q5, q3\n"
      "vadd.i32 q7, q7, q4\n"
      "vmul.i32 q5, q5, q0\n"
      "vmul.i32 q7, q7, q0\n"
      "vadd.i32 q5, q5, q1\n"
      "vadd.i32 q7, q7, q1\n"
      "vshl.s32 q5, q5, q2\n"
      "vshl.s32 q7, q7, q2\n"
      "vqmovn.s32 d10, q5\n"
      "vqmovn.s32 d14, q7\n"
      "vqmovun.s16 d10, q5\n"
      "vqmovun.s16 d14, q7\n"
      "vst1.8 {d10[0]}, [%[destination]]!\n"
      "vst1.8 {d14[0]}, [r1]!\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [stride] "+r"(stride), [shift] "+r"(shift),
        [destination] "+r"(destination), [offsets] "+r"(offsets),
        [source] "+r"(source), [destination_stride] "+r"(destination_stride),
        [rounding_offset] "+r"(rounding_offset)
      :
      : "r0", "r1", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9",
        "d10", "d11", "d12", "d13", "d14", "d15", "d16", "d17", "cc", "memory");
}

void qnt_2x8_2(const std::int32_t* source, std::int32_t count,
               std::int32_t stride, const std::int32_t* offsets,
               std::uint8_t* destination, std::int32_t destination_stride,
               std::int32_t multiplicative_offset, std::int32_t rounding_offset,
               std::int32_t shift) {
  asm volatile(
      "vdup.32 q0, %[multiplicative_offset]\n"
      "vdup.32 q1, %[rounding_offset]\n"
      "vdup.32 q2, %[shift]\n"
      "vld1.32 {d6[], d7[]}, [%[offsets]:32]!\n"
      "vld1.32 {d8[], d9[]}, [%[offsets]:32]!\n"
      "add r0, %[source], %[stride]\n"
      "add r1, %[destination], %[destination_stride]\n"
      "subs %[count], %[count], #2\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"
      "vld1.32 {d10, d11, d12, d13}, [%[source]:128]!\n"
      "vld1.32 {d14, d15, d16, d17}, [r0:128]!\n"
      "pld [%[source]]\n"
      "pld [r0]\n"
      "vadd.i32 q5, q5, q3\n"
      "vadd.i32 q6, q6, q3\n"
      "vadd.i32 q7, q7, q4\n"
      "vadd.i32 q8, q8, q4\n"
      "vmul.i32 q5, q5, q0\n"
      "vmul.i32 q6, q6, q0\n"
      "vmul.i32 q7, q7, q0\n"
      "vmul.i32 q8, q8, q0\n"
      "vadd.i32 q5, q5, q1\n"
      "vadd.i32 q6, q6, q1\n"
      "vadd.i32 q7, q7, q1\n"
      "vadd.i32 q8, q8, q1\n"
      "vshl.s32 q5, q5, q2\n"
      "vshl.s32 q6, q6, q2\n"
      "vshl.s32 q7, q7, q2\n"
      "vshl.s32 q8, q8, q2\n"
      "vqmovn.s32 d10, q5\n"
      "vqmovn.s32 d11, q6\n"
      "vqmovn.s32 d14, q7\n"
      "vqmovn.s32 d15, q8\n"
      "vqmovun.s16 d10, q5\n"
      "vqmovun.s16 d14, q7\n"
      "vst1.32 {d10}, [%[destination]]!\n"
      "vst1.32 {d14}, [r1]!\n"

      "bne 1b\n"
      "2:"
      "vld1.32 {d10}, [%[source]:64]!\n"
      "vld1.32 {d14}, [r0:64]!\n"
      "vadd.i32 q5, q5, q3\n"
      "vadd.i32 q7, q7, q4\n"
      "vmul.i32 q5, q5, q0\n"
      "vmul.i32 q7, q7, q0\n"
      "vadd.i32 q5, q5, q1\n"
      "vadd.i32 q7, q7, q1\n"
      "vshl.s32 q5, q5, q2\n"
      "vshl.s32 q7, q7, q2\n"
      "vqmovn.s32 d10, q5\n"
      "vqmovn.s32 d14, q7\n"
      "vqmovun.s16 d10, q5\n"
      "vqmovun.s16 d14, q7\n"
      "vst1.16 {d10[0]}, [%[destination]]!\n"
      "vst1.16 {d14[0]}, [r1]!\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [stride] "+r"(stride), [shift] "+r"(shift),
        [destination] "+r"(destination), [offsets] "+r"(offsets),
        [source] "+r"(source), [destination_stride] "+r"(destination_stride),
        [rounding_offset] "+r"(rounding_offset)
      :
      : "r0", "r1", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9",
        "d10", "d11", "d12", "d13", "d14", "d15", "d16", "d17", "cc", "memory");
}

void qnt_2x8_3(const std::int32_t* source, std::int32_t count,
               std::int32_t stride, const std::int32_t* offsets,
               std::uint8_t* destination, std::int32_t destination_stride,
               std::int32_t multiplicative_offset, std::int32_t rounding_offset,
               std::int32_t shift) {
  asm volatile(
      "vdup.32 q0, %[multiplicative_offset]\n"
      "vdup.32 q1, %[rounding_offset]\n"
      "vdup.32 q2, %[shift]\n"
      "vld1.32 {d6[], d7[]}, [%[offsets]:32]!\n"
      "vld1.32 {d8[], d9[]}, [%[offsets]:32]!\n"
      "add r0, %[source], %[stride]\n"
      "add r1, %[destination], %[destination_stride]\n"
      "subs %[count], %[count], #3\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"
      "vld1.32 {d10, d11, d12, d13}, [%[source]:128]!\n"
      "vld1.32 {d14, d15, d16, d17}, [r0:128]!\n"
      "pld [%[source]]\n"
      "pld [r0]\n"
      "vadd.i32 q5, q5, q3\n"
      "vadd.i32 q6, q6, q3\n"
      "vadd.i32 q7, q7, q4\n"
      "vadd.i32 q8, q8, q4\n"
      "vmul.i32 q5, q5, q0\n"
      "vmul.i32 q6, q6, q0\n"
      "vmul.i32 q7, q7, q0\n"
      "vmul.i32 q8, q8, q0\n"
      "vadd.i32 q5, q5, q1\n"
      "vadd.i32 q6, q6, q1\n"
      "vadd.i32 q7, q7, q1\n"
      "vadd.i32 q8, q8, q1\n"
      "vshl.s32 q5, q5, q2\n"
      "vshl.s32 q6, q6, q2\n"
      "vshl.s32 q7, q7, q2\n"
      "vshl.s32 q8, q8, q2\n"
      "vqmovn.s32 d10, q5\n"
      "vqmovn.s32 d11, q6\n"
      "vqmovn.s32 d14, q7\n"
      "vqmovn.s32 d15, q8\n"
      "vqmovun.s16 d10, q5\n"
      "vqmovun.s16 d14, q7\n"
      "vst1.32 {d10}, [%[destination]]!\n"
      "vst1.32 {d14}, [r1]!\n"

      "bne 1b\n"
      "2:"
      "vld1.32 {d10}, [%[source]:64]!\n"
      "vld1.32 {d11[0]}, [%[source]]!\n"
      "vld1.32 {d14}, [r0:64]!\n"
      "vld1.32 {d15[0]}, [r0]!\n"
      "vadd.i32 q5, q5, q3\n"
      "vadd.i32 q7, q7, q4\n"
      "vmul.i32 q5, q5, q0\n"
      "vmul.i32 q7, q7, q0\n"
      "vadd.i32 q5, q5, q1\n"
      "vadd.i32 q7, q7, q1\n"
      "vshl.s32 q5, q5, q2\n"
      "vshl.s32 q7, q7, q2\n"
      "vqmovn.s32 d10, q5\n"
      "vqmovn.s32 d14, q7\n"
      "vqmovun.s16 d10, q5\n"
      "vqmovun.s16 d14, q7\n"
      "vst1.16 {d10[0]}, [%[destination]]!\n"
      "vst1.8 {d10[2]}, [%[destination]]!\n"
      "vst1.16 {d14[0]}, [r1]!\n"
      "vst1.8 {d14[2]}, [r1]!\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [stride] "+r"(stride), [shift] "+r"(shift),
        [destination] "+r"(destination), [offsets] "+r"(offsets),
        [source] "+r"(source), [destination_stride] "+r"(destination_stride),
        [rounding_offset] "+r"(rounding_offset)
      :
      : "r0", "r1", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9",
        "d10", "d11", "d12", "d13", "d14", "d15", "d16", "d17", "cc", "memory");
}

void qnt_2x8_4(const std::int32_t* source, std::int32_t count,
               std::int32_t stride, const std::int32_t* offsets,
               std::uint8_t* destination, std::int32_t destination_stride,
               std::int32_t multiplicative_offset, std::int32_t rounding_offset,
               std::int32_t shift) {
  asm volatile(
      "vdup.32 q0, %[multiplicative_offset]\n"
      "vdup.32 q1, %[rounding_offset]\n"
      "vdup.32 q2, %[shift]\n"
      "vld1.32 {d6[], d7[]}, [%[offsets]:32]!\n"
      "vld1.32 {d8[], d9[]}, [%[offsets]:32]!\n"
      "add r0, %[source], %[stride]\n"
      "add r1, %[destination], %[destination_stride]\n"
      "subs %[count], %[count], #4\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"
      "vld1.32 {d10, d11, d12, d13}, [%[source]:128]!\n"
      "vld1.32 {d14, d15, d16, d17}, [r0:128]!\n"
      "pld [%[source]]\n"
      "pld [r0]\n"
      "vadd.i32 q5, q5, q3\n"
      "vadd.i32 q6, q6, q3\n"
      "vadd.i32 q7, q7, q4\n"
      "vadd.i32 q8, q8, q4\n"
      "vmul.i32 q5, q5, q0\n"
      "vmul.i32 q6, q6, q0\n"
      "vmul.i32 q7, q7, q0\n"
      "vmul.i32 q8, q8, q0\n"
      "vadd.i32 q5, q5, q1\n"
      "vadd.i32 q6, q6, q1\n"
      "vadd.i32 q7, q7, q1\n"
      "vadd.i32 q8, q8, q1\n"
      "vshl.s32 q5, q5, q2\n"
      "vshl.s32 q6, q6, q2\n"
      "vshl.s32 q7, q7, q2\n"
      "vshl.s32 q8, q8, q2\n"
      "vqmovn.s32 d10, q5\n"
      "vqmovn.s32 d11, q6\n"
      "vqmovn.s32 d14, q7\n"
      "vqmovn.s32 d15, q8\n"
      "vqmovun.s16 d10, q5\n"
      "vqmovun.s16 d14, q7\n"
      "vst1.32 {d10}, [%[destination]]!\n"
      "vst1.32 {d14}, [r1]!\n"

      "bne 1b\n"
      "2:"
      "vld1.32 {d10, d11}, [%[source]:64]!\n"
      "vld1.32 {d14, d15}, [r0:64]!\n"
      "vadd.i32 q5, q5, q3\n"
      "vadd.i32 q7, q7, q4\n"
      "vmul.i32 q5, q5, q0\n"
      "vmul.i32 q7, q7, q0\n"
      "vadd.i32 q5, q5, q1\n"
      "vadd.i32 q7, q7, q1\n"
      "vshl.s32 q5, q5, q2\n"
      "vshl.s32 q7, q7, q2\n"
      "vqmovn.s32 d10, q5\n"
      "vqmovn.s32 d14, q7\n"
      "vqmovun.s16 d10, q5\n"
      "vqmovun.s16 d14, q7\n"
      "vst1.32 {d10[0]}, [%[destination]]!\n"
      "vst1.32 {d14[0]}, [r1]!\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [stride] "+r"(stride), [shift] "+r"(shift),
        [destination] "+r"(destination), [offsets] "+r"(offsets),
        [source] "+r"(source), [destination_stride] "+r"(destination_stride),
        [rounding_offset] "+r"(rounding_offset)
      :
      : "r0", "r1", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9",
        "d10", "d11", "d12", "d13", "d14", "d15", "d16", "d17", "cc", "memory");
}

void qnt_2x8_5(const std::int32_t* source, std::int32_t count,
               std::int32_t stride, const std::int32_t* offsets,
               std::uint8_t* destination, std::int32_t destination_stride,
               std::int32_t multiplicative_offset, std::int32_t rounding_offset,
               std::int32_t shift) {
  asm volatile(
      "vdup.32 q0, %[multiplicative_offset]\n"
      "vdup.32 q1, %[rounding_offset]\n"
      "vdup.32 q2, %[shift]\n"
      "vld1.32 {d6[], d7[]}, [%[offsets]:32]!\n"
      "vld1.32 {d8[], d9[]}, [%[offsets]:32]!\n"
      "add r0, %[source], %[stride]\n"
      "add r1, %[destination], %[destination_stride]\n"
      "subs %[count], %[count], #5\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"
      "vld1.32 {d10, d11, d12, d13}, [%[source]:128]!\n"
      "vld1.32 {d14, d15, d16, d17}, [r0:128]!\n"
      "pld [%[source]]\n"
      "pld [r0]\n"
      "vadd.i32 q5, q5, q3\n"
      "vadd.i32 q6, q6, q3\n"
      "vadd.i32 q7, q7, q4\n"
      "vadd.i32 q8, q8, q4\n"
      "vmul.i32 q5, q5, q0\n"
      "vmul.i32 q6, q6, q0\n"
      "vmul.i32 q7, q7, q0\n"
      "vmul.i32 q8, q8, q0\n"
      "vadd.i32 q5, q5, q1\n"
      "vadd.i32 q6, q6, q1\n"
      "vadd.i32 q7, q7, q1\n"
      "vadd.i32 q8, q8, q1\n"
      "vshl.s32 q5, q5, q2\n"
      "vshl.s32 q6, q6, q2\n"
      "vshl.s32 q7, q7, q2\n"
      "vshl.s32 q8, q8, q2\n"
      "vqmovn.s32 d10, q5\n"
      "vqmovn.s32 d11, q6\n"
      "vqmovn.s32 d14, q7\n"
      "vqmovn.s32 d15, q8\n"
      "vqmovun.s16 d10, q5\n"
      "vqmovun.s16 d14, q7\n"
      "vst1.32 {d10}, [%[destination]]!\n"
      "vst1.32 {d14}, [r1]!\n"

      "bne 1b\n"
      "2:"
      "vld1.32 {d10, d11}, [%[source]:64]!\n"
      "vld1.32 {d12[0]}, [%[source]]!\n"
      "vld1.32 {d14, d15}, [r0:64]!\n"
      "vld1.32 {d16[0]}, [r0]!\n"
      "vadd.i32 q5, q5, q3\n"
      "vadd.i32 q6, q6, q3\n"
      "vadd.i32 q7, q7, q4\n"
      "vadd.i32 q8, q8, q4\n"
      "vmul.i32 q5, q5, q0\n"
      "vmul.i32 q6, q6, q0\n"
      "vmul.i32 q7, q7, q0\n"
      "vmul.i32 q8, q8, q0\n"
      "vadd.i32 q5, q5, q1\n"
      "vadd.i32 q6, q6, q1\n"
      "vadd.i32 q7, q7, q1\n"
      "vadd.i32 q8, q8, q1\n"
      "vshl.s32 q5, q5, q2\n"
      "vshl.s32 q6, q6, q2\n"
      "vshl.s32 q7, q7, q2\n"
      "vshl.s32 q8, q8, q2\n"
      "vqmovn.s32 d10, q5\n"
      "vqmovn.s32 d11, q6\n"
      "vqmovn.s32 d14, q7\n"
      "vqmovn.s32 d15, q8\n"
      "vqmovun.s16 d10, q5\n"
      "vqmovun.s16 d14, q7\n"
      "vst1.32 {d10[0]}, [%[destination]]!\n"
      "vst1.8 {d10[4]}, [%[destination]]!\n"
      "vst1.32 {d14[0]}, [r1]!\n"
      "vst1.8 {d14[4]}, [r1]!\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [stride] "+r"(stride), [shift] "+r"(shift),
        [destination] "+r"(destination), [offsets] "+r"(offsets),
        [source] "+r"(source), [destination_stride] "+r"(destination_stride),
        [rounding_offset] "+r"(rounding_offset)
      :
      : "r0", "r1", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9",
        "d10", "d11", "d12", "d13", "d14", "d15", "d16", "d17", "cc", "memory");
}

void qnt_2x8_6(const std::int32_t* source, std::int32_t count,
               std::int32_t stride, const std::int32_t* offsets,
               std::uint8_t* destination, std::int32_t destination_stride,
               std::int32_t multiplicative_offset, std::int32_t rounding_offset,
               std::int32_t shift) {
  asm volatile(
      "vdup.32 q0, %[multiplicative_offset]\n"
      "vdup.32 q1, %[rounding_offset]\n"
      "vdup.32 q2, %[shift]\n"
      "vld1.32 {d6[], d7[]}, [%[offsets]:32]!\n"
      "vld1.32 {d8[], d9[]}, [%[offsets]:32]!\n"
      "add r0, %[source], %[stride]\n"
      "add r1, %[destination], %[destination_stride]\n"
      "subs %[count], %[count], #6\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"
      "vld1.32 {d10, d11, d12, d13}, [%[source]:128]!\n"
      "vld1.32 {d14, d15, d16, d17}, [r0:128]!\n"
      "pld [%[source]]\n"
      "pld [r0]\n"
      "vadd.i32 q5, q5, q3\n"
      "vadd.i32 q6, q6, q3\n"
      "vadd.i32 q7, q7, q4\n"
      "vadd.i32 q8, q8, q4\n"
      "vmul.i32 q5, q5, q0\n"
      "vmul.i32 q6, q6, q0\n"
      "vmul.i32 q7, q7, q0\n"
      "vmul.i32 q8, q8, q0\n"
      "vadd.i32 q5, q5, q1\n"
      "vadd.i32 q6, q6, q1\n"
      "vadd.i32 q7, q7, q1\n"
      "vadd.i32 q8, q8, q1\n"
      "vshl.s32 q5, q5, q2\n"
      "vshl.s32 q6, q6, q2\n"
      "vshl.s32 q7, q7, q2\n"
      "vshl.s32 q8, q8, q2\n"
      "vqmovn.s32 d10, q5\n"
      "vqmovn.s32 d11, q6\n"
      "vqmovn.s32 d14, q7\n"
      "vqmovn.s32 d15, q8\n"
      "vqmovun.s16 d10, q5\n"
      "vqmovun.s16 d14, q7\n"
      "vst1.32 {d10}, [%[destination]]!\n"
      "vst1.32 {d14}, [r1]!\n"

      "bne 1b\n"
      "2:"
      "vld1.32 {d10, d11, d12}, [%[source]:64]!\n"
      "vld1.32 {d14, d15, d16}, [r0:64]!\n"
      "vadd.i32 q5, q5, q3\n"
      "vadd.i32 q6, q6, q3\n"
      "vadd.i32 q7, q7, q4\n"
      "vadd.i32 q8, q8, q4\n"
      "vmul.i32 q5, q5, q0\n"
      "vmul.i32 q6, q6, q0\n"
      "vmul.i32 q7, q7, q0\n"
      "vmul.i32 q8, q8, q0\n"
      "vadd.i32 q5, q5, q1\n"
      "vadd.i32 q6, q6, q1\n"
      "vadd.i32 q7, q7, q1\n"
      "vadd.i32 q8, q8, q1\n"
      "vshl.s32 q5, q5, q2\n"
      "vshl.s32 q6, q6, q2\n"
      "vshl.s32 q7, q7, q2\n"
      "vshl.s32 q8, q8, q2\n"
      "vqmovn.s32 d10, q5\n"
      "vqmovn.s32 d11, q6\n"
      "vqmovn.s32 d14, q7\n"
      "vqmovn.s32 d15, q8\n"
      "vqmovun.s16 d10, q5\n"
      "vqmovun.s16 d14, q7\n"
      "vst1.32 {d10[0]}, [%[destination]]!\n"
      "vst1.16 {d10[2]}, [%[destination]]!\n"
      "vst1.32 {d14[0]}, [r1]!\n"
      "vst1.16 {d14[2]}, [r1]!\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [stride] "+r"(stride), [shift] "+r"(shift),
        [destination] "+r"(destination), [offsets] "+r"(offsets),
        [source] "+r"(source), [destination_stride] "+r"(destination_stride),
        [rounding_offset] "+r"(rounding_offset)
      :
      : "r0", "r1", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9",
        "d10", "d11", "d12", "d13", "d14", "d15", "d16", "d17", "cc", "memory");
}

void qnt_2x8_7(const std::int32_t* source, std::int32_t count,
               std::int32_t stride, const std::int32_t* offsets,
               std::uint8_t* destination, std::int32_t destination_stride,
               std::int32_t multiplicative_offset, std::int32_t rounding_offset,
               std::int32_t shift) {
  asm volatile(
      "vdup.32 q0, %[multiplicative_offset]\n"
      "vdup.32 q1, %[rounding_offset]\n"
      "vdup.32 q2, %[shift]\n"
      "vld1.32 {d6[], d7[]}, [%[offsets]:32]!\n"
      "vld1.32 {d8[], d9[]}, [%[offsets]:32]!\n"
      "add r0, %[source], %[stride]\n"
      "add r1, %[destination], %[destination_stride]\n"
      "subs %[count], %[count], #7\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"
      "vld1.32 {d10, d11, d12, d13}, [%[source]:128]!\n"
      "vld1.32 {d14, d15, d16, d17}, [r0:128]!\n"
      "pld [%[source]]\n"
      "pld [r0]\n"
      "vadd.i32 q5, q5, q3\n"
      "vadd.i32 q6, q6, q3\n"
      "vadd.i32 q7, q7, q4\n"
      "vadd.i32 q8, q8, q4\n"
      "vmul.i32 q5, q5, q0\n"
      "vmul.i32 q6, q6, q0\n"
      "vmul.i32 q7, q7, q0\n"
      "vmul.i32 q8, q8, q0\n"
      "vadd.i32 q5, q5, q1\n"
      "vadd.i32 q6, q6, q1\n"
      "vadd.i32 q7, q7, q1\n"
      "vadd.i32 q8, q8, q1\n"
      "vshl.s32 q5, q5, q2\n"
      "vshl.s32 q6, q6, q2\n"
      "vshl.s32 q7, q7, q2\n"
      "vshl.s32 q8, q8, q2\n"
      "vqmovn.s32 d10, q5\n"
      "vqmovn.s32 d11, q6\n"
      "vqmovn.s32 d14, q7\n"
      "vqmovn.s32 d15, q8\n"
      "vqmovun.s16 d10, q5\n"
      "vqmovun.s16 d14, q7\n"
      "vst1.32 {d10}, [%[destination]]!\n"
      "vst1.32 {d14}, [r1]!\n"

      "bne 1b\n"
      "2:"
      "vld1.32 {d10, d11, d12}, [%[source]:64]!\n"
      "vld1.32 {d13[0]}, [%[source]]!\n"
      "vld1.32 {d14, d15, d16}, [r0:64]!\n"
      "vld1.32 {d17[0]}, [r0]!\n"
      "vadd.i32 q5, q5, q3\n"
      "vadd.i32 q6, q6, q3\n"
      "vadd.i32 q7, q7, q4\n"
      "vadd.i32 q8, q8, q4\n"
      "vmul.i32 q5, q5, q0\n"
      "vmul.i32 q6, q6, q0\n"
      "vmul.i32 q7, q7, q0\n"
      "vmul.i32 q8, q8, q0\n"
      "vadd.i32 q5, q5, q1\n"
      "vadd.i32 q6, q6, q1\n"
      "vadd.i32 q7, q7, q1\n"
      "vadd.i32 q8, q8, q1\n"
      "vshl.s32 q5, q5, q2\n"
      "vshl.s32 q6, q6, q2\n"
      "vshl.s32 q7, q7, q2\n"
      "vshl.s32 q8, q8, q2\n"
      "vqmovn.s32 d10, q5\n"
      "vqmovn.s32 d11, q6\n"
      "vqmovn.s32 d14, q7\n"
      "vqmovn.s32 d15, q8\n"
      "vqmovun.s16 d10, q5\n"
      "vqmovun.s16 d14, q7\n"
      "vst1.32 {d10[0]}, [%[destination]]!\n"
      "vst1.16 {d10[2]}, [%[destination]]!\n"
      "vst1.8 {d10[6]}, [%[destination]]!\n"
      "vst1.32 {d14[0]}, [r1]!\n"
      "vst1.16 {d14[2]}, [r1]!\n"
      "vst1.8 {d14[6]}, [r1]!\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [stride] "+r"(stride), [shift] "+r"(shift),
        [destination] "+r"(destination), [offsets] "+r"(offsets),
        [source] "+r"(source), [destination_stride] "+r"(destination_stride),
        [rounding_offset] "+r"(rounding_offset)
      :
      : "r0", "r1", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9",
        "d10", "d11", "d12", "d13", "d14", "d15", "d16", "d17", "cc", "memory");
}

void qnt_3x8(const std::int32_t* source, std::int32_t count,
             std::int32_t stride, const std::int32_t* offsets,
             std::uint8_t* destination, std::int32_t destination_stride,
             std::int32_t multiplicative_offset, std::int32_t rounding_offset,
             std::int32_t shift) {
  asm volatile(
      "vdup.32 q0, %[multiplicative_offset]\n"
      "vdup.32 q1, %[rounding_offset]\n"
      "vdup.32 q2, %[shift]\n"
      "vld1.32 {d6[], d7[]}, [%[offsets]:32]!\n"
      "vld1.32 {d8[], d9[]}, [%[offsets]:32]!\n"
      "vld1.32 {d10[], d11[]}, [%[offsets]:32]!\n"
      "add r0, %[source], %[stride]\n"
      "add r1, %[destination], %[destination_stride]\n"
      "add r2, r0, %[stride]\n"
      "add r3, r1, %[destination_stride]\n"

      "1:"
      "subs %[count], %[count], #8\n"
      "vld1.32 {d12, d13, d14, d15}, [%[source]:128]!\n"
      "vld1.32 {d16, d17, d18, d19}, [r0:128]!\n"
      "vld1.32 {d20, d21, d22, d23}, [r2:128]!\n"
      "pld [%[source]]\n"
      "pld [r0]\n"
      "pld [r2]\n"
      "vadd.i32 q6, q6, q3\n"
      "vadd.i32 q7, q7, q3\n"
      "vadd.i32 q8, q8, q4\n"
      "vadd.i32 q9, q9, q4\n"
      "vadd.i32 q10, q10, q5\n"
      "vadd.i32 q11, q11, q5\n"
      "vmul.i32 q6, q6, q0\n"
      "vmul.i32 q7, q7, q0\n"
      "vmul.i32 q8, q8, q0\n"
      "vmul.i32 q9, q9, q0\n"
      "vmul.i32 q10, q10, q0\n"
      "vmul.i32 q11, q11, q0\n"
      "vadd.i32 q6, q6, q1\n"
      "vadd.i32 q7, q7, q1\n"
      "vadd.i32 q8, q8, q1\n"
      "vadd.i32 q9, q9, q1\n"
      "vadd.i32 q10, q10, q1\n"
      "vadd.i32 q11, q11, q1\n"
      "vshl.s32 q6, q6, q2\n"
      "vshl.s32 q7, q7, q2\n"
      "vshl.s32 q8, q8, q2\n"
      "vshl.s32 q9, q9, q2\n"
      "vshl.s32 q10, q10, q2\n"
      "vshl.s32 q11, q11, q2\n"
      "vqmovn.s32 d12, q6\n"
      "vqmovn.s32 d13, q7\n"
      "vqmovn.s32 d16, q8\n"
      "vqmovn.s32 d17, q9\n"
      "vqmovn.s32 d20, q10\n"
      "vqmovn.s32 d21, q11\n"
      "vqmovun.s16 d12, q6\n"
      "vqmovun.s16 d16, q8\n"
      "vqmovun.s16 d20, q10\n"
      "vst1.32 {d12}, [%[destination]]!\n"
      "vst1.32 {d16}, [r1]!\n"
      "vst1.32 {d20}, [r3]!\n"

      "bne 1b\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [stride] "+r"(stride), [shift] "+r"(shift),
        [destination] "+r"(destination), [offsets] "+r"(offsets),
        [source] "+r"(source), [destination_stride] "+r"(destination_stride),
        [rounding_offset] "+r"(rounding_offset)
      :
      : "r0", "r1", "r2", "r3", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7",
        "d8", "d9", "d10", "d11", "d12", "d13", "d14", "d15", "d16", "d17",
        "d18", "d19", "d20", "d21", "d22", "d23", "cc", "memory");
}

void qnt_3x8_1(const std::int32_t* source, std::int32_t count,
               std::int32_t stride, const std::int32_t* offsets,
               std::uint8_t* destination, std::int32_t destination_stride,
               std::int32_t multiplicative_offset, std::int32_t rounding_offset,
               std::int32_t shift) {
  asm volatile(
      "vdup.32 q0, %[multiplicative_offset]\n"
      "vdup.32 q1, %[rounding_offset]\n"
      "vdup.32 q2, %[shift]\n"
      "vld1.32 {d6[], d7[]}, [%[offsets]:32]!\n"
      "vld1.32 {d8[], d9[]}, [%[offsets]:32]!\n"
      "vld1.32 {d10[], d11[]}, [%[offsets]:32]!\n"
      "add r0, %[source], %[stride]\n"
      "add r1, %[destination], %[destination_stride]\n"
      "add r2, r0, %[stride]\n"
      "add r3, r1, %[destination_stride]\n"
      "subs %[count], %[count], #1\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"
      "vld1.32 {d12, d13, d14, d15}, [%[source]:128]!\n"
      "vld1.32 {d16, d17, d18, d19}, [r0:128]!\n"
      "vld1.32 {d20, d21, d22, d23}, [r2:128]!\n"
      "pld [%[source]]\n"
      "pld [r0]\n"
      "pld [r2]\n"
      "vadd.i32 q6, q6, q3\n"
      "vadd.i32 q7, q7, q3\n"
      "vadd.i32 q8, q8, q4\n"
      "vadd.i32 q9, q9, q4\n"
      "vadd.i32 q10, q10, q5\n"
      "vadd.i32 q11, q11, q5\n"
      "vmul.i32 q6, q6, q0\n"
      "vmul.i32 q7, q7, q0\n"
      "vmul.i32 q8, q8, q0\n"
      "vmul.i32 q9, q9, q0\n"
      "vmul.i32 q10, q10, q0\n"
      "vmul.i32 q11, q11, q0\n"
      "vadd.i32 q6, q6, q1\n"
      "vadd.i32 q7, q7, q1\n"
      "vadd.i32 q8, q8, q1\n"
      "vadd.i32 q9, q9, q1\n"
      "vadd.i32 q10, q10, q1\n"
      "vadd.i32 q11, q11, q1\n"
      "vshl.s32 q6, q6, q2\n"
      "vshl.s32 q7, q7, q2\n"
      "vshl.s32 q8, q8, q2\n"
      "vshl.s32 q9, q9, q2\n"
      "vshl.s32 q10, q10, q2\n"
      "vshl.s32 q11, q11, q2\n"
      "vqmovn.s32 d12, q6\n"
      "vqmovn.s32 d13, q7\n"
      "vqmovn.s32 d16, q8\n"
      "vqmovn.s32 d17, q9\n"
      "vqmovn.s32 d20, q10\n"
      "vqmovn.s32 d21, q11\n"
      "vqmovun.s16 d12, q6\n"
      "vqmovun.s16 d16, q8\n"
      "vqmovun.s16 d20, q10\n"
      "vst1.32 {d12}, [%[destination]]!\n"
      "vst1.32 {d16}, [r1]!\n"
      "vst1.32 {d20}, [r3]!\n"

      "bne 1b\n"
      "2:"
      "vld1.32 {d12[0]}, [%[source]]!\n"
      "vld1.32 {d16[0]}, [r0]!\n"
      "vld1.32 {d20[0]}, [r2]!\n"
      "vadd.i32 q6, q6, q3\n"
      "vadd.i32 q8, q8, q4\n"
      "vadd.i32 q10, q10, q5\n"
      "vmul.i32 q6, q6, q0\n"
      "vmul.i32 q8, q8, q0\n"
      "vmul.i32 q10, q10, q0\n"
      "vadd.i32 q6, q6, q1\n"
      "vadd.i32 q8, q8, q1\n"
      "vadd.i32 q10, q10, q1\n"
      "vshl.s32 q6, q6, q2\n"
      "vshl.s32 q8, q8, q2\n"
      "vshl.s32 q10, q10, q2\n"
      "vqmovn.s32 d12, q6\n"
      "vqmovn.s32 d16, q8\n"
      "vqmovn.s32 d20, q10\n"
      "vqmovun.s16 d12, q6\n"
      "vqmovun.s16 d16, q8\n"
      "vqmovun.s16 d20, q10\n"
      "vst1.8 {d12[0]}, [%[destination]]!\n"
      "vst1.8 {d16[0]}, [r1]!\n"
      "vst1.8 {d20[0]}, [r3]!\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [stride] "+r"(stride), [shift] "+r"(shift),
        [destination] "+r"(destination), [offsets] "+r"(offsets),
        [source] "+r"(source), [destination_stride] "+r"(destination_stride),
        [rounding_offset] "+r"(rounding_offset)
      :
      : "r0", "r1", "r2", "r3", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7",
        "d8", "d9", "d10", "d11", "d12", "d13", "d14", "d15", "d16", "d17",
        "d18", "d19", "d20", "d21", "d22", "d23", "cc", "memory");
}

void qnt_3x8_2(const std::int32_t* source, std::int32_t count,
               std::int32_t stride, const std::int32_t* offsets,
               std::uint8_t* destination, std::int32_t destination_stride,
               std::int32_t multiplicative_offset, std::int32_t rounding_offset,
               std::int32_t shift) {
  asm volatile(
      "vdup.32 q0, %[multiplicative_offset]\n"
      "vdup.32 q1, %[rounding_offset]\n"
      "vdup.32 q2, %[shift]\n"
      "vld1.32 {d6[], d7[]}, [%[offsets]:32]!\n"
      "vld1.32 {d8[], d9[]}, [%[offsets]:32]!\n"
      "vld1.32 {d10[], d11[]}, [%[offsets]:32]!\n"
      "add r0, %[source], %[stride]\n"
      "add r1, %[destination], %[destination_stride]\n"
      "add r2, r0, %[stride]\n"
      "add r3, r1, %[destination_stride]\n"
      "subs %[count], %[count], #2\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"
      "vld1.32 {d12, d13, d14, d15}, [%[source]:128]!\n"
      "vld1.32 {d16, d17, d18, d19}, [r0:128]!\n"
      "vld1.32 {d20, d21, d22, d23}, [r2:128]!\n"
      "pld [%[source]]\n"
      "pld [r0]\n"
      "pld [r2]\n"
      "vadd.i32 q6, q6, q3\n"
      "vadd.i32 q7, q7, q3\n"
      "vadd.i32 q8, q8, q4\n"
      "vadd.i32 q9, q9, q4\n"
      "vadd.i32 q10, q10, q5\n"
      "vadd.i32 q11, q11, q5\n"
      "vmul.i32 q6, q6, q0\n"
      "vmul.i32 q7, q7, q0\n"
      "vmul.i32 q8, q8, q0\n"
      "vmul.i32 q9, q9, q0\n"
      "vmul.i32 q10, q10, q0\n"
      "vmul.i32 q11, q11, q0\n"
      "vadd.i32 q6, q6, q1\n"
      "vadd.i32 q7, q7, q1\n"
      "vadd.i32 q8, q8, q1\n"
      "vadd.i32 q9, q9, q1\n"
      "vadd.i32 q10, q10, q1\n"
      "vadd.i32 q11, q11, q1\n"
      "vshl.s32 q6, q6, q2\n"
      "vshl.s32 q7, q7, q2\n"
      "vshl.s32 q8, q8, q2\n"
      "vshl.s32 q9, q9, q2\n"
      "vshl.s32 q10, q10, q2\n"
      "vshl.s32 q11, q11, q2\n"
      "vqmovn.s32 d12, q6\n"
      "vqmovn.s32 d13, q7\n"
      "vqmovn.s32 d16, q8\n"
      "vqmovn.s32 d17, q9\n"
      "vqmovn.s32 d20, q10\n"
      "vqmovn.s32 d21, q11\n"
      "vqmovun.s16 d12, q6\n"
      "vqmovun.s16 d16, q8\n"
      "vqmovun.s16 d20, q10\n"
      "vst1.32 {d12}, [%[destination]]!\n"
      "vst1.32 {d16}, [r1]!\n"
      "vst1.32 {d20}, [r3]!\n"

      "bne 1b\n"
      "2:"
      "vld1.32 {d12}, [%[source]:64]!\n"
      "vld1.32 {d16}, [r0:64]!\n"
      "vld1.32 {d20}, [r2:64]!\n"
      "vadd.i32 q6, q6, q3\n"
      "vadd.i32 q8, q8, q4\n"
      "vadd.i32 q10, q10, q5\n"
      "vmul.i32 q6, q6, q0\n"
      "vmul.i32 q8, q8, q0\n"
      "vmul.i32 q10, q10, q0\n"
      "vadd.i32 q6, q6, q1\n"
      "vadd.i32 q8, q8, q1\n"
      "vadd.i32 q10, q10, q1\n"
      "vshl.s32 q6, q6, q2\n"
      "vshl.s32 q8, q8, q2\n"
      "vshl.s32 q10, q10, q2\n"
      "vqmovn.s32 d12, q6\n"
      "vqmovn.s32 d16, q8\n"
      "vqmovn.s32 d20, q10\n"
      "vqmovun.s16 d12, q6\n"
      "vqmovun.s16 d16, q8\n"
      "vqmovun.s16 d20, q10\n"
      "vst1.16 {d12[0]}, [%[destination]]!\n"
      "vst1.16 {d16[0]}, [r1]!\n"
      "vst1.16 {d20[0]}, [r3]!\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [stride] "+r"(stride), [shift] "+r"(shift),
        [destination] "+r"(destination), [offsets] "+r"(offsets),
        [source] "+r"(source), [destination_stride] "+r"(destination_stride),
        [rounding_offset] "+r"(rounding_offset)
      :
      : "r0", "r1", "r2", "r3", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7",
        "d8", "d9", "d10", "d11", "d12", "d13", "d14", "d15", "d16", "d17",
        "d18", "d19", "d20", "d21", "d22", "d23", "cc", "memory");
}

void qnt_3x8_3(const std::int32_t* source, std::int32_t count,
               std::int32_t stride, const std::int32_t* offsets,
               std::uint8_t* destination, std::int32_t destination_stride,
               std::int32_t multiplicative_offset, std::int32_t rounding_offset,
               std::int32_t shift) {
  asm volatile(
      "vdup.32 q0, %[multiplicative_offset]\n"
      "vdup.32 q1, %[rounding_offset]\n"
      "vdup.32 q2, %[shift]\n"
      "vld1.32 {d6[], d7[]}, [%[offsets]:32]!\n"
      "vld1.32 {d8[], d9[]}, [%[offsets]:32]!\n"
      "vld1.32 {d10[], d11[]}, [%[offsets]:32]!\n"
      "add r0, %[source], %[stride]\n"
      "add r1, %[destination], %[destination_stride]\n"
      "add r2, r0, %[stride]\n"
      "add r3, r1, %[destination_stride]\n"
      "subs %[count], %[count], #3\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"
      "vld1.32 {d12, d13, d14, d15}, [%[source]:128]!\n"
      "vld1.32 {d16, d17, d18, d19}, [r0:128]!\n"
      "vld1.32 {d20, d21, d22, d23}, [r2:128]!\n"
      "pld [%[source]]\n"
      "pld [r0]\n"
      "pld [r2]\n"
      "vadd.i32 q6, q6, q3\n"
      "vadd.i32 q7, q7, q3\n"
      "vadd.i32 q8, q8, q4\n"
      "vadd.i32 q9, q9, q4\n"
      "vadd.i32 q10, q10, q5\n"
      "vadd.i32 q11, q11, q5\n"
      "vmul.i32 q6, q6, q0\n"
      "vmul.i32 q7, q7, q0\n"
      "vmul.i32 q8, q8, q0\n"
      "vmul.i32 q9, q9, q0\n"
      "vmul.i32 q10, q10, q0\n"
      "vmul.i32 q11, q11, q0\n"
      "vadd.i32 q6, q6, q1\n"
      "vadd.i32 q7, q7, q1\n"
      "vadd.i32 q8, q8, q1\n"
      "vadd.i32 q9, q9, q1\n"
      "vadd.i32 q10, q10, q1\n"
      "vadd.i32 q11, q11, q1\n"
      "vshl.s32 q6, q6, q2\n"
      "vshl.s32 q7, q7, q2\n"
      "vshl.s32 q8, q8, q2\n"
      "vshl.s32 q9, q9, q2\n"
      "vshl.s32 q10, q10, q2\n"
      "vshl.s32 q11, q11, q2\n"
      "vqmovn.s32 d12, q6\n"
      "vqmovn.s32 d13, q7\n"
      "vqmovn.s32 d16, q8\n"
      "vqmovn.s32 d17, q9\n"
      "vqmovn.s32 d20, q10\n"
      "vqmovn.s32 d21, q11\n"
      "vqmovun.s16 d12, q6\n"
      "vqmovun.s16 d16, q8\n"
      "vqmovun.s16 d20, q10\n"
      "vst1.32 {d12}, [%[destination]]!\n"
      "vst1.32 {d16}, [r1]!\n"
      "vst1.32 {d20}, [r3]!\n"

      "bne 1b\n"
      "2:"
      "vld1.32 {d12}, [%[source]:64]!\n"
      "vld1.32 {d13[0]}, [%[source]]!\n"
      "vld1.32 {d16}, [r0:64]!\n"
      "vld1.32 {d17[0]}, [r0]!\n"
      "vld1.32 {d20}, [r2:64]!\n"
      "vld1.32 {d21[0]}, [r2]!\n"
      "vadd.i32 q6, q6, q3\n"
      "vadd.i32 q8, q8, q4\n"
      "vadd.i32 q10, q10, q5\n"
      "vmul.i32 q6, q6, q0\n"
      "vmul.i32 q8, q8, q0\n"
      "vmul.i32 q10, q10, q0\n"
      "vadd.i32 q6, q6, q1\n"
      "vadd.i32 q8, q8, q1\n"
      "vadd.i32 q10, q10, q1\n"
      "vshl.s32 q6, q6, q2\n"
      "vshl.s32 q8, q8, q2\n"
      "vshl.s32 q10, q10, q2\n"
      "vqmovn.s32 d12, q6\n"
      "vqmovn.s32 d16, q8\n"
      "vqmovn.s32 d20, q10\n"
      "vqmovun.s16 d12, q6\n"
      "vqmovun.s16 d16, q8\n"
      "vqmovun.s16 d20, q10\n"
      "vst1.16 {d12[0]}, [%[destination]]!\n"
      "vst1.8 {d12[2]}, [%[destination]]!\n"
      "vst1.16 {d16[0]}, [r1]!\n"
      "vst1.8 {d16[2]}, [r1]!\n"
      "vst1.16 {d20[0]}, [r3]!\n"
      "vst1.8 {d20[2]}, [r3]!\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [stride] "+r"(stride), [shift] "+r"(shift),
        [destination] "+r"(destination), [offsets] "+r"(offsets),
        [source] "+r"(source), [destination_stride] "+r"(destination_stride),
        [rounding_offset] "+r"(rounding_offset)
      :
      : "r0", "r1", "r2", "r3", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7",
        "d8", "d9", "d10", "d11", "d12", "d13", "d14", "d15", "d16", "d17",
        "d18", "d19", "d20", "d21", "d22", "d23", "cc", "memory");
}

void qnt_3x8_4(const std::int32_t* source, std::int32_t count,
               std::int32_t stride, const std::int32_t* offsets,
               std::uint8_t* destination, std::int32_t destination_stride,
               std::int32_t multiplicative_offset, std::int32_t rounding_offset,
               std::int32_t shift) {
  asm volatile(
      "vdup.32 q0, %[multiplicative_offset]\n"
      "vdup.32 q1, %[rounding_offset]\n"
      "vdup.32 q2, %[shift]\n"
      "vld1.32 {d6[], d7[]}, [%[offsets]:32]!\n"
      "vld1.32 {d8[], d9[]}, [%[offsets]:32]!\n"
      "vld1.32 {d10[], d11[]}, [%[offsets]:32]!\n"
      "add r0, %[source], %[stride]\n"
      "add r1, %[destination], %[destination_stride]\n"
      "add r2, r0, %[stride]\n"
      "add r3, r1, %[destination_stride]\n"
      "subs %[count], %[count], #4\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"
      "vld1.32 {d12, d13, d14, d15}, [%[source]:128]!\n"
      "vld1.32 {d16, d17, d18, d19}, [r0:128]!\n"
      "vld1.32 {d20, d21, d22, d23}, [r2:128]!\n"
      "pld [%[source]]\n"
      "pld [r0]\n"
      "pld [r2]\n"
      "vadd.i32 q6, q6, q3\n"
      "vadd.i32 q7, q7, q3\n"
      "vadd.i32 q8, q8, q4\n"
      "vadd.i32 q9, q9, q4\n"
      "vadd.i32 q10, q10, q5\n"
      "vadd.i32 q11, q11, q5\n"
      "vmul.i32 q6, q6, q0\n"
      "vmul.i32 q7, q7, q0\n"
      "vmul.i32 q8, q8, q0\n"
      "vmul.i32 q9, q9, q0\n"
      "vmul.i32 q10, q10, q0\n"
      "vmul.i32 q11, q11, q0\n"
      "vadd.i32 q6, q6, q1\n"
      "vadd.i32 q7, q7, q1\n"
      "vadd.i32 q8, q8, q1\n"
      "vadd.i32 q9, q9, q1\n"
      "vadd.i32 q10, q10, q1\n"
      "vadd.i32 q11, q11, q1\n"
      "vshl.s32 q6, q6, q2\n"
      "vshl.s32 q7, q7, q2\n"
      "vshl.s32 q8, q8, q2\n"
      "vshl.s32 q9, q9, q2\n"
      "vshl.s32 q10, q10, q2\n"
      "vshl.s32 q11, q11, q2\n"
      "vqmovn.s32 d12, q6\n"
      "vqmovn.s32 d13, q7\n"
      "vqmovn.s32 d16, q8\n"
      "vqmovn.s32 d17, q9\n"
      "vqmovn.s32 d20, q10\n"
      "vqmovn.s32 d21, q11\n"
      "vqmovun.s16 d12, q6\n"
      "vqmovun.s16 d16, q8\n"
      "vqmovun.s16 d20, q10\n"
      "vst1.32 {d12}, [%[destination]]!\n"
      "vst1.32 {d16}, [r1]!\n"
      "vst1.32 {d20}, [r3]!\n"

      "bne 1b\n"
      "2:"
      "vld1.32 {d12, d13}, [%[source]:64]!\n"
      "vld1.32 {d16, d17}, [r0:64]!\n"
      "vld1.32 {d20, d21}, [r2:64]!\n"
      "vadd.i32 q6, q6, q3\n"
      "vadd.i32 q8, q8, q4\n"
      "vadd.i32 q10, q10, q5\n"
      "vmul.i32 q6, q6, q0\n"
      "vmul.i32 q8, q8, q0\n"
      "vmul.i32 q10, q10, q0\n"
      "vadd.i32 q6, q6, q1\n"
      "vadd.i32 q8, q8, q1\n"
      "vadd.i32 q10, q10, q1\n"
      "vshl.s32 q6, q6, q2\n"
      "vshl.s32 q8, q8, q2\n"
      "vshl.s32 q10, q10, q2\n"
      "vqmovn.s32 d12, q6\n"
      "vqmovn.s32 d16, q8\n"
      "vqmovn.s32 d20, q10\n"
      "vqmovun.s16 d12, q6\n"
      "vqmovun.s16 d16, q8\n"
      "vqmovun.s16 d20, q10\n"
      "vst1.32 {d12[0]}, [%[destination]]!\n"
      "vst1.32 {d16[0]}, [r1]!\n"
      "vst1.32 {d20[0]}, [r3]!\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [stride] "+r"(stride), [shift] "+r"(shift),
        [destination] "+r"(destination), [offsets] "+r"(offsets),
        [source] "+r"(source), [destination_stride] "+r"(destination_stride),
        [rounding_offset] "+r"(rounding_offset)
      :
      : "r0", "r1", "r2", "r3", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7",
        "d8", "d9", "d10", "d11", "d12", "d13", "d14", "d15", "d16", "d17",
        "d18", "d19", "d20", "d21", "d22", "d23", "cc", "memory");
}

void qnt_3x8_5(const std::int32_t* source, std::int32_t count,
               std::int32_t stride, const std::int32_t* offsets,
               std::uint8_t* destination, std::int32_t destination_stride,
               std::int32_t multiplicative_offset, std::int32_t rounding_offset,
               std::int32_t shift) {
  asm volatile(
      "vdup.32 q0, %[multiplicative_offset]\n"
      "vdup.32 q1, %[rounding_offset]\n"
      "vdup.32 q2, %[shift]\n"
      "vld1.32 {d6[], d7[]}, [%[offsets]:32]!\n"
      "vld1.32 {d8[], d9[]}, [%[offsets]:32]!\n"
      "vld1.32 {d10[], d11[]}, [%[offsets]:32]!\n"
      "add r0, %[source], %[stride]\n"
      "add r1, %[destination], %[destination_stride]\n"
      "add r2, r0, %[stride]\n"
      "add r3, r1, %[destination_stride]\n"
      "subs %[count], %[count], #5\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"
      "vld1.32 {d12, d13, d14, d15}, [%[source]:128]!\n"
      "vld1.32 {d16, d17, d18, d19}, [r0:128]!\n"
      "vld1.32 {d20, d21, d22, d23}, [r2:128]!\n"
      "pld [%[source]]\n"
      "pld [r0]\n"
      "pld [r2]\n"
      "vadd.i32 q6, q6, q3\n"
      "vadd.i32 q7, q7, q3\n"
      "vadd.i32 q8, q8, q4\n"
      "vadd.i32 q9, q9, q4\n"
      "vadd.i32 q10, q10, q5\n"
      "vadd.i32 q11, q11, q5\n"
      "vmul.i32 q6, q6, q0\n"
      "vmul.i32 q7, q7, q0\n"
      "vmul.i32 q8, q8, q0\n"
      "vmul.i32 q9, q9, q0\n"
      "vmul.i32 q10, q10, q0\n"
      "vmul.i32 q11, q11, q0\n"
      "vadd.i32 q6, q6, q1\n"
      "vadd.i32 q7, q7, q1\n"
      "vadd.i32 q8, q8, q1\n"
      "vadd.i32 q9, q9, q1\n"
      "vadd.i32 q10, q10, q1\n"
      "vadd.i32 q11, q11, q1\n"
      "vshl.s32 q6, q6, q2\n"
      "vshl.s32 q7, q7, q2\n"
      "vshl.s32 q8, q8, q2\n"
      "vshl.s32 q9, q9, q2\n"
      "vshl.s32 q10, q10, q2\n"
      "vshl.s32 q11, q11, q2\n"
      "vqmovn.s32 d12, q6\n"
      "vqmovn.s32 d13, q7\n"
      "vqmovn.s32 d16, q8\n"
      "vqmovn.s32 d17, q9\n"
      "vqmovn.s32 d20, q10\n"
      "vqmovn.s32 d21, q11\n"
      "vqmovun.s16 d12, q6\n"
      "vqmovun.s16 d16, q8\n"
      "vqmovun.s16 d20, q10\n"
      "vst1.32 {d12}, [%[destination]]!\n"
      "vst1.32 {d16}, [r1]!\n"
      "vst1.32 {d20}, [r3]!\n"

      "bne 1b\n"
      "2:"
      "vld1.32 {d12, d13}, [%[source]:64]!\n"
      "vld1.32 {d14[0]}, [%[source]]!\n"
      "vld1.32 {d16, d17}, [r0:64]!\n"
      "vld1.32 {d18[0]}, [r0]!\n"
      "vld1.32 {d20, d21}, [r2:64]!\n"
      "vld1.32 {d22[0]}, [r2]!\n"
      "vadd.i32 q6, q6, q3\n"
      "vadd.i32 q7, q7, q3\n"
      "vadd.i32 q8, q8, q4\n"
      "vadd.i32 q9, q9, q4\n"
      "vadd.i32 q10, q10, q5\n"
      "vadd.i32 q11, q11, q5\n"
      "vmul.i32 q6, q6, q0\n"
      "vmul.i32 q7, q7, q0\n"
      "vmul.i32 q8, q8, q0\n"
      "vmul.i32 q9, q9, q0\n"
      "vmul.i32 q10, q10, q0\n"
      "vmul.i32 q11, q11, q0\n"
      "vadd.i32 q6, q6, q1\n"
      "vadd.i32 q7, q7, q1\n"
      "vadd.i32 q8, q8, q1\n"
      "vadd.i32 q9, q9, q1\n"
      "vadd.i32 q10, q10, q1\n"
      "vadd.i32 q11, q11, q1\n"
      "vshl.s32 q6, q6, q2\n"
      "vshl.s32 q7, q7, q2\n"
      "vshl.s32 q8, q8, q2\n"
      "vshl.s32 q9, q9, q2\n"
      "vshl.s32 q10, q10, q2\n"
      "vshl.s32 q11, q11, q2\n"
      "vqmovn.s32 d12, q6\n"
      "vqmovn.s32 d13, q7\n"
      "vqmovn.s32 d16, q8\n"
      "vqmovn.s32 d17, q9\n"
      "vqmovn.s32 d20, q10\n"
      "vqmovn.s32 d21, q11\n"
      "vqmovun.s16 d12, q6\n"
      "vqmovun.s16 d16, q8\n"
      "vqmovun.s16 d20, q10\n"
      "vst1.32 {d12[0]}, [%[destination]]!\n"
      "vst1.8 {d12[4]}, [%[destination]]!\n"
      "vst1.32 {d16[0]}, [r1]!\n"
      "vst1.8 {d16[4]}, [r1]!\n"
      "vst1.32 {d20[0]}, [r3]!\n"
      "vst1.8 {d20[4]}, [r3]!\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [stride] "+r"(stride), [shift] "+r"(shift),
        [destination] "+r"(destination), [offsets] "+r"(offsets),
        [source] "+r"(source), [destination_stride] "+r"(destination_stride),
        [rounding_offset] "+r"(rounding_offset)
      :
      : "r0", "r1", "r2", "r3", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7",
        "d8", "d9", "d10", "d11", "d12", "d13", "d14", "d15", "d16", "d17",
        "d18", "d19", "d20", "d21", "d22", "d23", "cc", "memory");
}

void qnt_3x8_6(const std::int32_t* source, std::int32_t count,
               std::int32_t stride, const std::int32_t* offsets,
               std::uint8_t* destination, std::int32_t destination_stride,
               std::int32_t multiplicative_offset, std::int32_t rounding_offset,
               std::int32_t shift) {
  asm volatile(
      "vdup.32 q0, %[multiplicative_offset]\n"
      "vdup.32 q1, %[rounding_offset]\n"
      "vdup.32 q2, %[shift]\n"
      "vld1.32 {d6[], d7[]}, [%[offsets]:32]!\n"
      "vld1.32 {d8[], d9[]}, [%[offsets]:32]!\n"
      "vld1.32 {d10[], d11[]}, [%[offsets]:32]!\n"
      "add r0, %[source], %[stride]\n"
      "add r1, %[destination], %[destination_stride]\n"
      "add r2, r0, %[stride]\n"
      "add r3, r1, %[destination_stride]\n"
      "subs %[count], %[count], #6\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"
      "vld1.32 {d12, d13, d14, d15}, [%[source]:128]!\n"
      "vld1.32 {d16, d17, d18, d19}, [r0:128]!\n"
      "vld1.32 {d20, d21, d22, d23}, [r2:128]!\n"
      "pld [%[source]]\n"
      "pld [r0]\n"
      "pld [r2]\n"
      "vadd.i32 q6, q6, q3\n"
      "vadd.i32 q7, q7, q3\n"
      "vadd.i32 q8, q8, q4\n"
      "vadd.i32 q9, q9, q4\n"
      "vadd.i32 q10, q10, q5\n"
      "vadd.i32 q11, q11, q5\n"
      "vmul.i32 q6, q6, q0\n"
      "vmul.i32 q7, q7, q0\n"
      "vmul.i32 q8, q8, q0\n"
      "vmul.i32 q9, q9, q0\n"
      "vmul.i32 q10, q10, q0\n"
      "vmul.i32 q11, q11, q0\n"
      "vadd.i32 q6, q6, q1\n"
      "vadd.i32 q7, q7, q1\n"
      "vadd.i32 q8, q8, q1\n"
      "vadd.i32 q9, q9, q1\n"
      "vadd.i32 q10, q10, q1\n"
      "vadd.i32 q11, q11, q1\n"
      "vshl.s32 q6, q6, q2\n"
      "vshl.s32 q7, q7, q2\n"
      "vshl.s32 q8, q8, q2\n"
      "vshl.s32 q9, q9, q2\n"
      "vshl.s32 q10, q10, q2\n"
      "vshl.s32 q11, q11, q2\n"
      "vqmovn.s32 d12, q6\n"
      "vqmovn.s32 d13, q7\n"
      "vqmovn.s32 d16, q8\n"
      "vqmovn.s32 d17, q9\n"
      "vqmovn.s32 d20, q10\n"
      "vqmovn.s32 d21, q11\n"
      "vqmovun.s16 d12, q6\n"
      "vqmovun.s16 d16, q8\n"
      "vqmovun.s16 d20, q10\n"
      "vst1.32 {d12}, [%[destination]]!\n"
      "vst1.32 {d16}, [r1]!\n"
      "vst1.32 {d20}, [r3]!\n"

      "bne 1b\n"
      "2:"
      "vld1.32 {d12, d13, d14}, [%[source]:64]!\n"
      "vld1.32 {d16, d17, d18}, [r0:64]!\n"
      "vld1.32 {d20, d21, d22}, [r2:64]!\n"
      "vadd.i32 q6, q6, q3\n"
      "vadd.i32 q7, q7, q3\n"
      "vadd.i32 q8, q8, q4\n"
      "vadd.i32 q9, q9, q4\n"
      "vadd.i32 q10, q10, q5\n"
      "vadd.i32 q11, q11, q5\n"
      "vmul.i32 q6, q6, q0\n"
      "vmul.i32 q7, q7, q0\n"
      "vmul.i32 q8, q8, q0\n"
      "vmul.i32 q9, q9, q0\n"
      "vmul.i32 q10, q10, q0\n"
      "vmul.i32 q11, q11, q0\n"
      "vadd.i32 q6, q6, q1\n"
      "vadd.i32 q7, q7, q1\n"
      "vadd.i32 q8, q8, q1\n"
      "vadd.i32 q9, q9, q1\n"
      "vadd.i32 q10, q10, q1\n"
      "vadd.i32 q11, q11, q1\n"
      "vshl.s32 q6, q6, q2\n"
      "vshl.s32 q7, q7, q2\n"
      "vshl.s32 q8, q8, q2\n"
      "vshl.s32 q9, q9, q2\n"
      "vshl.s32 q10, q10, q2\n"
      "vshl.s32 q11, q11, q2\n"
      "vqmovn.s32 d12, q6\n"
      "vqmovn.s32 d13, q7\n"
      "vqmovn.s32 d16, q8\n"
      "vqmovn.s32 d17, q9\n"
      "vqmovn.s32 d20, q10\n"
      "vqmovn.s32 d21, q11\n"
      "vqmovun.s16 d12, q6\n"
      "vqmovun.s16 d16, q8\n"
      "vqmovun.s16 d20, q10\n"
      "vst1.32 {d12[0]}, [%[destination]]!\n"
      "vst1.16 {d12[2]}, [%[destination]]!\n"
      "vst1.32 {d16[0]}, [r1]!\n"
      "vst1.16 {d16[2]}, [r1]!\n"
      "vst1.32 {d20[0]}, [r3]!\n"
      "vst1.16 {d20[2]}, [r3]!\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [stride] "+r"(stride), [shift] "+r"(shift),
        [destination] "+r"(destination), [offsets] "+r"(offsets),
        [source] "+r"(source), [destination_stride] "+r"(destination_stride),
        [rounding_offset] "+r"(rounding_offset)
      :
      : "r0", "r1", "r2", "r3", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7",
        "d8", "d9", "d10", "d11", "d12", "d13", "d14", "d15", "d16", "d17",
        "d18", "d19", "d20", "d21", "d22", "d23", "cc", "memory");
}

void qnt_3x8_7(const std::int32_t* source, std::int32_t count,
               std::int32_t stride, const std::int32_t* offsets,
               std::uint8_t* destination, std::int32_t destination_stride,
               std::int32_t multiplicative_offset, std::int32_t rounding_offset,
               std::int32_t shift) {
  asm volatile(
      "vdup.32 q0, %[multiplicative_offset]\n"
      "vdup.32 q1, %[rounding_offset]\n"
      "vdup.32 q2, %[shift]\n"
      "vld1.32 {d6[], d7[]}, [%[offsets]:32]!\n"
      "vld1.32 {d8[], d9[]}, [%[offsets]:32]!\n"
      "vld1.32 {d10[], d11[]}, [%[offsets]:32]!\n"
      "add r0, %[source], %[stride]\n"
      "add r1, %[destination], %[destination_stride]\n"
      "add r2, r0, %[stride]\n"
      "add r3, r1, %[destination_stride]\n"
      "subs %[count], %[count], #7\n"
      "beq 2f\n"

      "1:"
      "subs %[count], %[count], #8\n"
      "vld1.32 {d12, d13, d14, d15}, [%[source]:128]!\n"
      "vld1.32 {d16, d17, d18, d19}, [r0:128]!\n"
      "vld1.32 {d20, d21, d22, d23}, [r2:128]!\n"
      "pld [%[source]]\n"
      "pld [r0]\n"
      "pld [r2]\n"
      "vadd.i32 q6, q6, q3\n"
      "vadd.i32 q7, q7, q3\n"
      "vadd.i32 q8, q8, q4\n"
      "vadd.i32 q9, q9, q4\n"
      "vadd.i32 q10, q10, q5\n"
      "vadd.i32 q11, q11, q5\n"
      "vmul.i32 q6, q6, q0\n"
      "vmul.i32 q7, q7, q0\n"
      "vmul.i32 q8, q8, q0\n"
      "vmul.i32 q9, q9, q0\n"
      "vmul.i32 q10, q10, q0\n"
      "vmul.i32 q11, q11, q0\n"
      "vadd.i32 q6, q6, q1\n"
      "vadd.i32 q7, q7, q1\n"
      "vadd.i32 q8, q8, q1\n"
      "vadd.i32 q9, q9, q1\n"
      "vadd.i32 q10, q10, q1\n"
      "vadd.i32 q11, q11, q1\n"
      "vshl.s32 q6, q6, q2\n"
      "vshl.s32 q7, q7, q2\n"
      "vshl.s32 q8, q8, q2\n"
      "vshl.s32 q9, q9, q2\n"
      "vshl.s32 q10, q10, q2\n"
      "vshl.s32 q11, q11, q2\n"
      "vqmovn.s32 d12, q6\n"
      "vqmovn.s32 d13, q7\n"
      "vqmovn.s32 d16, q8\n"
      "vqmovn.s32 d17, q9\n"
      "vqmovn.s32 d20, q10\n"
      "vqmovn.s32 d21, q11\n"
      "vqmovun.s16 d12, q6\n"
      "vqmovun.s16 d16, q8\n"
      "vqmovun.s16 d20, q10\n"
      "vst1.32 {d12}, [%[destination]]!\n"
      "vst1.32 {d16}, [r1]!\n"
      "vst1.32 {d20}, [r3]!\n"

      "bne 1b\n"
      "2:"
      "vld1.32 {d12, d13, d14}, [%[source]:64]!\n"
      "vld1.32 {d15[0]}, [%[source]]!\n"
      "vld1.32 {d16, d17, d18}, [r0:64]!\n"
      "vld1.32 {d19[0]}, [r0]!\n"
      "vld1.32 {d20, d21, d22}, [r2:64]!\n"
      "vld1.32 {d23[0]}, [r2]!\n"
      "vadd.i32 q6, q6, q3\n"
      "vadd.i32 q7, q7, q3\n"
      "vadd.i32 q8, q8, q4\n"
      "vadd.i32 q9, q9, q4\n"
      "vadd.i32 q10, q10, q5\n"
      "vadd.i32 q11, q11, q5\n"
      "vmul.i32 q6, q6, q0\n"
      "vmul.i32 q7, q7, q0\n"
      "vmul.i32 q8, q8, q0\n"
      "vmul.i32 q9, q9, q0\n"
      "vmul.i32 q10, q10, q0\n"
      "vmul.i32 q11, q11, q0\n"
      "vadd.i32 q6, q6, q1\n"
      "vadd.i32 q7, q7, q1\n"
      "vadd.i32 q8, q8, q1\n"
      "vadd.i32 q9, q9, q1\n"
      "vadd.i32 q10, q10, q1\n"
      "vadd.i32 q11, q11, q1\n"
      "vshl.s32 q6, q6, q2\n"
      "vshl.s32 q7, q7, q2\n"
      "vshl.s32 q8, q8, q2\n"
      "vshl.s32 q9, q9, q2\n"
      "vshl.s32 q10, q10, q2\n"
      "vshl.s32 q11, q11, q2\n"
      "vqmovn.s32 d12, q6\n"
      "vqmovn.s32 d13, q7\n"
      "vqmovn.s32 d16, q8\n"
      "vqmovn.s32 d17, q9\n"
      "vqmovn.s32 d20, q10\n"
      "vqmovn.s32 d21, q11\n"
      "vqmovun.s16 d12, q6\n"
      "vqmovun.s16 d16, q8\n"
      "vqmovun.s16 d20, q10\n"
      "vst1.32 {d12[0]}, [%[destination]]!\n"
      "vst1.16 {d12[2]}, [%[destination]]!\n"
      "vst1.8 {d12[6]}, [%[destination]]!\n"
      "vst1.32 {d16[0]}, [r1]!\n"
      "vst1.16 {d16[2]}, [r1]!\n"
      "vst1.8 {d16[6]}, [r1]!\n"
      "vst1.32 {d20[0]}, [r3]!\n"
      "vst1.16 {d20[2]}, [r3]!\n"
      "vst1.8 {d20[6]}, [r3]!\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [stride] "+r"(stride), [shift] "+r"(shift),
        [destination] "+r"(destination), [offsets] "+r"(offsets),
        [source] "+r"(source), [destination_stride] "+r"(destination_stride),
        [rounding_offset] "+r"(rounding_offset)
      :
      : "r0", "r1", "r2", "r3", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7",
        "d8", "d9", "d10", "d11", "d12", "d13", "d14", "d15", "d16", "d17",
        "d18", "d19", "d20", "d21", "d22", "d23", "cc", "memory");
}

void multi_qnt_1x8_aligned(const std::int32_t* source, std::int32_t count,
                           std::int32_t stride, const std::int32_t* offsets,
                           std::uint8_t* destination,
                           std::int32_t destination_stride,
                           std::int32_t multiplicative_offset,
                           std::int32_t rounding_offset, std::int32_t shift) {
  switch (count % 8) {
    case 0:
      qnt_1x8_aligned(source, count, stride, offsets, destination,
                      destination_stride, multiplicative_offset,
                      rounding_offset, shift);
      break;
    case 1:
      qnt_1x8_1_aligned(source, count, stride, offsets, destination,
                        destination_stride, multiplicative_offset,
                        rounding_offset, shift);
      break;
    case 2:
      qnt_1x8_2_aligned(source, count, stride, offsets, destination,
                        destination_stride, multiplicative_offset,
                        rounding_offset, shift);
      break;
    case 3:
      qnt_1x8_3_aligned(source, count, stride, offsets, destination,
                        destination_stride, multiplicative_offset,
                        rounding_offset, shift);
      break;
    case 4:
      qnt_1x8_4_aligned(source, count, stride, offsets, destination,
                        destination_stride, multiplicative_offset,
                        rounding_offset, shift);
      break;
    case 5:
      qnt_1x8_5_aligned(source, count, stride, offsets, destination,
                        destination_stride, multiplicative_offset,
                        rounding_offset, shift);
      break;
    case 6:
      qnt_1x8_6_aligned(source, count, stride, offsets, destination,
                        destination_stride, multiplicative_offset,
                        rounding_offset, shift);
      break;
    case 7:
      qnt_1x8_7_aligned(source, count, stride, offsets, destination,
                        destination_stride, multiplicative_offset,
                        rounding_offset, shift);
      break;
  }
}

void multi_qnt_2x8_aligned(const std::int32_t* source, std::int32_t count,
                           std::int32_t stride, const std::int32_t* offsets,
                           std::uint8_t* destination,
                           std::int32_t destination_stride,
                           std::int32_t multiplicative_offset,
                           std::int32_t rounding_offset, std::int32_t shift) {
  switch (count % 8) {
    case 0:
      qnt_2x8_aligned(source, count, stride, offsets, destination,
                      destination_stride, multiplicative_offset,
                      rounding_offset, shift);
      break;
    case 1:
      qnt_2x8_1_aligned(source, count, stride, offsets, destination,
                        destination_stride, multiplicative_offset,
                        rounding_offset, shift);
      break;
    case 2:
      qnt_2x8_2_aligned(source, count, stride, offsets, destination,
                        destination_stride, multiplicative_offset,
                        rounding_offset, shift);
      break;
    case 3:
      qnt_2x8_3_aligned(source, count, stride, offsets, destination,
                        destination_stride, multiplicative_offset,
                        rounding_offset, shift);
      break;
    case 4:
      qnt_2x8_4_aligned(source, count, stride, offsets, destination,
                        destination_stride, multiplicative_offset,
                        rounding_offset, shift);
      break;
    case 5:
      qnt_2x8_5_aligned(source, count, stride, offsets, destination,
                        destination_stride, multiplicative_offset,
                        rounding_offset, shift);
      break;
    case 6:
      qnt_2x8_6_aligned(source, count, stride, offsets, destination,
                        destination_stride, multiplicative_offset,
                        rounding_offset, shift);
      break;
    case 7:
      qnt_2x8_7_aligned(source, count, stride, offsets, destination,
                        destination_stride, multiplicative_offset,
                        rounding_offset, shift);
      break;
  }
}

void multi_qnt_3x8_aligned(const std::int32_t* source, std::int32_t count,
                           std::int32_t stride, const std::int32_t* offsets,
                           std::uint8_t* destination,
                           std::int32_t destination_stride,
                           std::int32_t multiplicative_offset,
                           std::int32_t rounding_offset, std::int32_t shift) {
  switch (count % 8) {
    case 0:
      qnt_3x8_aligned(source, count, stride, offsets, destination,
                      destination_stride, multiplicative_offset,
                      rounding_offset, shift);
      break;
    case 1:
      qnt_3x8_1_aligned(source, count, stride, offsets, destination,
                        destination_stride, multiplicative_offset,
                        rounding_offset, shift);
      break;
    case 2:
      qnt_3x8_2_aligned(source, count, stride, offsets, destination,
                        destination_stride, multiplicative_offset,
                        rounding_offset, shift);
      break;
    case 3:
      qnt_3x8_3_aligned(source, count, stride, offsets, destination,
                        destination_stride, multiplicative_offset,
                        rounding_offset, shift);
      break;
    case 4:
      qnt_3x8_4_aligned(source, count, stride, offsets, destination,
                        destination_stride, multiplicative_offset,
                        rounding_offset, shift);
      break;
    case 5:
      qnt_3x8_5_aligned(source, count, stride, offsets, destination,
                        destination_stride, multiplicative_offset,
                        rounding_offset, shift);
      break;
    case 6:
      qnt_3x8_6_aligned(source, count, stride, offsets, destination,
                        destination_stride, multiplicative_offset,
                        rounding_offset, shift);
      break;
    case 7:
      qnt_3x8_7_aligned(source, count, stride, offsets, destination,
                        destination_stride, multiplicative_offset,
                        rounding_offset, shift);
      break;
  }
}

void multi_qnt_1x8(const std::int32_t* source, std::int32_t count,
                   std::int32_t stride, const std::int32_t* offsets,
                   std::uint8_t* destination, std::int32_t destination_stride,
                   std::int32_t multiplicative_offset,
                   std::int32_t rounding_offset, std::int32_t shift) {
  switch (count % 8) {
    case 0:
      qnt_1x8(source, count, stride, offsets, destination, destination_stride,
              multiplicative_offset, rounding_offset, shift);
      break;
    case 1:
      qnt_1x8_1(source, count, stride, offsets, destination, destination_stride,
                multiplicative_offset, rounding_offset, shift);
      break;
    case 2:
      qnt_1x8_2(source, count, stride, offsets, destination, destination_stride,
                multiplicative_offset, rounding_offset, shift);
      break;
    case 3:
      qnt_1x8_3(source, count, stride, offsets, destination, destination_stride,
                multiplicative_offset, rounding_offset, shift);
      break;
    case 4:
      qnt_1x8_4(source, count, stride, offsets, destination, destination_stride,
                multiplicative_offset, rounding_offset, shift);
      break;
    case 5:
      qnt_1x8_5(source, count, stride, offsets, destination, destination_stride,
                multiplicative_offset, rounding_offset, shift);
      break;
    case 6:
      qnt_1x8_6(source, count, stride, offsets, destination, destination_stride,
                multiplicative_offset, rounding_offset, shift);
      break;
    case 7:
      qnt_1x8_7(source, count, stride, offsets, destination, destination_stride,
                multiplicative_offset, rounding_offset, shift);
      break;
  }
}

void multi_qnt_2x8(const std::int32_t* source, std::int32_t count,
                   std::int32_t stride, const std::int32_t* offsets,
                   std::uint8_t* destination, std::int32_t destination_stride,
                   std::int32_t multiplicative_offset,
                   std::int32_t rounding_offset, std::int32_t shift) {
  switch (count % 8) {
    case 0:
      qnt_2x8(source, count, stride, offsets, destination, destination_stride,
              multiplicative_offset, rounding_offset, shift);
      break;
    case 1:
      qnt_2x8_1(source, count, stride, offsets, destination, destination_stride,
                multiplicative_offset, rounding_offset, shift);
      break;
    case 2:
      qnt_2x8_2(source, count, stride, offsets, destination, destination_stride,
                multiplicative_offset, rounding_offset, shift);
      break;
    case 3:
      qnt_2x8_3(source, count, stride, offsets, destination, destination_stride,
                multiplicative_offset, rounding_offset, shift);
      break;
    case 4:
      qnt_2x8_4(source, count, stride, offsets, destination, destination_stride,
                multiplicative_offset, rounding_offset, shift);
      break;
    case 5:
      qnt_2x8_5(source, count, stride, offsets, destination, destination_stride,
                multiplicative_offset, rounding_offset, shift);
      break;
    case 6:
      qnt_2x8_6(source, count, stride, offsets, destination, destination_stride,
                multiplicative_offset, rounding_offset, shift);
      break;
    case 7:
      qnt_2x8_7(source, count, stride, offsets, destination, destination_stride,
                multiplicative_offset, rounding_offset, shift);
      break;
  }
}

void multi_qnt_3x8(const std::int32_t* source, std::int32_t count,
                   std::int32_t stride, const std::int32_t* offsets,
                   std::uint8_t* destination, std::int32_t destination_stride,
                   std::int32_t multiplicative_offset,
                   std::int32_t rounding_offset, std::int32_t shift) {
  switch (count % 8) {
    case 0:
      qnt_3x8(source, count, stride, offsets, destination, destination_stride,
              multiplicative_offset, rounding_offset, shift);
      break;
    case 1:
      qnt_3x8_1(source, count, stride, offsets, destination, destination_stride,
                multiplicative_offset, rounding_offset, shift);
      break;
    case 2:
      qnt_3x8_2(source, count, stride, offsets, destination, destination_stride,
                multiplicative_offset, rounding_offset, shift);
      break;
    case 3:
      qnt_3x8_3(source, count, stride, offsets, destination, destination_stride,
                multiplicative_offset, rounding_offset, shift);
      break;
    case 4:
      qnt_3x8_4(source, count, stride, offsets, destination, destination_stride,
                multiplicative_offset, rounding_offset, shift);
      break;
    case 5:
      qnt_3x8_5(source, count, stride, offsets, destination, destination_stride,
                multiplicative_offset, rounding_offset, shift);
      break;
    case 6:
      qnt_3x8_6(source, count, stride, offsets, destination, destination_stride,
                multiplicative_offset, rounding_offset, shift);
      break;
    case 7:
      qnt_3x8_7(source, count, stride, offsets, destination, destination_stride,
                multiplicative_offset, rounding_offset, shift);
      break;
  }
}

}  // namespace internal
}  // namespace meta
}  // namespace gemmlowp

#else
#warning "Meta gemm for arm32 requires: GEMMLOWP_NEON_32!"
#endif

#endif  // GEMMLOWP_META_SINGLE_THREAD_GEMM_ARM32_H_
