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

#ifndef GEMMLOWP_META_SINGLE_THREAD_GEMM_ARM64_H_
#define GEMMLOWP_META_SINGLE_THREAD_GEMM_ARM64_H_

#ifdef GEMMLOWP_NEON_64

#include <cassert>

namespace gemmlowp {
namespace meta {
namespace internal {

void zip_1x8_aligned(const std::uint8_t* source, std::int32_t count,
                     std::int32_t stride, std::uint8_t* destination,
                     std::int32_t multiplicative_offset,
                     std::int32_t additive_offset) {
  asm volatile(
      "movi v4.8h, #0\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store.
      "ld1 {v0.8b}, [%x[source]], #8\n"
      "uaddw v4.8h, v4.8h, v0.8b\n"
      "st1 {v0.8b}, [%x[destination]], #8\n"

      "bne 1b\n"

      // Aggregator Reduction.
      "mov v1.s[0], %w[multiplicative_offset]\n"
      "dup v2.2s, %w[additive_offset]\n"
      "uaddlp v4.4s, v4.8h\n"
      "addp v4.4s, v4.4s, v4.4s\n"
      "addp v3.4s, v4.4s, v4.4s\n"
      "mul v3.2s, v3.2s, v1.s[0]\n"
      "add v3.2s, v3.2s, v2.2s\n"
      "st1 {v3.2s}, [%x[destination]]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "v0", "v1", "v2", "v3", "v4", "cc", "memory");
}

void zip_1x8_1_aligned(const std::uint8_t* source, std::int32_t count,
                       std::int32_t stride, std::uint8_t* destination,
                       std::int32_t multiplicative_offset,
                       std::int32_t additive_offset) {
  asm volatile(
      "sub %x[count], %x[count], #1\n"
      "movi v4.8h, #0\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store.
      "ld1 {v0.8b}, [%x[source]], #8\n"
      "uaddw v4.8h, v4.8h, v0.8b\n"
      "st1 {v0.8b}, [%x[destination]], #8\n"

      "bne 1b\n"

      // Leftover Load Aggregate Store.
      "movi v0.8b, #0\n"
      "ld1 {v0.b}[0], [%x[source]], #1\n"
      "uaddw v4.8h, v4.8h, v0.8b\n"
      "st1 {v0.8b}, [%x[destination]], #8\n"

      // Aggregator Reduction.
      "mov v1.s[0], %w[multiplicative_offset]\n"
      "dup v2.2s, %w[additive_offset]\n"
      "uaddlp v4.4s, v4.8h\n"
      "addp v4.4s, v4.4s, v4.4s\n"
      "addp v3.4s, v4.4s, v4.4s\n"
      "mul v3.2s, v3.2s, v1.s[0]\n"
      "add v3.2s, v3.2s, v2.2s\n"
      "st1 {v3.2s}, [%x[destination]]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "v0", "v1", "v2", "v3", "v4", "cc", "memory");
}

void zip_1x8_2_aligned(const std::uint8_t* source, std::int32_t count,
                       std::int32_t stride, std::uint8_t* destination,
                       std::int32_t multiplicative_offset,
                       std::int32_t additive_offset) {
  asm volatile(
      "sub %x[count], %x[count], #2\n"
      "movi v4.8h, #0\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store.
      "ld1 {v0.8b}, [%x[source]], #8\n"
      "uaddw v4.8h, v4.8h, v0.8b\n"
      "st1 {v0.8b}, [%x[destination]], #8\n"

      "bne 1b\n"

      // Leftover Load Aggregate Store.
      "movi v0.8b, #0\n"
      "ld1 {v0.h}[0], [%x[source]], #2\n"
      "uaddw v4.8h, v4.8h, v0.8b\n"
      "st1 {v0.8b}, [%x[destination]], #8\n"

      // Aggregator Reduction.
      "mov v1.s[0], %w[multiplicative_offset]\n"
      "dup v2.2s, %w[additive_offset]\n"
      "uaddlp v4.4s, v4.8h\n"
      "addp v4.4s, v4.4s, v4.4s\n"
      "addp v3.4s, v4.4s, v4.4s\n"
      "mul v3.2s, v3.2s, v1.s[0]\n"
      "add v3.2s, v3.2s, v2.2s\n"
      "st1 {v3.2s}, [%x[destination]]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "v0", "v1", "v2", "v3", "v4", "cc", "memory");
}

void zip_1x8_3_aligned(const std::uint8_t* source, std::int32_t count,
                       std::int32_t stride, std::uint8_t* destination,
                       std::int32_t multiplicative_offset,
                       std::int32_t additive_offset) {
  asm volatile(
      "sub %x[count], %x[count], #3\n"
      "movi v4.8h, #0\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store.
      "ld1 {v0.8b}, [%x[source]], #8\n"
      "uaddw v4.8h, v4.8h, v0.8b\n"
      "st1 {v0.8b}, [%x[destination]], #8\n"

      "bne 1b\n"

      // Leftover Load Aggregate Store.
      "movi v0.8b, #0\n"
      "ld1 {v0.h}[0], [%x[source]], #2\n"
      "ld1 {v0.b}[2], [%x[source]], #1\n"
      "uaddw v4.8h, v4.8h, v0.8b\n"
      "st1 {v0.8b}, [%x[destination]], #8\n"

      // Aggregator Reduction.
      "mov v1.s[0], %w[multiplicative_offset]\n"
      "dup v2.2s, %w[additive_offset]\n"
      "uaddlp v4.4s, v4.8h\n"
      "addp v4.4s, v4.4s, v4.4s\n"
      "addp v3.4s, v4.4s, v4.4s\n"
      "mul v3.2s, v3.2s, v1.s[0]\n"
      "add v3.2s, v3.2s, v2.2s\n"
      "st1 {v3.2s}, [%x[destination]]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "v0", "v1", "v2", "v3", "v4", "cc", "memory");
}

void zip_1x8_4_aligned(const std::uint8_t* source, std::int32_t count,
                       std::int32_t stride, std::uint8_t* destination,
                       std::int32_t multiplicative_offset,
                       std::int32_t additive_offset) {
  asm volatile(
      "sub %x[count], %x[count], #4\n"
      "movi v4.8h, #0\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store.
      "ld1 {v0.8b}, [%x[source]], #8\n"
      "uaddw v4.8h, v4.8h, v0.8b\n"
      "st1 {v0.8b}, [%x[destination]], #8\n"

      "bne 1b\n"

      // Leftover Load Aggregate Store.
      "movi v0.8b, #0\n"
      "ld1 {v0.s}[0], [%x[source]], #4\n"
      "uaddw v4.8h, v4.8h, v0.8b\n"
      "st1 {v0.8b}, [%x[destination]], #8\n"

      // Aggregator Reduction.
      "mov v1.s[0], %w[multiplicative_offset]\n"
      "dup v2.2s, %w[additive_offset]\n"
      "uaddlp v4.4s, v4.8h\n"
      "addp v4.4s, v4.4s, v4.4s\n"
      "addp v3.4s, v4.4s, v4.4s\n"
      "mul v3.2s, v3.2s, v1.s[0]\n"
      "add v3.2s, v3.2s, v2.2s\n"
      "st1 {v3.2s}, [%x[destination]]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "v0", "v1", "v2", "v3", "v4", "cc", "memory");
}

void zip_1x8_5_aligned(const std::uint8_t* source, std::int32_t count,
                       std::int32_t stride, std::uint8_t* destination,
                       std::int32_t multiplicative_offset,
                       std::int32_t additive_offset) {
  asm volatile(
      "sub %x[count], %x[count], #5\n"
      "movi v4.8h, #0\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store.
      "ld1 {v0.8b}, [%x[source]], #8\n"
      "uaddw v4.8h, v4.8h, v0.8b\n"
      "st1 {v0.8b}, [%x[destination]], #8\n"

      "bne 1b\n"

      // Leftover Load Aggregate Store.
      "movi v0.8b, #0\n"
      "ld1 {v0.s}[0], [%x[source]], #4\n"
      "ld1 {v0.b}[4], [%x[source]], #1\n"
      "uaddw v4.8h, v4.8h, v0.8b\n"
      "st1 {v0.8b}, [%x[destination]], #8\n"

      // Aggregator Reduction.
      "mov v1.s[0], %w[multiplicative_offset]\n"
      "dup v2.2s, %w[additive_offset]\n"
      "uaddlp v4.4s, v4.8h\n"
      "addp v4.4s, v4.4s, v4.4s\n"
      "addp v3.4s, v4.4s, v4.4s\n"
      "mul v3.2s, v3.2s, v1.s[0]\n"
      "add v3.2s, v3.2s, v2.2s\n"
      "st1 {v3.2s}, [%x[destination]]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "v0", "v1", "v2", "v3", "v4", "cc", "memory");
}

void zip_1x8_6_aligned(const std::uint8_t* source, std::int32_t count,
                       std::int32_t stride, std::uint8_t* destination,
                       std::int32_t multiplicative_offset,
                       std::int32_t additive_offset) {
  asm volatile(
      "sub %x[count], %x[count], #6\n"
      "movi v4.8h, #0\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store.
      "ld1 {v0.8b}, [%x[source]], #8\n"
      "uaddw v4.8h, v4.8h, v0.8b\n"
      "st1 {v0.8b}, [%x[destination]], #8\n"

      "bne 1b\n"

      // Leftover Load Aggregate Store.
      "movi v0.8b, #0\n"
      "ld1 {v0.s}[0], [%x[source]], #4\n"
      "ld1 {v0.h}[2], [%x[source]], #2\n"
      "uaddw v4.8h, v4.8h, v0.8b\n"
      "st1 {v0.8b}, [%x[destination]], #8\n"

      // Aggregator Reduction.
      "mov v1.s[0], %w[multiplicative_offset]\n"
      "dup v2.2s, %w[additive_offset]\n"
      "uaddlp v4.4s, v4.8h\n"
      "addp v4.4s, v4.4s, v4.4s\n"
      "addp v3.4s, v4.4s, v4.4s\n"
      "mul v3.2s, v3.2s, v1.s[0]\n"
      "add v3.2s, v3.2s, v2.2s\n"
      "st1 {v3.2s}, [%x[destination]]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "v0", "v1", "v2", "v3", "v4", "cc", "memory");
}

void zip_1x8_7_aligned(const std::uint8_t* source, std::int32_t count,
                       std::int32_t stride, std::uint8_t* destination,
                       std::int32_t multiplicative_offset,
                       std::int32_t additive_offset) {
  asm volatile(
      "sub %x[count], %x[count], #7\n"
      "movi v4.8h, #0\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store.
      "ld1 {v0.8b}, [%x[source]], #8\n"
      "uaddw v4.8h, v4.8h, v0.8b\n"
      "st1 {v0.8b}, [%x[destination]], #8\n"

      "bne 1b\n"

      // Leftover Load Aggregate Store.
      "movi v0.8b, #0\n"
      "ld1 {v0.s}[0], [%x[source]], #4\n"
      "ld1 {v0.h}[2], [%x[source]], #2\n"
      "ld1 {v0.b}[6], [%x[source]], #1\n"
      "uaddw v4.8h, v4.8h, v0.8b\n"
      "st1 {v0.8b}, [%x[destination]], #8\n"

      // Aggregator Reduction.
      "mov v1.s[0], %w[multiplicative_offset]\n"
      "dup v2.2s, %w[additive_offset]\n"
      "uaddlp v4.4s, v4.8h\n"
      "addp v4.4s, v4.4s, v4.4s\n"
      "addp v3.4s, v4.4s, v4.4s\n"
      "mul v3.2s, v3.2s, v1.s[0]\n"
      "add v3.2s, v3.2s, v2.2s\n"
      "st1 {v3.2s}, [%x[destination]]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "v0", "v1", "v2", "v3", "v4", "cc", "memory");
}

void zip_2x8_aligned(const std::uint8_t* source, std::int32_t count,
                     std::int32_t stride, std::uint8_t* destination,
                     std::int32_t multiplicative_offset,
                     std::int32_t additive_offset) {
  asm volatile(
      "add x0, %x[source], %x[stride]\n"
      "movi v4.8h, #0\n"
      "movi v5.8h, #0\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store.
      "ld1 {v0.8b}, [%x[source]], #8\n"
      "ld1 {v1.8b}, [x0], #8\n"
      "uaddw v4.8h, v4.8h, v0.8b\n"
      "uaddw v5.8h, v5.8h, v1.8b\n"
      "st1 {v0.8b, v1.8b}, [%x[destination]], #16\n"

      "bne 1b\n"

      // Aggregator Reduction.
      "mov v2.s[0], %w[multiplicative_offset]\n"
      "dup v3.2s, %w[additive_offset]\n"
      "uaddlp v4.4s, v4.8h\n"
      "uaddlp v5.4s, v5.8h\n"
      "addp v4.4s, v4.4s, v5.4s\n"
      "addp v6.4s, v4.4s, v4.4s\n"
      "mul v6.2s, v6.2s, v2.s[0]\n"
      "add v6.2s, v6.2s, v3.2s\n"
      "st1 {v6.2s}, [%x[destination]]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "x0", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "cc", "memory");
}

void zip_2x8_1_aligned(const std::uint8_t* source, std::int32_t count,
                       std::int32_t stride, std::uint8_t* destination,
                       std::int32_t multiplicative_offset,
                       std::int32_t additive_offset) {
  asm volatile(
      "add x0, %x[source], %x[stride]\n"
      "sub %x[count], %x[count], #1\n"
      "movi v4.8h, #0\n"
      "movi v5.8h, #0\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store.
      "ld1 {v0.8b}, [%x[source]], #8\n"
      "ld1 {v1.8b}, [x0], #8\n"
      "uaddw v4.8h, v4.8h, v0.8b\n"
      "uaddw v5.8h, v5.8h, v1.8b\n"
      "st1 {v0.8b, v1.8b}, [%x[destination]], #16\n"

      "bne 1b\n"

      // Leftover Load Aggregate Store.
      "movi v0.8b, #0\n"
      "movi v1.8b, #0\n"
      "ld1 {v0.b}[0], [%x[source]], #1\n"
      "ld1 {v1.b}[0], [x0], #1\n"
      "uaddw v4.8h, v4.8h, v0.8b\n"
      "uaddw v5.8h, v5.8h, v1.8b\n"
      "st1 {v0.8b, v1.8b}, [%x[destination]], #16\n"

      // Aggregator Reduction.
      "mov v2.s[0], %w[multiplicative_offset]\n"
      "dup v3.2s, %w[additive_offset]\n"
      "uaddlp v4.4s, v4.8h\n"
      "uaddlp v5.4s, v5.8h\n"
      "addp v4.4s, v4.4s, v5.4s\n"
      "addp v6.4s, v4.4s, v4.4s\n"
      "mul v6.2s, v6.2s, v2.s[0]\n"
      "add v6.2s, v6.2s, v3.2s\n"
      "st1 {v6.2s}, [%x[destination]]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "x0", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "cc", "memory");
}

void zip_2x8_2_aligned(const std::uint8_t* source, std::int32_t count,
                       std::int32_t stride, std::uint8_t* destination,
                       std::int32_t multiplicative_offset,
                       std::int32_t additive_offset) {
  asm volatile(
      "add x0, %x[source], %x[stride]\n"
      "sub %x[count], %x[count], #2\n"
      "movi v4.8h, #0\n"
      "movi v5.8h, #0\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store.
      "ld1 {v0.8b}, [%x[source]], #8\n"
      "ld1 {v1.8b}, [x0], #8\n"
      "uaddw v4.8h, v4.8h, v0.8b\n"
      "uaddw v5.8h, v5.8h, v1.8b\n"
      "st1 {v0.8b, v1.8b}, [%x[destination]], #16\n"

      "bne 1b\n"

      // Leftover Load Aggregate Store.
      "movi v0.8b, #0\n"
      "movi v1.8b, #0\n"
      "ld1 {v0.h}[0], [%x[source]], #2\n"
      "ld1 {v1.h}[0], [x0], #2\n"
      "uaddw v4.8h, v4.8h, v0.8b\n"
      "uaddw v5.8h, v5.8h, v1.8b\n"
      "st1 {v0.8b, v1.8b}, [%x[destination]], #16\n"

      // Aggregator Reduction.
      "mov v2.s[0], %w[multiplicative_offset]\n"
      "dup v3.2s, %w[additive_offset]\n"
      "uaddlp v4.4s, v4.8h\n"
      "uaddlp v5.4s, v5.8h\n"
      "addp v4.4s, v4.4s, v5.4s\n"
      "addp v6.4s, v4.4s, v4.4s\n"
      "mul v6.2s, v6.2s, v2.s[0]\n"
      "add v6.2s, v6.2s, v3.2s\n"
      "st1 {v6.2s}, [%x[destination]]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "x0", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "cc", "memory");
}

void zip_2x8_3_aligned(const std::uint8_t* source, std::int32_t count,
                       std::int32_t stride, std::uint8_t* destination,
                       std::int32_t multiplicative_offset,
                       std::int32_t additive_offset) {
  asm volatile(
      "add x0, %x[source], %x[stride]\n"
      "sub %x[count], %x[count], #3\n"
      "movi v4.8h, #0\n"
      "movi v5.8h, #0\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store.
      "ld1 {v0.8b}, [%x[source]], #8\n"
      "ld1 {v1.8b}, [x0], #8\n"
      "uaddw v4.8h, v4.8h, v0.8b\n"
      "uaddw v5.8h, v5.8h, v1.8b\n"
      "st1 {v0.8b, v1.8b}, [%x[destination]], #16\n"

      "bne 1b\n"

      // Leftover Load Aggregate Store.
      "movi v0.8b, #0\n"
      "movi v1.8b, #0\n"
      "ld1 {v0.h}[0], [%x[source]], #2\n"
      "ld1 {v0.b}[2], [%x[source]], #1\n"
      "ld1 {v1.h}[0], [x0], #2\n"
      "ld1 {v1.b}[2], [x0], #1\n"
      "uaddw v4.8h, v4.8h, v0.8b\n"
      "uaddw v5.8h, v5.8h, v1.8b\n"
      "st1 {v0.8b, v1.8b}, [%x[destination]], #16\n"

      // Aggregator Reduction.
      "mov v2.s[0], %w[multiplicative_offset]\n"
      "dup v3.2s, %w[additive_offset]\n"
      "uaddlp v4.4s, v4.8h\n"
      "uaddlp v5.4s, v5.8h\n"
      "addp v4.4s, v4.4s, v5.4s\n"
      "addp v6.4s, v4.4s, v4.4s\n"
      "mul v6.2s, v6.2s, v2.s[0]\n"
      "add v6.2s, v6.2s, v3.2s\n"
      "st1 {v6.2s}, [%x[destination]]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "x0", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "cc", "memory");
}

void zip_2x8_4_aligned(const std::uint8_t* source, std::int32_t count,
                       std::int32_t stride, std::uint8_t* destination,
                       std::int32_t multiplicative_offset,
                       std::int32_t additive_offset) {
  asm volatile(
      "add x0, %x[source], %x[stride]\n"
      "sub %x[count], %x[count], #4\n"
      "movi v4.8h, #0\n"
      "movi v5.8h, #0\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store.
      "ld1 {v0.8b}, [%x[source]], #8\n"
      "ld1 {v1.8b}, [x0], #8\n"
      "uaddw v4.8h, v4.8h, v0.8b\n"
      "uaddw v5.8h, v5.8h, v1.8b\n"
      "st1 {v0.8b, v1.8b}, [%x[destination]], #16\n"

      "bne 1b\n"

      // Leftover Load Aggregate Store.
      "movi v0.8b, #0\n"
      "movi v1.8b, #0\n"
      "ld1 {v0.s}[0], [%x[source]], #4\n"
      "ld1 {v1.s}[0], [x0], #4\n"
      "uaddw v4.8h, v4.8h, v0.8b\n"
      "uaddw v5.8h, v5.8h, v1.8b\n"
      "st1 {v0.8b, v1.8b}, [%x[destination]], #16\n"

      // Aggregator Reduction.
      "mov v2.s[0], %w[multiplicative_offset]\n"
      "dup v3.2s, %w[additive_offset]\n"
      "uaddlp v4.4s, v4.8h\n"
      "uaddlp v5.4s, v5.8h\n"
      "addp v4.4s, v4.4s, v5.4s\n"
      "addp v6.4s, v4.4s, v4.4s\n"
      "mul v6.2s, v6.2s, v2.s[0]\n"
      "add v6.2s, v6.2s, v3.2s\n"
      "st1 {v6.2s}, [%x[destination]]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "x0", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "cc", "memory");
}

void zip_2x8_5_aligned(const std::uint8_t* source, std::int32_t count,
                       std::int32_t stride, std::uint8_t* destination,
                       std::int32_t multiplicative_offset,
                       std::int32_t additive_offset) {
  asm volatile(
      "add x0, %x[source], %x[stride]\n"
      "sub %x[count], %x[count], #5\n"
      "movi v4.8h, #0\n"
      "movi v5.8h, #0\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store.
      "ld1 {v0.8b}, [%x[source]], #8\n"
      "ld1 {v1.8b}, [x0], #8\n"
      "uaddw v4.8h, v4.8h, v0.8b\n"
      "uaddw v5.8h, v5.8h, v1.8b\n"
      "st1 {v0.8b, v1.8b}, [%x[destination]], #16\n"

      "bne 1b\n"

      // Leftover Load Aggregate Store.
      "movi v0.8b, #0\n"
      "movi v1.8b, #0\n"
      "ld1 {v0.s}[0], [%x[source]], #4\n"
      "ld1 {v0.b}[4], [%x[source]], #1\n"
      "ld1 {v1.s}[0], [x0], #4\n"
      "ld1 {v1.b}[4], [x0], #1\n"
      "uaddw v4.8h, v4.8h, v0.8b\n"
      "uaddw v5.8h, v5.8h, v1.8b\n"
      "st1 {v0.8b, v1.8b}, [%x[destination]], #16\n"

      // Aggregator Reduction.
      "mov v2.s[0], %w[multiplicative_offset]\n"
      "dup v3.2s, %w[additive_offset]\n"
      "uaddlp v4.4s, v4.8h\n"
      "uaddlp v5.4s, v5.8h\n"
      "addp v4.4s, v4.4s, v5.4s\n"
      "addp v6.4s, v4.4s, v4.4s\n"
      "mul v6.2s, v6.2s, v2.s[0]\n"
      "add v6.2s, v6.2s, v3.2s\n"
      "st1 {v6.2s}, [%x[destination]]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "x0", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "cc", "memory");
}

void zip_2x8_6_aligned(const std::uint8_t* source, std::int32_t count,
                       std::int32_t stride, std::uint8_t* destination,
                       std::int32_t multiplicative_offset,
                       std::int32_t additive_offset) {
  asm volatile(
      "add x0, %x[source], %x[stride]\n"
      "sub %x[count], %x[count], #6\n"
      "movi v4.8h, #0\n"
      "movi v5.8h, #0\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store.
      "ld1 {v0.8b}, [%x[source]], #8\n"
      "ld1 {v1.8b}, [x0], #8\n"
      "uaddw v4.8h, v4.8h, v0.8b\n"
      "uaddw v5.8h, v5.8h, v1.8b\n"
      "st1 {v0.8b, v1.8b}, [%x[destination]], #16\n"

      "bne 1b\n"

      // Leftover Load Aggregate Store.
      "movi v0.8b, #0\n"
      "movi v1.8b, #0\n"
      "ld1 {v0.s}[0], [%x[source]], #4\n"
      "ld1 {v0.h}[2], [%x[source]], #2\n"
      "ld1 {v1.s}[0], [x0], #4\n"
      "ld1 {v1.h}[2], [x0], #2\n"
      "uaddw v4.8h, v4.8h, v0.8b\n"
      "uaddw v5.8h, v5.8h, v1.8b\n"
      "st1 {v0.8b, v1.8b}, [%x[destination]], #16\n"

      // Aggregator Reduction.
      "mov v2.s[0], %w[multiplicative_offset]\n"
      "dup v3.2s, %w[additive_offset]\n"
      "uaddlp v4.4s, v4.8h\n"
      "uaddlp v5.4s, v5.8h\n"
      "addp v4.4s, v4.4s, v5.4s\n"
      "addp v6.4s, v4.4s, v4.4s\n"
      "mul v6.2s, v6.2s, v2.s[0]\n"
      "add v6.2s, v6.2s, v3.2s\n"
      "st1 {v6.2s}, [%x[destination]]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "x0", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "cc", "memory");
}

void zip_2x8_7_aligned(const std::uint8_t* source, std::int32_t count,
                       std::int32_t stride, std::uint8_t* destination,
                       std::int32_t multiplicative_offset,
                       std::int32_t additive_offset) {
  asm volatile(
      "add x0, %x[source], %x[stride]\n"
      "sub %x[count], %x[count], #7\n"
      "movi v4.8h, #0\n"
      "movi v5.8h, #0\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store.
      "ld1 {v0.8b}, [%x[source]], #8\n"
      "ld1 {v1.8b}, [x0], #8\n"
      "uaddw v4.8h, v4.8h, v0.8b\n"
      "uaddw v5.8h, v5.8h, v1.8b\n"
      "st1 {v0.8b, v1.8b}, [%x[destination]], #16\n"

      "bne 1b\n"

      // Leftover Load Aggregate Store.
      "movi v0.8b, #0\n"
      "movi v1.8b, #0\n"
      "ld1 {v0.s}[0], [%x[source]], #4\n"
      "ld1 {v0.h}[2], [%x[source]], #2\n"
      "ld1 {v0.b}[6], [%x[source]], #1\n"
      "ld1 {v1.s}[0], [x0], #4\n"
      "ld1 {v1.h}[2], [x0], #2\n"
      "ld1 {v1.b}[6], [x0], #1\n"
      "uaddw v4.8h, v4.8h, v0.8b\n"
      "uaddw v5.8h, v5.8h, v1.8b\n"
      "st1 {v0.8b, v1.8b}, [%x[destination]], #16\n"

      // Aggregator Reduction.
      "mov v2.s[0], %w[multiplicative_offset]\n"
      "dup v3.2s, %w[additive_offset]\n"
      "uaddlp v4.4s, v4.8h\n"
      "uaddlp v5.4s, v5.8h\n"
      "addp v4.4s, v4.4s, v5.4s\n"
      "addp v6.4s, v4.4s, v4.4s\n"
      "mul v6.2s, v6.2s, v2.s[0]\n"
      "add v6.2s, v6.2s, v3.2s\n"
      "st1 {v6.2s}, [%x[destination]]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "x0", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "cc", "memory");
}

void zip_3x8_aligned(const std::uint8_t* source, std::int32_t count,
                     std::int32_t stride, std::uint8_t* destination,
                     std::int32_t multiplicative_offset,
                     std::int32_t additive_offset) {
  asm volatile(
      "add x0, %x[source], %x[stride]\n"
      "add x1, x0, %x[stride]\n"
      "movi v4.8h, #0\n"
      "movi v5.8h, #0\n"
      "movi v6.8h, #0\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store.
      "ld1 {v0.8b}, [%x[source]], #8\n"
      "ld1 {v1.8b}, [x0], #8\n"
      "ld1 {v2.8b}, [x1], #8\n"
      "uaddw v4.8h, v4.8h, v0.8b\n"
      "uaddw v5.8h, v5.8h, v1.8b\n"
      "uaddw v6.8h, v6.8h, v2.8b\n"
      "st1 {v0.8b, v1.8b, v2.8b}, [%x[destination]], #24\n"

      "bne 1b\n"

      // Aggregator Reduction.
      "mov v3.s[0], %w[multiplicative_offset]\n"
      "dup v7.4s, %w[additive_offset]\n"
      "uaddlp v4.4s, v4.8h\n"
      "uaddlp v5.4s, v5.8h\n"
      "uaddlp v6.4s, v6.8h\n"
      "addp v4.4s, v4.4s, v5.4s\n"
      "addp v6.4s, v6.4s, v6.4s\n"
      "addp v8.4s, v4.4s, v6.4s\n"
      "mul v8.4s, v8.4s, v3.s[0]\n"
      "add v8.4s, v8.4s, v7.4s\n"
      "st1 {v8.4s}, [%x[destination]]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "x0", "x1", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "cc",
        "memory");
}

void zip_3x8_1_aligned(const std::uint8_t* source, std::int32_t count,
                       std::int32_t stride, std::uint8_t* destination,
                       std::int32_t multiplicative_offset,
                       std::int32_t additive_offset) {
  asm volatile(
      "add x0, %x[source], %x[stride]\n"
      "add x1, x0, %x[stride]\n"
      "sub %x[count], %x[count], #1\n"
      "movi v4.8h, #0\n"
      "movi v5.8h, #0\n"
      "movi v6.8h, #0\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store.
      "ld1 {v0.8b}, [%x[source]], #8\n"
      "ld1 {v1.8b}, [x0], #8\n"
      "ld1 {v2.8b}, [x1], #8\n"
      "uaddw v4.8h, v4.8h, v0.8b\n"
      "uaddw v5.8h, v5.8h, v1.8b\n"
      "uaddw v6.8h, v6.8h, v2.8b\n"
      "st1 {v0.8b, v1.8b, v2.8b}, [%x[destination]], #24\n"

      "bne 1b\n"

      // Leftover Load Aggregate Store.
      "movi v0.8b, #0\n"
      "movi v1.8b, #0\n"
      "movi v2.8b, #0\n"
      "ld1 {v0.b}[0], [%x[source]], #1\n"
      "ld1 {v1.b}[0], [x0], #1\n"
      "ld1 {v2.b}[0], [x1], #1\n"
      "uaddw v4.8h, v4.8h, v0.8b\n"
      "uaddw v5.8h, v5.8h, v1.8b\n"
      "uaddw v6.8h, v6.8h, v2.8b\n"
      "st1 {v0.8b, v1.8b, v2.8b}, [%x[destination]], #24\n"

      // Aggregator Reduction.
      "mov v3.s[0], %w[multiplicative_offset]\n"
      "dup v7.4s, %w[additive_offset]\n"
      "uaddlp v4.4s, v4.8h\n"
      "uaddlp v5.4s, v5.8h\n"
      "uaddlp v6.4s, v6.8h\n"
      "addp v4.4s, v4.4s, v5.4s\n"
      "addp v6.4s, v6.4s, v6.4s\n"
      "addp v8.4s, v4.4s, v6.4s\n"
      "mul v8.4s, v8.4s, v3.s[0]\n"
      "add v8.4s, v8.4s, v7.4s\n"
      "st1 {v8.4s}, [%x[destination]]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "x0", "x1", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "cc",
        "memory");
}

void zip_3x8_2_aligned(const std::uint8_t* source, std::int32_t count,
                       std::int32_t stride, std::uint8_t* destination,
                       std::int32_t multiplicative_offset,
                       std::int32_t additive_offset) {
  asm volatile(
      "add x0, %x[source], %x[stride]\n"
      "add x1, x0, %x[stride]\n"
      "sub %x[count], %x[count], #2\n"
      "movi v4.8h, #0\n"
      "movi v5.8h, #0\n"
      "movi v6.8h, #0\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store.
      "ld1 {v0.8b}, [%x[source]], #8\n"
      "ld1 {v1.8b}, [x0], #8\n"
      "ld1 {v2.8b}, [x1], #8\n"
      "uaddw v4.8h, v4.8h, v0.8b\n"
      "uaddw v5.8h, v5.8h, v1.8b\n"
      "uaddw v6.8h, v6.8h, v2.8b\n"
      "st1 {v0.8b, v1.8b, v2.8b}, [%x[destination]], #24\n"

      "bne 1b\n"

      // Leftover Load Aggregate Store.
      "movi v0.8b, #0\n"
      "movi v1.8b, #0\n"
      "movi v2.8b, #0\n"
      "ld1 {v0.h}[0], [%x[source]], #2\n"
      "ld1 {v1.h}[0], [x0], #2\n"
      "ld1 {v2.h}[0], [x1], #2\n"
      "uaddw v4.8h, v4.8h, v0.8b\n"
      "uaddw v5.8h, v5.8h, v1.8b\n"
      "uaddw v6.8h, v6.8h, v2.8b\n"
      "st1 {v0.8b, v1.8b, v2.8b}, [%x[destination]], #24\n"

      // Aggregator Reduction.
      "mov v3.s[0], %w[multiplicative_offset]\n"
      "dup v7.4s, %w[additive_offset]\n"
      "uaddlp v4.4s, v4.8h\n"
      "uaddlp v5.4s, v5.8h\n"
      "uaddlp v6.4s, v6.8h\n"
      "addp v4.4s, v4.4s, v5.4s\n"
      "addp v6.4s, v6.4s, v6.4s\n"
      "addp v8.4s, v4.4s, v6.4s\n"
      "mul v8.4s, v8.4s, v3.s[0]\n"
      "add v8.4s, v8.4s, v7.4s\n"
      "st1 {v8.4s}, [%x[destination]]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "x0", "x1", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "cc",
        "memory");
}

void zip_3x8_3_aligned(const std::uint8_t* source, std::int32_t count,
                       std::int32_t stride, std::uint8_t* destination,
                       std::int32_t multiplicative_offset,
                       std::int32_t additive_offset) {
  asm volatile(
      "add x0, %x[source], %x[stride]\n"
      "add x1, x0, %x[stride]\n"
      "sub %x[count], %x[count], #3\n"
      "movi v4.8h, #0\n"
      "movi v5.8h, #0\n"
      "movi v6.8h, #0\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store.
      "ld1 {v0.8b}, [%x[source]], #8\n"
      "ld1 {v1.8b}, [x0], #8\n"
      "ld1 {v2.8b}, [x1], #8\n"
      "uaddw v4.8h, v4.8h, v0.8b\n"
      "uaddw v5.8h, v5.8h, v1.8b\n"
      "uaddw v6.8h, v6.8h, v2.8b\n"
      "st1 {v0.8b, v1.8b, v2.8b}, [%x[destination]], #24\n"

      "bne 1b\n"

      // Leftover Load Aggregate Store.
      "movi v0.8b, #0\n"
      "movi v1.8b, #0\n"
      "movi v2.8b, #0\n"
      "ld1 {v0.h}[0], [%x[source]], #2\n"
      "ld1 {v0.b}[2], [%x[source]], #1\n"
      "ld1 {v1.h}[0], [x0], #2\n"
      "ld1 {v1.b}[2], [x0], #1\n"
      "ld1 {v2.h}[0], [x1], #2\n"
      "ld1 {v2.b}[2], [x1], #1\n"
      "uaddw v4.8h, v4.8h, v0.8b\n"
      "uaddw v5.8h, v5.8h, v1.8b\n"
      "uaddw v6.8h, v6.8h, v2.8b\n"
      "st1 {v0.8b, v1.8b, v2.8b}, [%x[destination]], #24\n"

      // Aggregator Reduction.
      "mov v3.s[0], %w[multiplicative_offset]\n"
      "dup v7.4s, %w[additive_offset]\n"
      "uaddlp v4.4s, v4.8h\n"
      "uaddlp v5.4s, v5.8h\n"
      "uaddlp v6.4s, v6.8h\n"
      "addp v4.4s, v4.4s, v5.4s\n"
      "addp v6.4s, v6.4s, v6.4s\n"
      "addp v8.4s, v4.4s, v6.4s\n"
      "mul v8.4s, v8.4s, v3.s[0]\n"
      "add v8.4s, v8.4s, v7.4s\n"
      "st1 {v8.4s}, [%x[destination]]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "x0", "x1", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "cc",
        "memory");
}

void zip_3x8_4_aligned(const std::uint8_t* source, std::int32_t count,
                       std::int32_t stride, std::uint8_t* destination,
                       std::int32_t multiplicative_offset,
                       std::int32_t additive_offset) {
  asm volatile(
      "add x0, %x[source], %x[stride]\n"
      "add x1, x0, %x[stride]\n"
      "sub %x[count], %x[count], #4\n"
      "movi v4.8h, #0\n"
      "movi v5.8h, #0\n"
      "movi v6.8h, #0\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store.
      "ld1 {v0.8b}, [%x[source]], #8\n"
      "ld1 {v1.8b}, [x0], #8\n"
      "ld1 {v2.8b}, [x1], #8\n"
      "uaddw v4.8h, v4.8h, v0.8b\n"
      "uaddw v5.8h, v5.8h, v1.8b\n"
      "uaddw v6.8h, v6.8h, v2.8b\n"
      "st1 {v0.8b, v1.8b, v2.8b}, [%x[destination]], #24\n"

      "bne 1b\n"

      // Leftover Load Aggregate Store.
      "movi v0.8b, #0\n"
      "movi v1.8b, #0\n"
      "movi v2.8b, #0\n"
      "ld1 {v0.s}[0], [%x[source]], #4\n"
      "ld1 {v1.s}[0], [x0], #4\n"
      "ld1 {v2.s}[0], [x1], #4\n"
      "uaddw v4.8h, v4.8h, v0.8b\n"
      "uaddw v5.8h, v5.8h, v1.8b\n"
      "uaddw v6.8h, v6.8h, v2.8b\n"
      "st1 {v0.8b, v1.8b, v2.8b}, [%x[destination]], #24\n"

      // Aggregator Reduction.
      "mov v3.s[0], %w[multiplicative_offset]\n"
      "dup v7.4s, %w[additive_offset]\n"
      "uaddlp v4.4s, v4.8h\n"
      "uaddlp v5.4s, v5.8h\n"
      "uaddlp v6.4s, v6.8h\n"
      "addp v4.4s, v4.4s, v5.4s\n"
      "addp v6.4s, v6.4s, v6.4s\n"
      "addp v8.4s, v4.4s, v6.4s\n"
      "mul v8.4s, v8.4s, v3.s[0]\n"
      "add v8.4s, v8.4s, v7.4s\n"
      "st1 {v8.4s}, [%x[destination]]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "x0", "x1", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "cc",
        "memory");
}

void zip_3x8_5_aligned(const std::uint8_t* source, std::int32_t count,
                       std::int32_t stride, std::uint8_t* destination,
                       std::int32_t multiplicative_offset,
                       std::int32_t additive_offset) {
  asm volatile(
      "add x0, %x[source], %x[stride]\n"
      "add x1, x0, %x[stride]\n"
      "sub %x[count], %x[count], #5\n"
      "movi v4.8h, #0\n"
      "movi v5.8h, #0\n"
      "movi v6.8h, #0\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store.
      "ld1 {v0.8b}, [%x[source]], #8\n"
      "ld1 {v1.8b}, [x0], #8\n"
      "ld1 {v2.8b}, [x1], #8\n"
      "uaddw v4.8h, v4.8h, v0.8b\n"
      "uaddw v5.8h, v5.8h, v1.8b\n"
      "uaddw v6.8h, v6.8h, v2.8b\n"
      "st1 {v0.8b, v1.8b, v2.8b}, [%x[destination]], #24\n"

      "bne 1b\n"

      // Leftover Load Aggregate Store.
      "movi v0.8b, #0\n"
      "movi v1.8b, #0\n"
      "movi v2.8b, #0\n"
      "ld1 {v0.s}[0], [%x[source]], #4\n"
      "ld1 {v0.b}[4], [%x[source]], #1\n"
      "ld1 {v1.s}[0], [x0], #4\n"
      "ld1 {v1.b}[4], [x0], #1\n"
      "ld1 {v2.s}[0], [x1], #4\n"
      "ld1 {v2.b}[4], [x1], #1\n"
      "uaddw v4.8h, v4.8h, v0.8b\n"
      "uaddw v5.8h, v5.8h, v1.8b\n"
      "uaddw v6.8h, v6.8h, v2.8b\n"
      "st1 {v0.8b, v1.8b, v2.8b}, [%x[destination]], #24\n"

      // Aggregator Reduction.
      "mov v3.s[0], %w[multiplicative_offset]\n"
      "dup v7.4s, %w[additive_offset]\n"
      "uaddlp v4.4s, v4.8h\n"
      "uaddlp v5.4s, v5.8h\n"
      "uaddlp v6.4s, v6.8h\n"
      "addp v4.4s, v4.4s, v5.4s\n"
      "addp v6.4s, v6.4s, v6.4s\n"
      "addp v8.4s, v4.4s, v6.4s\n"
      "mul v8.4s, v8.4s, v3.s[0]\n"
      "add v8.4s, v8.4s, v7.4s\n"
      "st1 {v8.4s}, [%x[destination]]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "x0", "x1", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "cc",
        "memory");
}

void zip_3x8_6_aligned(const std::uint8_t* source, std::int32_t count,
                       std::int32_t stride, std::uint8_t* destination,
                       std::int32_t multiplicative_offset,
                       std::int32_t additive_offset) {
  asm volatile(
      "add x0, %x[source], %x[stride]\n"
      "add x1, x0, %x[stride]\n"
      "sub %x[count], %x[count], #6\n"
      "movi v4.8h, #0\n"
      "movi v5.8h, #0\n"
      "movi v6.8h, #0\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store.
      "ld1 {v0.8b}, [%x[source]], #8\n"
      "ld1 {v1.8b}, [x0], #8\n"
      "ld1 {v2.8b}, [x1], #8\n"
      "uaddw v4.8h, v4.8h, v0.8b\n"
      "uaddw v5.8h, v5.8h, v1.8b\n"
      "uaddw v6.8h, v6.8h, v2.8b\n"
      "st1 {v0.8b, v1.8b, v2.8b}, [%x[destination]], #24\n"

      "bne 1b\n"

      // Leftover Load Aggregate Store.
      "movi v0.8b, #0\n"
      "movi v1.8b, #0\n"
      "movi v2.8b, #0\n"
      "ld1 {v0.s}[0], [%x[source]], #4\n"
      "ld1 {v0.h}[2], [%x[source]], #2\n"
      "ld1 {v1.s}[0], [x0], #4\n"
      "ld1 {v1.h}[2], [x0], #2\n"
      "ld1 {v2.s}[0], [x1], #4\n"
      "ld1 {v2.h}[2], [x1], #2\n"
      "uaddw v4.8h, v4.8h, v0.8b\n"
      "uaddw v5.8h, v5.8h, v1.8b\n"
      "uaddw v6.8h, v6.8h, v2.8b\n"
      "st1 {v0.8b, v1.8b, v2.8b}, [%x[destination]], #24\n"

      // Aggregator Reduction.
      "mov v3.s[0], %w[multiplicative_offset]\n"
      "dup v7.4s, %w[additive_offset]\n"
      "uaddlp v4.4s, v4.8h\n"
      "uaddlp v5.4s, v5.8h\n"
      "uaddlp v6.4s, v6.8h\n"
      "addp v4.4s, v4.4s, v5.4s\n"
      "addp v6.4s, v6.4s, v6.4s\n"
      "addp v8.4s, v4.4s, v6.4s\n"
      "mul v8.4s, v8.4s, v3.s[0]\n"
      "add v8.4s, v8.4s, v7.4s\n"
      "st1 {v8.4s}, [%x[destination]]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "x0", "x1", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "cc",
        "memory");
}

void zip_3x8_7_aligned(const std::uint8_t* source, std::int32_t count,
                       std::int32_t stride, std::uint8_t* destination,
                       std::int32_t multiplicative_offset,
                       std::int32_t additive_offset) {
  asm volatile(
      "add x0, %x[source], %x[stride]\n"
      "add x1, x0, %x[stride]\n"
      "sub %x[count], %x[count], #7\n"
      "movi v4.8h, #0\n"
      "movi v5.8h, #0\n"
      "movi v6.8h, #0\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store.
      "ld1 {v0.8b}, [%x[source]], #8\n"
      "ld1 {v1.8b}, [x0], #8\n"
      "ld1 {v2.8b}, [x1], #8\n"
      "uaddw v4.8h, v4.8h, v0.8b\n"
      "uaddw v5.8h, v5.8h, v1.8b\n"
      "uaddw v6.8h, v6.8h, v2.8b\n"
      "st1 {v0.8b, v1.8b, v2.8b}, [%x[destination]], #24\n"

      "bne 1b\n"

      // Leftover Load Aggregate Store.
      "movi v0.8b, #0\n"
      "movi v1.8b, #0\n"
      "movi v2.8b, #0\n"
      "ld1 {v0.s}[0], [%x[source]], #4\n"
      "ld1 {v0.h}[2], [%x[source]], #2\n"
      "ld1 {v0.b}[6], [%x[source]], #1\n"
      "ld1 {v1.s}[0], [x0], #4\n"
      "ld1 {v1.h}[2], [x0], #2\n"
      "ld1 {v1.b}[6], [x0], #1\n"
      "ld1 {v2.s}[0], [x1], #4\n"
      "ld1 {v2.h}[2], [x1], #2\n"
      "ld1 {v2.b}[6], [x1], #1\n"
      "uaddw v4.8h, v4.8h, v0.8b\n"
      "uaddw v5.8h, v5.8h, v1.8b\n"
      "uaddw v6.8h, v6.8h, v2.8b\n"
      "st1 {v0.8b, v1.8b, v2.8b}, [%x[destination]], #24\n"

      // Aggregator Reduction.
      "mov v3.s[0], %w[multiplicative_offset]\n"
      "dup v7.4s, %w[additive_offset]\n"
      "uaddlp v4.4s, v4.8h\n"
      "uaddlp v5.4s, v5.8h\n"
      "uaddlp v6.4s, v6.8h\n"
      "addp v4.4s, v4.4s, v5.4s\n"
      "addp v6.4s, v6.4s, v6.4s\n"
      "addp v8.4s, v4.4s, v6.4s\n"
      "mul v8.4s, v8.4s, v3.s[0]\n"
      "add v8.4s, v8.4s, v7.4s\n"
      "st1 {v8.4s}, [%x[destination]]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "x0", "x1", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "cc",
        "memory");
}

void zip_4x8_aligned(const std::uint8_t* source, std::int32_t count,
                     std::int32_t stride, std::uint8_t* destination,
                     std::int32_t multiplicative_offset,
                     std::int32_t additive_offset) {
  asm volatile(
      "add x0, %x[source], %x[stride]\n"
      "add x1, x0, %x[stride]\n"
      "add x2, x1, %x[stride]\n"
      "movi v4.8h, #0\n"
      "movi v5.8h, #0\n"
      "movi v6.8h, #0\n"
      "movi v7.8h, #0\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store.
      "ld1 {v0.8b}, [%x[source]], #8\n"
      "ld1 {v1.8b}, [x0], #8\n"
      "ld1 {v2.8b}, [x1], #8\n"
      "ld1 {v3.8b}, [x2], #8\n"
      "uaddw v4.8h, v4.8h, v0.8b\n"
      "uaddw v5.8h, v5.8h, v1.8b\n"
      "uaddw v6.8h, v6.8h, v2.8b\n"
      "uaddw v7.8h, v7.8h, v3.8b\n"
      "st1 {v0.8b, v1.8b, v2.8b, v3.8b}, [%x[destination]], #32\n"

      "bne 1b\n"

      // Aggregator Reduction.
      "mov v8.s[0], %w[multiplicative_offset]\n"
      "dup v9.4s, %w[additive_offset]\n"
      "uaddlp v4.4s, v4.8h\n"
      "uaddlp v5.4s, v5.8h\n"
      "uaddlp v6.4s, v6.8h\n"
      "uaddlp v7.4s, v7.8h\n"
      "addp v4.4s, v4.4s, v5.4s\n"
      "addp v6.4s, v6.4s, v7.4s\n"
      "addp v10.4s, v4.4s, v6.4s\n"
      "mul v10.4s, v10.4s, v8.s[0]\n"
      "add v10.4s, v10.4s, v9.4s\n"
      "st1 {v10.4s}, [%x[destination]]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "x0", "x1", "x2", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8",
        "v9", "v10", "cc", "memory");
}

void zip_4x8_1_aligned(const std::uint8_t* source, std::int32_t count,
                       std::int32_t stride, std::uint8_t* destination,
                       std::int32_t multiplicative_offset,
                       std::int32_t additive_offset) {
  asm volatile(
      "add x0, %x[source], %x[stride]\n"
      "add x1, x0, %x[stride]\n"
      "add x2, x1, %x[stride]\n"
      "sub %x[count], %x[count], #1\n"
      "movi v4.8h, #0\n"
      "movi v5.8h, #0\n"
      "movi v6.8h, #0\n"
      "movi v7.8h, #0\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store.
      "ld1 {v0.8b}, [%x[source]], #8\n"
      "ld1 {v1.8b}, [x0], #8\n"
      "ld1 {v2.8b}, [x1], #8\n"
      "ld1 {v3.8b}, [x2], #8\n"
      "uaddw v4.8h, v4.8h, v0.8b\n"
      "uaddw v5.8h, v5.8h, v1.8b\n"
      "uaddw v6.8h, v6.8h, v2.8b\n"
      "uaddw v7.8h, v7.8h, v3.8b\n"
      "st1 {v0.8b, v1.8b, v2.8b, v3.8b}, [%x[destination]], #32\n"

      "bne 1b\n"

      // Leftover Load Aggregate Store.
      "movi v0.8b, #0\n"
      "movi v1.8b, #0\n"
      "movi v2.8b, #0\n"
      "movi v3.8b, #0\n"
      "ld1 {v0.b}[0], [%x[source]], #1\n"
      "ld1 {v1.b}[0], [x0], #1\n"
      "ld1 {v2.b}[0], [x1], #1\n"
      "ld1 {v3.b}[0], [x2], #1\n"
      "uaddw v4.8h, v4.8h, v0.8b\n"
      "uaddw v5.8h, v5.8h, v1.8b\n"
      "uaddw v6.8h, v6.8h, v2.8b\n"
      "uaddw v7.8h, v7.8h, v3.8b\n"
      "st1 {v0.8b, v1.8b, v2.8b, v3.8b}, [%x[destination]], #32\n"

      // Aggregator Reduction.
      "mov v8.s[0], %w[multiplicative_offset]\n"
      "dup v9.4s, %w[additive_offset]\n"
      "uaddlp v4.4s, v4.8h\n"
      "uaddlp v5.4s, v5.8h\n"
      "uaddlp v6.4s, v6.8h\n"
      "uaddlp v7.4s, v7.8h\n"
      "addp v4.4s, v4.4s, v5.4s\n"
      "addp v6.4s, v6.4s, v7.4s\n"
      "addp v10.4s, v4.4s, v6.4s\n"
      "mul v10.4s, v10.4s, v8.s[0]\n"
      "add v10.4s, v10.4s, v9.4s\n"
      "st1 {v10.4s}, [%x[destination]]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "x0", "x1", "x2", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8",
        "v9", "v10", "cc", "memory");
}

void zip_4x8_2_aligned(const std::uint8_t* source, std::int32_t count,
                       std::int32_t stride, std::uint8_t* destination,
                       std::int32_t multiplicative_offset,
                       std::int32_t additive_offset) {
  asm volatile(
      "add x0, %x[source], %x[stride]\n"
      "add x1, x0, %x[stride]\n"
      "add x2, x1, %x[stride]\n"
      "sub %x[count], %x[count], #2\n"
      "movi v4.8h, #0\n"
      "movi v5.8h, #0\n"
      "movi v6.8h, #0\n"
      "movi v7.8h, #0\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store.
      "ld1 {v0.8b}, [%x[source]], #8\n"
      "ld1 {v1.8b}, [x0], #8\n"
      "ld1 {v2.8b}, [x1], #8\n"
      "ld1 {v3.8b}, [x2], #8\n"
      "uaddw v4.8h, v4.8h, v0.8b\n"
      "uaddw v5.8h, v5.8h, v1.8b\n"
      "uaddw v6.8h, v6.8h, v2.8b\n"
      "uaddw v7.8h, v7.8h, v3.8b\n"
      "st1 {v0.8b, v1.8b, v2.8b, v3.8b}, [%x[destination]], #32\n"

      "bne 1b\n"

      // Leftover Load Aggregate Store.
      "movi v0.8b, #0\n"
      "movi v1.8b, #0\n"
      "movi v2.8b, #0\n"
      "movi v3.8b, #0\n"
      "ld1 {v0.h}[0], [%x[source]], #2\n"
      "ld1 {v1.h}[0], [x0], #2\n"
      "ld1 {v2.h}[0], [x1], #2\n"
      "ld1 {v3.h}[0], [x2], #2\n"
      "uaddw v4.8h, v4.8h, v0.8b\n"
      "uaddw v5.8h, v5.8h, v1.8b\n"
      "uaddw v6.8h, v6.8h, v2.8b\n"
      "uaddw v7.8h, v7.8h, v3.8b\n"
      "st1 {v0.8b, v1.8b, v2.8b, v3.8b}, [%x[destination]], #32\n"

      // Aggregator Reduction.
      "mov v8.s[0], %w[multiplicative_offset]\n"
      "dup v9.4s, %w[additive_offset]\n"
      "uaddlp v4.4s, v4.8h\n"
      "uaddlp v5.4s, v5.8h\n"
      "uaddlp v6.4s, v6.8h\n"
      "uaddlp v7.4s, v7.8h\n"
      "addp v4.4s, v4.4s, v5.4s\n"
      "addp v6.4s, v6.4s, v7.4s\n"
      "addp v10.4s, v4.4s, v6.4s\n"
      "mul v10.4s, v10.4s, v8.s[0]\n"
      "add v10.4s, v10.4s, v9.4s\n"
      "st1 {v10.4s}, [%x[destination]]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "x0", "x1", "x2", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8",
        "v9", "v10", "cc", "memory");
}

void zip_4x8_3_aligned(const std::uint8_t* source, std::int32_t count,
                       std::int32_t stride, std::uint8_t* destination,
                       std::int32_t multiplicative_offset,
                       std::int32_t additive_offset) {
  asm volatile(
      "add x0, %x[source], %x[stride]\n"
      "add x1, x0, %x[stride]\n"
      "add x2, x1, %x[stride]\n"
      "sub %x[count], %x[count], #3\n"
      "movi v4.8h, #0\n"
      "movi v5.8h, #0\n"
      "movi v6.8h, #0\n"
      "movi v7.8h, #0\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store.
      "ld1 {v0.8b}, [%x[source]], #8\n"
      "ld1 {v1.8b}, [x0], #8\n"
      "ld1 {v2.8b}, [x1], #8\n"
      "ld1 {v3.8b}, [x2], #8\n"
      "uaddw v4.8h, v4.8h, v0.8b\n"
      "uaddw v5.8h, v5.8h, v1.8b\n"
      "uaddw v6.8h, v6.8h, v2.8b\n"
      "uaddw v7.8h, v7.8h, v3.8b\n"
      "st1 {v0.8b, v1.8b, v2.8b, v3.8b}, [%x[destination]], #32\n"

      "bne 1b\n"

      // Leftover Load Aggregate Store.
      "movi v0.8b, #0\n"
      "movi v1.8b, #0\n"
      "movi v2.8b, #0\n"
      "movi v3.8b, #0\n"
      "ld1 {v0.h}[0], [%x[source]], #2\n"
      "ld1 {v0.b}[2], [%x[source]], #1\n"
      "ld1 {v1.h}[0], [x0], #2\n"
      "ld1 {v1.b}[2], [x0], #1\n"
      "ld1 {v2.h}[0], [x1], #2\n"
      "ld1 {v2.b}[2], [x1], #1\n"
      "ld1 {v3.h}[0], [x2], #2\n"
      "ld1 {v3.b}[2], [x2], #1\n"
      "uaddw v4.8h, v4.8h, v0.8b\n"
      "uaddw v5.8h, v5.8h, v1.8b\n"
      "uaddw v6.8h, v6.8h, v2.8b\n"
      "uaddw v7.8h, v7.8h, v3.8b\n"
      "st1 {v0.8b, v1.8b, v2.8b, v3.8b}, [%x[destination]], #32\n"

      // Aggregator Reduction.
      "mov v8.s[0], %w[multiplicative_offset]\n"
      "dup v9.4s, %w[additive_offset]\n"
      "uaddlp v4.4s, v4.8h\n"
      "uaddlp v5.4s, v5.8h\n"
      "uaddlp v6.4s, v6.8h\n"
      "uaddlp v7.4s, v7.8h\n"
      "addp v4.4s, v4.4s, v5.4s\n"
      "addp v6.4s, v6.4s, v7.4s\n"
      "addp v10.4s, v4.4s, v6.4s\n"
      "mul v10.4s, v10.4s, v8.s[0]\n"
      "add v10.4s, v10.4s, v9.4s\n"
      "st1 {v10.4s}, [%x[destination]]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "x0", "x1", "x2", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8",
        "v9", "v10", "cc", "memory");
}

void zip_4x8_4_aligned(const std::uint8_t* source, std::int32_t count,
                       std::int32_t stride, std::uint8_t* destination,
                       std::int32_t multiplicative_offset,
                       std::int32_t additive_offset) {
  asm volatile(
      "add x0, %x[source], %x[stride]\n"
      "add x1, x0, %x[stride]\n"
      "add x2, x1, %x[stride]\n"
      "sub %x[count], %x[count], #4\n"
      "movi v4.8h, #0\n"
      "movi v5.8h, #0\n"
      "movi v6.8h, #0\n"
      "movi v7.8h, #0\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store.
      "ld1 {v0.8b}, [%x[source]], #8\n"
      "ld1 {v1.8b}, [x0], #8\n"
      "ld1 {v2.8b}, [x1], #8\n"
      "ld1 {v3.8b}, [x2], #8\n"
      "uaddw v4.8h, v4.8h, v0.8b\n"
      "uaddw v5.8h, v5.8h, v1.8b\n"
      "uaddw v6.8h, v6.8h, v2.8b\n"
      "uaddw v7.8h, v7.8h, v3.8b\n"
      "st1 {v0.8b, v1.8b, v2.8b, v3.8b}, [%x[destination]], #32\n"

      "bne 1b\n"

      // Leftover Load Aggregate Store.
      "movi v0.8b, #0\n"
      "movi v1.8b, #0\n"
      "movi v2.8b, #0\n"
      "movi v3.8b, #0\n"
      "ld1 {v0.s}[0], [%x[source]], #4\n"
      "ld1 {v1.s}[0], [x0], #4\n"
      "ld1 {v2.s}[0], [x1], #4\n"
      "ld1 {v3.s}[0], [x2], #4\n"
      "uaddw v4.8h, v4.8h, v0.8b\n"
      "uaddw v5.8h, v5.8h, v1.8b\n"
      "uaddw v6.8h, v6.8h, v2.8b\n"
      "uaddw v7.8h, v7.8h, v3.8b\n"
      "st1 {v0.8b, v1.8b, v2.8b, v3.8b}, [%x[destination]], #32\n"

      // Aggregator Reduction.
      "mov v8.s[0], %w[multiplicative_offset]\n"
      "dup v9.4s, %w[additive_offset]\n"
      "uaddlp v4.4s, v4.8h\n"
      "uaddlp v5.4s, v5.8h\n"
      "uaddlp v6.4s, v6.8h\n"
      "uaddlp v7.4s, v7.8h\n"
      "addp v4.4s, v4.4s, v5.4s\n"
      "addp v6.4s, v6.4s, v7.4s\n"
      "addp v10.4s, v4.4s, v6.4s\n"
      "mul v10.4s, v10.4s, v8.s[0]\n"
      "add v10.4s, v10.4s, v9.4s\n"
      "st1 {v10.4s}, [%x[destination]]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "x0", "x1", "x2", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8",
        "v9", "v10", "cc", "memory");
}

void zip_4x8_5_aligned(const std::uint8_t* source, std::int32_t count,
                       std::int32_t stride, std::uint8_t* destination,
                       std::int32_t multiplicative_offset,
                       std::int32_t additive_offset) {
  asm volatile(
      "add x0, %x[source], %x[stride]\n"
      "add x1, x0, %x[stride]\n"
      "add x2, x1, %x[stride]\n"
      "sub %x[count], %x[count], #5\n"
      "movi v4.8h, #0\n"
      "movi v5.8h, #0\n"
      "movi v6.8h, #0\n"
      "movi v7.8h, #0\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store.
      "ld1 {v0.8b}, [%x[source]], #8\n"
      "ld1 {v1.8b}, [x0], #8\n"
      "ld1 {v2.8b}, [x1], #8\n"
      "ld1 {v3.8b}, [x2], #8\n"
      "uaddw v4.8h, v4.8h, v0.8b\n"
      "uaddw v5.8h, v5.8h, v1.8b\n"
      "uaddw v6.8h, v6.8h, v2.8b\n"
      "uaddw v7.8h, v7.8h, v3.8b\n"
      "st1 {v0.8b, v1.8b, v2.8b, v3.8b}, [%x[destination]], #32\n"

      "bne 1b\n"

      // Leftover Load Aggregate Store.
      "movi v0.8b, #0\n"
      "movi v1.8b, #0\n"
      "movi v2.8b, #0\n"
      "movi v3.8b, #0\n"
      "ld1 {v0.s}[0], [%x[source]], #4\n"
      "ld1 {v0.b}[4], [%x[source]], #1\n"
      "ld1 {v1.s}[0], [x0], #4\n"
      "ld1 {v1.b}[4], [x0], #1\n"
      "ld1 {v2.s}[0], [x1], #4\n"
      "ld1 {v2.b}[4], [x1], #1\n"
      "ld1 {v3.s}[0], [x2], #4\n"
      "ld1 {v3.b}[4], [x2], #1\n"
      "uaddw v4.8h, v4.8h, v0.8b\n"
      "uaddw v5.8h, v5.8h, v1.8b\n"
      "uaddw v6.8h, v6.8h, v2.8b\n"
      "uaddw v7.8h, v7.8h, v3.8b\n"
      "st1 {v0.8b, v1.8b, v2.8b, v3.8b}, [%x[destination]], #32\n"

      // Aggregator Reduction.
      "mov v8.s[0], %w[multiplicative_offset]\n"
      "dup v9.4s, %w[additive_offset]\n"
      "uaddlp v4.4s, v4.8h\n"
      "uaddlp v5.4s, v5.8h\n"
      "uaddlp v6.4s, v6.8h\n"
      "uaddlp v7.4s, v7.8h\n"
      "addp v4.4s, v4.4s, v5.4s\n"
      "addp v6.4s, v6.4s, v7.4s\n"
      "addp v10.4s, v4.4s, v6.4s\n"
      "mul v10.4s, v10.4s, v8.s[0]\n"
      "add v10.4s, v10.4s, v9.4s\n"
      "st1 {v10.4s}, [%x[destination]]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "x0", "x1", "x2", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8",
        "v9", "v10", "cc", "memory");
}

void zip_4x8_6_aligned(const std::uint8_t* source, std::int32_t count,
                       std::int32_t stride, std::uint8_t* destination,
                       std::int32_t multiplicative_offset,
                       std::int32_t additive_offset) {
  asm volatile(
      "add x0, %x[source], %x[stride]\n"
      "add x1, x0, %x[stride]\n"
      "add x2, x1, %x[stride]\n"
      "sub %x[count], %x[count], #6\n"
      "movi v4.8h, #0\n"
      "movi v5.8h, #0\n"
      "movi v6.8h, #0\n"
      "movi v7.8h, #0\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store.
      "ld1 {v0.8b}, [%x[source]], #8\n"
      "ld1 {v1.8b}, [x0], #8\n"
      "ld1 {v2.8b}, [x1], #8\n"
      "ld1 {v3.8b}, [x2], #8\n"
      "uaddw v4.8h, v4.8h, v0.8b\n"
      "uaddw v5.8h, v5.8h, v1.8b\n"
      "uaddw v6.8h, v6.8h, v2.8b\n"
      "uaddw v7.8h, v7.8h, v3.8b\n"
      "st1 {v0.8b, v1.8b, v2.8b, v3.8b}, [%x[destination]], #32\n"

      "bne 1b\n"

      // Leftover Load Aggregate Store.
      "movi v0.8b, #0\n"
      "movi v1.8b, #0\n"
      "movi v2.8b, #0\n"
      "movi v3.8b, #0\n"
      "ld1 {v0.s}[0], [%x[source]], #4\n"
      "ld1 {v0.h}[2], [%x[source]], #2\n"
      "ld1 {v1.s}[0], [x0], #4\n"
      "ld1 {v1.h}[2], [x0], #2\n"
      "ld1 {v2.s}[0], [x1], #4\n"
      "ld1 {v2.h}[2], [x1], #2\n"
      "ld1 {v3.s}[0], [x2], #4\n"
      "ld1 {v3.h}[2], [x2], #2\n"
      "uaddw v4.8h, v4.8h, v0.8b\n"
      "uaddw v5.8h, v5.8h, v1.8b\n"
      "uaddw v6.8h, v6.8h, v2.8b\n"
      "uaddw v7.8h, v7.8h, v3.8b\n"
      "st1 {v0.8b, v1.8b, v2.8b, v3.8b}, [%x[destination]], #32\n"

      // Aggregator Reduction.
      "mov v8.s[0], %w[multiplicative_offset]\n"
      "dup v9.4s, %w[additive_offset]\n"
      "uaddlp v4.4s, v4.8h\n"
      "uaddlp v5.4s, v5.8h\n"
      "uaddlp v6.4s, v6.8h\n"
      "uaddlp v7.4s, v7.8h\n"
      "addp v4.4s, v4.4s, v5.4s\n"
      "addp v6.4s, v6.4s, v7.4s\n"
      "addp v10.4s, v4.4s, v6.4s\n"
      "mul v10.4s, v10.4s, v8.s[0]\n"
      "add v10.4s, v10.4s, v9.4s\n"
      "st1 {v10.4s}, [%x[destination]]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "x0", "x1", "x2", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8",
        "v9", "v10", "cc", "memory");
}

void zip_4x8_7_aligned(const std::uint8_t* source, std::int32_t count,
                       std::int32_t stride, std::uint8_t* destination,
                       std::int32_t multiplicative_offset,
                       std::int32_t additive_offset) {
  asm volatile(
      "add x0, %x[source], %x[stride]\n"
      "add x1, x0, %x[stride]\n"
      "add x2, x1, %x[stride]\n"
      "sub %x[count], %x[count], #7\n"
      "movi v4.8h, #0\n"
      "movi v5.8h, #0\n"
      "movi v6.8h, #0\n"
      "movi v7.8h, #0\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store.
      "ld1 {v0.8b}, [%x[source]], #8\n"
      "ld1 {v1.8b}, [x0], #8\n"
      "ld1 {v2.8b}, [x1], #8\n"
      "ld1 {v3.8b}, [x2], #8\n"
      "uaddw v4.8h, v4.8h, v0.8b\n"
      "uaddw v5.8h, v5.8h, v1.8b\n"
      "uaddw v6.8h, v6.8h, v2.8b\n"
      "uaddw v7.8h, v7.8h, v3.8b\n"
      "st1 {v0.8b, v1.8b, v2.8b, v3.8b}, [%x[destination]], #32\n"

      "bne 1b\n"

      // Leftover Load Aggregate Store.
      "movi v0.8b, #0\n"
      "movi v1.8b, #0\n"
      "movi v2.8b, #0\n"
      "movi v3.8b, #0\n"
      "ld1 {v0.s}[0], [%x[source]], #4\n"
      "ld1 {v0.h}[2], [%x[source]], #2\n"
      "ld1 {v0.b}[6], [%x[source]], #1\n"
      "ld1 {v1.s}[0], [x0], #4\n"
      "ld1 {v1.h}[2], [x0], #2\n"
      "ld1 {v1.b}[6], [x0], #1\n"
      "ld1 {v2.s}[0], [x1], #4\n"
      "ld1 {v2.h}[2], [x1], #2\n"
      "ld1 {v2.b}[6], [x1], #1\n"
      "ld1 {v3.s}[0], [x2], #4\n"
      "ld1 {v3.h}[2], [x2], #2\n"
      "ld1 {v3.b}[6], [x2], #1\n"
      "uaddw v4.8h, v4.8h, v0.8b\n"
      "uaddw v5.8h, v5.8h, v1.8b\n"
      "uaddw v6.8h, v6.8h, v2.8b\n"
      "uaddw v7.8h, v7.8h, v3.8b\n"
      "st1 {v0.8b, v1.8b, v2.8b, v3.8b}, [%x[destination]], #32\n"

      // Aggregator Reduction.
      "mov v8.s[0], %w[multiplicative_offset]\n"
      "dup v9.4s, %w[additive_offset]\n"
      "uaddlp v4.4s, v4.8h\n"
      "uaddlp v5.4s, v5.8h\n"
      "uaddlp v6.4s, v6.8h\n"
      "uaddlp v7.4s, v7.8h\n"
      "addp v4.4s, v4.4s, v5.4s\n"
      "addp v6.4s, v6.4s, v7.4s\n"
      "addp v10.4s, v4.4s, v6.4s\n"
      "mul v10.4s, v10.4s, v8.s[0]\n"
      "add v10.4s, v10.4s, v9.4s\n"
      "st1 {v10.4s}, [%x[destination]]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "x0", "x1", "x2", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8",
        "v9", "v10", "cc", "memory");
}

void zip_1x8(const std::uint8_t* source, std::int32_t count,
             std::int32_t stride, std::uint8_t* destination,
             std::int32_t multiplicative_offset, std::int32_t additive_offset) {
  asm volatile(
      "movi v4.8h, #0\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store.
      "ld1 {v0.8b}, [%x[source]], #8\n"
      "uaddw v4.8h, v4.8h, v0.8b\n"
      "st1 {v0.8b}, [%x[destination]], #8\n"

      "bne 1b\n"

      // Aggregator Reduction.
      "mov v1.s[0], %w[multiplicative_offset]\n"
      "dup v2.2s, %w[additive_offset]\n"
      "uaddlp v4.4s, v4.8h\n"
      "addp v4.4s, v4.4s, v4.4s\n"
      "addp v3.4s, v4.4s, v4.4s\n"
      "mul v3.2s, v3.2s, v1.s[0]\n"
      "add v3.2s, v3.2s, v2.2s\n"
      "st1 {v3.2s}, [%x[destination]]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "v0", "v1", "v2", "v3", "v4", "cc", "memory");
}

void zip_1x8_1(const std::uint8_t* source, std::int32_t count,
               std::int32_t stride, std::uint8_t* destination,
               std::int32_t multiplicative_offset,
               std::int32_t additive_offset) {
  asm volatile(
      "sub %x[count], %x[count], #1\n"
      "movi v4.8h, #0\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store.
      "ld1 {v0.8b}, [%x[source]], #8\n"
      "uaddw v4.8h, v4.8h, v0.8b\n"
      "st1 {v0.8b}, [%x[destination]], #8\n"

      "bne 1b\n"

      // Leftover Load Aggregate Store.
      "movi v0.8b, #0\n"
      "ld1 {v0.b}[0], [%x[source]], #1\n"
      "uaddw v4.8h, v4.8h, v0.8b\n"
      "st1 {v0.8b}, [%x[destination]], #8\n"

      // Aggregator Reduction.
      "mov v1.s[0], %w[multiplicative_offset]\n"
      "dup v2.2s, %w[additive_offset]\n"
      "uaddlp v4.4s, v4.8h\n"
      "addp v4.4s, v4.4s, v4.4s\n"
      "addp v3.4s, v4.4s, v4.4s\n"
      "mul v3.2s, v3.2s, v1.s[0]\n"
      "add v3.2s, v3.2s, v2.2s\n"
      "st1 {v3.2s}, [%x[destination]]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "v0", "v1", "v2", "v3", "v4", "cc", "memory");
}

void zip_1x8_2(const std::uint8_t* source, std::int32_t count,
               std::int32_t stride, std::uint8_t* destination,
               std::int32_t multiplicative_offset,
               std::int32_t additive_offset) {
  asm volatile(
      "sub %x[count], %x[count], #2\n"
      "movi v4.8h, #0\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store.
      "ld1 {v0.8b}, [%x[source]], #8\n"
      "uaddw v4.8h, v4.8h, v0.8b\n"
      "st1 {v0.8b}, [%x[destination]], #8\n"

      "bne 1b\n"

      // Leftover Load Aggregate Store.
      "movi v0.8b, #0\n"
      "ld1 {v0.h}[0], [%x[source]], #2\n"
      "uaddw v4.8h, v4.8h, v0.8b\n"
      "st1 {v0.8b}, [%x[destination]], #8\n"

      // Aggregator Reduction.
      "mov v1.s[0], %w[multiplicative_offset]\n"
      "dup v2.2s, %w[additive_offset]\n"
      "uaddlp v4.4s, v4.8h\n"
      "addp v4.4s, v4.4s, v4.4s\n"
      "addp v3.4s, v4.4s, v4.4s\n"
      "mul v3.2s, v3.2s, v1.s[0]\n"
      "add v3.2s, v3.2s, v2.2s\n"
      "st1 {v3.2s}, [%x[destination]]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "v0", "v1", "v2", "v3", "v4", "cc", "memory");
}

void zip_1x8_3(const std::uint8_t* source, std::int32_t count,
               std::int32_t stride, std::uint8_t* destination,
               std::int32_t multiplicative_offset,
               std::int32_t additive_offset) {
  asm volatile(
      "sub %x[count], %x[count], #3\n"
      "movi v4.8h, #0\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store.
      "ld1 {v0.8b}, [%x[source]], #8\n"
      "uaddw v4.8h, v4.8h, v0.8b\n"
      "st1 {v0.8b}, [%x[destination]], #8\n"

      "bne 1b\n"

      // Leftover Load Aggregate Store.
      "movi v0.8b, #0\n"
      "ld1 {v0.h}[0], [%x[source]], #2\n"
      "ld1 {v0.b}[2], [%x[source]], #1\n"
      "uaddw v4.8h, v4.8h, v0.8b\n"
      "st1 {v0.8b}, [%x[destination]], #8\n"

      // Aggregator Reduction.
      "mov v1.s[0], %w[multiplicative_offset]\n"
      "dup v2.2s, %w[additive_offset]\n"
      "uaddlp v4.4s, v4.8h\n"
      "addp v4.4s, v4.4s, v4.4s\n"
      "addp v3.4s, v4.4s, v4.4s\n"
      "mul v3.2s, v3.2s, v1.s[0]\n"
      "add v3.2s, v3.2s, v2.2s\n"
      "st1 {v3.2s}, [%x[destination]]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "v0", "v1", "v2", "v3", "v4", "cc", "memory");
}

void zip_1x8_4(const std::uint8_t* source, std::int32_t count,
               std::int32_t stride, std::uint8_t* destination,
               std::int32_t multiplicative_offset,
               std::int32_t additive_offset) {
  asm volatile(
      "sub %x[count], %x[count], #4\n"
      "movi v4.8h, #0\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store.
      "ld1 {v0.8b}, [%x[source]], #8\n"
      "uaddw v4.8h, v4.8h, v0.8b\n"
      "st1 {v0.8b}, [%x[destination]], #8\n"

      "bne 1b\n"

      // Leftover Load Aggregate Store.
      "movi v0.8b, #0\n"
      "ld1 {v0.s}[0], [%x[source]], #4\n"
      "uaddw v4.8h, v4.8h, v0.8b\n"
      "st1 {v0.8b}, [%x[destination]], #8\n"

      // Aggregator Reduction.
      "mov v1.s[0], %w[multiplicative_offset]\n"
      "dup v2.2s, %w[additive_offset]\n"
      "uaddlp v4.4s, v4.8h\n"
      "addp v4.4s, v4.4s, v4.4s\n"
      "addp v3.4s, v4.4s, v4.4s\n"
      "mul v3.2s, v3.2s, v1.s[0]\n"
      "add v3.2s, v3.2s, v2.2s\n"
      "st1 {v3.2s}, [%x[destination]]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "v0", "v1", "v2", "v3", "v4", "cc", "memory");
}

void zip_1x8_5(const std::uint8_t* source, std::int32_t count,
               std::int32_t stride, std::uint8_t* destination,
               std::int32_t multiplicative_offset,
               std::int32_t additive_offset) {
  asm volatile(
      "sub %x[count], %x[count], #5\n"
      "movi v4.8h, #0\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store.
      "ld1 {v0.8b}, [%x[source]], #8\n"
      "uaddw v4.8h, v4.8h, v0.8b\n"
      "st1 {v0.8b}, [%x[destination]], #8\n"

      "bne 1b\n"

      // Leftover Load Aggregate Store.
      "movi v0.8b, #0\n"
      "ld1 {v0.s}[0], [%x[source]], #4\n"
      "ld1 {v0.b}[4], [%x[source]], #1\n"
      "uaddw v4.8h, v4.8h, v0.8b\n"
      "st1 {v0.8b}, [%x[destination]], #8\n"

      // Aggregator Reduction.
      "mov v1.s[0], %w[multiplicative_offset]\n"
      "dup v2.2s, %w[additive_offset]\n"
      "uaddlp v4.4s, v4.8h\n"
      "addp v4.4s, v4.4s, v4.4s\n"
      "addp v3.4s, v4.4s, v4.4s\n"
      "mul v3.2s, v3.2s, v1.s[0]\n"
      "add v3.2s, v3.2s, v2.2s\n"
      "st1 {v3.2s}, [%x[destination]]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "v0", "v1", "v2", "v3", "v4", "cc", "memory");
}

void zip_1x8_6(const std::uint8_t* source, std::int32_t count,
               std::int32_t stride, std::uint8_t* destination,
               std::int32_t multiplicative_offset,
               std::int32_t additive_offset) {
  asm volatile(
      "sub %x[count], %x[count], #6\n"
      "movi v4.8h, #0\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store.
      "ld1 {v0.8b}, [%x[source]], #8\n"
      "uaddw v4.8h, v4.8h, v0.8b\n"
      "st1 {v0.8b}, [%x[destination]], #8\n"

      "bne 1b\n"

      // Leftover Load Aggregate Store.
      "movi v0.8b, #0\n"
      "ld1 {v0.s}[0], [%x[source]], #4\n"
      "ld1 {v0.h}[2], [%x[source]], #2\n"
      "uaddw v4.8h, v4.8h, v0.8b\n"
      "st1 {v0.8b}, [%x[destination]], #8\n"

      // Aggregator Reduction.
      "mov v1.s[0], %w[multiplicative_offset]\n"
      "dup v2.2s, %w[additive_offset]\n"
      "uaddlp v4.4s, v4.8h\n"
      "addp v4.4s, v4.4s, v4.4s\n"
      "addp v3.4s, v4.4s, v4.4s\n"
      "mul v3.2s, v3.2s, v1.s[0]\n"
      "add v3.2s, v3.2s, v2.2s\n"
      "st1 {v3.2s}, [%x[destination]]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "v0", "v1", "v2", "v3", "v4", "cc", "memory");
}

void zip_1x8_7(const std::uint8_t* source, std::int32_t count,
               std::int32_t stride, std::uint8_t* destination,
               std::int32_t multiplicative_offset,
               std::int32_t additive_offset) {
  asm volatile(
      "sub %x[count], %x[count], #7\n"
      "movi v4.8h, #0\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store.
      "ld1 {v0.8b}, [%x[source]], #8\n"
      "uaddw v4.8h, v4.8h, v0.8b\n"
      "st1 {v0.8b}, [%x[destination]], #8\n"

      "bne 1b\n"

      // Leftover Load Aggregate Store.
      "movi v0.8b, #0\n"
      "ld1 {v0.s}[0], [%x[source]], #4\n"
      "ld1 {v0.h}[2], [%x[source]], #2\n"
      "ld1 {v0.b}[6], [%x[source]], #1\n"
      "uaddw v4.8h, v4.8h, v0.8b\n"
      "st1 {v0.8b}, [%x[destination]], #8\n"

      // Aggregator Reduction.
      "mov v1.s[0], %w[multiplicative_offset]\n"
      "dup v2.2s, %w[additive_offset]\n"
      "uaddlp v4.4s, v4.8h\n"
      "addp v4.4s, v4.4s, v4.4s\n"
      "addp v3.4s, v4.4s, v4.4s\n"
      "mul v3.2s, v3.2s, v1.s[0]\n"
      "add v3.2s, v3.2s, v2.2s\n"
      "st1 {v3.2s}, [%x[destination]]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "v0", "v1", "v2", "v3", "v4", "cc", "memory");
}

void zip_2x8(const std::uint8_t* source, std::int32_t count,
             std::int32_t stride, std::uint8_t* destination,
             std::int32_t multiplicative_offset, std::int32_t additive_offset) {
  asm volatile(
      "add x0, %x[source], %x[stride]\n"
      "movi v4.8h, #0\n"
      "movi v5.8h, #0\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store.
      "ld1 {v0.8b}, [%x[source]], #8\n"
      "ld1 {v1.8b}, [x0], #8\n"
      "uaddw v4.8h, v4.8h, v0.8b\n"
      "uaddw v5.8h, v5.8h, v1.8b\n"
      "st1 {v0.8b, v1.8b}, [%x[destination]], #16\n"

      "bne 1b\n"

      // Aggregator Reduction.
      "mov v2.s[0], %w[multiplicative_offset]\n"
      "dup v3.2s, %w[additive_offset]\n"
      "uaddlp v4.4s, v4.8h\n"
      "uaddlp v5.4s, v5.8h\n"
      "addp v4.4s, v4.4s, v5.4s\n"
      "addp v6.4s, v4.4s, v4.4s\n"
      "mul v6.2s, v6.2s, v2.s[0]\n"
      "add v6.2s, v6.2s, v3.2s\n"
      "st1 {v6.2s}, [%x[destination]]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "x0", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "cc", "memory");
}

void zip_2x8_1(const std::uint8_t* source, std::int32_t count,
               std::int32_t stride, std::uint8_t* destination,
               std::int32_t multiplicative_offset,
               std::int32_t additive_offset) {
  asm volatile(
      "add x0, %x[source], %x[stride]\n"
      "sub %x[count], %x[count], #1\n"
      "movi v4.8h, #0\n"
      "movi v5.8h, #0\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store.
      "ld1 {v0.8b}, [%x[source]], #8\n"
      "ld1 {v1.8b}, [x0], #8\n"
      "uaddw v4.8h, v4.8h, v0.8b\n"
      "uaddw v5.8h, v5.8h, v1.8b\n"
      "st1 {v0.8b, v1.8b}, [%x[destination]], #16\n"

      "bne 1b\n"

      // Leftover Load Aggregate Store.
      "movi v0.8b, #0\n"
      "movi v1.8b, #0\n"
      "ld1 {v0.b}[0], [%x[source]], #1\n"
      "ld1 {v1.b}[0], [x0], #1\n"
      "uaddw v4.8h, v4.8h, v0.8b\n"
      "uaddw v5.8h, v5.8h, v1.8b\n"
      "st1 {v0.8b, v1.8b}, [%x[destination]], #16\n"

      // Aggregator Reduction.
      "mov v2.s[0], %w[multiplicative_offset]\n"
      "dup v3.2s, %w[additive_offset]\n"
      "uaddlp v4.4s, v4.8h\n"
      "uaddlp v5.4s, v5.8h\n"
      "addp v4.4s, v4.4s, v5.4s\n"
      "addp v6.4s, v4.4s, v4.4s\n"
      "mul v6.2s, v6.2s, v2.s[0]\n"
      "add v6.2s, v6.2s, v3.2s\n"
      "st1 {v6.2s}, [%x[destination]]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "x0", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "cc", "memory");
}

void zip_2x8_2(const std::uint8_t* source, std::int32_t count,
               std::int32_t stride, std::uint8_t* destination,
               std::int32_t multiplicative_offset,
               std::int32_t additive_offset) {
  asm volatile(
      "add x0, %x[source], %x[stride]\n"
      "sub %x[count], %x[count], #2\n"
      "movi v4.8h, #0\n"
      "movi v5.8h, #0\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store.
      "ld1 {v0.8b}, [%x[source]], #8\n"
      "ld1 {v1.8b}, [x0], #8\n"
      "uaddw v4.8h, v4.8h, v0.8b\n"
      "uaddw v5.8h, v5.8h, v1.8b\n"
      "st1 {v0.8b, v1.8b}, [%x[destination]], #16\n"

      "bne 1b\n"

      // Leftover Load Aggregate Store.
      "movi v0.8b, #0\n"
      "movi v1.8b, #0\n"
      "ld1 {v0.h}[0], [%x[source]], #2\n"
      "ld1 {v1.h}[0], [x0], #2\n"
      "uaddw v4.8h, v4.8h, v0.8b\n"
      "uaddw v5.8h, v5.8h, v1.8b\n"
      "st1 {v0.8b, v1.8b}, [%x[destination]], #16\n"

      // Aggregator Reduction.
      "mov v2.s[0], %w[multiplicative_offset]\n"
      "dup v3.2s, %w[additive_offset]\n"
      "uaddlp v4.4s, v4.8h\n"
      "uaddlp v5.4s, v5.8h\n"
      "addp v4.4s, v4.4s, v5.4s\n"
      "addp v6.4s, v4.4s, v4.4s\n"
      "mul v6.2s, v6.2s, v2.s[0]\n"
      "add v6.2s, v6.2s, v3.2s\n"
      "st1 {v6.2s}, [%x[destination]]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "x0", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "cc", "memory");
}

void zip_2x8_3(const std::uint8_t* source, std::int32_t count,
               std::int32_t stride, std::uint8_t* destination,
               std::int32_t multiplicative_offset,
               std::int32_t additive_offset) {
  asm volatile(
      "add x0, %x[source], %x[stride]\n"
      "sub %x[count], %x[count], #3\n"
      "movi v4.8h, #0\n"
      "movi v5.8h, #0\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store.
      "ld1 {v0.8b}, [%x[source]], #8\n"
      "ld1 {v1.8b}, [x0], #8\n"
      "uaddw v4.8h, v4.8h, v0.8b\n"
      "uaddw v5.8h, v5.8h, v1.8b\n"
      "st1 {v0.8b, v1.8b}, [%x[destination]], #16\n"

      "bne 1b\n"

      // Leftover Load Aggregate Store.
      "movi v0.8b, #0\n"
      "movi v1.8b, #0\n"
      "ld1 {v0.h}[0], [%x[source]], #2\n"
      "ld1 {v0.b}[2], [%x[source]], #1\n"
      "ld1 {v1.h}[0], [x0], #2\n"
      "ld1 {v1.b}[2], [x0], #1\n"
      "uaddw v4.8h, v4.8h, v0.8b\n"
      "uaddw v5.8h, v5.8h, v1.8b\n"
      "st1 {v0.8b, v1.8b}, [%x[destination]], #16\n"

      // Aggregator Reduction.
      "mov v2.s[0], %w[multiplicative_offset]\n"
      "dup v3.2s, %w[additive_offset]\n"
      "uaddlp v4.4s, v4.8h\n"
      "uaddlp v5.4s, v5.8h\n"
      "addp v4.4s, v4.4s, v5.4s\n"
      "addp v6.4s, v4.4s, v4.4s\n"
      "mul v6.2s, v6.2s, v2.s[0]\n"
      "add v6.2s, v6.2s, v3.2s\n"
      "st1 {v6.2s}, [%x[destination]]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "x0", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "cc", "memory");
}

void zip_2x8_4(const std::uint8_t* source, std::int32_t count,
               std::int32_t stride, std::uint8_t* destination,
               std::int32_t multiplicative_offset,
               std::int32_t additive_offset) {
  asm volatile(
      "add x0, %x[source], %x[stride]\n"
      "sub %x[count], %x[count], #4\n"
      "movi v4.8h, #0\n"
      "movi v5.8h, #0\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store.
      "ld1 {v0.8b}, [%x[source]], #8\n"
      "ld1 {v1.8b}, [x0], #8\n"
      "uaddw v4.8h, v4.8h, v0.8b\n"
      "uaddw v5.8h, v5.8h, v1.8b\n"
      "st1 {v0.8b, v1.8b}, [%x[destination]], #16\n"

      "bne 1b\n"

      // Leftover Load Aggregate Store.
      "movi v0.8b, #0\n"
      "movi v1.8b, #0\n"
      "ld1 {v0.s}[0], [%x[source]], #4\n"
      "ld1 {v1.s}[0], [x0], #4\n"
      "uaddw v4.8h, v4.8h, v0.8b\n"
      "uaddw v5.8h, v5.8h, v1.8b\n"
      "st1 {v0.8b, v1.8b}, [%x[destination]], #16\n"

      // Aggregator Reduction.
      "mov v2.s[0], %w[multiplicative_offset]\n"
      "dup v3.2s, %w[additive_offset]\n"
      "uaddlp v4.4s, v4.8h\n"
      "uaddlp v5.4s, v5.8h\n"
      "addp v4.4s, v4.4s, v5.4s\n"
      "addp v6.4s, v4.4s, v4.4s\n"
      "mul v6.2s, v6.2s, v2.s[0]\n"
      "add v6.2s, v6.2s, v3.2s\n"
      "st1 {v6.2s}, [%x[destination]]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "x0", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "cc", "memory");
}

void zip_2x8_5(const std::uint8_t* source, std::int32_t count,
               std::int32_t stride, std::uint8_t* destination,
               std::int32_t multiplicative_offset,
               std::int32_t additive_offset) {
  asm volatile(
      "add x0, %x[source], %x[stride]\n"
      "sub %x[count], %x[count], #5\n"
      "movi v4.8h, #0\n"
      "movi v5.8h, #0\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store.
      "ld1 {v0.8b}, [%x[source]], #8\n"
      "ld1 {v1.8b}, [x0], #8\n"
      "uaddw v4.8h, v4.8h, v0.8b\n"
      "uaddw v5.8h, v5.8h, v1.8b\n"
      "st1 {v0.8b, v1.8b}, [%x[destination]], #16\n"

      "bne 1b\n"

      // Leftover Load Aggregate Store.
      "movi v0.8b, #0\n"
      "movi v1.8b, #0\n"
      "ld1 {v0.s}[0], [%x[source]], #4\n"
      "ld1 {v0.b}[4], [%x[source]], #1\n"
      "ld1 {v1.s}[0], [x0], #4\n"
      "ld1 {v1.b}[4], [x0], #1\n"
      "uaddw v4.8h, v4.8h, v0.8b\n"
      "uaddw v5.8h, v5.8h, v1.8b\n"
      "st1 {v0.8b, v1.8b}, [%x[destination]], #16\n"

      // Aggregator Reduction.
      "mov v2.s[0], %w[multiplicative_offset]\n"
      "dup v3.2s, %w[additive_offset]\n"
      "uaddlp v4.4s, v4.8h\n"
      "uaddlp v5.4s, v5.8h\n"
      "addp v4.4s, v4.4s, v5.4s\n"
      "addp v6.4s, v4.4s, v4.4s\n"
      "mul v6.2s, v6.2s, v2.s[0]\n"
      "add v6.2s, v6.2s, v3.2s\n"
      "st1 {v6.2s}, [%x[destination]]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "x0", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "cc", "memory");
}

void zip_2x8_6(const std::uint8_t* source, std::int32_t count,
               std::int32_t stride, std::uint8_t* destination,
               std::int32_t multiplicative_offset,
               std::int32_t additive_offset) {
  asm volatile(
      "add x0, %x[source], %x[stride]\n"
      "sub %x[count], %x[count], #6\n"
      "movi v4.8h, #0\n"
      "movi v5.8h, #0\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store.
      "ld1 {v0.8b}, [%x[source]], #8\n"
      "ld1 {v1.8b}, [x0], #8\n"
      "uaddw v4.8h, v4.8h, v0.8b\n"
      "uaddw v5.8h, v5.8h, v1.8b\n"
      "st1 {v0.8b, v1.8b}, [%x[destination]], #16\n"

      "bne 1b\n"

      // Leftover Load Aggregate Store.
      "movi v0.8b, #0\n"
      "movi v1.8b, #0\n"
      "ld1 {v0.s}[0], [%x[source]], #4\n"
      "ld1 {v0.h}[2], [%x[source]], #2\n"
      "ld1 {v1.s}[0], [x0], #4\n"
      "ld1 {v1.h}[2], [x0], #2\n"
      "uaddw v4.8h, v4.8h, v0.8b\n"
      "uaddw v5.8h, v5.8h, v1.8b\n"
      "st1 {v0.8b, v1.8b}, [%x[destination]], #16\n"

      // Aggregator Reduction.
      "mov v2.s[0], %w[multiplicative_offset]\n"
      "dup v3.2s, %w[additive_offset]\n"
      "uaddlp v4.4s, v4.8h\n"
      "uaddlp v5.4s, v5.8h\n"
      "addp v4.4s, v4.4s, v5.4s\n"
      "addp v6.4s, v4.4s, v4.4s\n"
      "mul v6.2s, v6.2s, v2.s[0]\n"
      "add v6.2s, v6.2s, v3.2s\n"
      "st1 {v6.2s}, [%x[destination]]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "x0", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "cc", "memory");
}

void zip_2x8_7(const std::uint8_t* source, std::int32_t count,
               std::int32_t stride, std::uint8_t* destination,
               std::int32_t multiplicative_offset,
               std::int32_t additive_offset) {
  asm volatile(
      "add x0, %x[source], %x[stride]\n"
      "sub %x[count], %x[count], #7\n"
      "movi v4.8h, #0\n"
      "movi v5.8h, #0\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store.
      "ld1 {v0.8b}, [%x[source]], #8\n"
      "ld1 {v1.8b}, [x0], #8\n"
      "uaddw v4.8h, v4.8h, v0.8b\n"
      "uaddw v5.8h, v5.8h, v1.8b\n"
      "st1 {v0.8b, v1.8b}, [%x[destination]], #16\n"

      "bne 1b\n"

      // Leftover Load Aggregate Store.
      "movi v0.8b, #0\n"
      "movi v1.8b, #0\n"
      "ld1 {v0.s}[0], [%x[source]], #4\n"
      "ld1 {v0.h}[2], [%x[source]], #2\n"
      "ld1 {v0.b}[6], [%x[source]], #1\n"
      "ld1 {v1.s}[0], [x0], #4\n"
      "ld1 {v1.h}[2], [x0], #2\n"
      "ld1 {v1.b}[6], [x0], #1\n"
      "uaddw v4.8h, v4.8h, v0.8b\n"
      "uaddw v5.8h, v5.8h, v1.8b\n"
      "st1 {v0.8b, v1.8b}, [%x[destination]], #16\n"

      // Aggregator Reduction.
      "mov v2.s[0], %w[multiplicative_offset]\n"
      "dup v3.2s, %w[additive_offset]\n"
      "uaddlp v4.4s, v4.8h\n"
      "uaddlp v5.4s, v5.8h\n"
      "addp v4.4s, v4.4s, v5.4s\n"
      "addp v6.4s, v4.4s, v4.4s\n"
      "mul v6.2s, v6.2s, v2.s[0]\n"
      "add v6.2s, v6.2s, v3.2s\n"
      "st1 {v6.2s}, [%x[destination]]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "x0", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "cc", "memory");
}

void zip_3x8(const std::uint8_t* source, std::int32_t count,
             std::int32_t stride, std::uint8_t* destination,
             std::int32_t multiplicative_offset, std::int32_t additive_offset) {
  asm volatile(
      "add x0, %x[source], %x[stride]\n"
      "add x1, x0, %x[stride]\n"
      "movi v4.8h, #0\n"
      "movi v5.8h, #0\n"
      "movi v6.8h, #0\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store.
      "ld1 {v0.8b}, [%x[source]], #8\n"
      "ld1 {v1.8b}, [x0], #8\n"
      "ld1 {v2.8b}, [x1], #8\n"
      "uaddw v4.8h, v4.8h, v0.8b\n"
      "uaddw v5.8h, v5.8h, v1.8b\n"
      "uaddw v6.8h, v6.8h, v2.8b\n"
      "st1 {v0.8b, v1.8b, v2.8b}, [%x[destination]], #24\n"

      "bne 1b\n"

      // Aggregator Reduction.
      "mov v3.s[0], %w[multiplicative_offset]\n"
      "dup v7.4s, %w[additive_offset]\n"
      "uaddlp v4.4s, v4.8h\n"
      "uaddlp v5.4s, v5.8h\n"
      "uaddlp v6.4s, v6.8h\n"
      "addp v4.4s, v4.4s, v5.4s\n"
      "addp v6.4s, v6.4s, v6.4s\n"
      "addp v8.4s, v4.4s, v6.4s\n"
      "mul v8.4s, v8.4s, v3.s[0]\n"
      "add v8.4s, v8.4s, v7.4s\n"
      "st1 {v8.4s}, [%x[destination]]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "x0", "x1", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "cc",
        "memory");
}

void zip_3x8_1(const std::uint8_t* source, std::int32_t count,
               std::int32_t stride, std::uint8_t* destination,
               std::int32_t multiplicative_offset,
               std::int32_t additive_offset) {
  asm volatile(
      "add x0, %x[source], %x[stride]\n"
      "add x1, x0, %x[stride]\n"
      "sub %x[count], %x[count], #1\n"
      "movi v4.8h, #0\n"
      "movi v5.8h, #0\n"
      "movi v6.8h, #0\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store.
      "ld1 {v0.8b}, [%x[source]], #8\n"
      "ld1 {v1.8b}, [x0], #8\n"
      "ld1 {v2.8b}, [x1], #8\n"
      "uaddw v4.8h, v4.8h, v0.8b\n"
      "uaddw v5.8h, v5.8h, v1.8b\n"
      "uaddw v6.8h, v6.8h, v2.8b\n"
      "st1 {v0.8b, v1.8b, v2.8b}, [%x[destination]], #24\n"

      "bne 1b\n"

      // Leftover Load Aggregate Store.
      "movi v0.8b, #0\n"
      "movi v1.8b, #0\n"
      "movi v2.8b, #0\n"
      "ld1 {v0.b}[0], [%x[source]], #1\n"
      "ld1 {v1.b}[0], [x0], #1\n"
      "ld1 {v2.b}[0], [x1], #1\n"
      "uaddw v4.8h, v4.8h, v0.8b\n"
      "uaddw v5.8h, v5.8h, v1.8b\n"
      "uaddw v6.8h, v6.8h, v2.8b\n"
      "st1 {v0.8b, v1.8b, v2.8b}, [%x[destination]], #24\n"

      // Aggregator Reduction.
      "mov v3.s[0], %w[multiplicative_offset]\n"
      "dup v7.4s, %w[additive_offset]\n"
      "uaddlp v4.4s, v4.8h\n"
      "uaddlp v5.4s, v5.8h\n"
      "uaddlp v6.4s, v6.8h\n"
      "addp v4.4s, v4.4s, v5.4s\n"
      "addp v6.4s, v6.4s, v6.4s\n"
      "addp v8.4s, v4.4s, v6.4s\n"
      "mul v8.4s, v8.4s, v3.s[0]\n"
      "add v8.4s, v8.4s, v7.4s\n"
      "st1 {v8.4s}, [%x[destination]]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "x0", "x1", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "cc",
        "memory");
}

void zip_3x8_2(const std::uint8_t* source, std::int32_t count,
               std::int32_t stride, std::uint8_t* destination,
               std::int32_t multiplicative_offset,
               std::int32_t additive_offset) {
  asm volatile(
      "add x0, %x[source], %x[stride]\n"
      "add x1, x0, %x[stride]\n"
      "sub %x[count], %x[count], #2\n"
      "movi v4.8h, #0\n"
      "movi v5.8h, #0\n"
      "movi v6.8h, #0\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store.
      "ld1 {v0.8b}, [%x[source]], #8\n"
      "ld1 {v1.8b}, [x0], #8\n"
      "ld1 {v2.8b}, [x1], #8\n"
      "uaddw v4.8h, v4.8h, v0.8b\n"
      "uaddw v5.8h, v5.8h, v1.8b\n"
      "uaddw v6.8h, v6.8h, v2.8b\n"
      "st1 {v0.8b, v1.8b, v2.8b}, [%x[destination]], #24\n"

      "bne 1b\n"

      // Leftover Load Aggregate Store.
      "movi v0.8b, #0\n"
      "movi v1.8b, #0\n"
      "movi v2.8b, #0\n"
      "ld1 {v0.h}[0], [%x[source]], #2\n"
      "ld1 {v1.h}[0], [x0], #2\n"
      "ld1 {v2.h}[0], [x1], #2\n"
      "uaddw v4.8h, v4.8h, v0.8b\n"
      "uaddw v5.8h, v5.8h, v1.8b\n"
      "uaddw v6.8h, v6.8h, v2.8b\n"
      "st1 {v0.8b, v1.8b, v2.8b}, [%x[destination]], #24\n"

      // Aggregator Reduction.
      "mov v3.s[0], %w[multiplicative_offset]\n"
      "dup v7.4s, %w[additive_offset]\n"
      "uaddlp v4.4s, v4.8h\n"
      "uaddlp v5.4s, v5.8h\n"
      "uaddlp v6.4s, v6.8h\n"
      "addp v4.4s, v4.4s, v5.4s\n"
      "addp v6.4s, v6.4s, v6.4s\n"
      "addp v8.4s, v4.4s, v6.4s\n"
      "mul v8.4s, v8.4s, v3.s[0]\n"
      "add v8.4s, v8.4s, v7.4s\n"
      "st1 {v8.4s}, [%x[destination]]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "x0", "x1", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "cc",
        "memory");
}

void zip_3x8_3(const std::uint8_t* source, std::int32_t count,
               std::int32_t stride, std::uint8_t* destination,
               std::int32_t multiplicative_offset,
               std::int32_t additive_offset) {
  asm volatile(
      "add x0, %x[source], %x[stride]\n"
      "add x1, x0, %x[stride]\n"
      "sub %x[count], %x[count], #3\n"
      "movi v4.8h, #0\n"
      "movi v5.8h, #0\n"
      "movi v6.8h, #0\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store.
      "ld1 {v0.8b}, [%x[source]], #8\n"
      "ld1 {v1.8b}, [x0], #8\n"
      "ld1 {v2.8b}, [x1], #8\n"
      "uaddw v4.8h, v4.8h, v0.8b\n"
      "uaddw v5.8h, v5.8h, v1.8b\n"
      "uaddw v6.8h, v6.8h, v2.8b\n"
      "st1 {v0.8b, v1.8b, v2.8b}, [%x[destination]], #24\n"

      "bne 1b\n"

      // Leftover Load Aggregate Store.
      "movi v0.8b, #0\n"
      "movi v1.8b, #0\n"
      "movi v2.8b, #0\n"
      "ld1 {v0.h}[0], [%x[source]], #2\n"
      "ld1 {v0.b}[2], [%x[source]], #1\n"
      "ld1 {v1.h}[0], [x0], #2\n"
      "ld1 {v1.b}[2], [x0], #1\n"
      "ld1 {v2.h}[0], [x1], #2\n"
      "ld1 {v2.b}[2], [x1], #1\n"
      "uaddw v4.8h, v4.8h, v0.8b\n"
      "uaddw v5.8h, v5.8h, v1.8b\n"
      "uaddw v6.8h, v6.8h, v2.8b\n"
      "st1 {v0.8b, v1.8b, v2.8b}, [%x[destination]], #24\n"

      // Aggregator Reduction.
      "mov v3.s[0], %w[multiplicative_offset]\n"
      "dup v7.4s, %w[additive_offset]\n"
      "uaddlp v4.4s, v4.8h\n"
      "uaddlp v5.4s, v5.8h\n"
      "uaddlp v6.4s, v6.8h\n"
      "addp v4.4s, v4.4s, v5.4s\n"
      "addp v6.4s, v6.4s, v6.4s\n"
      "addp v8.4s, v4.4s, v6.4s\n"
      "mul v8.4s, v8.4s, v3.s[0]\n"
      "add v8.4s, v8.4s, v7.4s\n"
      "st1 {v8.4s}, [%x[destination]]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "x0", "x1", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "cc",
        "memory");
}

void zip_3x8_4(const std::uint8_t* source, std::int32_t count,
               std::int32_t stride, std::uint8_t* destination,
               std::int32_t multiplicative_offset,
               std::int32_t additive_offset) {
  asm volatile(
      "add x0, %x[source], %x[stride]\n"
      "add x1, x0, %x[stride]\n"
      "sub %x[count], %x[count], #4\n"
      "movi v4.8h, #0\n"
      "movi v5.8h, #0\n"
      "movi v6.8h, #0\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store.
      "ld1 {v0.8b}, [%x[source]], #8\n"
      "ld1 {v1.8b}, [x0], #8\n"
      "ld1 {v2.8b}, [x1], #8\n"
      "uaddw v4.8h, v4.8h, v0.8b\n"
      "uaddw v5.8h, v5.8h, v1.8b\n"
      "uaddw v6.8h, v6.8h, v2.8b\n"
      "st1 {v0.8b, v1.8b, v2.8b}, [%x[destination]], #24\n"

      "bne 1b\n"

      // Leftover Load Aggregate Store.
      "movi v0.8b, #0\n"
      "movi v1.8b, #0\n"
      "movi v2.8b, #0\n"
      "ld1 {v0.s}[0], [%x[source]], #4\n"
      "ld1 {v1.s}[0], [x0], #4\n"
      "ld1 {v2.s}[0], [x1], #4\n"
      "uaddw v4.8h, v4.8h, v0.8b\n"
      "uaddw v5.8h, v5.8h, v1.8b\n"
      "uaddw v6.8h, v6.8h, v2.8b\n"
      "st1 {v0.8b, v1.8b, v2.8b}, [%x[destination]], #24\n"

      // Aggregator Reduction.
      "mov v3.s[0], %w[multiplicative_offset]\n"
      "dup v7.4s, %w[additive_offset]\n"
      "uaddlp v4.4s, v4.8h\n"
      "uaddlp v5.4s, v5.8h\n"
      "uaddlp v6.4s, v6.8h\n"
      "addp v4.4s, v4.4s, v5.4s\n"
      "addp v6.4s, v6.4s, v6.4s\n"
      "addp v8.4s, v4.4s, v6.4s\n"
      "mul v8.4s, v8.4s, v3.s[0]\n"
      "add v8.4s, v8.4s, v7.4s\n"
      "st1 {v8.4s}, [%x[destination]]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "x0", "x1", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "cc",
        "memory");
}

void zip_3x8_5(const std::uint8_t* source, std::int32_t count,
               std::int32_t stride, std::uint8_t* destination,
               std::int32_t multiplicative_offset,
               std::int32_t additive_offset) {
  asm volatile(
      "add x0, %x[source], %x[stride]\n"
      "add x1, x0, %x[stride]\n"
      "sub %x[count], %x[count], #5\n"
      "movi v4.8h, #0\n"
      "movi v5.8h, #0\n"
      "movi v6.8h, #0\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store.
      "ld1 {v0.8b}, [%x[source]], #8\n"
      "ld1 {v1.8b}, [x0], #8\n"
      "ld1 {v2.8b}, [x1], #8\n"
      "uaddw v4.8h, v4.8h, v0.8b\n"
      "uaddw v5.8h, v5.8h, v1.8b\n"
      "uaddw v6.8h, v6.8h, v2.8b\n"
      "st1 {v0.8b, v1.8b, v2.8b}, [%x[destination]], #24\n"

      "bne 1b\n"

      // Leftover Load Aggregate Store.
      "movi v0.8b, #0\n"
      "movi v1.8b, #0\n"
      "movi v2.8b, #0\n"
      "ld1 {v0.s}[0], [%x[source]], #4\n"
      "ld1 {v0.b}[4], [%x[source]], #1\n"
      "ld1 {v1.s}[0], [x0], #4\n"
      "ld1 {v1.b}[4], [x0], #1\n"
      "ld1 {v2.s}[0], [x1], #4\n"
      "ld1 {v2.b}[4], [x1], #1\n"
      "uaddw v4.8h, v4.8h, v0.8b\n"
      "uaddw v5.8h, v5.8h, v1.8b\n"
      "uaddw v6.8h, v6.8h, v2.8b\n"
      "st1 {v0.8b, v1.8b, v2.8b}, [%x[destination]], #24\n"

      // Aggregator Reduction.
      "mov v3.s[0], %w[multiplicative_offset]\n"
      "dup v7.4s, %w[additive_offset]\n"
      "uaddlp v4.4s, v4.8h\n"
      "uaddlp v5.4s, v5.8h\n"
      "uaddlp v6.4s, v6.8h\n"
      "addp v4.4s, v4.4s, v5.4s\n"
      "addp v6.4s, v6.4s, v6.4s\n"
      "addp v8.4s, v4.4s, v6.4s\n"
      "mul v8.4s, v8.4s, v3.s[0]\n"
      "add v8.4s, v8.4s, v7.4s\n"
      "st1 {v8.4s}, [%x[destination]]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "x0", "x1", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "cc",
        "memory");
}

void zip_3x8_6(const std::uint8_t* source, std::int32_t count,
               std::int32_t stride, std::uint8_t* destination,
               std::int32_t multiplicative_offset,
               std::int32_t additive_offset) {
  asm volatile(
      "add x0, %x[source], %x[stride]\n"
      "add x1, x0, %x[stride]\n"
      "sub %x[count], %x[count], #6\n"
      "movi v4.8h, #0\n"
      "movi v5.8h, #0\n"
      "movi v6.8h, #0\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store.
      "ld1 {v0.8b}, [%x[source]], #8\n"
      "ld1 {v1.8b}, [x0], #8\n"
      "ld1 {v2.8b}, [x1], #8\n"
      "uaddw v4.8h, v4.8h, v0.8b\n"
      "uaddw v5.8h, v5.8h, v1.8b\n"
      "uaddw v6.8h, v6.8h, v2.8b\n"
      "st1 {v0.8b, v1.8b, v2.8b}, [%x[destination]], #24\n"

      "bne 1b\n"

      // Leftover Load Aggregate Store.
      "movi v0.8b, #0\n"
      "movi v1.8b, #0\n"
      "movi v2.8b, #0\n"
      "ld1 {v0.s}[0], [%x[source]], #4\n"
      "ld1 {v0.h}[2], [%x[source]], #2\n"
      "ld1 {v1.s}[0], [x0], #4\n"
      "ld1 {v1.h}[2], [x0], #2\n"
      "ld1 {v2.s}[0], [x1], #4\n"
      "ld1 {v2.h}[2], [x1], #2\n"
      "uaddw v4.8h, v4.8h, v0.8b\n"
      "uaddw v5.8h, v5.8h, v1.8b\n"
      "uaddw v6.8h, v6.8h, v2.8b\n"
      "st1 {v0.8b, v1.8b, v2.8b}, [%x[destination]], #24\n"

      // Aggregator Reduction.
      "mov v3.s[0], %w[multiplicative_offset]\n"
      "dup v7.4s, %w[additive_offset]\n"
      "uaddlp v4.4s, v4.8h\n"
      "uaddlp v5.4s, v5.8h\n"
      "uaddlp v6.4s, v6.8h\n"
      "addp v4.4s, v4.4s, v5.4s\n"
      "addp v6.4s, v6.4s, v6.4s\n"
      "addp v8.4s, v4.4s, v6.4s\n"
      "mul v8.4s, v8.4s, v3.s[0]\n"
      "add v8.4s, v8.4s, v7.4s\n"
      "st1 {v8.4s}, [%x[destination]]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "x0", "x1", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "cc",
        "memory");
}

void zip_3x8_7(const std::uint8_t* source, std::int32_t count,
               std::int32_t stride, std::uint8_t* destination,
               std::int32_t multiplicative_offset,
               std::int32_t additive_offset) {
  asm volatile(
      "add x0, %x[source], %x[stride]\n"
      "add x1, x0, %x[stride]\n"
      "sub %x[count], %x[count], #7\n"
      "movi v4.8h, #0\n"
      "movi v5.8h, #0\n"
      "movi v6.8h, #0\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store.
      "ld1 {v0.8b}, [%x[source]], #8\n"
      "ld1 {v1.8b}, [x0], #8\n"
      "ld1 {v2.8b}, [x1], #8\n"
      "uaddw v4.8h, v4.8h, v0.8b\n"
      "uaddw v5.8h, v5.8h, v1.8b\n"
      "uaddw v6.8h, v6.8h, v2.8b\n"
      "st1 {v0.8b, v1.8b, v2.8b}, [%x[destination]], #24\n"

      "bne 1b\n"

      // Leftover Load Aggregate Store.
      "movi v0.8b, #0\n"
      "movi v1.8b, #0\n"
      "movi v2.8b, #0\n"
      "ld1 {v0.s}[0], [%x[source]], #4\n"
      "ld1 {v0.h}[2], [%x[source]], #2\n"
      "ld1 {v0.b}[6], [%x[source]], #1\n"
      "ld1 {v1.s}[0], [x0], #4\n"
      "ld1 {v1.h}[2], [x0], #2\n"
      "ld1 {v1.b}[6], [x0], #1\n"
      "ld1 {v2.s}[0], [x1], #4\n"
      "ld1 {v2.h}[2], [x1], #2\n"
      "ld1 {v2.b}[6], [x1], #1\n"
      "uaddw v4.8h, v4.8h, v0.8b\n"
      "uaddw v5.8h, v5.8h, v1.8b\n"
      "uaddw v6.8h, v6.8h, v2.8b\n"
      "st1 {v0.8b, v1.8b, v2.8b}, [%x[destination]], #24\n"

      // Aggregator Reduction.
      "mov v3.s[0], %w[multiplicative_offset]\n"
      "dup v7.4s, %w[additive_offset]\n"
      "uaddlp v4.4s, v4.8h\n"
      "uaddlp v5.4s, v5.8h\n"
      "uaddlp v6.4s, v6.8h\n"
      "addp v4.4s, v4.4s, v5.4s\n"
      "addp v6.4s, v6.4s, v6.4s\n"
      "addp v8.4s, v4.4s, v6.4s\n"
      "mul v8.4s, v8.4s, v3.s[0]\n"
      "add v8.4s, v8.4s, v7.4s\n"
      "st1 {v8.4s}, [%x[destination]]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "x0", "x1", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "cc",
        "memory");
}

void zip_4x8(const std::uint8_t* source, std::int32_t count,
             std::int32_t stride, std::uint8_t* destination,
             std::int32_t multiplicative_offset, std::int32_t additive_offset) {
  asm volatile(
      "add x0, %x[source], %x[stride]\n"
      "add x1, x0, %x[stride]\n"
      "add x2, x1, %x[stride]\n"
      "movi v4.8h, #0\n"
      "movi v5.8h, #0\n"
      "movi v6.8h, #0\n"
      "movi v7.8h, #0\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store.
      "ld1 {v0.8b}, [%x[source]], #8\n"
      "ld1 {v1.8b}, [x0], #8\n"
      "ld1 {v2.8b}, [x1], #8\n"
      "ld1 {v3.8b}, [x2], #8\n"
      "uaddw v4.8h, v4.8h, v0.8b\n"
      "uaddw v5.8h, v5.8h, v1.8b\n"
      "uaddw v6.8h, v6.8h, v2.8b\n"
      "uaddw v7.8h, v7.8h, v3.8b\n"
      "st1 {v0.8b, v1.8b, v2.8b, v3.8b}, [%x[destination]], #32\n"

      "bne 1b\n"

      // Aggregator Reduction.
      "mov v8.s[0], %w[multiplicative_offset]\n"
      "dup v9.4s, %w[additive_offset]\n"
      "uaddlp v4.4s, v4.8h\n"
      "uaddlp v5.4s, v5.8h\n"
      "uaddlp v6.4s, v6.8h\n"
      "uaddlp v7.4s, v7.8h\n"
      "addp v4.4s, v4.4s, v5.4s\n"
      "addp v6.4s, v6.4s, v7.4s\n"
      "addp v10.4s, v4.4s, v6.4s\n"
      "mul v10.4s, v10.4s, v8.s[0]\n"
      "add v10.4s, v10.4s, v9.4s\n"
      "st1 {v10.4s}, [%x[destination]]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "x0", "x1", "x2", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8",
        "v9", "v10", "cc", "memory");
}

void zip_4x8_1(const std::uint8_t* source, std::int32_t count,
               std::int32_t stride, std::uint8_t* destination,
               std::int32_t multiplicative_offset,
               std::int32_t additive_offset) {
  asm volatile(
      "add x0, %x[source], %x[stride]\n"
      "add x1, x0, %x[stride]\n"
      "add x2, x1, %x[stride]\n"
      "sub %x[count], %x[count], #1\n"
      "movi v4.8h, #0\n"
      "movi v5.8h, #0\n"
      "movi v6.8h, #0\n"
      "movi v7.8h, #0\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store.
      "ld1 {v0.8b}, [%x[source]], #8\n"
      "ld1 {v1.8b}, [x0], #8\n"
      "ld1 {v2.8b}, [x1], #8\n"
      "ld1 {v3.8b}, [x2], #8\n"
      "uaddw v4.8h, v4.8h, v0.8b\n"
      "uaddw v5.8h, v5.8h, v1.8b\n"
      "uaddw v6.8h, v6.8h, v2.8b\n"
      "uaddw v7.8h, v7.8h, v3.8b\n"
      "st1 {v0.8b, v1.8b, v2.8b, v3.8b}, [%x[destination]], #32\n"

      "bne 1b\n"

      // Leftover Load Aggregate Store.
      "movi v0.8b, #0\n"
      "movi v1.8b, #0\n"
      "movi v2.8b, #0\n"
      "movi v3.8b, #0\n"
      "ld1 {v0.b}[0], [%x[source]], #1\n"
      "ld1 {v1.b}[0], [x0], #1\n"
      "ld1 {v2.b}[0], [x1], #1\n"
      "ld1 {v3.b}[0], [x2], #1\n"
      "uaddw v4.8h, v4.8h, v0.8b\n"
      "uaddw v5.8h, v5.8h, v1.8b\n"
      "uaddw v6.8h, v6.8h, v2.8b\n"
      "uaddw v7.8h, v7.8h, v3.8b\n"
      "st1 {v0.8b, v1.8b, v2.8b, v3.8b}, [%x[destination]], #32\n"

      // Aggregator Reduction.
      "mov v8.s[0], %w[multiplicative_offset]\n"
      "dup v9.4s, %w[additive_offset]\n"
      "uaddlp v4.4s, v4.8h\n"
      "uaddlp v5.4s, v5.8h\n"
      "uaddlp v6.4s, v6.8h\n"
      "uaddlp v7.4s, v7.8h\n"
      "addp v4.4s, v4.4s, v5.4s\n"
      "addp v6.4s, v6.4s, v7.4s\n"
      "addp v10.4s, v4.4s, v6.4s\n"
      "mul v10.4s, v10.4s, v8.s[0]\n"
      "add v10.4s, v10.4s, v9.4s\n"
      "st1 {v10.4s}, [%x[destination]]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "x0", "x1", "x2", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8",
        "v9", "v10", "cc", "memory");
}

void zip_4x8_2(const std::uint8_t* source, std::int32_t count,
               std::int32_t stride, std::uint8_t* destination,
               std::int32_t multiplicative_offset,
               std::int32_t additive_offset) {
  asm volatile(
      "add x0, %x[source], %x[stride]\n"
      "add x1, x0, %x[stride]\n"
      "add x2, x1, %x[stride]\n"
      "sub %x[count], %x[count], #2\n"
      "movi v4.8h, #0\n"
      "movi v5.8h, #0\n"
      "movi v6.8h, #0\n"
      "movi v7.8h, #0\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store.
      "ld1 {v0.8b}, [%x[source]], #8\n"
      "ld1 {v1.8b}, [x0], #8\n"
      "ld1 {v2.8b}, [x1], #8\n"
      "ld1 {v3.8b}, [x2], #8\n"
      "uaddw v4.8h, v4.8h, v0.8b\n"
      "uaddw v5.8h, v5.8h, v1.8b\n"
      "uaddw v6.8h, v6.8h, v2.8b\n"
      "uaddw v7.8h, v7.8h, v3.8b\n"
      "st1 {v0.8b, v1.8b, v2.8b, v3.8b}, [%x[destination]], #32\n"

      "bne 1b\n"

      // Leftover Load Aggregate Store.
      "movi v0.8b, #0\n"
      "movi v1.8b, #0\n"
      "movi v2.8b, #0\n"
      "movi v3.8b, #0\n"
      "ld1 {v0.h}[0], [%x[source]], #2\n"
      "ld1 {v1.h}[0], [x0], #2\n"
      "ld1 {v2.h}[0], [x1], #2\n"
      "ld1 {v3.h}[0], [x2], #2\n"
      "uaddw v4.8h, v4.8h, v0.8b\n"
      "uaddw v5.8h, v5.8h, v1.8b\n"
      "uaddw v6.8h, v6.8h, v2.8b\n"
      "uaddw v7.8h, v7.8h, v3.8b\n"
      "st1 {v0.8b, v1.8b, v2.8b, v3.8b}, [%x[destination]], #32\n"

      // Aggregator Reduction.
      "mov v8.s[0], %w[multiplicative_offset]\n"
      "dup v9.4s, %w[additive_offset]\n"
      "uaddlp v4.4s, v4.8h\n"
      "uaddlp v5.4s, v5.8h\n"
      "uaddlp v6.4s, v6.8h\n"
      "uaddlp v7.4s, v7.8h\n"
      "addp v4.4s, v4.4s, v5.4s\n"
      "addp v6.4s, v6.4s, v7.4s\n"
      "addp v10.4s, v4.4s, v6.4s\n"
      "mul v10.4s, v10.4s, v8.s[0]\n"
      "add v10.4s, v10.4s, v9.4s\n"
      "st1 {v10.4s}, [%x[destination]]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "x0", "x1", "x2", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8",
        "v9", "v10", "cc", "memory");
}

void zip_4x8_3(const std::uint8_t* source, std::int32_t count,
               std::int32_t stride, std::uint8_t* destination,
               std::int32_t multiplicative_offset,
               std::int32_t additive_offset) {
  asm volatile(
      "add x0, %x[source], %x[stride]\n"
      "add x1, x0, %x[stride]\n"
      "add x2, x1, %x[stride]\n"
      "sub %x[count], %x[count], #3\n"
      "movi v4.8h, #0\n"
      "movi v5.8h, #0\n"
      "movi v6.8h, #0\n"
      "movi v7.8h, #0\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store.
      "ld1 {v0.8b}, [%x[source]], #8\n"
      "ld1 {v1.8b}, [x0], #8\n"
      "ld1 {v2.8b}, [x1], #8\n"
      "ld1 {v3.8b}, [x2], #8\n"
      "uaddw v4.8h, v4.8h, v0.8b\n"
      "uaddw v5.8h, v5.8h, v1.8b\n"
      "uaddw v6.8h, v6.8h, v2.8b\n"
      "uaddw v7.8h, v7.8h, v3.8b\n"
      "st1 {v0.8b, v1.8b, v2.8b, v3.8b}, [%x[destination]], #32\n"

      "bne 1b\n"

      // Leftover Load Aggregate Store.
      "movi v0.8b, #0\n"
      "movi v1.8b, #0\n"
      "movi v2.8b, #0\n"
      "movi v3.8b, #0\n"
      "ld1 {v0.h}[0], [%x[source]], #2\n"
      "ld1 {v0.b}[2], [%x[source]], #1\n"
      "ld1 {v1.h}[0], [x0], #2\n"
      "ld1 {v1.b}[2], [x0], #1\n"
      "ld1 {v2.h}[0], [x1], #2\n"
      "ld1 {v2.b}[2], [x1], #1\n"
      "ld1 {v3.h}[0], [x2], #2\n"
      "ld1 {v3.b}[2], [x2], #1\n"
      "uaddw v4.8h, v4.8h, v0.8b\n"
      "uaddw v5.8h, v5.8h, v1.8b\n"
      "uaddw v6.8h, v6.8h, v2.8b\n"
      "uaddw v7.8h, v7.8h, v3.8b\n"
      "st1 {v0.8b, v1.8b, v2.8b, v3.8b}, [%x[destination]], #32\n"

      // Aggregator Reduction.
      "mov v8.s[0], %w[multiplicative_offset]\n"
      "dup v9.4s, %w[additive_offset]\n"
      "uaddlp v4.4s, v4.8h\n"
      "uaddlp v5.4s, v5.8h\n"
      "uaddlp v6.4s, v6.8h\n"
      "uaddlp v7.4s, v7.8h\n"
      "addp v4.4s, v4.4s, v5.4s\n"
      "addp v6.4s, v6.4s, v7.4s\n"
      "addp v10.4s, v4.4s, v6.4s\n"
      "mul v10.4s, v10.4s, v8.s[0]\n"
      "add v10.4s, v10.4s, v9.4s\n"
      "st1 {v10.4s}, [%x[destination]]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "x0", "x1", "x2", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8",
        "v9", "v10", "cc", "memory");
}

void zip_4x8_4(const std::uint8_t* source, std::int32_t count,
               std::int32_t stride, std::uint8_t* destination,
               std::int32_t multiplicative_offset,
               std::int32_t additive_offset) {
  asm volatile(
      "add x0, %x[source], %x[stride]\n"
      "add x1, x0, %x[stride]\n"
      "add x2, x1, %x[stride]\n"
      "sub %x[count], %x[count], #4\n"
      "movi v4.8h, #0\n"
      "movi v5.8h, #0\n"
      "movi v6.8h, #0\n"
      "movi v7.8h, #0\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store.
      "ld1 {v0.8b}, [%x[source]], #8\n"
      "ld1 {v1.8b}, [x0], #8\n"
      "ld1 {v2.8b}, [x1], #8\n"
      "ld1 {v3.8b}, [x2], #8\n"
      "uaddw v4.8h, v4.8h, v0.8b\n"
      "uaddw v5.8h, v5.8h, v1.8b\n"
      "uaddw v6.8h, v6.8h, v2.8b\n"
      "uaddw v7.8h, v7.8h, v3.8b\n"
      "st1 {v0.8b, v1.8b, v2.8b, v3.8b}, [%x[destination]], #32\n"

      "bne 1b\n"

      // Leftover Load Aggregate Store.
      "movi v0.8b, #0\n"
      "movi v1.8b, #0\n"
      "movi v2.8b, #0\n"
      "movi v3.8b, #0\n"
      "ld1 {v0.s}[0], [%x[source]], #4\n"
      "ld1 {v1.s}[0], [x0], #4\n"
      "ld1 {v2.s}[0], [x1], #4\n"
      "ld1 {v3.s}[0], [x2], #4\n"
      "uaddw v4.8h, v4.8h, v0.8b\n"
      "uaddw v5.8h, v5.8h, v1.8b\n"
      "uaddw v6.8h, v6.8h, v2.8b\n"
      "uaddw v7.8h, v7.8h, v3.8b\n"
      "st1 {v0.8b, v1.8b, v2.8b, v3.8b}, [%x[destination]], #32\n"

      // Aggregator Reduction.
      "mov v8.s[0], %w[multiplicative_offset]\n"
      "dup v9.4s, %w[additive_offset]\n"
      "uaddlp v4.4s, v4.8h\n"
      "uaddlp v5.4s, v5.8h\n"
      "uaddlp v6.4s, v6.8h\n"
      "uaddlp v7.4s, v7.8h\n"
      "addp v4.4s, v4.4s, v5.4s\n"
      "addp v6.4s, v6.4s, v7.4s\n"
      "addp v10.4s, v4.4s, v6.4s\n"
      "mul v10.4s, v10.4s, v8.s[0]\n"
      "add v10.4s, v10.4s, v9.4s\n"
      "st1 {v10.4s}, [%x[destination]]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "x0", "x1", "x2", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8",
        "v9", "v10", "cc", "memory");
}

void zip_4x8_5(const std::uint8_t* source, std::int32_t count,
               std::int32_t stride, std::uint8_t* destination,
               std::int32_t multiplicative_offset,
               std::int32_t additive_offset) {
  asm volatile(
      "add x0, %x[source], %x[stride]\n"
      "add x1, x0, %x[stride]\n"
      "add x2, x1, %x[stride]\n"
      "sub %x[count], %x[count], #5\n"
      "movi v4.8h, #0\n"
      "movi v5.8h, #0\n"
      "movi v6.8h, #0\n"
      "movi v7.8h, #0\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store.
      "ld1 {v0.8b}, [%x[source]], #8\n"
      "ld1 {v1.8b}, [x0], #8\n"
      "ld1 {v2.8b}, [x1], #8\n"
      "ld1 {v3.8b}, [x2], #8\n"
      "uaddw v4.8h, v4.8h, v0.8b\n"
      "uaddw v5.8h, v5.8h, v1.8b\n"
      "uaddw v6.8h, v6.8h, v2.8b\n"
      "uaddw v7.8h, v7.8h, v3.8b\n"
      "st1 {v0.8b, v1.8b, v2.8b, v3.8b}, [%x[destination]], #32\n"

      "bne 1b\n"

      // Leftover Load Aggregate Store.
      "movi v0.8b, #0\n"
      "movi v1.8b, #0\n"
      "movi v2.8b, #0\n"
      "movi v3.8b, #0\n"
      "ld1 {v0.s}[0], [%x[source]], #4\n"
      "ld1 {v0.b}[4], [%x[source]], #1\n"
      "ld1 {v1.s}[0], [x0], #4\n"
      "ld1 {v1.b}[4], [x0], #1\n"
      "ld1 {v2.s}[0], [x1], #4\n"
      "ld1 {v2.b}[4], [x1], #1\n"
      "ld1 {v3.s}[0], [x2], #4\n"
      "ld1 {v3.b}[4], [x2], #1\n"
      "uaddw v4.8h, v4.8h, v0.8b\n"
      "uaddw v5.8h, v5.8h, v1.8b\n"
      "uaddw v6.8h, v6.8h, v2.8b\n"
      "uaddw v7.8h, v7.8h, v3.8b\n"
      "st1 {v0.8b, v1.8b, v2.8b, v3.8b}, [%x[destination]], #32\n"

      // Aggregator Reduction.
      "mov v8.s[0], %w[multiplicative_offset]\n"
      "dup v9.4s, %w[additive_offset]\n"
      "uaddlp v4.4s, v4.8h\n"
      "uaddlp v5.4s, v5.8h\n"
      "uaddlp v6.4s, v6.8h\n"
      "uaddlp v7.4s, v7.8h\n"
      "addp v4.4s, v4.4s, v5.4s\n"
      "addp v6.4s, v6.4s, v7.4s\n"
      "addp v10.4s, v4.4s, v6.4s\n"
      "mul v10.4s, v10.4s, v8.s[0]\n"
      "add v10.4s, v10.4s, v9.4s\n"
      "st1 {v10.4s}, [%x[destination]]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "x0", "x1", "x2", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8",
        "v9", "v10", "cc", "memory");
}

void zip_4x8_6(const std::uint8_t* source, std::int32_t count,
               std::int32_t stride, std::uint8_t* destination,
               std::int32_t multiplicative_offset,
               std::int32_t additive_offset) {
  asm volatile(
      "add x0, %x[source], %x[stride]\n"
      "add x1, x0, %x[stride]\n"
      "add x2, x1, %x[stride]\n"
      "sub %x[count], %x[count], #6\n"
      "movi v4.8h, #0\n"
      "movi v5.8h, #0\n"
      "movi v6.8h, #0\n"
      "movi v7.8h, #0\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store.
      "ld1 {v0.8b}, [%x[source]], #8\n"
      "ld1 {v1.8b}, [x0], #8\n"
      "ld1 {v2.8b}, [x1], #8\n"
      "ld1 {v3.8b}, [x2], #8\n"
      "uaddw v4.8h, v4.8h, v0.8b\n"
      "uaddw v5.8h, v5.8h, v1.8b\n"
      "uaddw v6.8h, v6.8h, v2.8b\n"
      "uaddw v7.8h, v7.8h, v3.8b\n"
      "st1 {v0.8b, v1.8b, v2.8b, v3.8b}, [%x[destination]], #32\n"

      "bne 1b\n"

      // Leftover Load Aggregate Store.
      "movi v0.8b, #0\n"
      "movi v1.8b, #0\n"
      "movi v2.8b, #0\n"
      "movi v3.8b, #0\n"
      "ld1 {v0.s}[0], [%x[source]], #4\n"
      "ld1 {v0.h}[2], [%x[source]], #2\n"
      "ld1 {v1.s}[0], [x0], #4\n"
      "ld1 {v1.h}[2], [x0], #2\n"
      "ld1 {v2.s}[0], [x1], #4\n"
      "ld1 {v2.h}[2], [x1], #2\n"
      "ld1 {v3.s}[0], [x2], #4\n"
      "ld1 {v3.h}[2], [x2], #2\n"
      "uaddw v4.8h, v4.8h, v0.8b\n"
      "uaddw v5.8h, v5.8h, v1.8b\n"
      "uaddw v6.8h, v6.8h, v2.8b\n"
      "uaddw v7.8h, v7.8h, v3.8b\n"
      "st1 {v0.8b, v1.8b, v2.8b, v3.8b}, [%x[destination]], #32\n"

      // Aggregator Reduction.
      "mov v8.s[0], %w[multiplicative_offset]\n"
      "dup v9.4s, %w[additive_offset]\n"
      "uaddlp v4.4s, v4.8h\n"
      "uaddlp v5.4s, v5.8h\n"
      "uaddlp v6.4s, v6.8h\n"
      "uaddlp v7.4s, v7.8h\n"
      "addp v4.4s, v4.4s, v5.4s\n"
      "addp v6.4s, v6.4s, v7.4s\n"
      "addp v10.4s, v4.4s, v6.4s\n"
      "mul v10.4s, v10.4s, v8.s[0]\n"
      "add v10.4s, v10.4s, v9.4s\n"
      "st1 {v10.4s}, [%x[destination]]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "x0", "x1", "x2", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8",
        "v9", "v10", "cc", "memory");
}

void zip_4x8_7(const std::uint8_t* source, std::int32_t count,
               std::int32_t stride, std::uint8_t* destination,
               std::int32_t multiplicative_offset,
               std::int32_t additive_offset) {
  asm volatile(
      "add x0, %x[source], %x[stride]\n"
      "add x1, x0, %x[stride]\n"
      "add x2, x1, %x[stride]\n"
      "sub %x[count], %x[count], #7\n"
      "movi v4.8h, #0\n"
      "movi v5.8h, #0\n"
      "movi v6.8h, #0\n"
      "movi v7.8h, #0\n"

      "1:"
      "subs %x[count], %x[count], #8\n"

      // Load Aggregate Store.
      "ld1 {v0.8b}, [%x[source]], #8\n"
      "ld1 {v1.8b}, [x0], #8\n"
      "ld1 {v2.8b}, [x1], #8\n"
      "ld1 {v3.8b}, [x2], #8\n"
      "uaddw v4.8h, v4.8h, v0.8b\n"
      "uaddw v5.8h, v5.8h, v1.8b\n"
      "uaddw v6.8h, v6.8h, v2.8b\n"
      "uaddw v7.8h, v7.8h, v3.8b\n"
      "st1 {v0.8b, v1.8b, v2.8b, v3.8b}, [%x[destination]], #32\n"

      "bne 1b\n"

      // Leftover Load Aggregate Store.
      "movi v0.8b, #0\n"
      "movi v1.8b, #0\n"
      "movi v2.8b, #0\n"
      "movi v3.8b, #0\n"
      "ld1 {v0.s}[0], [%x[source]], #4\n"
      "ld1 {v0.h}[2], [%x[source]], #2\n"
      "ld1 {v0.b}[6], [%x[source]], #1\n"
      "ld1 {v1.s}[0], [x0], #4\n"
      "ld1 {v1.h}[2], [x0], #2\n"
      "ld1 {v1.b}[6], [x0], #1\n"
      "ld1 {v2.s}[0], [x1], #4\n"
      "ld1 {v2.h}[2], [x1], #2\n"
      "ld1 {v2.b}[6], [x1], #1\n"
      "ld1 {v3.s}[0], [x2], #4\n"
      "ld1 {v3.h}[2], [x2], #2\n"
      "ld1 {v3.b}[6], [x2], #1\n"
      "uaddw v4.8h, v4.8h, v0.8b\n"
      "uaddw v5.8h, v5.8h, v1.8b\n"
      "uaddw v6.8h, v6.8h, v2.8b\n"
      "uaddw v7.8h, v7.8h, v3.8b\n"
      "st1 {v0.8b, v1.8b, v2.8b, v3.8b}, [%x[destination]], #32\n"

      // Aggregator Reduction.
      "mov v8.s[0], %w[multiplicative_offset]\n"
      "dup v9.4s, %w[additive_offset]\n"
      "uaddlp v4.4s, v4.8h\n"
      "uaddlp v5.4s, v5.8h\n"
      "uaddlp v6.4s, v6.8h\n"
      "uaddlp v7.4s, v7.8h\n"
      "addp v4.4s, v4.4s, v5.4s\n"
      "addp v6.4s, v6.4s, v7.4s\n"
      "addp v10.4s, v4.4s, v6.4s\n"
      "mul v10.4s, v10.4s, v8.s[0]\n"
      "add v10.4s, v10.4s, v9.4s\n"
      "st1 {v10.4s}, [%x[destination]]\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [additive_offset] "+r"(additive_offset), [stride] "+r"(stride),
        [destination] "+r"(destination), [source] "+r"(source)
      :
      : "x0", "x1", "x2", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8",
        "v9", "v10", "cc", "memory");
}

inline void mul_1x8_1x8_int32_rhsadd(const std::uint8_t* lhs,
                                     const std::uint8_t* rhs,
                                     std::int32_t count, std::int32_t* result,
                                     std::int32_t result_stride) {
  asm volatile(
      "prfm pldl1keep, [%x[lhs]]\n"
      "prfm pldl1keep, [%x[rhs]]\n"
      // Clear aggregators.
      "movi v0.4s, #0\n"

      // General NxM lanes loop.
      "1:"

      // Subtract counter.
      "subs %x[count], %x[count], #8\n"

      "ld1 {v1.8b}, [%x[lhs]], #8\n"
      "ld1 {v2.8b}, [%x[rhs]], #8\n"
      "prfm pldl1keep, [%x[lhs], #64]\n"
      "prfm pldl1keep, [%x[rhs], #64]\n"
      "umull v3.8h, v2.8b, v1.8b\n"
      "uadalp v0.4s, v3.8h\n"

      // Loop break.
      "bne 1b\n"

      "ld1 {v8.2s}, [%x[rhs]]\n"

      // Reduce rows.
      "addp v0.4s, v0.4s, v0.4s\n"
      "addp v1.4s, v0.4s, v0.4s\n"

      // Add rhs offset to aggregated rows.
      "add v1.2s, v1.2s, v8.2s\n"

      // Store reduced rows.
      "st1 {v1.s}[0], [%x[result]], %x[result_stride]\n"
      : [count] "+r"(count), [result_stride] "+r"(result_stride),
        [rhs] "+r"(rhs), [lhs] "+r"(lhs), [result] "+r"(result)
      :
      : "v0", "v1", "v2", "v3", "v8", "cc", "memory");
}

inline void mul_1x8_2x8_int32_rhsadd(const std::uint8_t* lhs,
                                     const std::uint8_t* rhs,
                                     std::int32_t count, std::int32_t* result,
                                     std::int32_t result_stride) {
  asm volatile(
      "prfm pldl1keep, [%x[lhs]]\n"
      "prfm pldl1keep, [%x[rhs]]\n"
      // Clear aggregators.
      "movi v0.4s, #0\n"
      "movi v1.4s, #0\n"

      // General NxM lanes loop.
      "1:"

      // Subtract counter.
      "subs %x[count], %x[count], #8\n"

      "ld1 {v2.8b}, [%x[lhs]], #8\n"
      "ld1 {v3.8b, v4.8b}, [%x[rhs]], #16\n"
      "prfm pldl1keep, [%x[lhs], #64]\n"
      "prfm pldl1keep, [%x[rhs], #64]\n"
      "umull v5.8h, v3.8b, v2.8b\n"
      "umull v6.8h, v4.8b, v2.8b\n"
      "uadalp v0.4s, v5.8h\n"
      "uadalp v1.4s, v6.8h\n"

      // Loop break.
      "bne 1b\n"

      "ld1 {v8.2s}, [%x[rhs]]\n"

      // Reduce rows.
      "addp v0.4s, v0.4s, v1.4s\n"
      "addp v2.4s, v0.4s, v0.4s\n"

      // Add rhs offset to aggregated rows.
      "add v2.2s, v2.2s, v8.2s\n"

      // Store reduced rows.
      "st1 {v2.2s}, [%x[result]], %x[result_stride]\n"
      : [count] "+r"(count), [result_stride] "+r"(result_stride),
        [rhs] "+r"(rhs), [lhs] "+r"(lhs), [result] "+r"(result)
      :
      : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v8", "cc", "memory");
}

inline void mul_1x8_3x8_int32_rhsadd(const std::uint8_t* lhs,
                                     const std::uint8_t* rhs,
                                     std::int32_t count, std::int32_t* result,
                                     std::int32_t result_stride) {
  asm volatile(
      "prfm pldl1keep, [%x[lhs]]\n"
      "prfm pldl1keep, [%x[rhs]]\n"
      // Clear aggregators.
      "movi v0.4s, #0\n"
      "movi v1.4s, #0\n"
      "movi v2.4s, #0\n"

      // General NxM lanes loop.
      "1:"

      // Subtract counter.
      "subs %x[count], %x[count], #8\n"

      "ld1 {v3.8b}, [%x[lhs]], #8\n"
      "ld1 {v4.8b, v5.8b, v6.8b}, [%x[rhs]], #24\n"
      "prfm pldl1keep, [%x[lhs], #64]\n"
      "prfm pldl1keep, [%x[rhs], #64]\n"
      "umull v7.8h, v4.8b, v3.8b\n"
      "umull v8.8h, v5.8b, v3.8b\n"
      "umull v9.8h, v6.8b, v3.8b\n"
      "uadalp v0.4s, v7.8h\n"
      "uadalp v1.4s, v8.8h\n"
      "uadalp v2.4s, v9.8h\n"

      // Loop break.
      "bne 1b\n"

      "ld1 {v4.4s}, [%x[rhs]]\n"

      // Reduce rows.
      "addp v0.4s, v0.4s, v1.4s\n"
      "addp v2.4s, v2.4s, v2.4s\n"
      "addp v0.4s, v0.4s, v2.4s\n"

      // Add rhs offset to aggregated rows.
      "add v0.4s, v0.4s, v4.4s\n"

      // Store reduced rows.
      "st1 {v0.2s}, [%x[result]], #8\n"
      "st1 {v0.s}[2], [%x[result]], %x[result_stride]\n"
      "sub %x[result], %x[result], #8\n"
      : [count] "+r"(count), [result_stride] "+r"(result_stride),
        [rhs] "+r"(rhs), [lhs] "+r"(lhs), [result] "+r"(result)
      :
      : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "cc",
        "memory");
}

inline void mul_2x8_1x8_int32_rhsadd(const std::uint8_t* lhs,
                                     const std::uint8_t* rhs,
                                     std::int32_t count, std::int32_t* result,
                                     std::int32_t result_stride) {
  asm volatile(
      "prfm pldl1keep, [%x[lhs]]\n"
      "prfm pldl1keep, [%x[rhs]]\n"
      // Clear aggregators.
      "movi v0.4s, #0\n"
      "movi v1.4s, #0\n"

      // General NxM lanes loop.
      "1:"

      // Subtract counter.
      "subs %x[count], %x[count], #8\n"

      "ld1 {v2.8b, v3.8b}, [%x[lhs]], #16\n"
      "ld1 {v4.8b}, [%x[rhs]], #8\n"
      "prfm pldl1keep, [%x[lhs], #64]\n"
      "prfm pldl1keep, [%x[rhs], #64]\n"
      "umull v5.8h, v4.8b, v2.8b\n"
      "umull v6.8h, v4.8b, v3.8b\n"
      "uadalp v0.4s, v5.8h\n"
      "uadalp v1.4s, v6.8h\n"

      // Loop break.
      "bne 1b\n"

      "ld1 {v8.2s}, [%x[rhs]]\n"

      // Reduce rows.
      "addp v0.4s, v0.4s, v0.4s\n"
      "addp v2.4s, v0.4s, v0.4s\n"
      "addp v1.4s, v1.4s, v1.4s\n"
      "addp v3.4s, v1.4s, v1.4s\n"

      // Add rhs offset to aggregated rows.
      "add v2.2s, v2.2s, v8.2s\n"
      "add v3.2s, v3.2s, v8.2s\n"

      // Store reduced rows.
      "st1 {v2.s}[0], [%x[result]], %x[result_stride]\n"
      "st1 {v3.s}[0], [%x[result]], %x[result_stride]\n"
      : [count] "+r"(count), [result_stride] "+r"(result_stride),
        [rhs] "+r"(rhs), [lhs] "+r"(lhs), [result] "+r"(result)
      :
      : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v8", "cc", "memory");
}

inline void mul_2x8_2x8_int32_rhsadd(const std::uint8_t* lhs,
                                     const std::uint8_t* rhs,
                                     std::int32_t count, std::int32_t* result,
                                     std::int32_t result_stride) {
  asm volatile(
      "prfm pldl1keep, [%x[lhs]]\n"
      "prfm pldl1keep, [%x[rhs]]\n"
      // Clear aggregators.
      "movi v0.4s, #0\n"
      "movi v1.4s, #0\n"
      "movi v2.4s, #0\n"
      "mov v3.16b, v0.16b\n"

      // General NxM lanes loop.
      "1:"

      // Subtract counter.
      "subs %x[count], %x[count], #8\n"

      "ld1 {v4.8b, v5.8b}, [%x[lhs]], #16\n"
      "ld1 {v6.8b, v7.8b}, [%x[rhs]], #16\n"
      "prfm pldl1keep, [%x[lhs], #64]\n"
      "prfm pldl1keep, [%x[rhs], #64]\n"
      "umull v8.8h, v6.8b, v4.8b\n"
      "umull v9.8h, v7.8b, v4.8b\n"
      "umull v10.8h, v6.8b, v5.8b\n"
      "umull v11.8h, v7.8b, v5.8b\n"
      "uadalp v0.4s, v8.8h\n"
      "uadalp v1.4s, v9.8h\n"
      "uadalp v2.4s, v10.8h\n"
      "uadalp v3.4s, v11.8h\n"

      // Loop break.
      "bne 1b\n"

      "ld1 {v8.2s}, [%x[rhs]]\n"

      // Reduce rows.
      "addp v0.4s, v0.4s, v1.4s\n"
      "addp v4.4s, v0.4s, v0.4s\n"
      "addp v2.4s, v2.4s, v3.4s\n"
      "addp v5.4s, v2.4s, v2.4s\n"

      // Add rhs offset to aggregated rows.
      "add v4.2s, v4.2s, v8.2s\n"
      "add v5.2s, v5.2s, v8.2s\n"

      // Store reduced rows.
      "st1 {v4.2s}, [%x[result]], %x[result_stride]\n"
      "st1 {v5.2s}, [%x[result]], %x[result_stride]\n"
      : [count] "+r"(count), [result_stride] "+r"(result_stride),
        [rhs] "+r"(rhs), [lhs] "+r"(lhs), [result] "+r"(result)
      :
      : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
        "v11", "cc", "memory");
}

inline void mul_2x8_3x8_int32_rhsadd(const std::uint8_t* lhs,
                                     const std::uint8_t* rhs,
                                     std::int32_t count, std::int32_t* result,
                                     std::int32_t result_stride) {
  asm volatile(
      "prfm pldl1keep, [%x[lhs]]\n"
      "prfm pldl1keep, [%x[rhs]]\n"
      // Clear aggregators.
      "movi v0.4s, #0\n"
      "movi v1.4s, #0\n"
      "movi v2.4s, #0\n"
      "mov v3.16b, v0.16b\n"
      "mov v4.16b, v1.16b\n"
      "mov v5.16b, v2.16b\n"

      // General NxM lanes loop.
      "1:"

      // Subtract counter.
      "subs %x[count], %x[count], #8\n"

      "ld1 {v6.8b, v7.8b}, [%x[lhs]], #16\n"
      "ld1 {v8.8b, v9.8b, v10.8b}, [%x[rhs]], #24\n"
      "prfm pldl1keep, [%x[lhs], #64]\n"
      "prfm pldl1keep, [%x[rhs], #64]\n"
      "umull v11.8h, v8.8b, v6.8b\n"
      "umull v12.8h, v9.8b, v6.8b\n"
      "umull v13.8h, v10.8b, v6.8b\n"
      "umull v14.8h, v8.8b, v7.8b\n"
      "umull v15.8h, v9.8b, v7.8b\n"
      "umull v16.8h, v10.8b, v7.8b\n"
      "uadalp v0.4s, v11.8h\n"
      "uadalp v1.4s, v12.8h\n"
      "uadalp v2.4s, v13.8h\n"
      "uadalp v3.4s, v14.8h\n"
      "uadalp v4.4s, v15.8h\n"
      "uadalp v5.4s, v16.8h\n"

      // Loop break.
      "bne 1b\n"

      "ld1 {v6.4s}, [%x[rhs]]\n"

      // Reduce rows.
      "addp v0.4s, v0.4s, v1.4s\n"
      "addp v2.4s, v2.4s, v2.4s\n"
      "addp v0.4s, v0.4s, v2.4s\n"
      "addp v3.4s, v3.4s, v4.4s\n"
      "addp v5.4s, v5.4s, v5.4s\n"
      "addp v3.4s, v3.4s, v5.4s\n"

      // Add rhs offset to aggregated rows.
      "add v0.4s, v0.4s, v6.4s\n"
      "add v3.4s, v3.4s, v6.4s\n"

      // Store reduced rows.
      "st1 {v0.2s}, [%x[result]], #8\n"
      "st1 {v0.s}[2], [%x[result]], %x[result_stride]\n"
      "sub %x[result], %x[result], #8\n"
      "st1 {v3.2s}, [%x[result]], #8\n"
      "st1 {v3.s}[2], [%x[result]], %x[result_stride]\n"
      "sub %x[result], %x[result], #8\n"
      : [count] "+r"(count), [result_stride] "+r"(result_stride),
        [rhs] "+r"(rhs), [lhs] "+r"(lhs), [result] "+r"(result)
      :
      : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
        "v11", "v12", "v13", "v14", "v15", "v16", "cc", "memory");
}

inline void mul_3x8_1x8_int32_rhsadd(const std::uint8_t* lhs,
                                     const std::uint8_t* rhs,
                                     std::int32_t count, std::int32_t* result,
                                     std::int32_t result_stride) {
  asm volatile(
      "prfm pldl1keep, [%x[lhs]]\n"
      "prfm pldl1keep, [%x[rhs]]\n"
      // Clear aggregators.
      "movi v0.4s, #0\n"
      "movi v1.4s, #0\n"
      "movi v2.4s, #0\n"

      // General NxM lanes loop.
      "1:"

      // Subtract counter.
      "subs %x[count], %x[count], #8\n"

      "ld1 {v3.8b, v4.8b, v5.8b}, [%x[lhs]], #24\n"
      "ld1 {v6.8b}, [%x[rhs]], #8\n"
      "prfm pldl1keep, [%x[lhs], #64]\n"
      "prfm pldl1keep, [%x[rhs], #64]\n"
      "umull v7.8h, v6.8b, v3.8b\n"
      "umull v8.8h, v6.8b, v4.8b\n"
      "umull v9.8h, v6.8b, v5.8b\n"
      "uadalp v0.4s, v7.8h\n"
      "uadalp v1.4s, v8.8h\n"
      "uadalp v2.4s, v9.8h\n"

      // Loop break.
      "bne 1b\n"

      "ld1 {v8.2s}, [%x[rhs]]\n"

      // Reduce rows.
      "addp v0.4s, v0.4s, v0.4s\n"
      "addp v3.4s, v0.4s, v0.4s\n"
      "addp v1.4s, v1.4s, v1.4s\n"
      "addp v4.4s, v1.4s, v1.4s\n"
      "addp v2.4s, v2.4s, v2.4s\n"
      "addp v5.4s, v2.4s, v2.4s\n"

      // Add rhs offset to aggregated rows.
      "add v3.2s, v3.2s, v8.2s\n"
      "add v4.2s, v4.2s, v8.2s\n"
      "add v5.2s, v5.2s, v8.2s\n"

      // Store reduced rows.
      "st1 {v3.s}[0], [%x[result]], %x[result_stride]\n"
      "st1 {v4.s}[0], [%x[result]], %x[result_stride]\n"
      "st1 {v5.s}[0], [%x[result]], %x[result_stride]\n"
      : [count] "+r"(count), [result_stride] "+r"(result_stride),
        [rhs] "+r"(rhs), [lhs] "+r"(lhs), [result] "+r"(result)
      :
      : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "cc",
        "memory");
}

inline void mul_3x8_2x8_int32_rhsadd(const std::uint8_t* lhs,
                                     const std::uint8_t* rhs,
                                     std::int32_t count, std::int32_t* result,
                                     std::int32_t result_stride) {
  asm volatile(
      "prfm pldl1keep, [%x[lhs]]\n"
      "prfm pldl1keep, [%x[rhs]]\n"
      // Clear aggregators.
      "movi v0.4s, #0\n"
      "movi v1.4s, #0\n"
      "movi v2.4s, #0\n"
      "mov v3.16b, v0.16b\n"
      "mov v4.16b, v1.16b\n"
      "mov v5.16b, v2.16b\n"

      // General NxM lanes loop.
      "1:"

      // Subtract counter.
      "subs %x[count], %x[count], #8\n"

      "ld1 {v6.8b, v7.8b, v8.8b}, [%x[lhs]], #24\n"
      "ld1 {v9.8b, v10.8b}, [%x[rhs]], #16\n"
      "prfm pldl1keep, [%x[lhs], #64]\n"
      "prfm pldl1keep, [%x[rhs], #64]\n"
      "umull v11.8h, v9.8b, v6.8b\n"
      "umull v12.8h, v10.8b, v6.8b\n"
      "umull v13.8h, v9.8b, v7.8b\n"
      "umull v14.8h, v10.8b, v7.8b\n"
      "umull v15.8h, v9.8b, v8.8b\n"
      "umull v16.8h, v10.8b, v8.8b\n"
      "uadalp v0.4s, v11.8h\n"
      "uadalp v1.4s, v12.8h\n"
      "uadalp v2.4s, v13.8h\n"
      "uadalp v3.4s, v14.8h\n"
      "uadalp v4.4s, v15.8h\n"
      "uadalp v5.4s, v16.8h\n"

      // Loop break.
      "bne 1b\n"

      "ld1 {v8.2s}, [%x[rhs]]\n"

      // Reduce rows.
      "addp v0.4s, v0.4s, v1.4s\n"
      "addp v6.4s, v0.4s, v0.4s\n"
      "addp v2.4s, v2.4s, v3.4s\n"
      "addp v7.4s, v2.4s, v2.4s\n"
      "addp v4.4s, v4.4s, v5.4s\n"
      "addp v9.4s, v4.4s, v4.4s\n"

      // Add rhs offset to aggregated rows.
      "add v6.2s, v6.2s, v8.2s\n"
      "add v7.2s, v7.2s, v8.2s\n"
      "add v9.2s, v9.2s, v8.2s\n"

      // Store reduced rows.
      "st1 {v6.2s}, [%x[result]], %x[result_stride]\n"
      "st1 {v7.2s}, [%x[result]], %x[result_stride]\n"
      "st1 {v9.2s}, [%x[result]], %x[result_stride]\n"
      : [count] "+r"(count), [result_stride] "+r"(result_stride),
        [rhs] "+r"(rhs), [lhs] "+r"(lhs), [result] "+r"(result)
      :
      : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
        "v11", "v12", "v13", "v14", "v15", "v16", "cc", "memory");
}

inline void mul_3x8_3x8_int32_rhsadd(const std::uint8_t* lhs,
                                     const std::uint8_t* rhs,
                                     std::int32_t count, std::int32_t* result,
                                     std::int32_t result_stride) {
  asm volatile(
      "prfm pldl1keep, [%x[lhs]]\n"
      "prfm pldl1keep, [%x[rhs]]\n"
      // Clear aggregators.
      "movi v0.4s, #0\n"
      "movi v1.4s, #0\n"
      "movi v2.4s, #0\n"
      "mov v3.16b, v0.16b\n"
      "mov v4.16b, v1.16b\n"
      "mov v5.16b, v2.16b\n"
      "mov v6.16b, v3.16b\n"
      "mov v7.16b, v4.16b\n"
      "mov v8.16b, v5.16b\n"

      // 3x3 lanes loop.
      "1:"

      "ld1 {v12.8b, v13.8b, v14.8b}, [%x[rhs]], #24\n"
      "ld1 {v9.8b}, [%x[lhs]], #8\n"
      "umull v15.8h, v9.8b, v12.8b\n"
      "ld1 {v10.8b}, [%x[lhs]], #8\n"
      "umull v16.8h, v9.8b, v13.8b\n"
      "ld1 {v11.8b}, [%x[lhs]], #8\n"
      "umull v17.8h, v9.8b, v14.8b\n"
      "prfm pldl1keep, [%x[lhs], #64]\n"
      "umull v18.8h, v10.8b, v12.8b\n"
      "prfm pldl1keep, [%x[rhs], #64]\n"
      "uadalp v0.4s, v15.8h\n"
      "uadalp v1.4s, v16.8h\n"
      "uadalp v2.4s, v17.8h\n"
      "uadalp v3.4s, v18.8h\n"
      "umull v15.8h, v10.8b, v13.8b\n"
      "umull v16.8h, v10.8b, v14.8b\n"
      "umull v17.8h, v11.8b, v12.8b\n"
      "umull v18.8h, v11.8b, v13.8b\n"

      // Subtract counter.
      "subs %x[count], %x[count], #8\n"

      "umull v9.8h, v11.8b, v14.8b\n"
      "uadalp v4.4s, v15.8h\n"
      "uadalp v5.4s, v16.8h\n"
      "uadalp v6.4s, v17.8h\n"
      "uadalp v7.4s, v18.8h\n"
      "uadalp v8.4s, v9.8h\n"

      // Loop break.
      "bne 1b\n"

      "ld1 {v9.4s}, [%x[rhs]]\n"

      // Reduce rows.
      "addp v0.4s, v0.4s, v1.4s\n"
      "addp v2.4s, v2.4s, v2.4s\n"
      "addp v0.4s, v0.4s, v2.4s\n"
      "addp v3.4s, v3.4s, v4.4s\n"
      "addp v5.4s, v5.4s, v5.4s\n"
      "addp v3.4s, v3.4s, v5.4s\n"
      "addp v6.4s, v6.4s, v7.4s\n"
      "addp v8.4s, v8.4s, v8.4s\n"
      "addp v6.4s, v6.4s, v8.4s\n"

      // Add rhs offset to aggregated rows.
      "add v0.4s, v0.4s, v9.4s\n"
      "add v3.4s, v3.4s, v9.4s\n"
      "add v6.4s, v6.4s, v9.4s\n"

      // Store reduced rows.
      "st1 {v0.2s}, [%x[result]], #8\n"
      "st1 {v0.s}[2], [%x[result]], %x[result_stride]\n"
      "sub %x[result], %x[result], #8\n"
      "st1 {v3.2s}, [%x[result]], #8\n"
      "st1 {v3.s}[2], [%x[result]], %x[result_stride]\n"
      "sub %x[result], %x[result], #8\n"
      "st1 {v6.2s}, [%x[result]], #8\n"
      "st1 {v6.s}[2], [%x[result]], %x[result_stride]\n"
      "sub %x[result], %x[result], #8\n"
      : [count] "+r"(count), [result_stride] "+r"(result_stride),
        [rhs] "+r"(rhs), [lhs] "+r"(lhs), [result] "+r"(result)
      :
      : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
        "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "cc", "memory");
}

inline void mul_1x8_4x8_int32_rhsadd(const std::uint8_t* lhs,
                                     const std::uint8_t* rhs,
                                     std::int32_t count, std::int32_t* result,
                                     std::int32_t result_stride) {
  asm volatile(
      "prfm pldl1keep, [%x[lhs]]\n"
      "prfm pldl1keep, [%x[rhs]]\n"
      // Clear aggregators.
      "movi v0.4s, #0\n"
      "movi v1.4s, #0\n"
      "movi v2.4s, #0\n"
      "mov v3.16b, v0.16b\n"

      // General NxM lanes loop.
      "1:"

      // Subtract counter.
      "subs %x[count], %x[count], #8\n"

      "ld1 {v4.8b}, [%x[lhs]], #8\n"
      "ld1 {v5.8b, v6.8b, v7.8b, v8.8b}, [%x[rhs]], #32\n"
      "prfm pldl1keep, [%x[lhs], #64]\n"
      "prfm pldl1keep, [%x[rhs], #64]\n"
      "umull v9.8h, v5.8b, v4.8b\n"
      "umull v10.8h, v6.8b, v4.8b\n"
      "umull v11.8h, v7.8b, v4.8b\n"
      "umull v12.8h, v8.8b, v4.8b\n"
      "uadalp v0.4s, v9.8h\n"
      "uadalp v1.4s, v10.8h\n"
      "uadalp v2.4s, v11.8h\n"
      "uadalp v3.4s, v12.8h\n"

      // Loop break.
      "bne 1b\n"

      "ld1 {v4.4s}, [%x[rhs]]\n"

      // Reduce rows.
      "addp v0.4s, v0.4s, v1.4s\n"
      "addp v2.4s, v2.4s, v3.4s\n"
      "addp v0.4s, v0.4s, v2.4s\n"

      // Add rhs offset to aggregated rows.
      "add v0.4s, v0.4s, v4.4s\n"

      // Store reduced rows.
      "st1 {v0.4s}, [%x[result]], %x[result_stride]\n"
      : [count] "+r"(count), [result_stride] "+r"(result_stride),
        [rhs] "+r"(rhs), [lhs] "+r"(lhs), [result] "+r"(result)
      :
      : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
        "v11", "v12", "cc", "memory");
}

inline void mul_1x8_1x8_int32_lhsadd_rhsadd(const std::uint8_t* lhs,
                                            const std::uint8_t* rhs,
                                            std::int32_t count,
                                            std::int32_t* result,
                                            std::int32_t result_stride) {
  asm volatile(
      "prfm pldl1keep, [%x[lhs]]\n"
      "prfm pldl1keep, [%x[rhs]]\n"
      // Clear aggregators.
      "movi v0.4s, #0\n"

      // General NxM lanes loop.
      "1:"

      // Subtract counter.
      "subs %x[count], %x[count], #8\n"

      "ld1 {v1.8b}, [%x[lhs]], #8\n"
      "ld1 {v2.8b}, [%x[rhs]], #8\n"
      "prfm pldl1keep, [%x[lhs], #64]\n"
      "prfm pldl1keep, [%x[rhs], #64]\n"
      "umull v3.8h, v2.8b, v1.8b\n"
      "uadalp v0.4s, v3.8h\n"

      // Loop break.
      "bne 1b\n"

      "ld1 {v8.2s}, [%x[lhs]]\n"
      "dup v9.2s, v8.s[0]\n"
      "ld1 {v10.2s}, [%x[rhs]]\n"

      // Reduce rows.
      "addp v0.4s, v0.4s, v0.4s\n"
      "addp v1.4s, v0.4s, v0.4s\n"

      // Add lhs offsets to aggregated rows.
      "add v1.2s, v1.2s, v9.2s\n"

      // Add rhs offset to aggregated rows.
      "add v1.2s, v1.2s, v10.2s\n"

      // Store reduced rows.
      "st1 {v1.s}[0], [%x[result]], %x[result_stride]\n"
      : [count] "+r"(count), [result_stride] "+r"(result_stride),
        [rhs] "+r"(rhs), [lhs] "+r"(lhs), [result] "+r"(result)
      :
      : "v0", "v1", "v2", "v3", "v8", "v9", "v10", "cc", "memory");
}

inline void mul_1x8_2x8_int32_lhsadd_rhsadd(const std::uint8_t* lhs,
                                            const std::uint8_t* rhs,
                                            std::int32_t count,
                                            std::int32_t* result,
                                            std::int32_t result_stride) {
  asm volatile(
      "prfm pldl1keep, [%x[lhs]]\n"
      "prfm pldl1keep, [%x[rhs]]\n"
      // Clear aggregators.
      "movi v0.4s, #0\n"
      "movi v1.4s, #0\n"

      // General NxM lanes loop.
      "1:"

      // Subtract counter.
      "subs %x[count], %x[count], #8\n"

      "ld1 {v2.8b}, [%x[lhs]], #8\n"
      "ld1 {v3.8b, v4.8b}, [%x[rhs]], #16\n"
      "prfm pldl1keep, [%x[lhs], #64]\n"
      "prfm pldl1keep, [%x[rhs], #64]\n"
      "umull v5.8h, v3.8b, v2.8b\n"
      "umull v6.8h, v4.8b, v2.8b\n"
      "uadalp v0.4s, v5.8h\n"
      "uadalp v1.4s, v6.8h\n"

      // Loop break.
      "bne 1b\n"

      "ld1 {v8.2s}, [%x[lhs]]\n"
      "dup v9.2s, v8.s[0]\n"
      "ld1 {v10.2s}, [%x[rhs]]\n"

      // Reduce rows.
      "addp v0.4s, v0.4s, v1.4s\n"
      "addp v2.4s, v0.4s, v0.4s\n"

      // Add lhs offsets to aggregated rows.
      "add v2.2s, v2.2s, v9.2s\n"

      // Add rhs offset to aggregated rows.
      "add v2.2s, v2.2s, v10.2s\n"

      // Store reduced rows.
      "st1 {v2.2s}, [%x[result]], %x[result_stride]\n"
      : [count] "+r"(count), [result_stride] "+r"(result_stride),
        [rhs] "+r"(rhs), [lhs] "+r"(lhs), [result] "+r"(result)
      :
      : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v8", "v9", "v10", "cc",
        "memory");
}

inline void mul_1x8_3x8_int32_lhsadd_rhsadd(const std::uint8_t* lhs,
                                            const std::uint8_t* rhs,
                                            std::int32_t count,
                                            std::int32_t* result,
                                            std::int32_t result_stride) {
  asm volatile(
      "prfm pldl1keep, [%x[lhs]]\n"
      "prfm pldl1keep, [%x[rhs]]\n"
      // Clear aggregators.
      "movi v0.4s, #0\n"
      "movi v1.4s, #0\n"
      "movi v2.4s, #0\n"

      // General NxM lanes loop.
      "1:"

      // Subtract counter.
      "subs %x[count], %x[count], #8\n"

      "ld1 {v3.8b}, [%x[lhs]], #8\n"
      "ld1 {v4.8b, v5.8b, v6.8b}, [%x[rhs]], #24\n"
      "prfm pldl1keep, [%x[lhs], #64]\n"
      "prfm pldl1keep, [%x[rhs], #64]\n"
      "umull v7.8h, v4.8b, v3.8b\n"
      "umull v8.8h, v5.8b, v3.8b\n"
      "umull v9.8h, v6.8b, v3.8b\n"
      "uadalp v0.4s, v7.8h\n"
      "uadalp v1.4s, v8.8h\n"
      "uadalp v2.4s, v9.8h\n"

      // Loop break.
      "bne 1b\n"

      "ld1 {v8.2s}, [%x[lhs]]\n"
      "dup v4.4s, v8.s[0]\n"
      "ld1 {v5.4s}, [%x[rhs]]\n"

      // Reduce rows.
      "addp v0.4s, v0.4s, v1.4s\n"
      "addp v2.4s, v2.4s, v2.4s\n"
      "addp v0.4s, v0.4s, v2.4s\n"

      // Add lhs offsets to aggregated rows.
      "add v0.4s, v0.4s, v4.4s\n"

      // Add rhs offset to aggregated rows.
      "add v0.4s, v0.4s, v5.4s\n"

      // Store reduced rows.
      "st1 {v0.2s}, [%x[result]], #8\n"
      "st1 {v0.s}[2], [%x[result]], %x[result_stride]\n"
      "sub %x[result], %x[result], #8\n"
      : [count] "+r"(count), [result_stride] "+r"(result_stride),
        [rhs] "+r"(rhs), [lhs] "+r"(lhs), [result] "+r"(result)
      :
      : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "cc",
        "memory");
}

inline void mul_2x8_1x8_int32_lhsadd_rhsadd(const std::uint8_t* lhs,
                                            const std::uint8_t* rhs,
                                            std::int32_t count,
                                            std::int32_t* result,
                                            std::int32_t result_stride) {
  asm volatile(
      "prfm pldl1keep, [%x[lhs]]\n"
      "prfm pldl1keep, [%x[rhs]]\n"
      // Clear aggregators.
      "movi v0.4s, #0\n"
      "movi v1.4s, #0\n"

      // General NxM lanes loop.
      "1:"

      // Subtract counter.
      "subs %x[count], %x[count], #8\n"

      "ld1 {v2.8b, v3.8b}, [%x[lhs]], #16\n"
      "ld1 {v4.8b}, [%x[rhs]], #8\n"
      "prfm pldl1keep, [%x[lhs], #64]\n"
      "prfm pldl1keep, [%x[rhs], #64]\n"
      "umull v5.8h, v4.8b, v2.8b\n"
      "umull v6.8h, v4.8b, v3.8b\n"
      "uadalp v0.4s, v5.8h\n"
      "uadalp v1.4s, v6.8h\n"

      // Loop break.
      "bne 1b\n"

      "ld1 {v8.2s}, [%x[lhs]]\n"
      "dup v9.2s, v8.s[0]\n"
      "dup v10.2s, v8.s[1]\n"
      "ld1 {v11.2s}, [%x[rhs]]\n"

      // Reduce rows.
      "addp v0.4s, v0.4s, v0.4s\n"
      "addp v2.4s, v0.4s, v0.4s\n"
      "addp v1.4s, v1.4s, v1.4s\n"
      "addp v3.4s, v1.4s, v1.4s\n"

      // Add lhs offsets to aggregated rows.
      "add v2.2s, v2.2s, v9.2s\n"
      "add v3.2s, v3.2s, v10.2s\n"

      // Add rhs offset to aggregated rows.
      "add v2.2s, v2.2s, v11.2s\n"
      "add v3.2s, v3.2s, v11.2s\n"

      // Store reduced rows.
      "st1 {v2.s}[0], [%x[result]], %x[result_stride]\n"
      "st1 {v3.s}[0], [%x[result]], %x[result_stride]\n"
      : [count] "+r"(count), [result_stride] "+r"(result_stride),
        [rhs] "+r"(rhs), [lhs] "+r"(lhs), [result] "+r"(result)
      :
      : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v8", "v9", "v10", "v11",
        "cc", "memory");
}

inline void mul_2x8_2x8_int32_lhsadd_rhsadd(const std::uint8_t* lhs,
                                            const std::uint8_t* rhs,
                                            std::int32_t count,
                                            std::int32_t* result,
                                            std::int32_t result_stride) {
  asm volatile(
      "prfm pldl1keep, [%x[lhs]]\n"
      "prfm pldl1keep, [%x[rhs]]\n"
      // Clear aggregators.
      "movi v0.4s, #0\n"
      "movi v1.4s, #0\n"
      "movi v2.4s, #0\n"
      "mov v3.16b, v0.16b\n"

      // General NxM lanes loop.
      "1:"

      // Subtract counter.
      "subs %x[count], %x[count], #8\n"

      "ld1 {v4.8b, v5.8b}, [%x[lhs]], #16\n"
      "ld1 {v6.8b, v7.8b}, [%x[rhs]], #16\n"
      "prfm pldl1keep, [%x[lhs], #64]\n"
      "prfm pldl1keep, [%x[rhs], #64]\n"
      "umull v8.8h, v6.8b, v4.8b\n"
      "umull v9.8h, v7.8b, v4.8b\n"
      "umull v10.8h, v6.8b, v5.8b\n"
      "umull v11.8h, v7.8b, v5.8b\n"
      "uadalp v0.4s, v8.8h\n"
      "uadalp v1.4s, v9.8h\n"
      "uadalp v2.4s, v10.8h\n"
      "uadalp v3.4s, v11.8h\n"

      // Loop break.
      "bne 1b\n"

      "ld1 {v8.2s}, [%x[lhs]]\n"
      "dup v9.2s, v8.s[0]\n"
      "dup v10.2s, v8.s[1]\n"
      "ld1 {v11.2s}, [%x[rhs]]\n"

      // Reduce rows.
      "addp v0.4s, v0.4s, v1.4s\n"
      "addp v4.4s, v0.4s, v0.4s\n"
      "addp v2.4s, v2.4s, v3.4s\n"
      "addp v5.4s, v2.4s, v2.4s\n"

      // Add lhs offsets to aggregated rows.
      "add v4.2s, v4.2s, v9.2s\n"
      "add v5.2s, v5.2s, v10.2s\n"

      // Add rhs offset to aggregated rows.
      "add v4.2s, v4.2s, v11.2s\n"
      "add v5.2s, v5.2s, v11.2s\n"

      // Store reduced rows.
      "st1 {v4.2s}, [%x[result]], %x[result_stride]\n"
      "st1 {v5.2s}, [%x[result]], %x[result_stride]\n"
      : [count] "+r"(count), [result_stride] "+r"(result_stride),
        [rhs] "+r"(rhs), [lhs] "+r"(lhs), [result] "+r"(result)
      :
      : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
        "v11", "cc", "memory");
}

inline void mul_2x8_3x8_int32_lhsadd_rhsadd(const std::uint8_t* lhs,
                                            const std::uint8_t* rhs,
                                            std::int32_t count,
                                            std::int32_t* result,
                                            std::int32_t result_stride) {
  asm volatile(
      "prfm pldl1keep, [%x[lhs]]\n"
      "prfm pldl1keep, [%x[rhs]]\n"
      // Clear aggregators.
      "movi v0.4s, #0\n"
      "movi v1.4s, #0\n"
      "movi v2.4s, #0\n"
      "mov v3.16b, v0.16b\n"
      "mov v4.16b, v1.16b\n"
      "mov v5.16b, v2.16b\n"

      // General NxM lanes loop.
      "1:"

      // Subtract counter.
      "subs %x[count], %x[count], #8\n"

      "ld1 {v6.8b, v7.8b}, [%x[lhs]], #16\n"
      "ld1 {v8.8b, v9.8b, v10.8b}, [%x[rhs]], #24\n"
      "prfm pldl1keep, [%x[lhs], #64]\n"
      "prfm pldl1keep, [%x[rhs], #64]\n"
      "umull v11.8h, v8.8b, v6.8b\n"
      "umull v12.8h, v9.8b, v6.8b\n"
      "umull v13.8h, v10.8b, v6.8b\n"
      "umull v14.8h, v8.8b, v7.8b\n"
      "umull v15.8h, v9.8b, v7.8b\n"
      "umull v16.8h, v10.8b, v7.8b\n"
      "uadalp v0.4s, v11.8h\n"
      "uadalp v1.4s, v12.8h\n"
      "uadalp v2.4s, v13.8h\n"
      "uadalp v3.4s, v14.8h\n"
      "uadalp v4.4s, v15.8h\n"
      "uadalp v5.4s, v16.8h\n"

      // Loop break.
      "bne 1b\n"

      "ld1 {v8.2s}, [%x[lhs]]\n"
      "dup v6.4s, v8.s[0]\n"
      "dup v7.4s, v8.s[1]\n"
      "ld1 {v9.4s}, [%x[rhs]]\n"

      // Reduce rows.
      "addp v0.4s, v0.4s, v1.4s\n"
      "addp v2.4s, v2.4s, v2.4s\n"
      "addp v0.4s, v0.4s, v2.4s\n"
      "addp v3.4s, v3.4s, v4.4s\n"
      "addp v5.4s, v5.4s, v5.4s\n"
      "addp v3.4s, v3.4s, v5.4s\n"

      // Add lhs offsets to aggregated rows.
      "add v0.4s, v0.4s, v6.4s\n"
      "add v3.4s, v3.4s, v7.4s\n"

      // Add rhs offset to aggregated rows.
      "add v0.4s, v0.4s, v9.4s\n"
      "add v3.4s, v3.4s, v9.4s\n"

      // Store reduced rows.
      "st1 {v0.2s}, [%x[result]], #8\n"
      "st1 {v0.s}[2], [%x[result]], %x[result_stride]\n"
      "sub %x[result], %x[result], #8\n"
      "st1 {v3.2s}, [%x[result]], #8\n"
      "st1 {v3.s}[2], [%x[result]], %x[result_stride]\n"
      "sub %x[result], %x[result], #8\n"
      : [count] "+r"(count), [result_stride] "+r"(result_stride),
        [rhs] "+r"(rhs), [lhs] "+r"(lhs), [result] "+r"(result)
      :
      : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
        "v11", "v12", "v13", "v14", "v15", "v16", "cc", "memory");
}

inline void mul_3x8_1x8_int32_lhsadd_rhsadd(const std::uint8_t* lhs,
                                            const std::uint8_t* rhs,
                                            std::int32_t count,
                                            std::int32_t* result,
                                            std::int32_t result_stride) {
  asm volatile(
      "prfm pldl1keep, [%x[lhs]]\n"
      "prfm pldl1keep, [%x[rhs]]\n"
      // Clear aggregators.
      "movi v0.4s, #0\n"
      "movi v1.4s, #0\n"
      "movi v2.4s, #0\n"

      // General NxM lanes loop.
      "1:"

      // Subtract counter.
      "subs %x[count], %x[count], #8\n"

      "ld1 {v3.8b, v4.8b, v5.8b}, [%x[lhs]], #24\n"
      "ld1 {v6.8b}, [%x[rhs]], #8\n"
      "prfm pldl1keep, [%x[lhs], #64]\n"
      "prfm pldl1keep, [%x[rhs], #64]\n"
      "umull v7.8h, v6.8b, v3.8b\n"
      "umull v8.8h, v6.8b, v4.8b\n"
      "umull v9.8h, v6.8b, v5.8b\n"
      "uadalp v0.4s, v7.8h\n"
      "uadalp v1.4s, v8.8h\n"
      "uadalp v2.4s, v9.8h\n"

      // Loop break.
      "bne 1b\n"

      "ld1 {v4.4s}, [%x[lhs]]\n"
      "dup v8.2s, v4.s[0]\n"
      "dup v9.2s, v4.s[1]\n"
      "dup v10.2s, v4.s[2]\n"
      "ld1 {v11.2s}, [%x[rhs]]\n"

      // Reduce rows.
      "addp v0.4s, v0.4s, v0.4s\n"
      "addp v3.4s, v0.4s, v0.4s\n"
      "addp v1.4s, v1.4s, v1.4s\n"
      "addp v5.4s, v1.4s, v1.4s\n"
      "addp v2.4s, v2.4s, v2.4s\n"
      "addp v6.4s, v2.4s, v2.4s\n"

      // Add lhs offsets to aggregated rows.
      "add v3.2s, v3.2s, v8.2s\n"
      "add v5.2s, v5.2s, v9.2s\n"
      "add v6.2s, v6.2s, v10.2s\n"

      // Add rhs offset to aggregated rows.
      "add v3.2s, v3.2s, v11.2s\n"
      "add v5.2s, v5.2s, v11.2s\n"
      "add v6.2s, v6.2s, v11.2s\n"

      // Store reduced rows.
      "st1 {v3.s}[0], [%x[result]], %x[result_stride]\n"
      "st1 {v5.s}[0], [%x[result]], %x[result_stride]\n"
      "st1 {v6.s}[0], [%x[result]], %x[result_stride]\n"
      : [count] "+r"(count), [result_stride] "+r"(result_stride),
        [rhs] "+r"(rhs), [lhs] "+r"(lhs), [result] "+r"(result)
      :
      : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
        "v11", "cc", "memory");
}

inline void mul_3x8_2x8_int32_lhsadd_rhsadd(const std::uint8_t* lhs,
                                            const std::uint8_t* rhs,
                                            std::int32_t count,
                                            std::int32_t* result,
                                            std::int32_t result_stride) {
  asm volatile(
      "prfm pldl1keep, [%x[lhs]]\n"
      "prfm pldl1keep, [%x[rhs]]\n"
      // Clear aggregators.
      "movi v0.4s, #0\n"
      "movi v1.4s, #0\n"
      "movi v2.4s, #0\n"
      "mov v3.16b, v0.16b\n"
      "mov v4.16b, v1.16b\n"
      "mov v5.16b, v2.16b\n"

      // General NxM lanes loop.
      "1:"

      // Subtract counter.
      "subs %x[count], %x[count], #8\n"

      "ld1 {v6.8b, v7.8b, v8.8b}, [%x[lhs]], #24\n"
      "ld1 {v9.8b, v10.8b}, [%x[rhs]], #16\n"
      "prfm pldl1keep, [%x[lhs], #64]\n"
      "prfm pldl1keep, [%x[rhs], #64]\n"
      "umull v11.8h, v9.8b, v6.8b\n"
      "umull v12.8h, v10.8b, v6.8b\n"
      "umull v13.8h, v9.8b, v7.8b\n"
      "umull v14.8h, v10.8b, v7.8b\n"
      "umull v15.8h, v9.8b, v8.8b\n"
      "umull v16.8h, v10.8b, v8.8b\n"
      "uadalp v0.4s, v11.8h\n"
      "uadalp v1.4s, v12.8h\n"
      "uadalp v2.4s, v13.8h\n"
      "uadalp v3.4s, v14.8h\n"
      "uadalp v4.4s, v15.8h\n"
      "uadalp v5.4s, v16.8h\n"

      // Loop break.
      "bne 1b\n"

      "ld1 {v6.4s}, [%x[lhs]]\n"
      "dup v8.2s, v6.s[0]\n"
      "dup v9.2s, v6.s[1]\n"
      "dup v10.2s, v6.s[2]\n"
      "ld1 {v11.2s}, [%x[rhs]]\n"

      // Reduce rows.
      "addp v0.4s, v0.4s, v1.4s\n"
      "addp v7.4s, v0.4s, v0.4s\n"
      "addp v2.4s, v2.4s, v3.4s\n"
      "addp v12.4s, v2.4s, v2.4s\n"
      "addp v4.4s, v4.4s, v5.4s\n"
      "addp v13.4s, v4.4s, v4.4s\n"

      // Add lhs offsets to aggregated rows.
      "add v7.2s, v7.2s, v8.2s\n"
      "add v12.2s, v12.2s, v9.2s\n"
      "add v13.2s, v13.2s, v10.2s\n"

      // Add rhs offset to aggregated rows.
      "add v7.2s, v7.2s, v11.2s\n"
      "add v12.2s, v12.2s, v11.2s\n"
      "add v13.2s, v13.2s, v11.2s\n"

      // Store reduced rows.
      "st1 {v7.2s}, [%x[result]], %x[result_stride]\n"
      "st1 {v12.2s}, [%x[result]], %x[result_stride]\n"
      "st1 {v13.2s}, [%x[result]], %x[result_stride]\n"
      : [count] "+r"(count), [result_stride] "+r"(result_stride),
        [rhs] "+r"(rhs), [lhs] "+r"(lhs), [result] "+r"(result)
      :
      : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
        "v11", "v12", "v13", "v14", "v15", "v16", "cc", "memory");
}

inline void mul_3x8_3x8_int32_lhsadd_rhsadd(const std::uint8_t* lhs,
                                            const std::uint8_t* rhs,
                                            std::int32_t count,
                                            std::int32_t* result,
                                            std::int32_t result_stride) {
  asm volatile(
      "prfm pldl1keep, [%x[lhs]]\n"
      "prfm pldl1keep, [%x[rhs]]\n"
      // Clear aggregators.
      "movi v0.4s, #0\n"
      "movi v1.4s, #0\n"
      "movi v2.4s, #0\n"
      "mov v3.16b, v0.16b\n"
      "mov v4.16b, v1.16b\n"
      "mov v5.16b, v2.16b\n"
      "mov v6.16b, v3.16b\n"
      "mov v7.16b, v4.16b\n"
      "mov v8.16b, v5.16b\n"

      // 3x3 lanes loop.
      "1:"

      "ld1 {v12.8b, v13.8b, v14.8b}, [%x[rhs]], #24\n"
      "ld1 {v9.8b}, [%x[lhs]], #8\n"
      "umull v15.8h, v9.8b, v12.8b\n"
      "ld1 {v10.8b}, [%x[lhs]], #8\n"
      "umull v16.8h, v9.8b, v13.8b\n"
      "ld1 {v11.8b}, [%x[lhs]], #8\n"
      "umull v17.8h, v9.8b, v14.8b\n"
      "prfm pldl1keep, [%x[lhs], #64]\n"
      "umull v18.8h, v10.8b, v12.8b\n"
      "prfm pldl1keep, [%x[rhs], #64]\n"
      "uadalp v0.4s, v15.8h\n"
      "uadalp v1.4s, v16.8h\n"
      "uadalp v2.4s, v17.8h\n"
      "uadalp v3.4s, v18.8h\n"
      "umull v15.8h, v10.8b, v13.8b\n"
      "umull v16.8h, v10.8b, v14.8b\n"
      "umull v17.8h, v11.8b, v12.8b\n"
      "umull v18.8h, v11.8b, v13.8b\n"

      // Subtract counter.
      "subs %x[count], %x[count], #8\n"

      "umull v9.8h, v11.8b, v14.8b\n"
      "uadalp v4.4s, v15.8h\n"
      "uadalp v5.4s, v16.8h\n"
      "uadalp v6.4s, v17.8h\n"
      "uadalp v7.4s, v18.8h\n"
      "uadalp v8.4s, v9.8h\n"

      // Loop break.
      "bne 1b\n"

      "ld1 {v9.4s}, [%x[lhs]]\n"
      "dup v10.4s, v9.s[0]\n"
      "dup v11.4s, v9.s[1]\n"
      "dup v12.4s, v9.s[2]\n"
      "ld1 {v13.4s}, [%x[rhs]]\n"

      // Reduce rows.
      "addp v0.4s, v0.4s, v1.4s\n"
      "addp v2.4s, v2.4s, v2.4s\n"
      "addp v0.4s, v0.4s, v2.4s\n"
      "addp v3.4s, v3.4s, v4.4s\n"
      "addp v5.4s, v5.4s, v5.4s\n"
      "addp v3.4s, v3.4s, v5.4s\n"
      "addp v6.4s, v6.4s, v7.4s\n"
      "addp v8.4s, v8.4s, v8.4s\n"
      "addp v6.4s, v6.4s, v8.4s\n"

      // Add lhs offsets to aggregated rows.
      "add v0.4s, v0.4s, v10.4s\n"
      "add v3.4s, v3.4s, v11.4s\n"
      "add v6.4s, v6.4s, v12.4s\n"

      // Add rhs offset to aggregated rows.
      "add v0.4s, v0.4s, v13.4s\n"
      "add v3.4s, v3.4s, v13.4s\n"
      "add v6.4s, v6.4s, v13.4s\n"

      // Store reduced rows.
      "st1 {v0.2s}, [%x[result]], #8\n"
      "st1 {v0.s}[2], [%x[result]], %x[result_stride]\n"
      "sub %x[result], %x[result], #8\n"
      "st1 {v3.2s}, [%x[result]], #8\n"
      "st1 {v3.s}[2], [%x[result]], %x[result_stride]\n"
      "sub %x[result], %x[result], #8\n"
      "st1 {v6.2s}, [%x[result]], #8\n"
      "st1 {v6.s}[2], [%x[result]], %x[result_stride]\n"
      "sub %x[result], %x[result], #8\n"
      : [count] "+r"(count), [result_stride] "+r"(result_stride),
        [rhs] "+r"(rhs), [lhs] "+r"(lhs), [result] "+r"(result)
      :
      : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
        "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "cc", "memory");
}

inline void mul_1x8_4x8_int32_lhsadd_rhsadd(const std::uint8_t* lhs,
                                            const std::uint8_t* rhs,
                                            std::int32_t count,
                                            std::int32_t* result,
                                            std::int32_t result_stride) {
  asm volatile(
      "prfm pldl1keep, [%x[lhs]]\n"
      "prfm pldl1keep, [%x[rhs]]\n"
      // Clear aggregators.
      "movi v0.4s, #0\n"
      "movi v1.4s, #0\n"
      "movi v2.4s, #0\n"
      "mov v3.16b, v0.16b\n"

      // General NxM lanes loop.
      "1:"

      // Subtract counter.
      "subs %x[count], %x[count], #8\n"

      "ld1 {v4.8b}, [%x[lhs]], #8\n"
      "ld1 {v5.8b, v6.8b, v7.8b, v8.8b}, [%x[rhs]], #32\n"
      "prfm pldl1keep, [%x[lhs], #64]\n"
      "prfm pldl1keep, [%x[rhs], #64]\n"
      "umull v9.8h, v5.8b, v4.8b\n"
      "umull v10.8h, v6.8b, v4.8b\n"
      "umull v11.8h, v7.8b, v4.8b\n"
      "umull v12.8h, v8.8b, v4.8b\n"
      "uadalp v0.4s, v9.8h\n"
      "uadalp v1.4s, v10.8h\n"
      "uadalp v2.4s, v11.8h\n"
      "uadalp v3.4s, v12.8h\n"

      // Loop break.
      "bne 1b\n"

      "ld1 {v8.2s}, [%x[lhs]]\n"
      "dup v4.4s, v8.s[0]\n"
      "ld1 {v5.4s}, [%x[rhs]]\n"

      // Reduce rows.
      "addp v0.4s, v0.4s, v1.4s\n"
      "addp v2.4s, v2.4s, v3.4s\n"
      "addp v0.4s, v0.4s, v2.4s\n"

      // Add lhs offsets to aggregated rows.
      "add v0.4s, v0.4s, v4.4s\n"

      // Add rhs offset to aggregated rows.
      "add v0.4s, v0.4s, v5.4s\n"

      // Store reduced rows.
      "st1 {v0.4s}, [%x[result]], %x[result_stride]\n"
      : [count] "+r"(count), [result_stride] "+r"(result_stride),
        [rhs] "+r"(rhs), [lhs] "+r"(lhs), [result] "+r"(result)
      :
      : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
        "v11", "v12", "cc", "memory");
}

inline void mul_1x8_1x8_float_lhsadd_rhsadd(const std::uint8_t* lhs,
                                            const std::uint8_t* rhs,
                                            std::int32_t count, float* result,
                                            std::int32_t result_stride,
                                            float result_scale) {
  asm volatile(
      "prfm pldl1keep, [%x[lhs]]\n"
      "prfm pldl1keep, [%x[rhs]]\n"
      // Clear aggregators.
      "movi v0.4s, #0\n"

      // General NxM lanes loop.
      "1:"

      // Subtract counter.
      "subs %x[count], %x[count], #8\n"

      "ld1 {v1.8b}, [%x[lhs]], #8\n"
      "ld1 {v2.8b}, [%x[rhs]], #8\n"
      "prfm pldl1keep, [%x[lhs], #64]\n"
      "prfm pldl1keep, [%x[rhs], #64]\n"
      "umull v3.8h, v2.8b, v1.8b\n"
      "uadalp v0.4s, v3.8h\n"

      // Loop break.
      "bne 1b\n"

      "ld1 {v8.2s}, [%x[lhs]]\n"
      "dup v9.2s, v8.s[0]\n"
      "ld1 {v10.2s}, [%x[rhs]]\n"
      "dup v11.2s, %w[result_scale]\n"

      // Reduce rows.
      "addp v0.4s, v0.4s, v0.4s\n"
      "addp v1.4s, v0.4s, v0.4s\n"

      // Add lhs offsets to aggregated rows.
      "add v1.2s, v1.2s, v9.2s\n"

      // Add rhs offset to aggregated rows.
      "add v1.2s, v1.2s, v10.2s\n"

      // Convert to float. Multiply by result scale.
      "scvtf v1.2s, v1.2s\n"
      "fmul v1.2s, v1.2s, v11.2s\n"

      // Store reduced rows.
      "st1 {v1.s}[0], [%x[result]], %x[result_stride]\n"
      : [count] "+r"(count), [result_scale] "+r"(result_scale),
        [result_stride] "+r"(result_stride), [rhs] "+r"(rhs), [lhs] "+r"(lhs),
        [result] "+r"(result)
      :
      : "v0", "v1", "v2", "v3", "v8", "v9", "v10", "v11", "cc", "memory");
}

inline void mul_1x8_2x8_float_lhsadd_rhsadd(const std::uint8_t* lhs,
                                            const std::uint8_t* rhs,
                                            std::int32_t count, float* result,
                                            std::int32_t result_stride,
                                            float result_scale) {
  asm volatile(
      "prfm pldl1keep, [%x[lhs]]\n"
      "prfm pldl1keep, [%x[rhs]]\n"
      // Clear aggregators.
      "movi v0.4s, #0\n"
      "movi v1.4s, #0\n"

      // General NxM lanes loop.
      "1:"

      // Subtract counter.
      "subs %x[count], %x[count], #8\n"

      "ld1 {v2.8b}, [%x[lhs]], #8\n"
      "ld1 {v3.8b, v4.8b}, [%x[rhs]], #16\n"
      "prfm pldl1keep, [%x[lhs], #64]\n"
      "prfm pldl1keep, [%x[rhs], #64]\n"
      "umull v5.8h, v3.8b, v2.8b\n"
      "umull v6.8h, v4.8b, v2.8b\n"
      "uadalp v0.4s, v5.8h\n"
      "uadalp v1.4s, v6.8h\n"

      // Loop break.
      "bne 1b\n"

      "ld1 {v8.2s}, [%x[lhs]]\n"
      "dup v9.2s, v8.s[0]\n"
      "ld1 {v10.2s}, [%x[rhs]]\n"
      "dup v11.2s, %w[result_scale]\n"

      // Reduce rows.
      "addp v0.4s, v0.4s, v1.4s\n"
      "addp v2.4s, v0.4s, v0.4s\n"

      // Add lhs offsets to aggregated rows.
      "add v2.2s, v2.2s, v9.2s\n"

      // Add rhs offset to aggregated rows.
      "add v2.2s, v2.2s, v10.2s\n"

      // Convert to float. Multiply by result scale.
      "scvtf v2.2s, v2.2s\n"
      "fmul v2.2s, v2.2s, v11.2s\n"

      // Store reduced rows.
      "st1 {v2.2s}, [%x[result]], %x[result_stride]\n"
      : [count] "+r"(count), [result_scale] "+r"(result_scale),
        [result_stride] "+r"(result_stride), [rhs] "+r"(rhs), [lhs] "+r"(lhs),
        [result] "+r"(result)
      :
      : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v8", "v9", "v10", "v11",
        "cc", "memory");
}

inline void mul_1x8_3x8_float_lhsadd_rhsadd(const std::uint8_t* lhs,
                                            const std::uint8_t* rhs,
                                            std::int32_t count, float* result,
                                            std::int32_t result_stride,
                                            float result_scale) {
  asm volatile(
      "prfm pldl1keep, [%x[lhs]]\n"
      "prfm pldl1keep, [%x[rhs]]\n"
      // Clear aggregators.
      "movi v0.4s, #0\n"
      "movi v1.4s, #0\n"
      "movi v2.4s, #0\n"

      // General NxM lanes loop.
      "1:"

      // Subtract counter.
      "subs %x[count], %x[count], #8\n"

      "ld1 {v3.8b}, [%x[lhs]], #8\n"
      "ld1 {v4.8b, v5.8b, v6.8b}, [%x[rhs]], #24\n"
      "prfm pldl1keep, [%x[lhs], #64]\n"
      "prfm pldl1keep, [%x[rhs], #64]\n"
      "umull v7.8h, v4.8b, v3.8b\n"
      "umull v8.8h, v5.8b, v3.8b\n"
      "umull v9.8h, v6.8b, v3.8b\n"
      "uadalp v0.4s, v7.8h\n"
      "uadalp v1.4s, v8.8h\n"
      "uadalp v2.4s, v9.8h\n"

      // Loop break.
      "bne 1b\n"

      "ld1 {v8.2s}, [%x[lhs]]\n"
      "dup v4.4s, v8.s[0]\n"
      "ld1 {v5.4s}, [%x[rhs]]\n"
      "dup v6.4s, %w[result_scale]\n"

      // Reduce rows.
      "addp v0.4s, v0.4s, v1.4s\n"
      "addp v2.4s, v2.4s, v2.4s\n"
      "addp v0.4s, v0.4s, v2.4s\n"

      // Add lhs offsets to aggregated rows.
      "add v0.4s, v0.4s, v4.4s\n"

      // Add rhs offset to aggregated rows.
      "add v0.4s, v0.4s, v5.4s\n"

      // Convert to float. Multiply by result scale.
      "scvtf v0.4s, v0.4s\n"
      "fmul v0.4s, v0.4s, v6.4s\n"

      // Store reduced rows.
      "st1 {v0.2s}, [%x[result]], #8\n"
      "st1 {v0.s}[2], [%x[result]], %x[result_stride]\n"
      "sub %x[result], %x[result], #8\n"
      : [count] "+r"(count), [result_scale] "+r"(result_scale),
        [result_stride] "+r"(result_stride), [rhs] "+r"(rhs), [lhs] "+r"(lhs),
        [result] "+r"(result)
      :
      : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "cc",
        "memory");
}

inline void mul_2x8_1x8_float_lhsadd_rhsadd(const std::uint8_t* lhs,
                                            const std::uint8_t* rhs,
                                            std::int32_t count, float* result,
                                            std::int32_t result_stride,
                                            float result_scale) {
  asm volatile(
      "prfm pldl1keep, [%x[lhs]]\n"
      "prfm pldl1keep, [%x[rhs]]\n"
      // Clear aggregators.
      "movi v0.4s, #0\n"
      "movi v1.4s, #0\n"

      // General NxM lanes loop.
      "1:"

      // Subtract counter.
      "subs %x[count], %x[count], #8\n"

      "ld1 {v2.8b, v3.8b}, [%x[lhs]], #16\n"
      "ld1 {v4.8b}, [%x[rhs]], #8\n"
      "prfm pldl1keep, [%x[lhs], #64]\n"
      "prfm pldl1keep, [%x[rhs], #64]\n"
      "umull v5.8h, v4.8b, v2.8b\n"
      "umull v6.8h, v4.8b, v3.8b\n"
      "uadalp v0.4s, v5.8h\n"
      "uadalp v1.4s, v6.8h\n"

      // Loop break.
      "bne 1b\n"

      "ld1 {v8.2s}, [%x[lhs]]\n"
      "dup v9.2s, v8.s[0]\n"
      "dup v10.2s, v8.s[1]\n"
      "ld1 {v11.2s}, [%x[rhs]]\n"
      "dup v12.2s, %w[result_scale]\n"

      // Reduce rows.
      "addp v0.4s, v0.4s, v0.4s\n"
      "addp v2.4s, v0.4s, v0.4s\n"
      "addp v1.4s, v1.4s, v1.4s\n"
      "addp v3.4s, v1.4s, v1.4s\n"

      // Add lhs offsets to aggregated rows.
      "add v2.2s, v2.2s, v9.2s\n"
      "add v3.2s, v3.2s, v10.2s\n"

      // Add rhs offset to aggregated rows.
      "add v2.2s, v2.2s, v11.2s\n"
      "add v3.2s, v3.2s, v11.2s\n"

      // Convert to float. Multiply by result scale.
      "scvtf v2.2s, v2.2s\n"
      "scvtf v3.2s, v3.2s\n"
      "fmul v2.2s, v2.2s, v12.2s\n"
      "fmul v3.2s, v3.2s, v12.2s\n"

      // Store reduced rows.
      "st1 {v2.s}[0], [%x[result]], %x[result_stride]\n"
      "st1 {v3.s}[0], [%x[result]], %x[result_stride]\n"
      : [count] "+r"(count), [result_scale] "+r"(result_scale),
        [result_stride] "+r"(result_stride), [rhs] "+r"(rhs), [lhs] "+r"(lhs),
        [result] "+r"(result)
      :
      : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v8", "v9", "v10", "v11",
        "v12", "cc", "memory");
}

inline void mul_2x8_2x8_float_lhsadd_rhsadd(const std::uint8_t* lhs,
                                            const std::uint8_t* rhs,
                                            std::int32_t count, float* result,
                                            std::int32_t result_stride,
                                            float result_scale) {
  asm volatile(
      "prfm pldl1keep, [%x[lhs]]\n"
      "prfm pldl1keep, [%x[rhs]]\n"
      // Clear aggregators.
      "movi v0.4s, #0\n"
      "movi v1.4s, #0\n"
      "movi v2.4s, #0\n"
      "mov v3.16b, v0.16b\n"

      // General NxM lanes loop.
      "1:"

      // Subtract counter.
      "subs %x[count], %x[count], #8\n"

      "ld1 {v4.8b, v5.8b}, [%x[lhs]], #16\n"
      "ld1 {v6.8b, v7.8b}, [%x[rhs]], #16\n"
      "prfm pldl1keep, [%x[lhs], #64]\n"
      "prfm pldl1keep, [%x[rhs], #64]\n"
      "umull v8.8h, v6.8b, v4.8b\n"
      "umull v9.8h, v7.8b, v4.8b\n"
      "umull v10.8h, v6.8b, v5.8b\n"
      "umull v11.8h, v7.8b, v5.8b\n"
      "uadalp v0.4s, v8.8h\n"
      "uadalp v1.4s, v9.8h\n"
      "uadalp v2.4s, v10.8h\n"
      "uadalp v3.4s, v11.8h\n"

      // Loop break.
      "bne 1b\n"

      "ld1 {v8.2s}, [%x[lhs]]\n"
      "dup v9.2s, v8.s[0]\n"
      "dup v10.2s, v8.s[1]\n"
      "ld1 {v11.2s}, [%x[rhs]]\n"
      "dup v12.2s, %w[result_scale]\n"

      // Reduce rows.
      "addp v0.4s, v0.4s, v1.4s\n"
      "addp v4.4s, v0.4s, v0.4s\n"
      "addp v2.4s, v2.4s, v3.4s\n"
      "addp v5.4s, v2.4s, v2.4s\n"

      // Add lhs offsets to aggregated rows.
      "add v4.2s, v4.2s, v9.2s\n"
      "add v5.2s, v5.2s, v10.2s\n"

      // Add rhs offset to aggregated rows.
      "add v4.2s, v4.2s, v11.2s\n"
      "add v5.2s, v5.2s, v11.2s\n"

      // Convert to float. Multiply by result scale.
      "scvtf v4.2s, v4.2s\n"
      "scvtf v5.2s, v5.2s\n"
      "fmul v4.2s, v4.2s, v12.2s\n"
      "fmul v5.2s, v5.2s, v12.2s\n"

      // Store reduced rows.
      "st1 {v4.2s}, [%x[result]], %x[result_stride]\n"
      "st1 {v5.2s}, [%x[result]], %x[result_stride]\n"
      : [count] "+r"(count), [result_scale] "+r"(result_scale),
        [result_stride] "+r"(result_stride), [rhs] "+r"(rhs), [lhs] "+r"(lhs),
        [result] "+r"(result)
      :
      : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
        "v11", "v12", "cc", "memory");
}

inline void mul_2x8_3x8_float_lhsadd_rhsadd(const std::uint8_t* lhs,
                                            const std::uint8_t* rhs,
                                            std::int32_t count, float* result,
                                            std::int32_t result_stride,
                                            float result_scale) {
  asm volatile(
      "prfm pldl1keep, [%x[lhs]]\n"
      "prfm pldl1keep, [%x[rhs]]\n"
      // Clear aggregators.
      "movi v0.4s, #0\n"
      "movi v1.4s, #0\n"
      "movi v2.4s, #0\n"
      "mov v3.16b, v0.16b\n"
      "mov v4.16b, v1.16b\n"
      "mov v5.16b, v2.16b\n"

      // General NxM lanes loop.
      "1:"

      // Subtract counter.
      "subs %x[count], %x[count], #8\n"

      "ld1 {v6.8b, v7.8b}, [%x[lhs]], #16\n"
      "ld1 {v8.8b, v9.8b, v10.8b}, [%x[rhs]], #24\n"
      "prfm pldl1keep, [%x[lhs], #64]\n"
      "prfm pldl1keep, [%x[rhs], #64]\n"
      "umull v11.8h, v8.8b, v6.8b\n"
      "umull v12.8h, v9.8b, v6.8b\n"
      "umull v13.8h, v10.8b, v6.8b\n"
      "umull v14.8h, v8.8b, v7.8b\n"
      "umull v15.8h, v9.8b, v7.8b\n"
      "umull v16.8h, v10.8b, v7.8b\n"
      "uadalp v0.4s, v11.8h\n"
      "uadalp v1.4s, v12.8h\n"
      "uadalp v2.4s, v13.8h\n"
      "uadalp v3.4s, v14.8h\n"
      "uadalp v4.4s, v15.8h\n"
      "uadalp v5.4s, v16.8h\n"

      // Loop break.
      "bne 1b\n"

      "ld1 {v8.2s}, [%x[lhs]]\n"
      "dup v6.4s, v8.s[0]\n"
      "dup v7.4s, v8.s[1]\n"
      "ld1 {v9.4s}, [%x[rhs]]\n"
      "dup v10.4s, %w[result_scale]\n"

      // Reduce rows.
      "addp v0.4s, v0.4s, v1.4s\n"
      "addp v2.4s, v2.4s, v2.4s\n"
      "addp v0.4s, v0.4s, v2.4s\n"
      "addp v3.4s, v3.4s, v4.4s\n"
      "addp v5.4s, v5.4s, v5.4s\n"
      "addp v3.4s, v3.4s, v5.4s\n"

      // Add lhs offsets to aggregated rows.
      "add v0.4s, v0.4s, v6.4s\n"
      "add v3.4s, v3.4s, v7.4s\n"

      // Add rhs offset to aggregated rows.
      "add v0.4s, v0.4s, v9.4s\n"
      "add v3.4s, v3.4s, v9.4s\n"

      // Convert to float. Multiply by result scale.
      "scvtf v0.4s, v0.4s\n"
      "scvtf v3.4s, v3.4s\n"
      "fmul v0.4s, v0.4s, v10.4s\n"
      "fmul v3.4s, v3.4s, v10.4s\n"

      // Store reduced rows.
      "st1 {v0.2s}, [%x[result]], #8\n"
      "st1 {v0.s}[2], [%x[result]], %x[result_stride]\n"
      "sub %x[result], %x[result], #8\n"
      "st1 {v3.2s}, [%x[result]], #8\n"
      "st1 {v3.s}[2], [%x[result]], %x[result_stride]\n"
      "sub %x[result], %x[result], #8\n"
      : [count] "+r"(count), [result_scale] "+r"(result_scale),
        [result_stride] "+r"(result_stride), [rhs] "+r"(rhs), [lhs] "+r"(lhs),
        [result] "+r"(result)
      :
      : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
        "v11", "v12", "v13", "v14", "v15", "v16", "cc", "memory");
}

inline void mul_3x8_1x8_float_lhsadd_rhsadd(const std::uint8_t* lhs,
                                            const std::uint8_t* rhs,
                                            std::int32_t count, float* result,
                                            std::int32_t result_stride,
                                            float result_scale) {
  asm volatile(
      "prfm pldl1keep, [%x[lhs]]\n"
      "prfm pldl1keep, [%x[rhs]]\n"
      // Clear aggregators.
      "movi v0.4s, #0\n"
      "movi v1.4s, #0\n"
      "movi v2.4s, #0\n"

      // General NxM lanes loop.
      "1:"

      // Subtract counter.
      "subs %x[count], %x[count], #8\n"

      "ld1 {v3.8b, v4.8b, v5.8b}, [%x[lhs]], #24\n"
      "ld1 {v6.8b}, [%x[rhs]], #8\n"
      "prfm pldl1keep, [%x[lhs], #64]\n"
      "prfm pldl1keep, [%x[rhs], #64]\n"
      "umull v7.8h, v6.8b, v3.8b\n"
      "umull v8.8h, v6.8b, v4.8b\n"
      "umull v9.8h, v6.8b, v5.8b\n"
      "uadalp v0.4s, v7.8h\n"
      "uadalp v1.4s, v8.8h\n"
      "uadalp v2.4s, v9.8h\n"

      // Loop break.
      "bne 1b\n"

      "ld1 {v4.4s}, [%x[lhs]]\n"
      "dup v8.2s, v4.s[0]\n"
      "dup v9.2s, v4.s[1]\n"
      "dup v10.2s, v4.s[2]\n"
      "ld1 {v11.2s}, [%x[rhs]]\n"
      "dup v12.2s, %w[result_scale]\n"

      // Reduce rows.
      "addp v0.4s, v0.4s, v0.4s\n"
      "addp v3.4s, v0.4s, v0.4s\n"
      "addp v1.4s, v1.4s, v1.4s\n"
      "addp v5.4s, v1.4s, v1.4s\n"
      "addp v2.4s, v2.4s, v2.4s\n"
      "addp v6.4s, v2.4s, v2.4s\n"

      // Add lhs offsets to aggregated rows.
      "add v3.2s, v3.2s, v8.2s\n"
      "add v5.2s, v5.2s, v9.2s\n"
      "add v6.2s, v6.2s, v10.2s\n"

      // Add rhs offset to aggregated rows.
      "add v3.2s, v3.2s, v11.2s\n"
      "add v5.2s, v5.2s, v11.2s\n"
      "add v6.2s, v6.2s, v11.2s\n"

      // Convert to float. Multiply by result scale.
      "scvtf v3.2s, v3.2s\n"
      "scvtf v5.2s, v5.2s\n"
      "scvtf v6.2s, v6.2s\n"
      "fmul v3.2s, v3.2s, v12.2s\n"
      "fmul v5.2s, v5.2s, v12.2s\n"
      "fmul v6.2s, v6.2s, v12.2s\n"

      // Store reduced rows.
      "st1 {v3.s}[0], [%x[result]], %x[result_stride]\n"
      "st1 {v5.s}[0], [%x[result]], %x[result_stride]\n"
      "st1 {v6.s}[0], [%x[result]], %x[result_stride]\n"
      : [count] "+r"(count), [result_scale] "+r"(result_scale),
        [result_stride] "+r"(result_stride), [rhs] "+r"(rhs), [lhs] "+r"(lhs),
        [result] "+r"(result)
      :
      : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
        "v11", "v12", "cc", "memory");
}

inline void mul_3x8_2x8_float_lhsadd_rhsadd(const std::uint8_t* lhs,
                                            const std::uint8_t* rhs,
                                            std::int32_t count, float* result,
                                            std::int32_t result_stride,
                                            float result_scale) {
  asm volatile(
      "prfm pldl1keep, [%x[lhs]]\n"
      "prfm pldl1keep, [%x[rhs]]\n"
      // Clear aggregators.
      "movi v0.4s, #0\n"
      "movi v1.4s, #0\n"
      "movi v2.4s, #0\n"
      "mov v3.16b, v0.16b\n"
      "mov v4.16b, v1.16b\n"
      "mov v5.16b, v2.16b\n"

      // General NxM lanes loop.
      "1:"

      // Subtract counter.
      "subs %x[count], %x[count], #8\n"

      "ld1 {v6.8b, v7.8b, v8.8b}, [%x[lhs]], #24\n"
      "ld1 {v9.8b, v10.8b}, [%x[rhs]], #16\n"
      "prfm pldl1keep, [%x[lhs], #64]\n"
      "prfm pldl1keep, [%x[rhs], #64]\n"
      "umull v11.8h, v9.8b, v6.8b\n"
      "umull v12.8h, v10.8b, v6.8b\n"
      "umull v13.8h, v9.8b, v7.8b\n"
      "umull v14.8h, v10.8b, v7.8b\n"
      "umull v15.8h, v9.8b, v8.8b\n"
      "umull v16.8h, v10.8b, v8.8b\n"
      "uadalp v0.4s, v11.8h\n"
      "uadalp v1.4s, v12.8h\n"
      "uadalp v2.4s, v13.8h\n"
      "uadalp v3.4s, v14.8h\n"
      "uadalp v4.4s, v15.8h\n"
      "uadalp v5.4s, v16.8h\n"

      // Loop break.
      "bne 1b\n"

      "ld1 {v6.4s}, [%x[lhs]]\n"
      "dup v8.2s, v6.s[0]\n"
      "dup v9.2s, v6.s[1]\n"
      "dup v10.2s, v6.s[2]\n"
      "ld1 {v11.2s}, [%x[rhs]]\n"
      "dup v12.2s, %w[result_scale]\n"

      // Reduce rows.
      "addp v0.4s, v0.4s, v1.4s\n"
      "addp v7.4s, v0.4s, v0.4s\n"
      "addp v2.4s, v2.4s, v3.4s\n"
      "addp v13.4s, v2.4s, v2.4s\n"
      "addp v4.4s, v4.4s, v5.4s\n"
      "addp v14.4s, v4.4s, v4.4s\n"

      // Add lhs offsets to aggregated rows.
      "add v7.2s, v7.2s, v8.2s\n"
      "add v13.2s, v13.2s, v9.2s\n"
      "add v14.2s, v14.2s, v10.2s\n"

      // Add rhs offset to aggregated rows.
      "add v7.2s, v7.2s, v11.2s\n"
      "add v13.2s, v13.2s, v11.2s\n"
      "add v14.2s, v14.2s, v11.2s\n"

      // Convert to float. Multiply by result scale.
      "scvtf v7.2s, v7.2s\n"
      "scvtf v13.2s, v13.2s\n"
      "scvtf v14.2s, v14.2s\n"
      "fmul v7.2s, v7.2s, v12.2s\n"
      "fmul v13.2s, v13.2s, v12.2s\n"
      "fmul v14.2s, v14.2s, v12.2s\n"

      // Store reduced rows.
      "st1 {v7.2s}, [%x[result]], %x[result_stride]\n"
      "st1 {v13.2s}, [%x[result]], %x[result_stride]\n"
      "st1 {v14.2s}, [%x[result]], %x[result_stride]\n"
      : [count] "+r"(count), [result_scale] "+r"(result_scale),
        [result_stride] "+r"(result_stride), [rhs] "+r"(rhs), [lhs] "+r"(lhs),
        [result] "+r"(result)
      :
      : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
        "v11", "v12", "v13", "v14", "v15", "v16", "cc", "memory");
}

inline void mul_3x8_3x8_float_lhsadd_rhsadd(const std::uint8_t* lhs,
                                            const std::uint8_t* rhs,
                                            std::int32_t count, float* result,
                                            std::int32_t result_stride,
                                            float result_scale) {
  asm volatile(
      "prfm pldl1keep, [%x[lhs]]\n"
      "prfm pldl1keep, [%x[rhs]]\n"
      // Clear aggregators.
      "movi v0.4s, #0\n"
      "movi v1.4s, #0\n"
      "movi v2.4s, #0\n"
      "mov v3.16b, v0.16b\n"
      "mov v4.16b, v1.16b\n"
      "mov v5.16b, v2.16b\n"
      "mov v6.16b, v3.16b\n"
      "mov v7.16b, v4.16b\n"
      "mov v8.16b, v5.16b\n"

      // 3x3 lanes loop.
      "1:"

      "ld1 {v12.8b, v13.8b, v14.8b}, [%x[rhs]], #24\n"
      "ld1 {v9.8b}, [%x[lhs]], #8\n"
      "umull v15.8h, v9.8b, v12.8b\n"
      "ld1 {v10.8b}, [%x[lhs]], #8\n"
      "umull v16.8h, v9.8b, v13.8b\n"
      "ld1 {v11.8b}, [%x[lhs]], #8\n"
      "umull v17.8h, v9.8b, v14.8b\n"
      "prfm pldl1keep, [%x[lhs], #64]\n"
      "umull v18.8h, v10.8b, v12.8b\n"
      "prfm pldl1keep, [%x[rhs], #64]\n"
      "uadalp v0.4s, v15.8h\n"
      "uadalp v1.4s, v16.8h\n"
      "uadalp v2.4s, v17.8h\n"
      "uadalp v3.4s, v18.8h\n"
      "umull v15.8h, v10.8b, v13.8b\n"
      "umull v16.8h, v10.8b, v14.8b\n"
      "umull v17.8h, v11.8b, v12.8b\n"
      "umull v18.8h, v11.8b, v13.8b\n"

      // Subtract counter.
      "subs %x[count], %x[count], #8\n"

      "umull v9.8h, v11.8b, v14.8b\n"
      "uadalp v4.4s, v15.8h\n"
      "uadalp v5.4s, v16.8h\n"
      "uadalp v6.4s, v17.8h\n"
      "uadalp v7.4s, v18.8h\n"
      "uadalp v8.4s, v9.8h\n"

      // Loop break.
      "bne 1b\n"

      "ld1 {v9.4s}, [%x[lhs]]\n"
      "dup v10.4s, v9.s[0]\n"
      "dup v11.4s, v9.s[1]\n"
      "dup v12.4s, v9.s[2]\n"
      "ld1 {v13.4s}, [%x[rhs]]\n"
      "dup v14.4s, %w[result_scale]\n"

      // Reduce rows.
      "addp v0.4s, v0.4s, v1.4s\n"
      "addp v2.4s, v2.4s, v2.4s\n"
      "addp v0.4s, v0.4s, v2.4s\n"
      "addp v3.4s, v3.4s, v4.4s\n"
      "addp v5.4s, v5.4s, v5.4s\n"
      "addp v3.4s, v3.4s, v5.4s\n"
      "addp v6.4s, v6.4s, v7.4s\n"
      "addp v8.4s, v8.4s, v8.4s\n"
      "addp v6.4s, v6.4s, v8.4s\n"

      // Add lhs offsets to aggregated rows.
      "add v0.4s, v0.4s, v10.4s\n"
      "add v3.4s, v3.4s, v11.4s\n"
      "add v6.4s, v6.4s, v12.4s\n"

      // Add rhs offset to aggregated rows.
      "add v0.4s, v0.4s, v13.4s\n"
      "add v3.4s, v3.4s, v13.4s\n"
      "add v6.4s, v6.4s, v13.4s\n"

      // Convert to float. Multiply by result scale.
      "scvtf v0.4s, v0.4s\n"
      "scvtf v3.4s, v3.4s\n"
      "scvtf v6.4s, v6.4s\n"
      "fmul v0.4s, v0.4s, v14.4s\n"
      "fmul v3.4s, v3.4s, v14.4s\n"
      "fmul v6.4s, v6.4s, v14.4s\n"

      // Store reduced rows.
      "st1 {v0.2s}, [%x[result]], #8\n"
      "st1 {v0.s}[2], [%x[result]], %x[result_stride]\n"
      "sub %x[result], %x[result], #8\n"
      "st1 {v3.2s}, [%x[result]], #8\n"
      "st1 {v3.s}[2], [%x[result]], %x[result_stride]\n"
      "sub %x[result], %x[result], #8\n"
      "st1 {v6.2s}, [%x[result]], #8\n"
      "st1 {v6.s}[2], [%x[result]], %x[result_stride]\n"
      "sub %x[result], %x[result], #8\n"
      : [count] "+r"(count), [result_scale] "+r"(result_scale),
        [result_stride] "+r"(result_stride), [rhs] "+r"(rhs), [lhs] "+r"(lhs),
        [result] "+r"(result)
      :
      : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
        "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "cc", "memory");
}

inline void mul_1x8_4x8_float_lhsadd_rhsadd(const std::uint8_t* lhs,
                                            const std::uint8_t* rhs,
                                            std::int32_t count, float* result,
                                            std::int32_t result_stride,
                                            float result_scale) {
  asm volatile(
      "prfm pldl1keep, [%x[lhs]]\n"
      "prfm pldl1keep, [%x[rhs]]\n"
      // Clear aggregators.
      "movi v0.4s, #0\n"
      "movi v1.4s, #0\n"
      "movi v2.4s, #0\n"
      "mov v3.16b, v0.16b\n"

      // General NxM lanes loop.
      "1:"

      // Subtract counter.
      "subs %x[count], %x[count], #8\n"

      "ld1 {v4.8b}, [%x[lhs]], #8\n"
      "ld1 {v5.8b, v6.8b, v7.8b, v8.8b}, [%x[rhs]], #32\n"
      "prfm pldl1keep, [%x[lhs], #64]\n"
      "prfm pldl1keep, [%x[rhs], #64]\n"
      "umull v9.8h, v5.8b, v4.8b\n"
      "umull v10.8h, v6.8b, v4.8b\n"
      "umull v11.8h, v7.8b, v4.8b\n"
      "umull v12.8h, v8.8b, v4.8b\n"
      "uadalp v0.4s, v9.8h\n"
      "uadalp v1.4s, v10.8h\n"
      "uadalp v2.4s, v11.8h\n"
      "uadalp v3.4s, v12.8h\n"

      // Loop break.
      "bne 1b\n"

      "ld1 {v8.2s}, [%x[lhs]]\n"
      "dup v4.4s, v8.s[0]\n"
      "ld1 {v5.4s}, [%x[rhs]]\n"
      "dup v6.4s, %w[result_scale]\n"

      // Reduce rows.
      "addp v0.4s, v0.4s, v1.4s\n"
      "addp v2.4s, v2.4s, v3.4s\n"
      "addp v0.4s, v0.4s, v2.4s\n"

      // Add lhs offsets to aggregated rows.
      "add v0.4s, v0.4s, v4.4s\n"

      // Add rhs offset to aggregated rows.
      "add v0.4s, v0.4s, v5.4s\n"

      // Convert to float. Multiply by result scale.
      "scvtf v0.4s, v0.4s\n"
      "fmul v0.4s, v0.4s, v6.4s\n"

      // Store reduced rows.
      "st1 {v0.4s}, [%x[result]], %x[result_stride]\n"
      : [count] "+r"(count), [result_scale] "+r"(result_scale),
        [result_stride] "+r"(result_stride), [rhs] "+r"(rhs), [lhs] "+r"(lhs),
        [result] "+r"(result)
      :
      : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
        "v11", "v12", "cc", "memory");
}

inline void mul_1x8_5x8_int32_rhsadd(const std::uint8_t* lhs,
                                     const std::uint8_t* rhs_1,
                                     const std::uint8_t* rhs_2,
                                     std::int32_t count, std::int32_t* result) {
  asm volatile(
      "prfm pldl1keep, [%x[lhs]]\n"
      "prfm pldl1keep, [%x[rhs_1]]\n"
      "prfm pldl1keep, [%x[rhs_2]]\n"

      // Clear aggregators.
      "movi v0.4s, #0\n"
      "movi v1.4s, #0\n"
      "movi v2.4s, #0\n"
      "mov v3.16b, v0.16b\n"
      "mov v4.16b, v1.16b\n"

      // General 1xM lanes loop.
      "1:"

      // Subtract counter.
      "subs %x[count], %x[count], #8\n"

      "ld1 {v9.8b}, [%x[lhs]], #8\n"
      "ld1 {v5.8b, v6.8b, v7.8b, v8.8b}, [%x[rhs_1]], #32\n"
      "prfm pldl1keep, [%x[lhs], #64]\n"
      "prfm pldl1keep, [%x[rhs_1], #128]\n"
      "umull v10.8h, v5.8b, v9.8b\n"
      "umull v11.8h, v6.8b, v9.8b\n"
      "umull v12.8h, v7.8b, v9.8b\n"
      "umull v13.8h, v8.8b, v9.8b\n"
      "ld1 {v5.8b}, [%x[rhs_2]], #8\n"
      "prfm pldl1keep, [%x[rhs_2], #32]\n"
      "uadalp v0.4s, v10.8h\n"
      "uadalp v1.4s, v11.8h\n"
      "uadalp v2.4s, v12.8h\n"
      "uadalp v3.4s, v13.8h\n"
      "umull v10.8h, v5.8b, v9.8b\n"
      "uadalp v4.4s, v10.8h\n"

      // Loop break.
      "bne 1b\n"

      "ld1 {v7.4s}, [%x[rhs_1]]\n"
      "ld1 {v8.2s}, [%x[rhs_2]]\n"

      // Horizontal reduce aggregators.
      "addp v0.4s, v0.4s, v1.4s\n"
      "addp v2.4s, v2.4s, v3.4s\n"
      "addp v4.4s, v4.4s, v4.4s\n"
      "addp v5.4s, v0.4s, v2.4s\n"
      "addp v6.4s, v4.4s, v4.4s\n"

      // Add rhs offset to aggregated rows.
      "add v5.4s, v5.4s, v7.4s\n"
      "add v6.2s, v6.2s, v8.2s\n"

      // Store results.
      "st1 {v5.4s}, [%x[result]], #16\n"
      "st1 {v6.s}[0], [%x[result]], #4\n"
      : [count] "+r"(count), [rhs_1] "+r"(rhs_1), [rhs_2] "+r"(rhs_2),
        [lhs] "+r"(lhs), [result] "+r"(result)
      :
      : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
        "v11", "v12", "v13", "cc", "memory");
}

inline void mul_1x8_6x8_int32_rhsadd(const std::uint8_t* lhs,
                                     const std::uint8_t* rhs_1,
                                     const std::uint8_t* rhs_2,
                                     std::int32_t count, std::int32_t* result) {
  asm volatile(
      "prfm pldl1keep, [%x[lhs]]\n"
      "prfm pldl1keep, [%x[rhs_1]]\n"
      "prfm pldl1keep, [%x[rhs_2]]\n"

      // Clear aggregators.
      "movi v0.4s, #0\n"
      "movi v1.4s, #0\n"
      "movi v2.4s, #0\n"
      "mov v3.16b, v0.16b\n"
      "mov v4.16b, v1.16b\n"
      "mov v5.16b, v2.16b\n"

      // General 1xM lanes loop.
      "1:"

      // Subtract counter.
      "subs %x[count], %x[count], #8\n"

      "ld1 {v10.8b}, [%x[lhs]], #8\n"
      "ld1 {v6.8b, v7.8b, v8.8b, v9.8b}, [%x[rhs_1]], #32\n"
      "prfm pldl1keep, [%x[lhs], #64]\n"
      "prfm pldl1keep, [%x[rhs_1], #128]\n"
      "umull v11.8h, v6.8b, v10.8b\n"
      "umull v12.8h, v7.8b, v10.8b\n"
      "umull v13.8h, v8.8b, v10.8b\n"
      "umull v14.8h, v9.8b, v10.8b\n"
      "ld1 {v6.8b, v7.8b}, [%x[rhs_2]], #16\n"
      "prfm pldl1keep, [%x[rhs_2], #64]\n"
      "uadalp v0.4s, v11.8h\n"
      "uadalp v1.4s, v12.8h\n"
      "uadalp v2.4s, v13.8h\n"
      "uadalp v3.4s, v14.8h\n"
      "umull v11.8h, v6.8b, v10.8b\n"
      "umull v12.8h, v7.8b, v10.8b\n"
      "uadalp v4.4s, v11.8h\n"
      "uadalp v5.4s, v12.8h\n"

      // Loop break.
      "bne 1b\n"

      "ld1 {v8.4s}, [%x[rhs_1]]\n"
      "ld1 {v9.2s}, [%x[rhs_2]]\n"

      // Horizontal reduce aggregators.
      "addp v0.4s, v0.4s, v1.4s\n"
      "addp v2.4s, v2.4s, v3.4s\n"
      "addp v4.4s, v4.4s, v5.4s\n"
      "addp v6.4s, v0.4s, v2.4s\n"
      "addp v7.4s, v4.4s, v4.4s\n"

      // Add rhs offset to aggregated rows.
      "add v6.4s, v6.4s, v8.4s\n"
      "add v7.2s, v7.2s, v9.2s\n"

      // Store results.
      "st1 {v6.4s}, [%x[result]], #16\n"
      "st1 {v7.2s}, [%x[result]], #8\n"
      : [count] "+r"(count), [rhs_1] "+r"(rhs_1), [rhs_2] "+r"(rhs_2),
        [lhs] "+r"(lhs), [result] "+r"(result)
      :
      : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
        "v11", "v12", "v13", "v14", "cc", "memory");
}

inline void mul_1x8_7x8_int32_rhsadd(const std::uint8_t* lhs,
                                     const std::uint8_t* rhs_1,
                                     const std::uint8_t* rhs_2,
                                     std::int32_t count, std::int32_t* result) {
  asm volatile(
      "prfm pldl1keep, [%x[lhs]]\n"
      "prfm pldl1keep, [%x[rhs_1]]\n"
      "prfm pldl1keep, [%x[rhs_2]]\n"

      // Clear aggregators.
      "movi v0.4s, #0\n"
      "movi v1.4s, #0\n"
      "movi v2.4s, #0\n"
      "mov v3.16b, v0.16b\n"
      "mov v4.16b, v1.16b\n"
      "mov v5.16b, v2.16b\n"
      "mov v6.16b, v3.16b\n"

      // General 1xM lanes loop.
      "1:"

      // Subtract counter.
      "subs %x[count], %x[count], #8\n"

      "ld1 {v11.8b}, [%x[lhs]], #8\n"
      "ld1 {v7.8b, v8.8b, v9.8b, v10.8b}, [%x[rhs_1]], #32\n"
      "prfm pldl1keep, [%x[lhs], #64]\n"
      "prfm pldl1keep, [%x[rhs_1], #128]\n"
      "umull v12.8h, v7.8b, v11.8b\n"
      "umull v13.8h, v8.8b, v11.8b\n"
      "umull v14.8h, v9.8b, v11.8b\n"
      "umull v15.8h, v10.8b, v11.8b\n"
      "ld1 {v7.8b, v8.8b, v9.8b}, [%x[rhs_2]], #24\n"
      "prfm pldl1keep, [%x[rhs_2], #96]\n"
      "uadalp v0.4s, v12.8h\n"
      "uadalp v1.4s, v13.8h\n"
      "uadalp v2.4s, v14.8h\n"
      "uadalp v3.4s, v15.8h\n"
      "umull v12.8h, v7.8b, v11.8b\n"
      "umull v13.8h, v8.8b, v11.8b\n"
      "umull v14.8h, v9.8b, v11.8b\n"
      "uadalp v4.4s, v12.8h\n"
      "uadalp v5.4s, v13.8h\n"
      "uadalp v6.4s, v14.8h\n"

      // Loop break.
      "bne 1b\n"

      "ld1 {v9.4s}, [%x[rhs_1]]\n"
      "ld1 {v10.4s}, [%x[rhs_2]]\n"

      // Horizontal reduce aggregators.
      "addp v0.4s, v0.4s, v1.4s\n"
      "addp v2.4s, v2.4s, v3.4s\n"
      "addp v4.4s, v4.4s, v5.4s\n"
      "addp v6.4s, v6.4s, v6.4s\n"
      "addp v7.4s, v0.4s, v2.4s\n"
      "addp v8.4s, v4.4s, v6.4s\n"

      // Add rhs offset to aggregated rows.
      "add v7.4s, v7.4s, v9.4s\n"
      "add v8.4s, v8.4s, v10.4s\n"

      // Store results.
      "st1 {v7.4s}, [%x[result]], #16\n"
      "st1 {v8.2s}, [%x[result]], #8\n"
      "st1 {v8.s}[2], [%x[result]], #4\n"
      : [count] "+r"(count), [rhs_1] "+r"(rhs_1), [rhs_2] "+r"(rhs_2),
        [lhs] "+r"(lhs), [result] "+r"(result)
      :
      : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
        "v11", "v12", "v13", "v14", "v15", "cc", "memory");
}

inline void mul_1x8_8x8_int32_rhsadd(const std::uint8_t* lhs,
                                     const std::uint8_t* rhs_1,
                                     const std::uint8_t* rhs_2,
                                     std::int32_t count, std::int32_t* result) {
  asm volatile(
      "prfm pldl1keep, [%x[lhs]]\n"
      "prfm pldl1keep, [%x[rhs_1]]\n"
      "prfm pldl1keep, [%x[rhs_2]]\n"

      // Clear aggregators.
      "movi v0.4s, #0\n"
      "movi v1.4s, #0\n"
      "movi v2.4s, #0\n"
      "mov v3.16b, v0.16b\n"
      "mov v4.16b, v1.16b\n"
      "mov v5.16b, v2.16b\n"
      "mov v6.16b, v3.16b\n"
      "mov v7.16b, v4.16b\n"

      // General 1xM lanes loop.
      "1:"

      // Subtract counter.
      "subs %x[count], %x[count], #8\n"

      "ld1 {v12.8b}, [%x[lhs]], #8\n"
      "ld1 {v8.8b, v9.8b, v10.8b, v11.8b}, [%x[rhs_1]], #32\n"
      "prfm pldl1keep, [%x[lhs], #64]\n"
      "prfm pldl1keep, [%x[rhs_1], #128]\n"
      "umull v13.8h, v8.8b, v12.8b\n"
      "umull v14.8h, v9.8b, v12.8b\n"
      "umull v15.8h, v10.8b, v12.8b\n"
      "umull v16.8h, v11.8b, v12.8b\n"
      "ld1 {v8.8b, v9.8b, v10.8b, v11.8b}, [%x[rhs_2]], #32\n"
      "prfm pldl1keep, [%x[rhs_2], #128]\n"
      "uadalp v0.4s, v13.8h\n"
      "uadalp v1.4s, v14.8h\n"
      "uadalp v2.4s, v15.8h\n"
      "uadalp v3.4s, v16.8h\n"
      "umull v13.8h, v8.8b, v12.8b\n"
      "umull v14.8h, v9.8b, v12.8b\n"
      "umull v15.8h, v10.8b, v12.8b\n"
      "umull v16.8h, v11.8b, v12.8b\n"
      "uadalp v4.4s, v13.8h\n"
      "uadalp v5.4s, v14.8h\n"
      "uadalp v6.4s, v15.8h\n"
      "uadalp v7.4s, v16.8h\n"

      // Loop break.
      "bne 1b\n"

      "ld1 {v10.4s}, [%x[rhs_1]]\n"
      "ld1 {v11.4s}, [%x[rhs_2]]\n"

      // Horizontal reduce aggregators.
      "addp v0.4s, v0.4s, v1.4s\n"
      "addp v2.4s, v2.4s, v3.4s\n"
      "addp v4.4s, v4.4s, v5.4s\n"
      "addp v6.4s, v6.4s, v7.4s\n"
      "addp v8.4s, v0.4s, v2.4s\n"
      "addp v9.4s, v4.4s, v6.4s\n"

      // Add rhs offset to aggregated rows.
      "add v8.4s, v8.4s, v10.4s\n"
      "add v9.4s, v9.4s, v11.4s\n"

      // Store results.
      "st1 {v8.4s}, [%x[result]], #16\n"
      "st1 {v9.4s}, [%x[result]], #16\n"
      : [count] "+r"(count), [rhs_1] "+r"(rhs_1), [rhs_2] "+r"(rhs_2),
        [lhs] "+r"(lhs), [result] "+r"(result)
      :
      : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
        "v11", "v12", "v13", "v14", "v15", "v16", "cc", "memory");
}

inline void mul_1x8_5x8_int32_lhsadd_rhsadd(const std::uint8_t* lhs,
                                            const std::uint8_t* rhs_1,
                                            const std::uint8_t* rhs_2,
                                            std::int32_t count,
                                            std::int32_t* result) {
  asm volatile(
      "prfm pldl1keep, [%x[lhs]]\n"
      "prfm pldl1keep, [%x[rhs_1]]\n"
      "prfm pldl1keep, [%x[rhs_2]]\n"

      // Clear aggregators.
      "movi v0.4s, #0\n"
      "movi v1.4s, #0\n"
      "movi v2.4s, #0\n"
      "mov v3.16b, v0.16b\n"
      "mov v4.16b, v1.16b\n"

      // General 1xM lanes loop.
      "1:"

      // Subtract counter.
      "subs %x[count], %x[count], #8\n"

      "ld1 {v9.8b}, [%x[lhs]], #8\n"
      "ld1 {v5.8b, v6.8b, v7.8b, v8.8b}, [%x[rhs_1]], #32\n"
      "prfm pldl1keep, [%x[lhs], #64]\n"
      "prfm pldl1keep, [%x[rhs_1], #128]\n"
      "umull v10.8h, v5.8b, v9.8b\n"
      "umull v11.8h, v6.8b, v9.8b\n"
      "umull v12.8h, v7.8b, v9.8b\n"
      "umull v13.8h, v8.8b, v9.8b\n"
      "ld1 {v5.8b}, [%x[rhs_2]], #8\n"
      "prfm pldl1keep, [%x[rhs_2], #32]\n"
      "uadalp v0.4s, v10.8h\n"
      "uadalp v1.4s, v11.8h\n"
      "uadalp v2.4s, v12.8h\n"
      "uadalp v3.4s, v13.8h\n"
      "umull v10.8h, v5.8b, v9.8b\n"
      "uadalp v4.4s, v10.8h\n"

      // Loop break.
      "bne 1b\n"

      "ld1r {v7.4s}, [%x[lhs]]\n"
      "ld1 {v8.4s}, [%x[rhs_1]]\n"
      "ld1 {v9.2s}, [%x[rhs_2]]\n"

      // Horizontal reduce aggregators.
      "addp v0.4s, v0.4s, v1.4s\n"
      "addp v2.4s, v2.4s, v3.4s\n"
      "addp v4.4s, v4.4s, v4.4s\n"
      "addp v5.4s, v0.4s, v2.4s\n"
      "addp v6.4s, v4.4s, v4.4s\n"

      // Add lhs offsets to aggregated rows.
      "add v5.4s, v5.4s, v7.4s\n"
      "add v6.2s, v6.2s, v7.2s\n"

      // Add rhs offset to aggregated rows.
      "add v5.4s, v5.4s, v8.4s\n"
      "add v6.2s, v6.2s, v9.2s\n"

      // Store results.
      "st1 {v5.4s}, [%x[result]], #16\n"
      "st1 {v6.s}[0], [%x[result]], #4\n"
      : [count] "+r"(count), [rhs_1] "+r"(rhs_1), [rhs_2] "+r"(rhs_2),
        [lhs] "+r"(lhs), [result] "+r"(result)
      :
      : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
        "v11", "v12", "v13", "cc", "memory");
}

inline void mul_1x8_6x8_int32_lhsadd_rhsadd(const std::uint8_t* lhs,
                                            const std::uint8_t* rhs_1,
                                            const std::uint8_t* rhs_2,
                                            std::int32_t count,
                                            std::int32_t* result) {
  asm volatile(
      "prfm pldl1keep, [%x[lhs]]\n"
      "prfm pldl1keep, [%x[rhs_1]]\n"
      "prfm pldl1keep, [%x[rhs_2]]\n"

      // Clear aggregators.
      "movi v0.4s, #0\n"
      "movi v1.4s, #0\n"
      "movi v2.4s, #0\n"
      "mov v3.16b, v0.16b\n"
      "mov v4.16b, v1.16b\n"
      "mov v5.16b, v2.16b\n"

      // General 1xM lanes loop.
      "1:"

      // Subtract counter.
      "subs %x[count], %x[count], #8\n"

      "ld1 {v10.8b}, [%x[lhs]], #8\n"
      "ld1 {v6.8b, v7.8b, v8.8b, v9.8b}, [%x[rhs_1]], #32\n"
      "prfm pldl1keep, [%x[lhs], #64]\n"
      "prfm pldl1keep, [%x[rhs_1], #128]\n"
      "umull v11.8h, v6.8b, v10.8b\n"
      "umull v12.8h, v7.8b, v10.8b\n"
      "umull v13.8h, v8.8b, v10.8b\n"
      "umull v14.8h, v9.8b, v10.8b\n"
      "ld1 {v6.8b, v7.8b}, [%x[rhs_2]], #16\n"
      "prfm pldl1keep, [%x[rhs_2], #64]\n"
      "uadalp v0.4s, v11.8h\n"
      "uadalp v1.4s, v12.8h\n"
      "uadalp v2.4s, v13.8h\n"
      "uadalp v3.4s, v14.8h\n"
      "umull v11.8h, v6.8b, v10.8b\n"
      "umull v12.8h, v7.8b, v10.8b\n"
      "uadalp v4.4s, v11.8h\n"
      "uadalp v5.4s, v12.8h\n"

      // Loop break.
      "bne 1b\n"

      "ld1r {v8.4s}, [%x[lhs]]\n"
      "ld1 {v9.4s}, [%x[rhs_1]]\n"
      "ld1 {v10.2s}, [%x[rhs_2]]\n"

      // Horizontal reduce aggregators.
      "addp v0.4s, v0.4s, v1.4s\n"
      "addp v2.4s, v2.4s, v3.4s\n"
      "addp v4.4s, v4.4s, v5.4s\n"
      "addp v6.4s, v0.4s, v2.4s\n"
      "addp v7.4s, v4.4s, v4.4s\n"

      // Add lhs offsets to aggregated rows.
      "add v6.4s, v6.4s, v8.4s\n"
      "add v7.2s, v7.2s, v8.2s\n"

      // Add rhs offset to aggregated rows.
      "add v6.4s, v6.4s, v9.4s\n"
      "add v7.2s, v7.2s, v10.2s\n"

      // Store results.
      "st1 {v6.4s}, [%x[result]], #16\n"
      "st1 {v7.2s}, [%x[result]], #8\n"
      : [count] "+r"(count), [rhs_1] "+r"(rhs_1), [rhs_2] "+r"(rhs_2),
        [lhs] "+r"(lhs), [result] "+r"(result)
      :
      : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
        "v11", "v12", "v13", "v14", "cc", "memory");
}

inline void mul_1x8_7x8_int32_lhsadd_rhsadd(const std::uint8_t* lhs,
                                            const std::uint8_t* rhs_1,
                                            const std::uint8_t* rhs_2,
                                            std::int32_t count,
                                            std::int32_t* result) {
  asm volatile(
      "prfm pldl1keep, [%x[lhs]]\n"
      "prfm pldl1keep, [%x[rhs_1]]\n"
      "prfm pldl1keep, [%x[rhs_2]]\n"

      // Clear aggregators.
      "movi v0.4s, #0\n"
      "movi v1.4s, #0\n"
      "movi v2.4s, #0\n"
      "mov v3.16b, v0.16b\n"
      "mov v4.16b, v1.16b\n"
      "mov v5.16b, v2.16b\n"
      "mov v6.16b, v3.16b\n"

      // General 1xM lanes loop.
      "1:"

      // Subtract counter.
      "subs %x[count], %x[count], #8\n"

      "ld1 {v11.8b}, [%x[lhs]], #8\n"
      "ld1 {v7.8b, v8.8b, v9.8b, v10.8b}, [%x[rhs_1]], #32\n"
      "prfm pldl1keep, [%x[lhs], #64]\n"
      "prfm pldl1keep, [%x[rhs_1], #128]\n"
      "umull v12.8h, v7.8b, v11.8b\n"
      "umull v13.8h, v8.8b, v11.8b\n"
      "umull v14.8h, v9.8b, v11.8b\n"
      "umull v15.8h, v10.8b, v11.8b\n"
      "ld1 {v7.8b, v8.8b, v9.8b}, [%x[rhs_2]], #24\n"
      "prfm pldl1keep, [%x[rhs_2], #96]\n"
      "uadalp v0.4s, v12.8h\n"
      "uadalp v1.4s, v13.8h\n"
      "uadalp v2.4s, v14.8h\n"
      "uadalp v3.4s, v15.8h\n"
      "umull v12.8h, v7.8b, v11.8b\n"
      "umull v13.8h, v8.8b, v11.8b\n"
      "umull v14.8h, v9.8b, v11.8b\n"
      "uadalp v4.4s, v12.8h\n"
      "uadalp v5.4s, v13.8h\n"
      "uadalp v6.4s, v14.8h\n"

      // Loop break.
      "bne 1b\n"

      "ld1r {v9.4s}, [%x[lhs]]\n"
      "ld1 {v10.4s}, [%x[rhs_1]]\n"
      "ld1 {v11.4s}, [%x[rhs_2]]\n"

      // Horizontal reduce aggregators.
      "addp v0.4s, v0.4s, v1.4s\n"
      "addp v2.4s, v2.4s, v3.4s\n"
      "addp v4.4s, v4.4s, v5.4s\n"
      "addp v6.4s, v6.4s, v6.4s\n"
      "addp v7.4s, v0.4s, v2.4s\n"
      "addp v8.4s, v4.4s, v6.4s\n"

      // Add lhs offsets to aggregated rows.
      "add v7.4s, v7.4s, v9.4s\n"
      "add v8.4s, v8.4s, v9.4s\n"

      // Add rhs offset to aggregated rows.
      "add v7.4s, v7.4s, v10.4s\n"
      "add v8.4s, v8.4s, v11.4s\n"

      // Store results.
      "st1 {v7.4s}, [%x[result]], #16\n"
      "st1 {v8.2s}, [%x[result]], #8\n"
      "st1 {v8.s}[2], [%x[result]], #4\n"
      : [count] "+r"(count), [rhs_1] "+r"(rhs_1), [rhs_2] "+r"(rhs_2),
        [lhs] "+r"(lhs), [result] "+r"(result)
      :
      : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
        "v11", "v12", "v13", "v14", "v15", "cc", "memory");
}

inline void mul_1x8_8x8_int32_lhsadd_rhsadd(const std::uint8_t* lhs,
                                            const std::uint8_t* rhs_1,
                                            const std::uint8_t* rhs_2,
                                            std::int32_t count,
                                            std::int32_t* result) {
  asm volatile(
      "prfm pldl1keep, [%x[lhs]]\n"
      "prfm pldl1keep, [%x[rhs_1]]\n"
      "prfm pldl1keep, [%x[rhs_2]]\n"

      // Clear aggregators.
      "movi v0.4s, #0\n"
      "movi v1.4s, #0\n"
      "movi v2.4s, #0\n"
      "mov v3.16b, v0.16b\n"
      "mov v4.16b, v1.16b\n"
      "mov v5.16b, v2.16b\n"
      "mov v6.16b, v3.16b\n"
      "mov v7.16b, v4.16b\n"

      // General 1xM lanes loop.
      "1:"

      // Subtract counter.
      "subs %x[count], %x[count], #8\n"

      "ld1 {v12.8b}, [%x[lhs]], #8\n"
      "ld1 {v8.8b, v9.8b, v10.8b, v11.8b}, [%x[rhs_1]], #32\n"
      "prfm pldl1keep, [%x[lhs], #64]\n"
      "prfm pldl1keep, [%x[rhs_1], #128]\n"
      "umull v13.8h, v8.8b, v12.8b\n"
      "umull v14.8h, v9.8b, v12.8b\n"
      "umull v15.8h, v10.8b, v12.8b\n"
      "umull v16.8h, v11.8b, v12.8b\n"
      "ld1 {v8.8b, v9.8b, v10.8b, v11.8b}, [%x[rhs_2]], #32\n"
      "prfm pldl1keep, [%x[rhs_2], #128]\n"
      "uadalp v0.4s, v13.8h\n"
      "uadalp v1.4s, v14.8h\n"
      "uadalp v2.4s, v15.8h\n"
      "uadalp v3.4s, v16.8h\n"
      "umull v13.8h, v8.8b, v12.8b\n"
      "umull v14.8h, v9.8b, v12.8b\n"
      "umull v15.8h, v10.8b, v12.8b\n"
      "umull v16.8h, v11.8b, v12.8b\n"
      "uadalp v4.4s, v13.8h\n"
      "uadalp v5.4s, v14.8h\n"
      "uadalp v6.4s, v15.8h\n"
      "uadalp v7.4s, v16.8h\n"

      // Loop break.
      "bne 1b\n"

      "ld1r {v10.4s}, [%x[lhs]]\n"
      "ld1 {v11.4s}, [%x[rhs_1]]\n"
      "ld1 {v12.4s}, [%x[rhs_2]]\n"

      // Horizontal reduce aggregators.
      "addp v0.4s, v0.4s, v1.4s\n"
      "addp v2.4s, v2.4s, v3.4s\n"
      "addp v4.4s, v4.4s, v5.4s\n"
      "addp v6.4s, v6.4s, v7.4s\n"
      "addp v8.4s, v0.4s, v2.4s\n"
      "addp v9.4s, v4.4s, v6.4s\n"

      // Add lhs offsets to aggregated rows.
      "add v8.4s, v8.4s, v10.4s\n"
      "add v9.4s, v9.4s, v10.4s\n"

      // Add rhs offset to aggregated rows.
      "add v8.4s, v8.4s, v11.4s\n"
      "add v9.4s, v9.4s, v12.4s\n"

      // Store results.
      "st1 {v8.4s}, [%x[result]], #16\n"
      "st1 {v9.4s}, [%x[result]], #16\n"
      : [count] "+r"(count), [rhs_1] "+r"(rhs_1), [rhs_2] "+r"(rhs_2),
        [lhs] "+r"(lhs), [result] "+r"(result)
      :
      : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
        "v11", "v12", "v13", "v14", "v15", "v16", "cc", "memory");
}

inline void mul_1x8_5x8_float_lhsadd_rhsadd(const std::uint8_t* lhs,
                                            const std::uint8_t* rhs_1,
                                            const std::uint8_t* rhs_2,
                                            std::int32_t count, float* result,
                                            float result_scale) {
  asm volatile(
      "prfm pldl1keep, [%x[lhs]]\n"
      "prfm pldl1keep, [%x[rhs_1]]\n"
      "prfm pldl1keep, [%x[rhs_2]]\n"

      // Clear aggregators.
      "movi v0.4s, #0\n"
      "movi v1.4s, #0\n"
      "movi v2.4s, #0\n"
      "mov v3.16b, v0.16b\n"
      "mov v4.16b, v1.16b\n"

      // General 1xM lanes loop.
      "1:"

      // Subtract counter.
      "subs %x[count], %x[count], #8\n"

      "ld1 {v9.8b}, [%x[lhs]], #8\n"
      "ld1 {v5.8b, v6.8b, v7.8b, v8.8b}, [%x[rhs_1]], #32\n"
      "prfm pldl1keep, [%x[lhs], #64]\n"
      "prfm pldl1keep, [%x[rhs_1], #128]\n"
      "umull v10.8h, v5.8b, v9.8b\n"
      "umull v11.8h, v6.8b, v9.8b\n"
      "umull v12.8h, v7.8b, v9.8b\n"
      "umull v13.8h, v8.8b, v9.8b\n"
      "ld1 {v5.8b}, [%x[rhs_2]], #8\n"
      "prfm pldl1keep, [%x[rhs_2], #32]\n"
      "uadalp v0.4s, v10.8h\n"
      "uadalp v1.4s, v11.8h\n"
      "uadalp v2.4s, v12.8h\n"
      "uadalp v3.4s, v13.8h\n"
      "umull v10.8h, v5.8b, v9.8b\n"
      "uadalp v4.4s, v10.8h\n"

      // Loop break.
      "bne 1b\n"

      "ld1r {v7.4s}, [%x[lhs]]\n"
      "ld1 {v8.4s}, [%x[rhs_1]]\n"
      "ld1 {v9.2s}, [%x[rhs_2]]\n"
      "dup v10.4s, %w[result_scale]\n"

      // Horizontal reduce aggregators.
      "addp v0.4s, v0.4s, v1.4s\n"
      "addp v2.4s, v2.4s, v3.4s\n"
      "addp v4.4s, v4.4s, v4.4s\n"
      "addp v5.4s, v0.4s, v2.4s\n"
      "addp v6.4s, v4.4s, v4.4s\n"

      // Add lhs offsets to aggregated rows.
      "add v5.4s, v5.4s, v7.4s\n"
      "add v6.2s, v6.2s, v7.2s\n"

      // Add rhs offset to aggregated rows.
      "add v5.4s, v5.4s, v8.4s\n"
      "add v6.2s, v6.2s, v9.2s\n"

      // Convert to float and scale.
      "scvtf v5.4s, v5.4s\n"
      "scvtf v6.2s, v6.2s\n"
      "fmul v5.4s, v5.4s, v10.4s\n"
      "fmul v6.2s, v6.2s, v10.2s\n"

      // Store results.
      "st1 {v5.4s}, [%x[result]], #16\n"
      "st1 {v6.s}[0], [%x[result]], #4\n"
      : [count] "+r"(count), [result_scale] "+r"(result_scale), [lhs] "+r"(lhs),
        [result] "+r"(result), [rhs_1] "+r"(rhs_1), [rhs_2] "+r"(rhs_2)
      :
      : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
        "v11", "v12", "v13", "cc", "memory");
}

inline void mul_1x8_6x8_float_lhsadd_rhsadd(const std::uint8_t* lhs,
                                            const std::uint8_t* rhs_1,
                                            const std::uint8_t* rhs_2,
                                            std::int32_t count, float* result,
                                            float result_scale) {
  asm volatile(
      "prfm pldl1keep, [%x[lhs]]\n"
      "prfm pldl1keep, [%x[rhs_1]]\n"
      "prfm pldl1keep, [%x[rhs_2]]\n"

      // Clear aggregators.
      "movi v0.4s, #0\n"
      "movi v1.4s, #0\n"
      "movi v2.4s, #0\n"
      "mov v3.16b, v0.16b\n"
      "mov v4.16b, v1.16b\n"
      "mov v5.16b, v2.16b\n"

      // General 1xM lanes loop.
      "1:"

      // Subtract counter.
      "subs %x[count], %x[count], #8\n"

      "ld1 {v10.8b}, [%x[lhs]], #8\n"
      "ld1 {v6.8b, v7.8b, v8.8b, v9.8b}, [%x[rhs_1]], #32\n"
      "prfm pldl1keep, [%x[lhs], #64]\n"
      "prfm pldl1keep, [%x[rhs_1], #128]\n"
      "umull v11.8h, v6.8b, v10.8b\n"
      "umull v12.8h, v7.8b, v10.8b\n"
      "umull v13.8h, v8.8b, v10.8b\n"
      "umull v14.8h, v9.8b, v10.8b\n"
      "ld1 {v6.8b, v7.8b}, [%x[rhs_2]], #16\n"
      "prfm pldl1keep, [%x[rhs_2], #64]\n"
      "uadalp v0.4s, v11.8h\n"
      "uadalp v1.4s, v12.8h\n"
      "uadalp v2.4s, v13.8h\n"
      "uadalp v3.4s, v14.8h\n"
      "umull v11.8h, v6.8b, v10.8b\n"
      "umull v12.8h, v7.8b, v10.8b\n"
      "uadalp v4.4s, v11.8h\n"
      "uadalp v5.4s, v12.8h\n"

      // Loop break.
      "bne 1b\n"

      "ld1r {v8.4s}, [%x[lhs]]\n"
      "ld1 {v9.4s}, [%x[rhs_1]]\n"
      "ld1 {v10.2s}, [%x[rhs_2]]\n"
      "dup v11.4s, %w[result_scale]\n"

      // Horizontal reduce aggregators.
      "addp v0.4s, v0.4s, v1.4s\n"
      "addp v2.4s, v2.4s, v3.4s\n"
      "addp v4.4s, v4.4s, v5.4s\n"
      "addp v6.4s, v0.4s, v2.4s\n"
      "addp v7.4s, v4.4s, v4.4s\n"

      // Add lhs offsets to aggregated rows.
      "add v6.4s, v6.4s, v8.4s\n"
      "add v7.2s, v7.2s, v8.2s\n"

      // Add rhs offset to aggregated rows.
      "add v6.4s, v6.4s, v9.4s\n"
      "add v7.2s, v7.2s, v10.2s\n"

      // Convert to float and scale.
      "scvtf v6.4s, v6.4s\n"
      "scvtf v7.2s, v7.2s\n"
      "fmul v6.4s, v6.4s, v11.4s\n"
      "fmul v7.2s, v7.2s, v11.2s\n"

      // Store results.
      "st1 {v6.4s}, [%x[result]], #16\n"
      "st1 {v7.2s}, [%x[result]], #8\n"
      : [count] "+r"(count), [result_scale] "+r"(result_scale), [lhs] "+r"(lhs),
        [result] "+r"(result), [rhs_1] "+r"(rhs_1), [rhs_2] "+r"(rhs_2)
      :
      : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
        "v11", "v12", "v13", "v14", "cc", "memory");
}

inline void mul_1x8_7x8_float_lhsadd_rhsadd(const std::uint8_t* lhs,
                                            const std::uint8_t* rhs_1,
                                            const std::uint8_t* rhs_2,
                                            std::int32_t count, float* result,
                                            float result_scale) {
  asm volatile(
      "prfm pldl1keep, [%x[lhs]]\n"
      "prfm pldl1keep, [%x[rhs_1]]\n"
      "prfm pldl1keep, [%x[rhs_2]]\n"

      // Clear aggregators.
      "movi v0.4s, #0\n"
      "movi v1.4s, #0\n"
      "movi v2.4s, #0\n"
      "mov v3.16b, v0.16b\n"
      "mov v4.16b, v1.16b\n"
      "mov v5.16b, v2.16b\n"
      "mov v6.16b, v3.16b\n"

      // General 1xM lanes loop.
      "1:"

      // Subtract counter.
      "subs %x[count], %x[count], #8\n"

      "ld1 {v11.8b}, [%x[lhs]], #8\n"
      "ld1 {v7.8b, v8.8b, v9.8b, v10.8b}, [%x[rhs_1]], #32\n"
      "prfm pldl1keep, [%x[lhs], #64]\n"
      "prfm pldl1keep, [%x[rhs_1], #128]\n"
      "umull v12.8h, v7.8b, v11.8b\n"
      "umull v13.8h, v8.8b, v11.8b\n"
      "umull v14.8h, v9.8b, v11.8b\n"
      "umull v15.8h, v10.8b, v11.8b\n"
      "ld1 {v7.8b, v8.8b, v9.8b}, [%x[rhs_2]], #24\n"
      "prfm pldl1keep, [%x[rhs_2], #96]\n"
      "uadalp v0.4s, v12.8h\n"
      "uadalp v1.4s, v13.8h\n"
      "uadalp v2.4s, v14.8h\n"
      "uadalp v3.4s, v15.8h\n"
      "umull v12.8h, v7.8b, v11.8b\n"
      "umull v13.8h, v8.8b, v11.8b\n"
      "umull v14.8h, v9.8b, v11.8b\n"
      "uadalp v4.4s, v12.8h\n"
      "uadalp v5.4s, v13.8h\n"
      "uadalp v6.4s, v14.8h\n"

      // Loop break.
      "bne 1b\n"

      "ld1r {v9.4s}, [%x[lhs]]\n"
      "ld1 {v10.4s}, [%x[rhs_1]]\n"
      "ld1 {v11.4s}, [%x[rhs_2]]\n"
      "dup v12.4s, %w[result_scale]\n"

      // Horizontal reduce aggregators.
      "addp v0.4s, v0.4s, v1.4s\n"
      "addp v2.4s, v2.4s, v3.4s\n"
      "addp v4.4s, v4.4s, v5.4s\n"
      "addp v6.4s, v6.4s, v6.4s\n"
      "addp v7.4s, v0.4s, v2.4s\n"
      "addp v8.4s, v4.4s, v6.4s\n"

      // Add lhs offsets to aggregated rows.
      "add v7.4s, v7.4s, v9.4s\n"
      "add v8.4s, v8.4s, v9.4s\n"

      // Add rhs offset to aggregated rows.
      "add v7.4s, v7.4s, v10.4s\n"
      "add v8.4s, v8.4s, v11.4s\n"

      // Convert to float and scale.
      "scvtf v7.4s, v7.4s\n"
      "scvtf v8.4s, v8.4s\n"
      "fmul v7.4s, v7.4s, v12.4s\n"
      "fmul v8.4s, v8.4s, v12.4s\n"

      // Store results.
      "st1 {v7.4s}, [%x[result]], #16\n"
      "st1 {v8.2s}, [%x[result]], #8\n"
      "st1 {v8.s}[2], [%x[result]], #4\n"
      : [count] "+r"(count), [result_scale] "+r"(result_scale), [lhs] "+r"(lhs),
        [result] "+r"(result), [rhs_1] "+r"(rhs_1), [rhs_2] "+r"(rhs_2)
      :
      : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
        "v11", "v12", "v13", "v14", "v15", "cc", "memory");
}

inline void mul_1x8_8x8_float_lhsadd_rhsadd(const std::uint8_t* lhs,
                                            const std::uint8_t* rhs_1,
                                            const std::uint8_t* rhs_2,
                                            std::int32_t count, float* result,
                                            float result_scale) {
  asm volatile(
      "prfm pldl1keep, [%x[lhs]]\n"
      "prfm pldl1keep, [%x[rhs_1]]\n"
      "prfm pldl1keep, [%x[rhs_2]]\n"

      // Clear aggregators.
      "movi v0.4s, #0\n"
      "movi v1.4s, #0\n"
      "movi v2.4s, #0\n"
      "mov v3.16b, v0.16b\n"
      "mov v4.16b, v1.16b\n"
      "mov v5.16b, v2.16b\n"
      "mov v6.16b, v3.16b\n"
      "mov v7.16b, v4.16b\n"

      // General 1xM lanes loop.
      "1:"

      // Subtract counter.
      "subs %x[count], %x[count], #8\n"

      "ld1 {v12.8b}, [%x[lhs]], #8\n"
      "ld1 {v8.8b, v9.8b, v10.8b, v11.8b}, [%x[rhs_1]], #32\n"
      "prfm pldl1keep, [%x[lhs], #64]\n"
      "prfm pldl1keep, [%x[rhs_1], #128]\n"
      "umull v13.8h, v8.8b, v12.8b\n"
      "umull v14.8h, v9.8b, v12.8b\n"
      "umull v15.8h, v10.8b, v12.8b\n"
      "umull v16.8h, v11.8b, v12.8b\n"
      "ld1 {v8.8b, v9.8b, v10.8b, v11.8b}, [%x[rhs_2]], #32\n"
      "prfm pldl1keep, [%x[rhs_2], #128]\n"
      "uadalp v0.4s, v13.8h\n"
      "uadalp v1.4s, v14.8h\n"
      "uadalp v2.4s, v15.8h\n"
      "uadalp v3.4s, v16.8h\n"
      "umull v13.8h, v8.8b, v12.8b\n"
      "umull v14.8h, v9.8b, v12.8b\n"
      "umull v15.8h, v10.8b, v12.8b\n"
      "umull v16.8h, v11.8b, v12.8b\n"
      "uadalp v4.4s, v13.8h\n"
      "uadalp v5.4s, v14.8h\n"
      "uadalp v6.4s, v15.8h\n"
      "uadalp v7.4s, v16.8h\n"

      // Loop break.
      "bne 1b\n"

      "ld1r {v10.4s}, [%x[lhs]]\n"
      "ld1 {v11.4s}, [%x[rhs_1]]\n"
      "ld1 {v12.4s}, [%x[rhs_2]]\n"
      "dup v13.4s, %w[result_scale]\n"

      // Horizontal reduce aggregators.
      "addp v0.4s, v0.4s, v1.4s\n"
      "addp v2.4s, v2.4s, v3.4s\n"
      "addp v4.4s, v4.4s, v5.4s\n"
      "addp v6.4s, v6.4s, v7.4s\n"
      "addp v8.4s, v0.4s, v2.4s\n"
      "addp v9.4s, v4.4s, v6.4s\n"

      // Add lhs offsets to aggregated rows.
      "add v8.4s, v8.4s, v10.4s\n"
      "add v9.4s, v9.4s, v10.4s\n"

      // Add rhs offset to aggregated rows.
      "add v8.4s, v8.4s, v11.4s\n"
      "add v9.4s, v9.4s, v12.4s\n"

      // Convert to float and scale.
      "scvtf v8.4s, v8.4s\n"
      "scvtf v9.4s, v9.4s\n"
      "fmul v8.4s, v8.4s, v13.4s\n"
      "fmul v9.4s, v9.4s, v13.4s\n"

      // Store results.
      "st1 {v8.4s}, [%x[result]], #16\n"
      "st1 {v9.4s}, [%x[result]], #16\n"
      : [count] "+r"(count), [result_scale] "+r"(result_scale), [lhs] "+r"(lhs),
        [result] "+r"(result), [rhs_1] "+r"(rhs_1), [rhs_2] "+r"(rhs_2)
      :
      : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
        "v11", "v12", "v13", "v14", "v15", "v16", "cc", "memory");
}

void qnt_1x8_aligned(const std::int32_t* source, std::int32_t count,
                     std::int32_t stride, const std::int32_t* offsets,
                     std::uint8_t* destination, std::int32_t destination_stride,
                     std::int32_t multiplicative_offset,
                     std::int32_t rounding_offset, std::int32_t shift) {
  asm volatile(
      "dup v0.4s, %w[multiplicative_offset]\n"
      "dup v1.4s, %w[rounding_offset]\n"
      "dup v2.4s, %w[shift]\n"
      "ld1r {v3.4s}, [%x[offsets]], #4\n"

      "1:"
      "subs %x[count], %x[count], #8\n"
      "ld1 {v4.4s, v5.4s}, [%x[source]], #32\n"
      "prfm pldl1keep, [%x[source]]\n"
      "add v4.4s, v4.4s, v3.4s\n"
      "add v5.4s, v5.4s, v3.4s\n"
      "mul v4.4s, v4.4s, v0.4s\n"
      "mul v5.4s, v5.4s, v0.4s\n"
      "add v4.4s, v4.4s, v1.4s\n"
      "add v5.4s, v5.4s, v1.4s\n"
      "sshl v4.4s, v4.4s, v2.4s\n"
      "sshl v5.4s, v5.4s, v2.4s\n"
      "sqxtn v4.4h, v4.4s\n"
      "sqxtn2 v4.8h, v5.4s\n"
      "sqxtun v4.8b, v4.8h\n"
      "st1 {v4.2s}, [%x[destination]], #8\n"

      "bne 1b\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [stride] "+r"(stride), [shift] "+r"(shift),
        [destination] "+r"(destination), [offsets] "+r"(offsets),
        [source] "+r"(source), [destination_stride] "+r"(destination_stride),
        [rounding_offset] "+r"(rounding_offset)
      :
      : "v0", "v1", "v2", "v3", "v4", "v5", "cc", "memory");
}

void qnt_1x8_1_aligned(const std::int32_t* source, std::int32_t count,
                       std::int32_t stride, const std::int32_t* offsets,
                       std::uint8_t* destination,
                       std::int32_t destination_stride,
                       std::int32_t multiplicative_offset,
                       std::int32_t rounding_offset, std::int32_t shift) {
  asm volatile(
      "dup v0.4s, %w[multiplicative_offset]\n"
      "dup v1.4s, %w[rounding_offset]\n"
      "dup v2.4s, %w[shift]\n"
      "ld1r {v3.4s}, [%x[offsets]], #4\n"
      "subs %x[count], %x[count], #1\n"
      "beq 2f\n"

      "1:"
      "subs %x[count], %x[count], #8\n"
      "ld1 {v4.4s, v5.4s}, [%x[source]], #32\n"
      "prfm pldl1keep, [%x[source]]\n"
      "add v4.4s, v4.4s, v3.4s\n"
      "add v5.4s, v5.4s, v3.4s\n"
      "mul v4.4s, v4.4s, v0.4s\n"
      "mul v5.4s, v5.4s, v0.4s\n"
      "add v4.4s, v4.4s, v1.4s\n"
      "add v5.4s, v5.4s, v1.4s\n"
      "sshl v4.4s, v4.4s, v2.4s\n"
      "sshl v5.4s, v5.4s, v2.4s\n"
      "sqxtn v4.4h, v4.4s\n"
      "sqxtn2 v4.8h, v5.4s\n"
      "sqxtun v4.8b, v4.8h\n"
      "st1 {v4.2s}, [%x[destination]], #8\n"

      "bne 1b\n"
      "2:"
      "ld1 {v4.s}[0], [%x[source]], #4\n"
      "add v4.4s, v4.4s, v3.4s\n"
      "mul v4.4s, v4.4s, v0.4s\n"
      "add v4.4s, v4.4s, v1.4s\n"
      "sshl v4.4s, v4.4s, v2.4s\n"
      "sqxtn v4.4h, v4.4s\n"
      "sqxtun v4.8b, v4.8h\n"
      "st1 {v4.b}[0], [%x[destination]], #1\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [stride] "+r"(stride), [shift] "+r"(shift),
        [destination] "+r"(destination), [offsets] "+r"(offsets),
        [source] "+r"(source), [destination_stride] "+r"(destination_stride),
        [rounding_offset] "+r"(rounding_offset)
      :
      : "v0", "v1", "v2", "v3", "v4", "v5", "cc", "memory");
}

void qnt_1x8_2_aligned(const std::int32_t* source, std::int32_t count,
                       std::int32_t stride, const std::int32_t* offsets,
                       std::uint8_t* destination,
                       std::int32_t destination_stride,
                       std::int32_t multiplicative_offset,
                       std::int32_t rounding_offset, std::int32_t shift) {
  asm volatile(
      "dup v0.4s, %w[multiplicative_offset]\n"
      "dup v1.4s, %w[rounding_offset]\n"
      "dup v2.4s, %w[shift]\n"
      "ld1r {v3.4s}, [%x[offsets]], #4\n"
      "subs %x[count], %x[count], #2\n"
      "beq 2f\n"

      "1:"
      "subs %x[count], %x[count], #8\n"
      "ld1 {v4.4s, v5.4s}, [%x[source]], #32\n"
      "prfm pldl1keep, [%x[source]]\n"
      "add v4.4s, v4.4s, v3.4s\n"
      "add v5.4s, v5.4s, v3.4s\n"
      "mul v4.4s, v4.4s, v0.4s\n"
      "mul v5.4s, v5.4s, v0.4s\n"
      "add v4.4s, v4.4s, v1.4s\n"
      "add v5.4s, v5.4s, v1.4s\n"
      "sshl v4.4s, v4.4s, v2.4s\n"
      "sshl v5.4s, v5.4s, v2.4s\n"
      "sqxtn v4.4h, v4.4s\n"
      "sqxtn2 v4.8h, v5.4s\n"
      "sqxtun v4.8b, v4.8h\n"
      "st1 {v4.2s}, [%x[destination]], #8\n"

      "bne 1b\n"
      "2:"
      "ld1 {v4.2s}, [%x[source]], #8\n"
      "add v4.4s, v4.4s, v3.4s\n"
      "mul v4.4s, v4.4s, v0.4s\n"
      "add v4.4s, v4.4s, v1.4s\n"
      "sshl v4.4s, v4.4s, v2.4s\n"
      "sqxtn v4.4h, v4.4s\n"
      "sqxtun v4.8b, v4.8h\n"
      "st1 {v4.h}[0], [%x[destination]], #2\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [stride] "+r"(stride), [shift] "+r"(shift),
        [destination] "+r"(destination), [offsets] "+r"(offsets),
        [source] "+r"(source), [destination_stride] "+r"(destination_stride),
        [rounding_offset] "+r"(rounding_offset)
      :
      : "v0", "v1", "v2", "v3", "v4", "v5", "cc", "memory");
}

void qnt_1x8_3_aligned(const std::int32_t* source, std::int32_t count,
                       std::int32_t stride, const std::int32_t* offsets,
                       std::uint8_t* destination,
                       std::int32_t destination_stride,
                       std::int32_t multiplicative_offset,
                       std::int32_t rounding_offset, std::int32_t shift) {
  asm volatile(
      "dup v0.4s, %w[multiplicative_offset]\n"
      "dup v1.4s, %w[rounding_offset]\n"
      "dup v2.4s, %w[shift]\n"
      "ld1r {v3.4s}, [%x[offsets]], #4\n"
      "subs %x[count], %x[count], #3\n"
      "beq 2f\n"

      "1:"
      "subs %x[count], %x[count], #8\n"
      "ld1 {v4.4s, v5.4s}, [%x[source]], #32\n"
      "prfm pldl1keep, [%x[source]]\n"
      "add v4.4s, v4.4s, v3.4s\n"
      "add v5.4s, v5.4s, v3.4s\n"
      "mul v4.4s, v4.4s, v0.4s\n"
      "mul v5.4s, v5.4s, v0.4s\n"
      "add v4.4s, v4.4s, v1.4s\n"
      "add v5.4s, v5.4s, v1.4s\n"
      "sshl v4.4s, v4.4s, v2.4s\n"
      "sshl v5.4s, v5.4s, v2.4s\n"
      "sqxtn v4.4h, v4.4s\n"
      "sqxtn2 v4.8h, v5.4s\n"
      "sqxtun v4.8b, v4.8h\n"
      "st1 {v4.2s}, [%x[destination]], #8\n"

      "bne 1b\n"
      "2:"
      "ld1 {v4.2s}, [%x[source]], #8\n"
      "ld1 {v4.s}[2], [%x[source]], #4\n"
      "add v4.4s, v4.4s, v3.4s\n"
      "mul v4.4s, v4.4s, v0.4s\n"
      "add v4.4s, v4.4s, v1.4s\n"
      "sshl v4.4s, v4.4s, v2.4s\n"
      "sqxtn v4.4h, v4.4s\n"
      "sqxtun v4.8b, v4.8h\n"
      "st1 {v4.h}[0], [%x[destination]], #2\n"
      "st1 {v4.b}[2], [%x[destination]], #1\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [stride] "+r"(stride), [shift] "+r"(shift),
        [destination] "+r"(destination), [offsets] "+r"(offsets),
        [source] "+r"(source), [destination_stride] "+r"(destination_stride),
        [rounding_offset] "+r"(rounding_offset)
      :
      : "v0", "v1", "v2", "v3", "v4", "v5", "cc", "memory");
}

void qnt_1x8_4_aligned(const std::int32_t* source, std::int32_t count,
                       std::int32_t stride, const std::int32_t* offsets,
                       std::uint8_t* destination,
                       std::int32_t destination_stride,
                       std::int32_t multiplicative_offset,
                       std::int32_t rounding_offset, std::int32_t shift) {
  asm volatile(
      "dup v0.4s, %w[multiplicative_offset]\n"
      "dup v1.4s, %w[rounding_offset]\n"
      "dup v2.4s, %w[shift]\n"
      "ld1r {v3.4s}, [%x[offsets]], #4\n"
      "subs %x[count], %x[count], #4\n"
      "beq 2f\n"

      "1:"
      "subs %x[count], %x[count], #8\n"
      "ld1 {v4.4s, v5.4s}, [%x[source]], #32\n"
      "prfm pldl1keep, [%x[source]]\n"
      "add v4.4s, v4.4s, v3.4s\n"
      "add v5.4s, v5.4s, v3.4s\n"
      "mul v4.4s, v4.4s, v0.4s\n"
      "mul v5.4s, v5.4s, v0.4s\n"
      "add v4.4s, v4.4s, v1.4s\n"
      "add v5.4s, v5.4s, v1.4s\n"
      "sshl v4.4s, v4.4s, v2.4s\n"
      "sshl v5.4s, v5.4s, v2.4s\n"
      "sqxtn v4.4h, v4.4s\n"
      "sqxtn2 v4.8h, v5.4s\n"
      "sqxtun v4.8b, v4.8h\n"
      "st1 {v4.2s}, [%x[destination]], #8\n"

      "bne 1b\n"
      "2:"
      "ld1 {v4.4s}, [%x[source]], #16\n"
      "add v4.4s, v4.4s, v3.4s\n"
      "mul v4.4s, v4.4s, v0.4s\n"
      "add v4.4s, v4.4s, v1.4s\n"
      "sshl v4.4s, v4.4s, v2.4s\n"
      "sqxtn v4.4h, v4.4s\n"
      "sqxtun v4.8b, v4.8h\n"
      "st1 {v4.s}[0], [%x[destination]], #4\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [stride] "+r"(stride), [shift] "+r"(shift),
        [destination] "+r"(destination), [offsets] "+r"(offsets),
        [source] "+r"(source), [destination_stride] "+r"(destination_stride),
        [rounding_offset] "+r"(rounding_offset)
      :
      : "v0", "v1", "v2", "v3", "v4", "v5", "cc", "memory");
}

void qnt_1x8_5_aligned(const std::int32_t* source, std::int32_t count,
                       std::int32_t stride, const std::int32_t* offsets,
                       std::uint8_t* destination,
                       std::int32_t destination_stride,
                       std::int32_t multiplicative_offset,
                       std::int32_t rounding_offset, std::int32_t shift) {
  asm volatile(
      "dup v0.4s, %w[multiplicative_offset]\n"
      "dup v1.4s, %w[rounding_offset]\n"
      "dup v2.4s, %w[shift]\n"
      "ld1r {v3.4s}, [%x[offsets]], #4\n"
      "subs %x[count], %x[count], #5\n"
      "beq 2f\n"

      "1:"
      "subs %x[count], %x[count], #8\n"
      "ld1 {v4.4s, v5.4s}, [%x[source]], #32\n"
      "prfm pldl1keep, [%x[source]]\n"
      "add v4.4s, v4.4s, v3.4s\n"
      "add v5.4s, v5.4s, v3.4s\n"
      "mul v4.4s, v4.4s, v0.4s\n"
      "mul v5.4s, v5.4s, v0.4s\n"
      "add v4.4s, v4.4s, v1.4s\n"
      "add v5.4s, v5.4s, v1.4s\n"
      "sshl v4.4s, v4.4s, v2.4s\n"
      "sshl v5.4s, v5.4s, v2.4s\n"
      "sqxtn v4.4h, v4.4s\n"
      "sqxtn2 v4.8h, v5.4s\n"
      "sqxtun v4.8b, v4.8h\n"
      "st1 {v4.2s}, [%x[destination]], #8\n"

      "bne 1b\n"
      "2:"
      "ld1 {v4.4s}, [%x[source]], #16\n"
      "ld1 {v5.s}[0], [%x[source]], #4\n"
      "add v4.4s, v4.4s, v3.4s\n"
      "add v5.4s, v5.4s, v3.4s\n"
      "mul v4.4s, v4.4s, v0.4s\n"
      "mul v5.4s, v5.4s, v0.4s\n"
      "add v4.4s, v4.4s, v1.4s\n"
      "add v5.4s, v5.4s, v1.4s\n"
      "sshl v4.4s, v4.4s, v2.4s\n"
      "sshl v5.4s, v5.4s, v2.4s\n"
      "sqxtn v4.4h, v4.4s\n"
      "sqxtn2 v4.8h, v5.4s\n"
      "sqxtun v4.8b, v4.8h\n"
      "st1 {v4.s}[0], [%x[destination]], #4\n"
      "st1 {v4.b}[4], [%x[destination]], #1\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [stride] "+r"(stride), [shift] "+r"(shift),
        [destination] "+r"(destination), [offsets] "+r"(offsets),
        [source] "+r"(source), [destination_stride] "+r"(destination_stride),
        [rounding_offset] "+r"(rounding_offset)
      :
      : "v0", "v1", "v2", "v3", "v4", "v5", "cc", "memory");
}

void qnt_1x8_6_aligned(const std::int32_t* source, std::int32_t count,
                       std::int32_t stride, const std::int32_t* offsets,
                       std::uint8_t* destination,
                       std::int32_t destination_stride,
                       std::int32_t multiplicative_offset,
                       std::int32_t rounding_offset, std::int32_t shift) {
  asm volatile(
      "dup v0.4s, %w[multiplicative_offset]\n"
      "dup v1.4s, %w[rounding_offset]\n"
      "dup v2.4s, %w[shift]\n"
      "ld1r {v3.4s}, [%x[offsets]], #4\n"
      "subs %x[count], %x[count], #6\n"
      "beq 2f\n"

      "1:"
      "subs %x[count], %x[count], #8\n"
      "ld1 {v4.4s, v5.4s}, [%x[source]], #32\n"
      "prfm pldl1keep, [%x[source]]\n"
      "add v4.4s, v4.4s, v3.4s\n"
      "add v5.4s, v5.4s, v3.4s\n"
      "mul v4.4s, v4.4s, v0.4s\n"
      "mul v5.4s, v5.4s, v0.4s\n"
      "add v4.4s, v4.4s, v1.4s\n"
      "add v5.4s, v5.4s, v1.4s\n"
      "sshl v4.4s, v4.4s, v2.4s\n"
      "sshl v5.4s, v5.4s, v2.4s\n"
      "sqxtn v4.4h, v4.4s\n"
      "sqxtn2 v4.8h, v5.4s\n"
      "sqxtun v4.8b, v4.8h\n"
      "st1 {v4.2s}, [%x[destination]], #8\n"

      "bne 1b\n"
      "2:"
      "ld1 {v4.4s}, [%x[source]], #16\n"
      "ld1 {v5.2s}, [%x[source]], #8\n"
      "add v4.4s, v4.4s, v3.4s\n"
      "add v5.4s, v5.4s, v3.4s\n"
      "mul v4.4s, v4.4s, v0.4s\n"
      "mul v5.4s, v5.4s, v0.4s\n"
      "add v4.4s, v4.4s, v1.4s\n"
      "add v5.4s, v5.4s, v1.4s\n"
      "sshl v4.4s, v4.4s, v2.4s\n"
      "sshl v5.4s, v5.4s, v2.4s\n"
      "sqxtn v4.4h, v4.4s\n"
      "sqxtn2 v4.8h, v5.4s\n"
      "sqxtun v4.8b, v4.8h\n"
      "st1 {v4.s}[0], [%x[destination]], #4\n"
      "st1 {v4.h}[2], [%x[destination]], #2\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [stride] "+r"(stride), [shift] "+r"(shift),
        [destination] "+r"(destination), [offsets] "+r"(offsets),
        [source] "+r"(source), [destination_stride] "+r"(destination_stride),
        [rounding_offset] "+r"(rounding_offset)
      :
      : "v0", "v1", "v2", "v3", "v4", "v5", "cc", "memory");
}

void qnt_1x8_7_aligned(const std::int32_t* source, std::int32_t count,
                       std::int32_t stride, const std::int32_t* offsets,
                       std::uint8_t* destination,
                       std::int32_t destination_stride,
                       std::int32_t multiplicative_offset,
                       std::int32_t rounding_offset, std::int32_t shift) {
  asm volatile(
      "dup v0.4s, %w[multiplicative_offset]\n"
      "dup v1.4s, %w[rounding_offset]\n"
      "dup v2.4s, %w[shift]\n"
      "ld1r {v3.4s}, [%x[offsets]], #4\n"
      "subs %x[count], %x[count], #7\n"
      "beq 2f\n"

      "1:"
      "subs %x[count], %x[count], #8\n"
      "ld1 {v4.4s, v5.4s}, [%x[source]], #32\n"
      "prfm pldl1keep, [%x[source]]\n"
      "add v4.4s, v4.4s, v3.4s\n"
      "add v5.4s, v5.4s, v3.4s\n"
      "mul v4.4s, v4.4s, v0.4s\n"
      "mul v5.4s, v5.4s, v0.4s\n"
      "add v4.4s, v4.4s, v1.4s\n"
      "add v5.4s, v5.4s, v1.4s\n"
      "sshl v4.4s, v4.4s, v2.4s\n"
      "sshl v5.4s, v5.4s, v2.4s\n"
      "sqxtn v4.4h, v4.4s\n"
      "sqxtn2 v4.8h, v5.4s\n"
      "sqxtun v4.8b, v4.8h\n"
      "st1 {v4.2s}, [%x[destination]], #8\n"

      "bne 1b\n"
      "2:"
      "ld1 {v4.4s}, [%x[source]], #16\n"
      "ld1 {v5.2s}, [%x[source]], #8\n"
      "ld1 {v5.s}[2], [%x[source]], #4\n"
      "add v4.4s, v4.4s, v3.4s\n"
      "add v5.4s, v5.4s, v3.4s\n"
      "mul v4.4s, v4.4s, v0.4s\n"
      "mul v5.4s, v5.4s, v0.4s\n"
      "add v4.4s, v4.4s, v1.4s\n"
      "add v5.4s, v5.4s, v1.4s\n"
      "sshl v4.4s, v4.4s, v2.4s\n"
      "sshl v5.4s, v5.4s, v2.4s\n"
      "sqxtn v4.4h, v4.4s\n"
      "sqxtn2 v4.8h, v5.4s\n"
      "sqxtun v4.8b, v4.8h\n"
      "st1 {v4.s}[0], [%x[destination]], #4\n"
      "st1 {v4.h}[2], [%x[destination]], #2\n"
      "st1 {v4.b}[6], [%x[destination]], #1\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [stride] "+r"(stride), [shift] "+r"(shift),
        [destination] "+r"(destination), [offsets] "+r"(offsets),
        [source] "+r"(source), [destination_stride] "+r"(destination_stride),
        [rounding_offset] "+r"(rounding_offset)
      :
      : "v0", "v1", "v2", "v3", "v4", "v5", "cc", "memory");
}

void qnt_2x8_aligned(const std::int32_t* source, std::int32_t count,
                     std::int32_t stride, const std::int32_t* offsets,
                     std::uint8_t* destination, std::int32_t destination_stride,
                     std::int32_t multiplicative_offset,
                     std::int32_t rounding_offset, std::int32_t shift) {
  asm volatile(
      "dup v0.4s, %w[multiplicative_offset]\n"
      "dup v1.4s, %w[rounding_offset]\n"
      "dup v2.4s, %w[shift]\n"
      "ld1r {v3.4s}, [%x[offsets]], #4\n"
      "ld1r {v4.4s}, [%x[offsets]], #4\n"
      "add x0, %x[source], %x[stride]\n"
      "add x1, %x[destination], %x[destination_stride]\n"

      "1:"
      "subs %x[count], %x[count], #8\n"
      "ld1 {v5.4s, v6.4s}, [%x[source]], #32\n"
      "ld1 {v7.4s, v8.4s}, [x0], #32\n"
      "prfm pldl1keep, [%x[source]]\n"
      "prfm pldl1keep, [x0]\n"
      "add v5.4s, v5.4s, v3.4s\n"
      "add v6.4s, v6.4s, v3.4s\n"
      "add v7.4s, v7.4s, v4.4s\n"
      "add v8.4s, v8.4s, v4.4s\n"
      "mul v5.4s, v5.4s, v0.4s\n"
      "mul v6.4s, v6.4s, v0.4s\n"
      "mul v7.4s, v7.4s, v0.4s\n"
      "mul v8.4s, v8.4s, v0.4s\n"
      "add v5.4s, v5.4s, v1.4s\n"
      "add v6.4s, v6.4s, v1.4s\n"
      "add v7.4s, v7.4s, v1.4s\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "sshl v5.4s, v5.4s, v2.4s\n"
      "sshl v6.4s, v6.4s, v2.4s\n"
      "sshl v7.4s, v7.4s, v2.4s\n"
      "sshl v8.4s, v8.4s, v2.4s\n"
      "sqxtn v5.4h, v5.4s\n"
      "sqxtn2 v5.8h, v6.4s\n"
      "sqxtn v7.4h, v7.4s\n"
      "sqxtn2 v7.8h, v8.4s\n"
      "sqxtun v5.8b, v5.8h\n"
      "sqxtun v7.8b, v7.8h\n"
      "st1 {v5.2s}, [%x[destination]], #8\n"
      "st1 {v7.2s}, [x1], #8\n"

      "bne 1b\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [stride] "+r"(stride), [shift] "+r"(shift),
        [destination] "+r"(destination), [offsets] "+r"(offsets),
        [source] "+r"(source), [destination_stride] "+r"(destination_stride),
        [rounding_offset] "+r"(rounding_offset)
      :
      : "x0", "x1", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "cc",
        "memory");
}

void qnt_2x8_1_aligned(const std::int32_t* source, std::int32_t count,
                       std::int32_t stride, const std::int32_t* offsets,
                       std::uint8_t* destination,
                       std::int32_t destination_stride,
                       std::int32_t multiplicative_offset,
                       std::int32_t rounding_offset, std::int32_t shift) {
  asm volatile(
      "dup v0.4s, %w[multiplicative_offset]\n"
      "dup v1.4s, %w[rounding_offset]\n"
      "dup v2.4s, %w[shift]\n"
      "ld1r {v3.4s}, [%x[offsets]], #4\n"
      "ld1r {v4.4s}, [%x[offsets]], #4\n"
      "add x0, %x[source], %x[stride]\n"
      "add x1, %x[destination], %x[destination_stride]\n"
      "subs %x[count], %x[count], #1\n"
      "beq 2f\n"

      "1:"
      "subs %x[count], %x[count], #8\n"
      "ld1 {v5.4s, v6.4s}, [%x[source]], #32\n"
      "ld1 {v7.4s, v8.4s}, [x0], #32\n"
      "prfm pldl1keep, [%x[source]]\n"
      "prfm pldl1keep, [x0]\n"
      "add v5.4s, v5.4s, v3.4s\n"
      "add v6.4s, v6.4s, v3.4s\n"
      "add v7.4s, v7.4s, v4.4s\n"
      "add v8.4s, v8.4s, v4.4s\n"
      "mul v5.4s, v5.4s, v0.4s\n"
      "mul v6.4s, v6.4s, v0.4s\n"
      "mul v7.4s, v7.4s, v0.4s\n"
      "mul v8.4s, v8.4s, v0.4s\n"
      "add v5.4s, v5.4s, v1.4s\n"
      "add v6.4s, v6.4s, v1.4s\n"
      "add v7.4s, v7.4s, v1.4s\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "sshl v5.4s, v5.4s, v2.4s\n"
      "sshl v6.4s, v6.4s, v2.4s\n"
      "sshl v7.4s, v7.4s, v2.4s\n"
      "sshl v8.4s, v8.4s, v2.4s\n"
      "sqxtn v5.4h, v5.4s\n"
      "sqxtn2 v5.8h, v6.4s\n"
      "sqxtn v7.4h, v7.4s\n"
      "sqxtn2 v7.8h, v8.4s\n"
      "sqxtun v5.8b, v5.8h\n"
      "sqxtun v7.8b, v7.8h\n"
      "st1 {v5.2s}, [%x[destination]], #8\n"
      "st1 {v7.2s}, [x1], #8\n"

      "bne 1b\n"
      "2:"
      "ld1 {v5.s}[0], [%x[source]], #4\n"
      "ld1 {v7.s}[0], [x0], #4\n"
      "add v5.4s, v5.4s, v3.4s\n"
      "add v7.4s, v7.4s, v4.4s\n"
      "mul v5.4s, v5.4s, v0.4s\n"
      "mul v7.4s, v7.4s, v0.4s\n"
      "add v5.4s, v5.4s, v1.4s\n"
      "add v7.4s, v7.4s, v1.4s\n"
      "sshl v5.4s, v5.4s, v2.4s\n"
      "sshl v7.4s, v7.4s, v2.4s\n"
      "sqxtn v5.4h, v5.4s\n"
      "sqxtn v7.4h, v7.4s\n"
      "sqxtun v5.8b, v5.8h\n"
      "sqxtun v7.8b, v7.8h\n"
      "st1 {v5.b}[0], [%x[destination]], #1\n"
      "st1 {v7.b}[0], [x1], #1\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [stride] "+r"(stride), [shift] "+r"(shift),
        [destination] "+r"(destination), [offsets] "+r"(offsets),
        [source] "+r"(source), [destination_stride] "+r"(destination_stride),
        [rounding_offset] "+r"(rounding_offset)
      :
      : "x0", "x1", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "cc",
        "memory");
}

void qnt_2x8_2_aligned(const std::int32_t* source, std::int32_t count,
                       std::int32_t stride, const std::int32_t* offsets,
                       std::uint8_t* destination,
                       std::int32_t destination_stride,
                       std::int32_t multiplicative_offset,
                       std::int32_t rounding_offset, std::int32_t shift) {
  asm volatile(
      "dup v0.4s, %w[multiplicative_offset]\n"
      "dup v1.4s, %w[rounding_offset]\n"
      "dup v2.4s, %w[shift]\n"
      "ld1r {v3.4s}, [%x[offsets]], #4\n"
      "ld1r {v4.4s}, [%x[offsets]], #4\n"
      "add x0, %x[source], %x[stride]\n"
      "add x1, %x[destination], %x[destination_stride]\n"
      "subs %x[count], %x[count], #2\n"
      "beq 2f\n"

      "1:"
      "subs %x[count], %x[count], #8\n"
      "ld1 {v5.4s, v6.4s}, [%x[source]], #32\n"
      "ld1 {v7.4s, v8.4s}, [x0], #32\n"
      "prfm pldl1keep, [%x[source]]\n"
      "prfm pldl1keep, [x0]\n"
      "add v5.4s, v5.4s, v3.4s\n"
      "add v6.4s, v6.4s, v3.4s\n"
      "add v7.4s, v7.4s, v4.4s\n"
      "add v8.4s, v8.4s, v4.4s\n"
      "mul v5.4s, v5.4s, v0.4s\n"
      "mul v6.4s, v6.4s, v0.4s\n"
      "mul v7.4s, v7.4s, v0.4s\n"
      "mul v8.4s, v8.4s, v0.4s\n"
      "add v5.4s, v5.4s, v1.4s\n"
      "add v6.4s, v6.4s, v1.4s\n"
      "add v7.4s, v7.4s, v1.4s\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "sshl v5.4s, v5.4s, v2.4s\n"
      "sshl v6.4s, v6.4s, v2.4s\n"
      "sshl v7.4s, v7.4s, v2.4s\n"
      "sshl v8.4s, v8.4s, v2.4s\n"
      "sqxtn v5.4h, v5.4s\n"
      "sqxtn2 v5.8h, v6.4s\n"
      "sqxtn v7.4h, v7.4s\n"
      "sqxtn2 v7.8h, v8.4s\n"
      "sqxtun v5.8b, v5.8h\n"
      "sqxtun v7.8b, v7.8h\n"
      "st1 {v5.2s}, [%x[destination]], #8\n"
      "st1 {v7.2s}, [x1], #8\n"

      "bne 1b\n"
      "2:"
      "ld1 {v5.2s}, [%x[source]], #8\n"
      "ld1 {v7.2s}, [x0], #8\n"
      "add v5.4s, v5.4s, v3.4s\n"
      "add v7.4s, v7.4s, v4.4s\n"
      "mul v5.4s, v5.4s, v0.4s\n"
      "mul v7.4s, v7.4s, v0.4s\n"
      "add v5.4s, v5.4s, v1.4s\n"
      "add v7.4s, v7.4s, v1.4s\n"
      "sshl v5.4s, v5.4s, v2.4s\n"
      "sshl v7.4s, v7.4s, v2.4s\n"
      "sqxtn v5.4h, v5.4s\n"
      "sqxtn v7.4h, v7.4s\n"
      "sqxtun v5.8b, v5.8h\n"
      "sqxtun v7.8b, v7.8h\n"
      "st1 {v5.h}[0], [%x[destination]], #2\n"
      "st1 {v7.h}[0], [x1], #2\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [stride] "+r"(stride), [shift] "+r"(shift),
        [destination] "+r"(destination), [offsets] "+r"(offsets),
        [source] "+r"(source), [destination_stride] "+r"(destination_stride),
        [rounding_offset] "+r"(rounding_offset)
      :
      : "x0", "x1", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "cc",
        "memory");
}

void qnt_2x8_3_aligned(const std::int32_t* source, std::int32_t count,
                       std::int32_t stride, const std::int32_t* offsets,
                       std::uint8_t* destination,
                       std::int32_t destination_stride,
                       std::int32_t multiplicative_offset,
                       std::int32_t rounding_offset, std::int32_t shift) {
  asm volatile(
      "dup v0.4s, %w[multiplicative_offset]\n"
      "dup v1.4s, %w[rounding_offset]\n"
      "dup v2.4s, %w[shift]\n"
      "ld1r {v3.4s}, [%x[offsets]], #4\n"
      "ld1r {v4.4s}, [%x[offsets]], #4\n"
      "add x0, %x[source], %x[stride]\n"
      "add x1, %x[destination], %x[destination_stride]\n"
      "subs %x[count], %x[count], #3\n"
      "beq 2f\n"

      "1:"
      "subs %x[count], %x[count], #8\n"
      "ld1 {v5.4s, v6.4s}, [%x[source]], #32\n"
      "ld1 {v7.4s, v8.4s}, [x0], #32\n"
      "prfm pldl1keep, [%x[source]]\n"
      "prfm pldl1keep, [x0]\n"
      "add v5.4s, v5.4s, v3.4s\n"
      "add v6.4s, v6.4s, v3.4s\n"
      "add v7.4s, v7.4s, v4.4s\n"
      "add v8.4s, v8.4s, v4.4s\n"
      "mul v5.4s, v5.4s, v0.4s\n"
      "mul v6.4s, v6.4s, v0.4s\n"
      "mul v7.4s, v7.4s, v0.4s\n"
      "mul v8.4s, v8.4s, v0.4s\n"
      "add v5.4s, v5.4s, v1.4s\n"
      "add v6.4s, v6.4s, v1.4s\n"
      "add v7.4s, v7.4s, v1.4s\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "sshl v5.4s, v5.4s, v2.4s\n"
      "sshl v6.4s, v6.4s, v2.4s\n"
      "sshl v7.4s, v7.4s, v2.4s\n"
      "sshl v8.4s, v8.4s, v2.4s\n"
      "sqxtn v5.4h, v5.4s\n"
      "sqxtn2 v5.8h, v6.4s\n"
      "sqxtn v7.4h, v7.4s\n"
      "sqxtn2 v7.8h, v8.4s\n"
      "sqxtun v5.8b, v5.8h\n"
      "sqxtun v7.8b, v7.8h\n"
      "st1 {v5.2s}, [%x[destination]], #8\n"
      "st1 {v7.2s}, [x1], #8\n"

      "bne 1b\n"
      "2:"
      "ld1 {v5.2s}, [%x[source]], #8\n"
      "ld1 {v5.s}[2], [%x[source]], #4\n"
      "ld1 {v7.2s}, [x0], #8\n"
      "ld1 {v7.s}[2], [x0], #4\n"
      "add v5.4s, v5.4s, v3.4s\n"
      "add v7.4s, v7.4s, v4.4s\n"
      "mul v5.4s, v5.4s, v0.4s\n"
      "mul v7.4s, v7.4s, v0.4s\n"
      "add v5.4s, v5.4s, v1.4s\n"
      "add v7.4s, v7.4s, v1.4s\n"
      "sshl v5.4s, v5.4s, v2.4s\n"
      "sshl v7.4s, v7.4s, v2.4s\n"
      "sqxtn v5.4h, v5.4s\n"
      "sqxtn v7.4h, v7.4s\n"
      "sqxtun v5.8b, v5.8h\n"
      "sqxtun v7.8b, v7.8h\n"
      "st1 {v5.h}[0], [%x[destination]], #2\n"
      "st1 {v5.b}[2], [%x[destination]], #1\n"
      "st1 {v7.h}[0], [x1], #2\n"
      "st1 {v7.b}[2], [x1], #1\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [stride] "+r"(stride), [shift] "+r"(shift),
        [destination] "+r"(destination), [offsets] "+r"(offsets),
        [source] "+r"(source), [destination_stride] "+r"(destination_stride),
        [rounding_offset] "+r"(rounding_offset)
      :
      : "x0", "x1", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "cc",
        "memory");
}

void qnt_2x8_4_aligned(const std::int32_t* source, std::int32_t count,
                       std::int32_t stride, const std::int32_t* offsets,
                       std::uint8_t* destination,
                       std::int32_t destination_stride,
                       std::int32_t multiplicative_offset,
                       std::int32_t rounding_offset, std::int32_t shift) {
  asm volatile(
      "dup v0.4s, %w[multiplicative_offset]\n"
      "dup v1.4s, %w[rounding_offset]\n"
      "dup v2.4s, %w[shift]\n"
      "ld1r {v3.4s}, [%x[offsets]], #4\n"
      "ld1r {v4.4s}, [%x[offsets]], #4\n"
      "add x0, %x[source], %x[stride]\n"
      "add x1, %x[destination], %x[destination_stride]\n"
      "subs %x[count], %x[count], #4\n"
      "beq 2f\n"

      "1:"
      "subs %x[count], %x[count], #8\n"
      "ld1 {v5.4s, v6.4s}, [%x[source]], #32\n"
      "ld1 {v7.4s, v8.4s}, [x0], #32\n"
      "prfm pldl1keep, [%x[source]]\n"
      "prfm pldl1keep, [x0]\n"
      "add v5.4s, v5.4s, v3.4s\n"
      "add v6.4s, v6.4s, v3.4s\n"
      "add v7.4s, v7.4s, v4.4s\n"
      "add v8.4s, v8.4s, v4.4s\n"
      "mul v5.4s, v5.4s, v0.4s\n"
      "mul v6.4s, v6.4s, v0.4s\n"
      "mul v7.4s, v7.4s, v0.4s\n"
      "mul v8.4s, v8.4s, v0.4s\n"
      "add v5.4s, v5.4s, v1.4s\n"
      "add v6.4s, v6.4s, v1.4s\n"
      "add v7.4s, v7.4s, v1.4s\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "sshl v5.4s, v5.4s, v2.4s\n"
      "sshl v6.4s, v6.4s, v2.4s\n"
      "sshl v7.4s, v7.4s, v2.4s\n"
      "sshl v8.4s, v8.4s, v2.4s\n"
      "sqxtn v5.4h, v5.4s\n"
      "sqxtn2 v5.8h, v6.4s\n"
      "sqxtn v7.4h, v7.4s\n"
      "sqxtn2 v7.8h, v8.4s\n"
      "sqxtun v5.8b, v5.8h\n"
      "sqxtun v7.8b, v7.8h\n"
      "st1 {v5.2s}, [%x[destination]], #8\n"
      "st1 {v7.2s}, [x1], #8\n"

      "bne 1b\n"
      "2:"
      "ld1 {v5.4s}, [%x[source]], #16\n"
      "ld1 {v7.4s}, [x0], #16\n"
      "add v5.4s, v5.4s, v3.4s\n"
      "add v7.4s, v7.4s, v4.4s\n"
      "mul v5.4s, v5.4s, v0.4s\n"
      "mul v7.4s, v7.4s, v0.4s\n"
      "add v5.4s, v5.4s, v1.4s\n"
      "add v7.4s, v7.4s, v1.4s\n"
      "sshl v5.4s, v5.4s, v2.4s\n"
      "sshl v7.4s, v7.4s, v2.4s\n"
      "sqxtn v5.4h, v5.4s\n"
      "sqxtn v7.4h, v7.4s\n"
      "sqxtun v5.8b, v5.8h\n"
      "sqxtun v7.8b, v7.8h\n"
      "st1 {v5.s}[0], [%x[destination]], #4\n"
      "st1 {v7.s}[0], [x1], #4\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [stride] "+r"(stride), [shift] "+r"(shift),
        [destination] "+r"(destination), [offsets] "+r"(offsets),
        [source] "+r"(source), [destination_stride] "+r"(destination_stride),
        [rounding_offset] "+r"(rounding_offset)
      :
      : "x0", "x1", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "cc",
        "memory");
}

void qnt_2x8_5_aligned(const std::int32_t* source, std::int32_t count,
                       std::int32_t stride, const std::int32_t* offsets,
                       std::uint8_t* destination,
                       std::int32_t destination_stride,
                       std::int32_t multiplicative_offset,
                       std::int32_t rounding_offset, std::int32_t shift) {
  asm volatile(
      "dup v0.4s, %w[multiplicative_offset]\n"
      "dup v1.4s, %w[rounding_offset]\n"
      "dup v2.4s, %w[shift]\n"
      "ld1r {v3.4s}, [%x[offsets]], #4\n"
      "ld1r {v4.4s}, [%x[offsets]], #4\n"
      "add x0, %x[source], %x[stride]\n"
      "add x1, %x[destination], %x[destination_stride]\n"
      "subs %x[count], %x[count], #5\n"
      "beq 2f\n"

      "1:"
      "subs %x[count], %x[count], #8\n"
      "ld1 {v5.4s, v6.4s}, [%x[source]], #32\n"
      "ld1 {v7.4s, v8.4s}, [x0], #32\n"
      "prfm pldl1keep, [%x[source]]\n"
      "prfm pldl1keep, [x0]\n"
      "add v5.4s, v5.4s, v3.4s\n"
      "add v6.4s, v6.4s, v3.4s\n"
      "add v7.4s, v7.4s, v4.4s\n"
      "add v8.4s, v8.4s, v4.4s\n"
      "mul v5.4s, v5.4s, v0.4s\n"
      "mul v6.4s, v6.4s, v0.4s\n"
      "mul v7.4s, v7.4s, v0.4s\n"
      "mul v8.4s, v8.4s, v0.4s\n"
      "add v5.4s, v5.4s, v1.4s\n"
      "add v6.4s, v6.4s, v1.4s\n"
      "add v7.4s, v7.4s, v1.4s\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "sshl v5.4s, v5.4s, v2.4s\n"
      "sshl v6.4s, v6.4s, v2.4s\n"
      "sshl v7.4s, v7.4s, v2.4s\n"
      "sshl v8.4s, v8.4s, v2.4s\n"
      "sqxtn v5.4h, v5.4s\n"
      "sqxtn2 v5.8h, v6.4s\n"
      "sqxtn v7.4h, v7.4s\n"
      "sqxtn2 v7.8h, v8.4s\n"
      "sqxtun v5.8b, v5.8h\n"
      "sqxtun v7.8b, v7.8h\n"
      "st1 {v5.2s}, [%x[destination]], #8\n"
      "st1 {v7.2s}, [x1], #8\n"

      "bne 1b\n"
      "2:"
      "ld1 {v5.4s}, [%x[source]], #16\n"
      "ld1 {v6.s}[0], [%x[source]], #4\n"
      "ld1 {v7.4s}, [x0], #16\n"
      "ld1 {v8.s}[0], [x0], #4\n"
      "add v5.4s, v5.4s, v3.4s\n"
      "add v6.4s, v6.4s, v3.4s\n"
      "add v7.4s, v7.4s, v4.4s\n"
      "add v8.4s, v8.4s, v4.4s\n"
      "mul v5.4s, v5.4s, v0.4s\n"
      "mul v6.4s, v6.4s, v0.4s\n"
      "mul v7.4s, v7.4s, v0.4s\n"
      "mul v8.4s, v8.4s, v0.4s\n"
      "add v5.4s, v5.4s, v1.4s\n"
      "add v6.4s, v6.4s, v1.4s\n"
      "add v7.4s, v7.4s, v1.4s\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "sshl v5.4s, v5.4s, v2.4s\n"
      "sshl v6.4s, v6.4s, v2.4s\n"
      "sshl v7.4s, v7.4s, v2.4s\n"
      "sshl v8.4s, v8.4s, v2.4s\n"
      "sqxtn v5.4h, v5.4s\n"
      "sqxtn2 v5.8h, v6.4s\n"
      "sqxtn v7.4h, v7.4s\n"
      "sqxtn2 v7.8h, v8.4s\n"
      "sqxtun v5.8b, v5.8h\n"
      "sqxtun v7.8b, v7.8h\n"
      "st1 {v5.s}[0], [%x[destination]], #4\n"
      "st1 {v5.b}[4], [%x[destination]], #1\n"
      "st1 {v7.s}[0], [x1], #4\n"
      "st1 {v7.b}[4], [x1], #1\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [stride] "+r"(stride), [shift] "+r"(shift),
        [destination] "+r"(destination), [offsets] "+r"(offsets),
        [source] "+r"(source), [destination_stride] "+r"(destination_stride),
        [rounding_offset] "+r"(rounding_offset)
      :
      : "x0", "x1", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "cc",
        "memory");
}

void qnt_2x8_6_aligned(const std::int32_t* source, std::int32_t count,
                       std::int32_t stride, const std::int32_t* offsets,
                       std::uint8_t* destination,
                       std::int32_t destination_stride,
                       std::int32_t multiplicative_offset,
                       std::int32_t rounding_offset, std::int32_t shift) {
  asm volatile(
      "dup v0.4s, %w[multiplicative_offset]\n"
      "dup v1.4s, %w[rounding_offset]\n"
      "dup v2.4s, %w[shift]\n"
      "ld1r {v3.4s}, [%x[offsets]], #4\n"
      "ld1r {v4.4s}, [%x[offsets]], #4\n"
      "add x0, %x[source], %x[stride]\n"
      "add x1, %x[destination], %x[destination_stride]\n"
      "subs %x[count], %x[count], #6\n"
      "beq 2f\n"

      "1:"
      "subs %x[count], %x[count], #8\n"
      "ld1 {v5.4s, v6.4s}, [%x[source]], #32\n"
      "ld1 {v7.4s, v8.4s}, [x0], #32\n"
      "prfm pldl1keep, [%x[source]]\n"
      "prfm pldl1keep, [x0]\n"
      "add v5.4s, v5.4s, v3.4s\n"
      "add v6.4s, v6.4s, v3.4s\n"
      "add v7.4s, v7.4s, v4.4s\n"
      "add v8.4s, v8.4s, v4.4s\n"
      "mul v5.4s, v5.4s, v0.4s\n"
      "mul v6.4s, v6.4s, v0.4s\n"
      "mul v7.4s, v7.4s, v0.4s\n"
      "mul v8.4s, v8.4s, v0.4s\n"
      "add v5.4s, v5.4s, v1.4s\n"
      "add v6.4s, v6.4s, v1.4s\n"
      "add v7.4s, v7.4s, v1.4s\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "sshl v5.4s, v5.4s, v2.4s\n"
      "sshl v6.4s, v6.4s, v2.4s\n"
      "sshl v7.4s, v7.4s, v2.4s\n"
      "sshl v8.4s, v8.4s, v2.4s\n"
      "sqxtn v5.4h, v5.4s\n"
      "sqxtn2 v5.8h, v6.4s\n"
      "sqxtn v7.4h, v7.4s\n"
      "sqxtn2 v7.8h, v8.4s\n"
      "sqxtun v5.8b, v5.8h\n"
      "sqxtun v7.8b, v7.8h\n"
      "st1 {v5.2s}, [%x[destination]], #8\n"
      "st1 {v7.2s}, [x1], #8\n"

      "bne 1b\n"
      "2:"
      "ld1 {v5.4s}, [%x[source]], #16\n"
      "ld1 {v6.2s}, [%x[source]], #8\n"
      "ld1 {v7.4s}, [x0], #16\n"
      "ld1 {v8.2s}, [x0], #8\n"
      "add v5.4s, v5.4s, v3.4s\n"
      "add v6.4s, v6.4s, v3.4s\n"
      "add v7.4s, v7.4s, v4.4s\n"
      "add v8.4s, v8.4s, v4.4s\n"
      "mul v5.4s, v5.4s, v0.4s\n"
      "mul v6.4s, v6.4s, v0.4s\n"
      "mul v7.4s, v7.4s, v0.4s\n"
      "mul v8.4s, v8.4s, v0.4s\n"
      "add v5.4s, v5.4s, v1.4s\n"
      "add v6.4s, v6.4s, v1.4s\n"
      "add v7.4s, v7.4s, v1.4s\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "sshl v5.4s, v5.4s, v2.4s\n"
      "sshl v6.4s, v6.4s, v2.4s\n"
      "sshl v7.4s, v7.4s, v2.4s\n"
      "sshl v8.4s, v8.4s, v2.4s\n"
      "sqxtn v5.4h, v5.4s\n"
      "sqxtn2 v5.8h, v6.4s\n"
      "sqxtn v7.4h, v7.4s\n"
      "sqxtn2 v7.8h, v8.4s\n"
      "sqxtun v5.8b, v5.8h\n"
      "sqxtun v7.8b, v7.8h\n"
      "st1 {v5.s}[0], [%x[destination]], #4\n"
      "st1 {v5.h}[2], [%x[destination]], #2\n"
      "st1 {v7.s}[0], [x1], #4\n"
      "st1 {v7.h}[2], [x1], #2\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [stride] "+r"(stride), [shift] "+r"(shift),
        [destination] "+r"(destination), [offsets] "+r"(offsets),
        [source] "+r"(source), [destination_stride] "+r"(destination_stride),
        [rounding_offset] "+r"(rounding_offset)
      :
      : "x0", "x1", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "cc",
        "memory");
}

void qnt_2x8_7_aligned(const std::int32_t* source, std::int32_t count,
                       std::int32_t stride, const std::int32_t* offsets,
                       std::uint8_t* destination,
                       std::int32_t destination_stride,
                       std::int32_t multiplicative_offset,
                       std::int32_t rounding_offset, std::int32_t shift) {
  asm volatile(
      "dup v0.4s, %w[multiplicative_offset]\n"
      "dup v1.4s, %w[rounding_offset]\n"
      "dup v2.4s, %w[shift]\n"
      "ld1r {v3.4s}, [%x[offsets]], #4\n"
      "ld1r {v4.4s}, [%x[offsets]], #4\n"
      "add x0, %x[source], %x[stride]\n"
      "add x1, %x[destination], %x[destination_stride]\n"
      "subs %x[count], %x[count], #7\n"
      "beq 2f\n"

      "1:"
      "subs %x[count], %x[count], #8\n"
      "ld1 {v5.4s, v6.4s}, [%x[source]], #32\n"
      "ld1 {v7.4s, v8.4s}, [x0], #32\n"
      "prfm pldl1keep, [%x[source]]\n"
      "prfm pldl1keep, [x0]\n"
      "add v5.4s, v5.4s, v3.4s\n"
      "add v6.4s, v6.4s, v3.4s\n"
      "add v7.4s, v7.4s, v4.4s\n"
      "add v8.4s, v8.4s, v4.4s\n"
      "mul v5.4s, v5.4s, v0.4s\n"
      "mul v6.4s, v6.4s, v0.4s\n"
      "mul v7.4s, v7.4s, v0.4s\n"
      "mul v8.4s, v8.4s, v0.4s\n"
      "add v5.4s, v5.4s, v1.4s\n"
      "add v6.4s, v6.4s, v1.4s\n"
      "add v7.4s, v7.4s, v1.4s\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "sshl v5.4s, v5.4s, v2.4s\n"
      "sshl v6.4s, v6.4s, v2.4s\n"
      "sshl v7.4s, v7.4s, v2.4s\n"
      "sshl v8.4s, v8.4s, v2.4s\n"
      "sqxtn v5.4h, v5.4s\n"
      "sqxtn2 v5.8h, v6.4s\n"
      "sqxtn v7.4h, v7.4s\n"
      "sqxtn2 v7.8h, v8.4s\n"
      "sqxtun v5.8b, v5.8h\n"
      "sqxtun v7.8b, v7.8h\n"
      "st1 {v5.2s}, [%x[destination]], #8\n"
      "st1 {v7.2s}, [x1], #8\n"

      "bne 1b\n"
      "2:"
      "ld1 {v5.4s}, [%x[source]], #16\n"
      "ld1 {v6.2s}, [%x[source]], #8\n"
      "ld1 {v6.s}[2], [%x[source]], #4\n"
      "ld1 {v7.4s}, [x0], #16\n"
      "ld1 {v8.2s}, [x0], #8\n"
      "ld1 {v8.s}[2], [x0], #4\n"
      "add v5.4s, v5.4s, v3.4s\n"
      "add v6.4s, v6.4s, v3.4s\n"
      "add v7.4s, v7.4s, v4.4s\n"
      "add v8.4s, v8.4s, v4.4s\n"
      "mul v5.4s, v5.4s, v0.4s\n"
      "mul v6.4s, v6.4s, v0.4s\n"
      "mul v7.4s, v7.4s, v0.4s\n"
      "mul v8.4s, v8.4s, v0.4s\n"
      "add v5.4s, v5.4s, v1.4s\n"
      "add v6.4s, v6.4s, v1.4s\n"
      "add v7.4s, v7.4s, v1.4s\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "sshl v5.4s, v5.4s, v2.4s\n"
      "sshl v6.4s, v6.4s, v2.4s\n"
      "sshl v7.4s, v7.4s, v2.4s\n"
      "sshl v8.4s, v8.4s, v2.4s\n"
      "sqxtn v5.4h, v5.4s\n"
      "sqxtn2 v5.8h, v6.4s\n"
      "sqxtn v7.4h, v7.4s\n"
      "sqxtn2 v7.8h, v8.4s\n"
      "sqxtun v5.8b, v5.8h\n"
      "sqxtun v7.8b, v7.8h\n"
      "st1 {v5.s}[0], [%x[destination]], #4\n"
      "st1 {v5.h}[2], [%x[destination]], #2\n"
      "st1 {v5.b}[6], [%x[destination]], #1\n"
      "st1 {v7.s}[0], [x1], #4\n"
      "st1 {v7.h}[2], [x1], #2\n"
      "st1 {v7.b}[6], [x1], #1\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [stride] "+r"(stride), [shift] "+r"(shift),
        [destination] "+r"(destination), [offsets] "+r"(offsets),
        [source] "+r"(source), [destination_stride] "+r"(destination_stride),
        [rounding_offset] "+r"(rounding_offset)
      :
      : "x0", "x1", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "cc",
        "memory");
}

void qnt_3x8_aligned(const std::int32_t* source, std::int32_t count,
                     std::int32_t stride, const std::int32_t* offsets,
                     std::uint8_t* destination, std::int32_t destination_stride,
                     std::int32_t multiplicative_offset,
                     std::int32_t rounding_offset, std::int32_t shift) {
  asm volatile(
      "dup v0.4s, %w[multiplicative_offset]\n"
      "dup v1.4s, %w[rounding_offset]\n"
      "dup v2.4s, %w[shift]\n"
      "ld1r {v3.4s}, [%x[offsets]], #4\n"
      "ld1r {v4.4s}, [%x[offsets]], #4\n"
      "ld1r {v5.4s}, [%x[offsets]], #4\n"
      "add x0, %x[source], %x[stride]\n"
      "add x1, %x[destination], %x[destination_stride]\n"
      "add x2, x0, %x[stride]\n"
      "add x3, x1, %x[destination_stride]\n"

      "1:"
      "subs %x[count], %x[count], #8\n"
      "ld1 {v6.4s, v7.4s}, [%x[source]], #32\n"
      "ld1 {v8.4s, v9.4s}, [x0], #32\n"
      "ld1 {v10.4s, v11.4s}, [x2], #32\n"
      "prfm pldl1keep, [%x[source]]\n"
      "prfm pldl1keep, [x0]\n"
      "prfm pldl1keep, [x2]\n"
      "add v6.4s, v6.4s, v3.4s\n"
      "add v7.4s, v7.4s, v3.4s\n"
      "add v8.4s, v8.4s, v4.4s\n"
      "add v9.4s, v9.4s, v4.4s\n"
      "add v10.4s, v10.4s, v5.4s\n"
      "add v11.4s, v11.4s, v5.4s\n"
      "mul v6.4s, v6.4s, v0.4s\n"
      "mul v7.4s, v7.4s, v0.4s\n"
      "mul v8.4s, v8.4s, v0.4s\n"
      "mul v9.4s, v9.4s, v0.4s\n"
      "mul v10.4s, v10.4s, v0.4s\n"
      "mul v11.4s, v11.4s, v0.4s\n"
      "add v6.4s, v6.4s, v1.4s\n"
      "add v7.4s, v7.4s, v1.4s\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "add v9.4s, v9.4s, v1.4s\n"
      "add v10.4s, v10.4s, v1.4s\n"
      "add v11.4s, v11.4s, v1.4s\n"
      "sshl v6.4s, v6.4s, v2.4s\n"
      "sshl v7.4s, v7.4s, v2.4s\n"
      "sshl v8.4s, v8.4s, v2.4s\n"
      "sshl v9.4s, v9.4s, v2.4s\n"
      "sshl v10.4s, v10.4s, v2.4s\n"
      "sshl v11.4s, v11.4s, v2.4s\n"
      "sqxtn v6.4h, v6.4s\n"
      "sqxtn2 v6.8h, v7.4s\n"
      "sqxtn v8.4h, v8.4s\n"
      "sqxtn2 v8.8h, v9.4s\n"
      "sqxtn v10.4h, v10.4s\n"
      "sqxtn2 v10.8h, v11.4s\n"
      "sqxtun v6.8b, v6.8h\n"
      "sqxtun v8.8b, v8.8h\n"
      "sqxtun v10.8b, v10.8h\n"
      "st1 {v6.2s}, [%x[destination]], #8\n"
      "st1 {v8.2s}, [x1], #8\n"
      "st1 {v10.2s}, [x3], #8\n"

      "bne 1b\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [stride] "+r"(stride), [shift] "+r"(shift),
        [destination] "+r"(destination), [offsets] "+r"(offsets),
        [source] "+r"(source), [destination_stride] "+r"(destination_stride),
        [rounding_offset] "+r"(rounding_offset)
      :
      : "x0", "x1", "x2", "x3", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
        "v8", "v9", "v10", "v11", "cc", "memory");
}

void qnt_3x8_1_aligned(const std::int32_t* source, std::int32_t count,
                       std::int32_t stride, const std::int32_t* offsets,
                       std::uint8_t* destination,
                       std::int32_t destination_stride,
                       std::int32_t multiplicative_offset,
                       std::int32_t rounding_offset, std::int32_t shift) {
  asm volatile(
      "dup v0.4s, %w[multiplicative_offset]\n"
      "dup v1.4s, %w[rounding_offset]\n"
      "dup v2.4s, %w[shift]\n"
      "ld1r {v3.4s}, [%x[offsets]], #4\n"
      "ld1r {v4.4s}, [%x[offsets]], #4\n"
      "ld1r {v5.4s}, [%x[offsets]], #4\n"
      "add x0, %x[source], %x[stride]\n"
      "add x1, %x[destination], %x[destination_stride]\n"
      "add x2, x0, %x[stride]\n"
      "add x3, x1, %x[destination_stride]\n"
      "subs %x[count], %x[count], #1\n"
      "beq 2f\n"

      "1:"
      "subs %x[count], %x[count], #8\n"
      "ld1 {v6.4s, v7.4s}, [%x[source]], #32\n"
      "ld1 {v8.4s, v9.4s}, [x0], #32\n"
      "ld1 {v10.4s, v11.4s}, [x2], #32\n"
      "prfm pldl1keep, [%x[source]]\n"
      "prfm pldl1keep, [x0]\n"
      "prfm pldl1keep, [x2]\n"
      "add v6.4s, v6.4s, v3.4s\n"
      "add v7.4s, v7.4s, v3.4s\n"
      "add v8.4s, v8.4s, v4.4s\n"
      "add v9.4s, v9.4s, v4.4s\n"
      "add v10.4s, v10.4s, v5.4s\n"
      "add v11.4s, v11.4s, v5.4s\n"
      "mul v6.4s, v6.4s, v0.4s\n"
      "mul v7.4s, v7.4s, v0.4s\n"
      "mul v8.4s, v8.4s, v0.4s\n"
      "mul v9.4s, v9.4s, v0.4s\n"
      "mul v10.4s, v10.4s, v0.4s\n"
      "mul v11.4s, v11.4s, v0.4s\n"
      "add v6.4s, v6.4s, v1.4s\n"
      "add v7.4s, v7.4s, v1.4s\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "add v9.4s, v9.4s, v1.4s\n"
      "add v10.4s, v10.4s, v1.4s\n"
      "add v11.4s, v11.4s, v1.4s\n"
      "sshl v6.4s, v6.4s, v2.4s\n"
      "sshl v7.4s, v7.4s, v2.4s\n"
      "sshl v8.4s, v8.4s, v2.4s\n"
      "sshl v9.4s, v9.4s, v2.4s\n"
      "sshl v10.4s, v10.4s, v2.4s\n"
      "sshl v11.4s, v11.4s, v2.4s\n"
      "sqxtn v6.4h, v6.4s\n"
      "sqxtn2 v6.8h, v7.4s\n"
      "sqxtn v8.4h, v8.4s\n"
      "sqxtn2 v8.8h, v9.4s\n"
      "sqxtn v10.4h, v10.4s\n"
      "sqxtn2 v10.8h, v11.4s\n"
      "sqxtun v6.8b, v6.8h\n"
      "sqxtun v8.8b, v8.8h\n"
      "sqxtun v10.8b, v10.8h\n"
      "st1 {v6.2s}, [%x[destination]], #8\n"
      "st1 {v8.2s}, [x1], #8\n"
      "st1 {v10.2s}, [x3], #8\n"

      "bne 1b\n"
      "2:"
      "ld1 {v6.s}[0], [%x[source]], #4\n"
      "ld1 {v8.s}[0], [x0], #4\n"
      "ld1 {v10.s}[0], [x2], #4\n"
      "add v6.4s, v6.4s, v3.4s\n"
      "add v8.4s, v8.4s, v4.4s\n"
      "add v10.4s, v10.4s, v5.4s\n"
      "mul v6.4s, v6.4s, v0.4s\n"
      "mul v8.4s, v8.4s, v0.4s\n"
      "mul v10.4s, v10.4s, v0.4s\n"
      "add v6.4s, v6.4s, v1.4s\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "add v10.4s, v10.4s, v1.4s\n"
      "sshl v6.4s, v6.4s, v2.4s\n"
      "sshl v8.4s, v8.4s, v2.4s\n"
      "sshl v10.4s, v10.4s, v2.4s\n"
      "sqxtn v6.4h, v6.4s\n"
      "sqxtn v8.4h, v8.4s\n"
      "sqxtn v10.4h, v10.4s\n"
      "sqxtun v6.8b, v6.8h\n"
      "sqxtun v8.8b, v8.8h\n"
      "sqxtun v10.8b, v10.8h\n"
      "st1 {v6.b}[0], [%x[destination]], #1\n"
      "st1 {v8.b}[0], [x1], #1\n"
      "st1 {v10.b}[0], [x3], #1\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [stride] "+r"(stride), [shift] "+r"(shift),
        [destination] "+r"(destination), [offsets] "+r"(offsets),
        [source] "+r"(source), [destination_stride] "+r"(destination_stride),
        [rounding_offset] "+r"(rounding_offset)
      :
      : "x0", "x1", "x2", "x3", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
        "v8", "v9", "v10", "v11", "cc", "memory");
}

void qnt_3x8_2_aligned(const std::int32_t* source, std::int32_t count,
                       std::int32_t stride, const std::int32_t* offsets,
                       std::uint8_t* destination,
                       std::int32_t destination_stride,
                       std::int32_t multiplicative_offset,
                       std::int32_t rounding_offset, std::int32_t shift) {
  asm volatile(
      "dup v0.4s, %w[multiplicative_offset]\n"
      "dup v1.4s, %w[rounding_offset]\n"
      "dup v2.4s, %w[shift]\n"
      "ld1r {v3.4s}, [%x[offsets]], #4\n"
      "ld1r {v4.4s}, [%x[offsets]], #4\n"
      "ld1r {v5.4s}, [%x[offsets]], #4\n"
      "add x0, %x[source], %x[stride]\n"
      "add x1, %x[destination], %x[destination_stride]\n"
      "add x2, x0, %x[stride]\n"
      "add x3, x1, %x[destination_stride]\n"
      "subs %x[count], %x[count], #2\n"
      "beq 2f\n"

      "1:"
      "subs %x[count], %x[count], #8\n"
      "ld1 {v6.4s, v7.4s}, [%x[source]], #32\n"
      "ld1 {v8.4s, v9.4s}, [x0], #32\n"
      "ld1 {v10.4s, v11.4s}, [x2], #32\n"
      "prfm pldl1keep, [%x[source]]\n"
      "prfm pldl1keep, [x0]\n"
      "prfm pldl1keep, [x2]\n"
      "add v6.4s, v6.4s, v3.4s\n"
      "add v7.4s, v7.4s, v3.4s\n"
      "add v8.4s, v8.4s, v4.4s\n"
      "add v9.4s, v9.4s, v4.4s\n"
      "add v10.4s, v10.4s, v5.4s\n"
      "add v11.4s, v11.4s, v5.4s\n"
      "mul v6.4s, v6.4s, v0.4s\n"
      "mul v7.4s, v7.4s, v0.4s\n"
      "mul v8.4s, v8.4s, v0.4s\n"
      "mul v9.4s, v9.4s, v0.4s\n"
      "mul v10.4s, v10.4s, v0.4s\n"
      "mul v11.4s, v11.4s, v0.4s\n"
      "add v6.4s, v6.4s, v1.4s\n"
      "add v7.4s, v7.4s, v1.4s\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "add v9.4s, v9.4s, v1.4s\n"
      "add v10.4s, v10.4s, v1.4s\n"
      "add v11.4s, v11.4s, v1.4s\n"
      "sshl v6.4s, v6.4s, v2.4s\n"
      "sshl v7.4s, v7.4s, v2.4s\n"
      "sshl v8.4s, v8.4s, v2.4s\n"
      "sshl v9.4s, v9.4s, v2.4s\n"
      "sshl v10.4s, v10.4s, v2.4s\n"
      "sshl v11.4s, v11.4s, v2.4s\n"
      "sqxtn v6.4h, v6.4s\n"
      "sqxtn2 v6.8h, v7.4s\n"
      "sqxtn v8.4h, v8.4s\n"
      "sqxtn2 v8.8h, v9.4s\n"
      "sqxtn v10.4h, v10.4s\n"
      "sqxtn2 v10.8h, v11.4s\n"
      "sqxtun v6.8b, v6.8h\n"
      "sqxtun v8.8b, v8.8h\n"
      "sqxtun v10.8b, v10.8h\n"
      "st1 {v6.2s}, [%x[destination]], #8\n"
      "st1 {v8.2s}, [x1], #8\n"
      "st1 {v10.2s}, [x3], #8\n"

      "bne 1b\n"
      "2:"
      "ld1 {v6.2s}, [%x[source]], #8\n"
      "ld1 {v8.2s}, [x0], #8\n"
      "ld1 {v10.2s}, [x2], #8\n"
      "add v6.4s, v6.4s, v3.4s\n"
      "add v8.4s, v8.4s, v4.4s\n"
      "add v10.4s, v10.4s, v5.4s\n"
      "mul v6.4s, v6.4s, v0.4s\n"
      "mul v8.4s, v8.4s, v0.4s\n"
      "mul v10.4s, v10.4s, v0.4s\n"
      "add v6.4s, v6.4s, v1.4s\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "add v10.4s, v10.4s, v1.4s\n"
      "sshl v6.4s, v6.4s, v2.4s\n"
      "sshl v8.4s, v8.4s, v2.4s\n"
      "sshl v10.4s, v10.4s, v2.4s\n"
      "sqxtn v6.4h, v6.4s\n"
      "sqxtn v8.4h, v8.4s\n"
      "sqxtn v10.4h, v10.4s\n"
      "sqxtun v6.8b, v6.8h\n"
      "sqxtun v8.8b, v8.8h\n"
      "sqxtun v10.8b, v10.8h\n"
      "st1 {v6.h}[0], [%x[destination]], #2\n"
      "st1 {v8.h}[0], [x1], #2\n"
      "st1 {v10.h}[0], [x3], #2\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [stride] "+r"(stride), [shift] "+r"(shift),
        [destination] "+r"(destination), [offsets] "+r"(offsets),
        [source] "+r"(source), [destination_stride] "+r"(destination_stride),
        [rounding_offset] "+r"(rounding_offset)
      :
      : "x0", "x1", "x2", "x3", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
        "v8", "v9", "v10", "v11", "cc", "memory");
}

void qnt_3x8_3_aligned(const std::int32_t* source, std::int32_t count,
                       std::int32_t stride, const std::int32_t* offsets,
                       std::uint8_t* destination,
                       std::int32_t destination_stride,
                       std::int32_t multiplicative_offset,
                       std::int32_t rounding_offset, std::int32_t shift) {
  asm volatile(
      "dup v0.4s, %w[multiplicative_offset]\n"
      "dup v1.4s, %w[rounding_offset]\n"
      "dup v2.4s, %w[shift]\n"
      "ld1r {v3.4s}, [%x[offsets]], #4\n"
      "ld1r {v4.4s}, [%x[offsets]], #4\n"
      "ld1r {v5.4s}, [%x[offsets]], #4\n"
      "add x0, %x[source], %x[stride]\n"
      "add x1, %x[destination], %x[destination_stride]\n"
      "add x2, x0, %x[stride]\n"
      "add x3, x1, %x[destination_stride]\n"
      "subs %x[count], %x[count], #3\n"
      "beq 2f\n"

      "1:"
      "subs %x[count], %x[count], #8\n"
      "ld1 {v6.4s, v7.4s}, [%x[source]], #32\n"
      "ld1 {v8.4s, v9.4s}, [x0], #32\n"
      "ld1 {v10.4s, v11.4s}, [x2], #32\n"
      "prfm pldl1keep, [%x[source]]\n"
      "prfm pldl1keep, [x0]\n"
      "prfm pldl1keep, [x2]\n"
      "add v6.4s, v6.4s, v3.4s\n"
      "add v7.4s, v7.4s, v3.4s\n"
      "add v8.4s, v8.4s, v4.4s\n"
      "add v9.4s, v9.4s, v4.4s\n"
      "add v10.4s, v10.4s, v5.4s\n"
      "add v11.4s, v11.4s, v5.4s\n"
      "mul v6.4s, v6.4s, v0.4s\n"
      "mul v7.4s, v7.4s, v0.4s\n"
      "mul v8.4s, v8.4s, v0.4s\n"
      "mul v9.4s, v9.4s, v0.4s\n"
      "mul v10.4s, v10.4s, v0.4s\n"
      "mul v11.4s, v11.4s, v0.4s\n"
      "add v6.4s, v6.4s, v1.4s\n"
      "add v7.4s, v7.4s, v1.4s\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "add v9.4s, v9.4s, v1.4s\n"
      "add v10.4s, v10.4s, v1.4s\n"
      "add v11.4s, v11.4s, v1.4s\n"
      "sshl v6.4s, v6.4s, v2.4s\n"
      "sshl v7.4s, v7.4s, v2.4s\n"
      "sshl v8.4s, v8.4s, v2.4s\n"
      "sshl v9.4s, v9.4s, v2.4s\n"
      "sshl v10.4s, v10.4s, v2.4s\n"
      "sshl v11.4s, v11.4s, v2.4s\n"
      "sqxtn v6.4h, v6.4s\n"
      "sqxtn2 v6.8h, v7.4s\n"
      "sqxtn v8.4h, v8.4s\n"
      "sqxtn2 v8.8h, v9.4s\n"
      "sqxtn v10.4h, v10.4s\n"
      "sqxtn2 v10.8h, v11.4s\n"
      "sqxtun v6.8b, v6.8h\n"
      "sqxtun v8.8b, v8.8h\n"
      "sqxtun v10.8b, v10.8h\n"
      "st1 {v6.2s}, [%x[destination]], #8\n"
      "st1 {v8.2s}, [x1], #8\n"
      "st1 {v10.2s}, [x3], #8\n"

      "bne 1b\n"
      "2:"
      "ld1 {v6.2s}, [%x[source]], #8\n"
      "ld1 {v6.s}[2], [%x[source]], #4\n"
      "ld1 {v8.2s}, [x0], #8\n"
      "ld1 {v8.s}[2], [x0], #4\n"
      "ld1 {v10.2s}, [x2], #8\n"
      "ld1 {v10.s}[2], [x2], #4\n"
      "add v6.4s, v6.4s, v3.4s\n"
      "add v8.4s, v8.4s, v4.4s\n"
      "add v10.4s, v10.4s, v5.4s\n"
      "mul v6.4s, v6.4s, v0.4s\n"
      "mul v8.4s, v8.4s, v0.4s\n"
      "mul v10.4s, v10.4s, v0.4s\n"
      "add v6.4s, v6.4s, v1.4s\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "add v10.4s, v10.4s, v1.4s\n"
      "sshl v6.4s, v6.4s, v2.4s\n"
      "sshl v8.4s, v8.4s, v2.4s\n"
      "sshl v10.4s, v10.4s, v2.4s\n"
      "sqxtn v6.4h, v6.4s\n"
      "sqxtn v8.4h, v8.4s\n"
      "sqxtn v10.4h, v10.4s\n"
      "sqxtun v6.8b, v6.8h\n"
      "sqxtun v8.8b, v8.8h\n"
      "sqxtun v10.8b, v10.8h\n"
      "st1 {v6.h}[0], [%x[destination]], #2\n"
      "st1 {v6.b}[2], [%x[destination]], #1\n"
      "st1 {v8.h}[0], [x1], #2\n"
      "st1 {v8.b}[2], [x1], #1\n"
      "st1 {v10.h}[0], [x3], #2\n"
      "st1 {v10.b}[2], [x3], #1\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [stride] "+r"(stride), [shift] "+r"(shift),
        [destination] "+r"(destination), [offsets] "+r"(offsets),
        [source] "+r"(source), [destination_stride] "+r"(destination_stride),
        [rounding_offset] "+r"(rounding_offset)
      :
      : "x0", "x1", "x2", "x3", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
        "v8", "v9", "v10", "v11", "cc", "memory");
}

void qnt_3x8_4_aligned(const std::int32_t* source, std::int32_t count,
                       std::int32_t stride, const std::int32_t* offsets,
                       std::uint8_t* destination,
                       std::int32_t destination_stride,
                       std::int32_t multiplicative_offset,
                       std::int32_t rounding_offset, std::int32_t shift) {
  asm volatile(
      "dup v0.4s, %w[multiplicative_offset]\n"
      "dup v1.4s, %w[rounding_offset]\n"
      "dup v2.4s, %w[shift]\n"
      "ld1r {v3.4s}, [%x[offsets]], #4\n"
      "ld1r {v4.4s}, [%x[offsets]], #4\n"
      "ld1r {v5.4s}, [%x[offsets]], #4\n"
      "add x0, %x[source], %x[stride]\n"
      "add x1, %x[destination], %x[destination_stride]\n"
      "add x2, x0, %x[stride]\n"
      "add x3, x1, %x[destination_stride]\n"
      "subs %x[count], %x[count], #4\n"
      "beq 2f\n"

      "1:"
      "subs %x[count], %x[count], #8\n"
      "ld1 {v6.4s, v7.4s}, [%x[source]], #32\n"
      "ld1 {v8.4s, v9.4s}, [x0], #32\n"
      "ld1 {v10.4s, v11.4s}, [x2], #32\n"
      "prfm pldl1keep, [%x[source]]\n"
      "prfm pldl1keep, [x0]\n"
      "prfm pldl1keep, [x2]\n"
      "add v6.4s, v6.4s, v3.4s\n"
      "add v7.4s, v7.4s, v3.4s\n"
      "add v8.4s, v8.4s, v4.4s\n"
      "add v9.4s, v9.4s, v4.4s\n"
      "add v10.4s, v10.4s, v5.4s\n"
      "add v11.4s, v11.4s, v5.4s\n"
      "mul v6.4s, v6.4s, v0.4s\n"
      "mul v7.4s, v7.4s, v0.4s\n"
      "mul v8.4s, v8.4s, v0.4s\n"
      "mul v9.4s, v9.4s, v0.4s\n"
      "mul v10.4s, v10.4s, v0.4s\n"
      "mul v11.4s, v11.4s, v0.4s\n"
      "add v6.4s, v6.4s, v1.4s\n"
      "add v7.4s, v7.4s, v1.4s\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "add v9.4s, v9.4s, v1.4s\n"
      "add v10.4s, v10.4s, v1.4s\n"
      "add v11.4s, v11.4s, v1.4s\n"
      "sshl v6.4s, v6.4s, v2.4s\n"
      "sshl v7.4s, v7.4s, v2.4s\n"
      "sshl v8.4s, v8.4s, v2.4s\n"
      "sshl v9.4s, v9.4s, v2.4s\n"
      "sshl v10.4s, v10.4s, v2.4s\n"
      "sshl v11.4s, v11.4s, v2.4s\n"
      "sqxtn v6.4h, v6.4s\n"
      "sqxtn2 v6.8h, v7.4s\n"
      "sqxtn v8.4h, v8.4s\n"
      "sqxtn2 v8.8h, v9.4s\n"
      "sqxtn v10.4h, v10.4s\n"
      "sqxtn2 v10.8h, v11.4s\n"
      "sqxtun v6.8b, v6.8h\n"
      "sqxtun v8.8b, v8.8h\n"
      "sqxtun v10.8b, v10.8h\n"
      "st1 {v6.2s}, [%x[destination]], #8\n"
      "st1 {v8.2s}, [x1], #8\n"
      "st1 {v10.2s}, [x3], #8\n"

      "bne 1b\n"
      "2:"
      "ld1 {v6.4s}, [%x[source]], #16\n"
      "ld1 {v8.4s}, [x0], #16\n"
      "ld1 {v10.4s}, [x2], #16\n"
      "add v6.4s, v6.4s, v3.4s\n"
      "add v8.4s, v8.4s, v4.4s\n"
      "add v10.4s, v10.4s, v5.4s\n"
      "mul v6.4s, v6.4s, v0.4s\n"
      "mul v8.4s, v8.4s, v0.4s\n"
      "mul v10.4s, v10.4s, v0.4s\n"
      "add v6.4s, v6.4s, v1.4s\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "add v10.4s, v10.4s, v1.4s\n"
      "sshl v6.4s, v6.4s, v2.4s\n"
      "sshl v8.4s, v8.4s, v2.4s\n"
      "sshl v10.4s, v10.4s, v2.4s\n"
      "sqxtn v6.4h, v6.4s\n"
      "sqxtn v8.4h, v8.4s\n"
      "sqxtn v10.4h, v10.4s\n"
      "sqxtun v6.8b, v6.8h\n"
      "sqxtun v8.8b, v8.8h\n"
      "sqxtun v10.8b, v10.8h\n"
      "st1 {v6.s}[0], [%x[destination]], #4\n"
      "st1 {v8.s}[0], [x1], #4\n"
      "st1 {v10.s}[0], [x3], #4\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [stride] "+r"(stride), [shift] "+r"(shift),
        [destination] "+r"(destination), [offsets] "+r"(offsets),
        [source] "+r"(source), [destination_stride] "+r"(destination_stride),
        [rounding_offset] "+r"(rounding_offset)
      :
      : "x0", "x1", "x2", "x3", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
        "v8", "v9", "v10", "v11", "cc", "memory");
}

void qnt_3x8_5_aligned(const std::int32_t* source, std::int32_t count,
                       std::int32_t stride, const std::int32_t* offsets,
                       std::uint8_t* destination,
                       std::int32_t destination_stride,
                       std::int32_t multiplicative_offset,
                       std::int32_t rounding_offset, std::int32_t shift) {
  asm volatile(
      "dup v0.4s, %w[multiplicative_offset]\n"
      "dup v1.4s, %w[rounding_offset]\n"
      "dup v2.4s, %w[shift]\n"
      "ld1r {v3.4s}, [%x[offsets]], #4\n"
      "ld1r {v4.4s}, [%x[offsets]], #4\n"
      "ld1r {v5.4s}, [%x[offsets]], #4\n"
      "add x0, %x[source], %x[stride]\n"
      "add x1, %x[destination], %x[destination_stride]\n"
      "add x2, x0, %x[stride]\n"
      "add x3, x1, %x[destination_stride]\n"
      "subs %x[count], %x[count], #5\n"
      "beq 2f\n"

      "1:"
      "subs %x[count], %x[count], #8\n"
      "ld1 {v6.4s, v7.4s}, [%x[source]], #32\n"
      "ld1 {v8.4s, v9.4s}, [x0], #32\n"
      "ld1 {v10.4s, v11.4s}, [x2], #32\n"
      "prfm pldl1keep, [%x[source]]\n"
      "prfm pldl1keep, [x0]\n"
      "prfm pldl1keep, [x2]\n"
      "add v6.4s, v6.4s, v3.4s\n"
      "add v7.4s, v7.4s, v3.4s\n"
      "add v8.4s, v8.4s, v4.4s\n"
      "add v9.4s, v9.4s, v4.4s\n"
      "add v10.4s, v10.4s, v5.4s\n"
      "add v11.4s, v11.4s, v5.4s\n"
      "mul v6.4s, v6.4s, v0.4s\n"
      "mul v7.4s, v7.4s, v0.4s\n"
      "mul v8.4s, v8.4s, v0.4s\n"
      "mul v9.4s, v9.4s, v0.4s\n"
      "mul v10.4s, v10.4s, v0.4s\n"
      "mul v11.4s, v11.4s, v0.4s\n"
      "add v6.4s, v6.4s, v1.4s\n"
      "add v7.4s, v7.4s, v1.4s\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "add v9.4s, v9.4s, v1.4s\n"
      "add v10.4s, v10.4s, v1.4s\n"
      "add v11.4s, v11.4s, v1.4s\n"
      "sshl v6.4s, v6.4s, v2.4s\n"
      "sshl v7.4s, v7.4s, v2.4s\n"
      "sshl v8.4s, v8.4s, v2.4s\n"
      "sshl v9.4s, v9.4s, v2.4s\n"
      "sshl v10.4s, v10.4s, v2.4s\n"
      "sshl v11.4s, v11.4s, v2.4s\n"
      "sqxtn v6.4h, v6.4s\n"
      "sqxtn2 v6.8h, v7.4s\n"
      "sqxtn v8.4h, v8.4s\n"
      "sqxtn2 v8.8h, v9.4s\n"
      "sqxtn v10.4h, v10.4s\n"
      "sqxtn2 v10.8h, v11.4s\n"
      "sqxtun v6.8b, v6.8h\n"
      "sqxtun v8.8b, v8.8h\n"
      "sqxtun v10.8b, v10.8h\n"
      "st1 {v6.2s}, [%x[destination]], #8\n"
      "st1 {v8.2s}, [x1], #8\n"
      "st1 {v10.2s}, [x3], #8\n"

      "bne 1b\n"
      "2:"
      "ld1 {v6.4s}, [%x[source]], #16\n"
      "ld1 {v7.s}[0], [%x[source]], #4\n"
      "ld1 {v8.4s}, [x0], #16\n"
      "ld1 {v9.s}[0], [x0], #4\n"
      "ld1 {v10.4s}, [x2], #16\n"
      "ld1 {v11.s}[0], [x2], #4\n"
      "add v6.4s, v6.4s, v3.4s\n"
      "add v7.4s, v7.4s, v3.4s\n"
      "add v8.4s, v8.4s, v4.4s\n"
      "add v9.4s, v9.4s, v4.4s\n"
      "add v10.4s, v10.4s, v5.4s\n"
      "add v11.4s, v11.4s, v5.4s\n"
      "mul v6.4s, v6.4s, v0.4s\n"
      "mul v7.4s, v7.4s, v0.4s\n"
      "mul v8.4s, v8.4s, v0.4s\n"
      "mul v9.4s, v9.4s, v0.4s\n"
      "mul v10.4s, v10.4s, v0.4s\n"
      "mul v11.4s, v11.4s, v0.4s\n"
      "add v6.4s, v6.4s, v1.4s\n"
      "add v7.4s, v7.4s, v1.4s\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "add v9.4s, v9.4s, v1.4s\n"
      "add v10.4s, v10.4s, v1.4s\n"
      "add v11.4s, v11.4s, v1.4s\n"
      "sshl v6.4s, v6.4s, v2.4s\n"
      "sshl v7.4s, v7.4s, v2.4s\n"
      "sshl v8.4s, v8.4s, v2.4s\n"
      "sshl v9.4s, v9.4s, v2.4s\n"
      "sshl v10.4s, v10.4s, v2.4s\n"
      "sshl v11.4s, v11.4s, v2.4s\n"
      "sqxtn v6.4h, v6.4s\n"
      "sqxtn2 v6.8h, v7.4s\n"
      "sqxtn v8.4h, v8.4s\n"
      "sqxtn2 v8.8h, v9.4s\n"
      "sqxtn v10.4h, v10.4s\n"
      "sqxtn2 v10.8h, v11.4s\n"
      "sqxtun v6.8b, v6.8h\n"
      "sqxtun v8.8b, v8.8h\n"
      "sqxtun v10.8b, v10.8h\n"
      "st1 {v6.s}[0], [%x[destination]], #4\n"
      "st1 {v6.b}[4], [%x[destination]], #1\n"
      "st1 {v8.s}[0], [x1], #4\n"
      "st1 {v8.b}[4], [x1], #1\n"
      "st1 {v10.s}[0], [x3], #4\n"
      "st1 {v10.b}[4], [x3], #1\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [stride] "+r"(stride), [shift] "+r"(shift),
        [destination] "+r"(destination), [offsets] "+r"(offsets),
        [source] "+r"(source), [destination_stride] "+r"(destination_stride),
        [rounding_offset] "+r"(rounding_offset)
      :
      : "x0", "x1", "x2", "x3", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
        "v8", "v9", "v10", "v11", "cc", "memory");
}

void qnt_3x8_6_aligned(const std::int32_t* source, std::int32_t count,
                       std::int32_t stride, const std::int32_t* offsets,
                       std::uint8_t* destination,
                       std::int32_t destination_stride,
                       std::int32_t multiplicative_offset,
                       std::int32_t rounding_offset, std::int32_t shift) {
  asm volatile(
      "dup v0.4s, %w[multiplicative_offset]\n"
      "dup v1.4s, %w[rounding_offset]\n"
      "dup v2.4s, %w[shift]\n"
      "ld1r {v3.4s}, [%x[offsets]], #4\n"
      "ld1r {v4.4s}, [%x[offsets]], #4\n"
      "ld1r {v5.4s}, [%x[offsets]], #4\n"
      "add x0, %x[source], %x[stride]\n"
      "add x1, %x[destination], %x[destination_stride]\n"
      "add x2, x0, %x[stride]\n"
      "add x3, x1, %x[destination_stride]\n"
      "subs %x[count], %x[count], #6\n"
      "beq 2f\n"

      "1:"
      "subs %x[count], %x[count], #8\n"
      "ld1 {v6.4s, v7.4s}, [%x[source]], #32\n"
      "ld1 {v8.4s, v9.4s}, [x0], #32\n"
      "ld1 {v10.4s, v11.4s}, [x2], #32\n"
      "prfm pldl1keep, [%x[source]]\n"
      "prfm pldl1keep, [x0]\n"
      "prfm pldl1keep, [x2]\n"
      "add v6.4s, v6.4s, v3.4s\n"
      "add v7.4s, v7.4s, v3.4s\n"
      "add v8.4s, v8.4s, v4.4s\n"
      "add v9.4s, v9.4s, v4.4s\n"
      "add v10.4s, v10.4s, v5.4s\n"
      "add v11.4s, v11.4s, v5.4s\n"
      "mul v6.4s, v6.4s, v0.4s\n"
      "mul v7.4s, v7.4s, v0.4s\n"
      "mul v8.4s, v8.4s, v0.4s\n"
      "mul v9.4s, v9.4s, v0.4s\n"
      "mul v10.4s, v10.4s, v0.4s\n"
      "mul v11.4s, v11.4s, v0.4s\n"
      "add v6.4s, v6.4s, v1.4s\n"
      "add v7.4s, v7.4s, v1.4s\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "add v9.4s, v9.4s, v1.4s\n"
      "add v10.4s, v10.4s, v1.4s\n"
      "add v11.4s, v11.4s, v1.4s\n"
      "sshl v6.4s, v6.4s, v2.4s\n"
      "sshl v7.4s, v7.4s, v2.4s\n"
      "sshl v8.4s, v8.4s, v2.4s\n"
      "sshl v9.4s, v9.4s, v2.4s\n"
      "sshl v10.4s, v10.4s, v2.4s\n"
      "sshl v11.4s, v11.4s, v2.4s\n"
      "sqxtn v6.4h, v6.4s\n"
      "sqxtn2 v6.8h, v7.4s\n"
      "sqxtn v8.4h, v8.4s\n"
      "sqxtn2 v8.8h, v9.4s\n"
      "sqxtn v10.4h, v10.4s\n"
      "sqxtn2 v10.8h, v11.4s\n"
      "sqxtun v6.8b, v6.8h\n"
      "sqxtun v8.8b, v8.8h\n"
      "sqxtun v10.8b, v10.8h\n"
      "st1 {v6.2s}, [%x[destination]], #8\n"
      "st1 {v8.2s}, [x1], #8\n"
      "st1 {v10.2s}, [x3], #8\n"

      "bne 1b\n"
      "2:"
      "ld1 {v6.4s}, [%x[source]], #16\n"
      "ld1 {v7.2s}, [%x[source]], #8\n"
      "ld1 {v8.4s}, [x0], #16\n"
      "ld1 {v9.2s}, [x0], #8\n"
      "ld1 {v10.4s}, [x2], #16\n"
      "ld1 {v11.2s}, [x2], #8\n"
      "add v6.4s, v6.4s, v3.4s\n"
      "add v7.4s, v7.4s, v3.4s\n"
      "add v8.4s, v8.4s, v4.4s\n"
      "add v9.4s, v9.4s, v4.4s\n"
      "add v10.4s, v10.4s, v5.4s\n"
      "add v11.4s, v11.4s, v5.4s\n"
      "mul v6.4s, v6.4s, v0.4s\n"
      "mul v7.4s, v7.4s, v0.4s\n"
      "mul v8.4s, v8.4s, v0.4s\n"
      "mul v9.4s, v9.4s, v0.4s\n"
      "mul v10.4s, v10.4s, v0.4s\n"
      "mul v11.4s, v11.4s, v0.4s\n"
      "add v6.4s, v6.4s, v1.4s\n"
      "add v7.4s, v7.4s, v1.4s\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "add v9.4s, v9.4s, v1.4s\n"
      "add v10.4s, v10.4s, v1.4s\n"
      "add v11.4s, v11.4s, v1.4s\n"
      "sshl v6.4s, v6.4s, v2.4s\n"
      "sshl v7.4s, v7.4s, v2.4s\n"
      "sshl v8.4s, v8.4s, v2.4s\n"
      "sshl v9.4s, v9.4s, v2.4s\n"
      "sshl v10.4s, v10.4s, v2.4s\n"
      "sshl v11.4s, v11.4s, v2.4s\n"
      "sqxtn v6.4h, v6.4s\n"
      "sqxtn2 v6.8h, v7.4s\n"
      "sqxtn v8.4h, v8.4s\n"
      "sqxtn2 v8.8h, v9.4s\n"
      "sqxtn v10.4h, v10.4s\n"
      "sqxtn2 v10.8h, v11.4s\n"
      "sqxtun v6.8b, v6.8h\n"
      "sqxtun v8.8b, v8.8h\n"
      "sqxtun v10.8b, v10.8h\n"
      "st1 {v6.s}[0], [%x[destination]], #4\n"
      "st1 {v6.h}[2], [%x[destination]], #2\n"
      "st1 {v8.s}[0], [x1], #4\n"
      "st1 {v8.h}[2], [x1], #2\n"
      "st1 {v10.s}[0], [x3], #4\n"
      "st1 {v10.h}[2], [x3], #2\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [stride] "+r"(stride), [shift] "+r"(shift),
        [destination] "+r"(destination), [offsets] "+r"(offsets),
        [source] "+r"(source), [destination_stride] "+r"(destination_stride),
        [rounding_offset] "+r"(rounding_offset)
      :
      : "x0", "x1", "x2", "x3", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
        "v8", "v9", "v10", "v11", "cc", "memory");
}

void qnt_3x8_7_aligned(const std::int32_t* source, std::int32_t count,
                       std::int32_t stride, const std::int32_t* offsets,
                       std::uint8_t* destination,
                       std::int32_t destination_stride,
                       std::int32_t multiplicative_offset,
                       std::int32_t rounding_offset, std::int32_t shift) {
  asm volatile(
      "dup v0.4s, %w[multiplicative_offset]\n"
      "dup v1.4s, %w[rounding_offset]\n"
      "dup v2.4s, %w[shift]\n"
      "ld1r {v3.4s}, [%x[offsets]], #4\n"
      "ld1r {v4.4s}, [%x[offsets]], #4\n"
      "ld1r {v5.4s}, [%x[offsets]], #4\n"
      "add x0, %x[source], %x[stride]\n"
      "add x1, %x[destination], %x[destination_stride]\n"
      "add x2, x0, %x[stride]\n"
      "add x3, x1, %x[destination_stride]\n"
      "subs %x[count], %x[count], #7\n"
      "beq 2f\n"

      "1:"
      "subs %x[count], %x[count], #8\n"
      "ld1 {v6.4s, v7.4s}, [%x[source]], #32\n"
      "ld1 {v8.4s, v9.4s}, [x0], #32\n"
      "ld1 {v10.4s, v11.4s}, [x2], #32\n"
      "prfm pldl1keep, [%x[source]]\n"
      "prfm pldl1keep, [x0]\n"
      "prfm pldl1keep, [x2]\n"
      "add v6.4s, v6.4s, v3.4s\n"
      "add v7.4s, v7.4s, v3.4s\n"
      "add v8.4s, v8.4s, v4.4s\n"
      "add v9.4s, v9.4s, v4.4s\n"
      "add v10.4s, v10.4s, v5.4s\n"
      "add v11.4s, v11.4s, v5.4s\n"
      "mul v6.4s, v6.4s, v0.4s\n"
      "mul v7.4s, v7.4s, v0.4s\n"
      "mul v8.4s, v8.4s, v0.4s\n"
      "mul v9.4s, v9.4s, v0.4s\n"
      "mul v10.4s, v10.4s, v0.4s\n"
      "mul v11.4s, v11.4s, v0.4s\n"
      "add v6.4s, v6.4s, v1.4s\n"
      "add v7.4s, v7.4s, v1.4s\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "add v9.4s, v9.4s, v1.4s\n"
      "add v10.4s, v10.4s, v1.4s\n"
      "add v11.4s, v11.4s, v1.4s\n"
      "sshl v6.4s, v6.4s, v2.4s\n"
      "sshl v7.4s, v7.4s, v2.4s\n"
      "sshl v8.4s, v8.4s, v2.4s\n"
      "sshl v9.4s, v9.4s, v2.4s\n"
      "sshl v10.4s, v10.4s, v2.4s\n"
      "sshl v11.4s, v11.4s, v2.4s\n"
      "sqxtn v6.4h, v6.4s\n"
      "sqxtn2 v6.8h, v7.4s\n"
      "sqxtn v8.4h, v8.4s\n"
      "sqxtn2 v8.8h, v9.4s\n"
      "sqxtn v10.4h, v10.4s\n"
      "sqxtn2 v10.8h, v11.4s\n"
      "sqxtun v6.8b, v6.8h\n"
      "sqxtun v8.8b, v8.8h\n"
      "sqxtun v10.8b, v10.8h\n"
      "st1 {v6.2s}, [%x[destination]], #8\n"
      "st1 {v8.2s}, [x1], #8\n"
      "st1 {v10.2s}, [x3], #8\n"

      "bne 1b\n"
      "2:"
      "ld1 {v6.4s}, [%x[source]], #16\n"
      "ld1 {v7.2s}, [%x[source]], #8\n"
      "ld1 {v7.s}[2], [%x[source]], #4\n"
      "ld1 {v8.4s}, [x0], #16\n"
      "ld1 {v9.2s}, [x0], #8\n"
      "ld1 {v9.s}[2], [x0], #4\n"
      "ld1 {v10.4s}, [x2], #16\n"
      "ld1 {v11.2s}, [x2], #8\n"
      "ld1 {v11.s}[2], [x2], #4\n"
      "add v6.4s, v6.4s, v3.4s\n"
      "add v7.4s, v7.4s, v3.4s\n"
      "add v8.4s, v8.4s, v4.4s\n"
      "add v9.4s, v9.4s, v4.4s\n"
      "add v10.4s, v10.4s, v5.4s\n"
      "add v11.4s, v11.4s, v5.4s\n"
      "mul v6.4s, v6.4s, v0.4s\n"
      "mul v7.4s, v7.4s, v0.4s\n"
      "mul v8.4s, v8.4s, v0.4s\n"
      "mul v9.4s, v9.4s, v0.4s\n"
      "mul v10.4s, v10.4s, v0.4s\n"
      "mul v11.4s, v11.4s, v0.4s\n"
      "add v6.4s, v6.4s, v1.4s\n"
      "add v7.4s, v7.4s, v1.4s\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "add v9.4s, v9.4s, v1.4s\n"
      "add v10.4s, v10.4s, v1.4s\n"
      "add v11.4s, v11.4s, v1.4s\n"
      "sshl v6.4s, v6.4s, v2.4s\n"
      "sshl v7.4s, v7.4s, v2.4s\n"
      "sshl v8.4s, v8.4s, v2.4s\n"
      "sshl v9.4s, v9.4s, v2.4s\n"
      "sshl v10.4s, v10.4s, v2.4s\n"
      "sshl v11.4s, v11.4s, v2.4s\n"
      "sqxtn v6.4h, v6.4s\n"
      "sqxtn2 v6.8h, v7.4s\n"
      "sqxtn v8.4h, v8.4s\n"
      "sqxtn2 v8.8h, v9.4s\n"
      "sqxtn v10.4h, v10.4s\n"
      "sqxtn2 v10.8h, v11.4s\n"
      "sqxtun v6.8b, v6.8h\n"
      "sqxtun v8.8b, v8.8h\n"
      "sqxtun v10.8b, v10.8h\n"
      "st1 {v6.s}[0], [%x[destination]], #4\n"
      "st1 {v6.h}[2], [%x[destination]], #2\n"
      "st1 {v6.b}[6], [%x[destination]], #1\n"
      "st1 {v8.s}[0], [x1], #4\n"
      "st1 {v8.h}[2], [x1], #2\n"
      "st1 {v8.b}[6], [x1], #1\n"
      "st1 {v10.s}[0], [x3], #4\n"
      "st1 {v10.h}[2], [x3], #2\n"
      "st1 {v10.b}[6], [x3], #1\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [stride] "+r"(stride), [shift] "+r"(shift),
        [destination] "+r"(destination), [offsets] "+r"(offsets),
        [source] "+r"(source), [destination_stride] "+r"(destination_stride),
        [rounding_offset] "+r"(rounding_offset)
      :
      : "x0", "x1", "x2", "x3", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
        "v8", "v9", "v10", "v11", "cc", "memory");
}

void qnt_1x8(const std::int32_t* source, std::int32_t count,
             std::int32_t stride, const std::int32_t* offsets,
             std::uint8_t* destination, std::int32_t destination_stride,
             std::int32_t multiplicative_offset, std::int32_t rounding_offset,
             std::int32_t shift) {
  asm volatile(
      "dup v0.4s, %w[multiplicative_offset]\n"
      "dup v1.4s, %w[rounding_offset]\n"
      "dup v2.4s, %w[shift]\n"
      "ld1r {v3.4s}, [%x[offsets]], #4\n"

      "1:"
      "subs %x[count], %x[count], #8\n"
      "ld1 {v4.4s, v5.4s}, [%x[source]], #32\n"
      "prfm pldl1keep, [%x[source]]\n"
      "add v4.4s, v4.4s, v3.4s\n"
      "add v5.4s, v5.4s, v3.4s\n"
      "mul v4.4s, v4.4s, v0.4s\n"
      "mul v5.4s, v5.4s, v0.4s\n"
      "add v4.4s, v4.4s, v1.4s\n"
      "add v5.4s, v5.4s, v1.4s\n"
      "sshl v4.4s, v4.4s, v2.4s\n"
      "sshl v5.4s, v5.4s, v2.4s\n"
      "sqxtn v4.4h, v4.4s\n"
      "sqxtn2 v4.8h, v5.4s\n"
      "sqxtun v4.8b, v4.8h\n"
      "st1 {v4.2s}, [%x[destination]], #8\n"

      "bne 1b\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [stride] "+r"(stride), [shift] "+r"(shift),
        [destination] "+r"(destination), [offsets] "+r"(offsets),
        [source] "+r"(source), [destination_stride] "+r"(destination_stride),
        [rounding_offset] "+r"(rounding_offset)
      :
      : "v0", "v1", "v2", "v3", "v4", "v5", "cc", "memory");
}

void qnt_1x8_1(const std::int32_t* source, std::int32_t count,
               std::int32_t stride, const std::int32_t* offsets,
               std::uint8_t* destination, std::int32_t destination_stride,
               std::int32_t multiplicative_offset, std::int32_t rounding_offset,
               std::int32_t shift) {
  asm volatile(
      "dup v0.4s, %w[multiplicative_offset]\n"
      "dup v1.4s, %w[rounding_offset]\n"
      "dup v2.4s, %w[shift]\n"
      "ld1r {v3.4s}, [%x[offsets]], #4\n"
      "subs %x[count], %x[count], #1\n"
      "beq 2f\n"

      "1:"
      "subs %x[count], %x[count], #8\n"
      "ld1 {v4.4s, v5.4s}, [%x[source]], #32\n"
      "prfm pldl1keep, [%x[source]]\n"
      "add v4.4s, v4.4s, v3.4s\n"
      "add v5.4s, v5.4s, v3.4s\n"
      "mul v4.4s, v4.4s, v0.4s\n"
      "mul v5.4s, v5.4s, v0.4s\n"
      "add v4.4s, v4.4s, v1.4s\n"
      "add v5.4s, v5.4s, v1.4s\n"
      "sshl v4.4s, v4.4s, v2.4s\n"
      "sshl v5.4s, v5.4s, v2.4s\n"
      "sqxtn v4.4h, v4.4s\n"
      "sqxtn2 v4.8h, v5.4s\n"
      "sqxtun v4.8b, v4.8h\n"
      "st1 {v4.2s}, [%x[destination]], #8\n"

      "bne 1b\n"
      "2:"
      "ld1 {v4.s}[0], [%x[source]], #4\n"
      "add v4.4s, v4.4s, v3.4s\n"
      "mul v4.4s, v4.4s, v0.4s\n"
      "add v4.4s, v4.4s, v1.4s\n"
      "sshl v4.4s, v4.4s, v2.4s\n"
      "sqxtn v4.4h, v4.4s\n"
      "sqxtun v4.8b, v4.8h\n"
      "st1 {v4.b}[0], [%x[destination]], #1\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [stride] "+r"(stride), [shift] "+r"(shift),
        [destination] "+r"(destination), [offsets] "+r"(offsets),
        [source] "+r"(source), [destination_stride] "+r"(destination_stride),
        [rounding_offset] "+r"(rounding_offset)
      :
      : "v0", "v1", "v2", "v3", "v4", "v5", "cc", "memory");
}

void qnt_1x8_2(const std::int32_t* source, std::int32_t count,
               std::int32_t stride, const std::int32_t* offsets,
               std::uint8_t* destination, std::int32_t destination_stride,
               std::int32_t multiplicative_offset, std::int32_t rounding_offset,
               std::int32_t shift) {
  asm volatile(
      "dup v0.4s, %w[multiplicative_offset]\n"
      "dup v1.4s, %w[rounding_offset]\n"
      "dup v2.4s, %w[shift]\n"
      "ld1r {v3.4s}, [%x[offsets]], #4\n"
      "subs %x[count], %x[count], #2\n"
      "beq 2f\n"

      "1:"
      "subs %x[count], %x[count], #8\n"
      "ld1 {v4.4s, v5.4s}, [%x[source]], #32\n"
      "prfm pldl1keep, [%x[source]]\n"
      "add v4.4s, v4.4s, v3.4s\n"
      "add v5.4s, v5.4s, v3.4s\n"
      "mul v4.4s, v4.4s, v0.4s\n"
      "mul v5.4s, v5.4s, v0.4s\n"
      "add v4.4s, v4.4s, v1.4s\n"
      "add v5.4s, v5.4s, v1.4s\n"
      "sshl v4.4s, v4.4s, v2.4s\n"
      "sshl v5.4s, v5.4s, v2.4s\n"
      "sqxtn v4.4h, v4.4s\n"
      "sqxtn2 v4.8h, v5.4s\n"
      "sqxtun v4.8b, v4.8h\n"
      "st1 {v4.2s}, [%x[destination]], #8\n"

      "bne 1b\n"
      "2:"
      "ld1 {v4.2s}, [%x[source]], #8\n"
      "add v4.4s, v4.4s, v3.4s\n"
      "mul v4.4s, v4.4s, v0.4s\n"
      "add v4.4s, v4.4s, v1.4s\n"
      "sshl v4.4s, v4.4s, v2.4s\n"
      "sqxtn v4.4h, v4.4s\n"
      "sqxtun v4.8b, v4.8h\n"
      "st1 {v4.h}[0], [%x[destination]], #2\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [stride] "+r"(stride), [shift] "+r"(shift),
        [destination] "+r"(destination), [offsets] "+r"(offsets),
        [source] "+r"(source), [destination_stride] "+r"(destination_stride),
        [rounding_offset] "+r"(rounding_offset)
      :
      : "v0", "v1", "v2", "v3", "v4", "v5", "cc", "memory");
}

void qnt_1x8_3(const std::int32_t* source, std::int32_t count,
               std::int32_t stride, const std::int32_t* offsets,
               std::uint8_t* destination, std::int32_t destination_stride,
               std::int32_t multiplicative_offset, std::int32_t rounding_offset,
               std::int32_t shift) {
  asm volatile(
      "dup v0.4s, %w[multiplicative_offset]\n"
      "dup v1.4s, %w[rounding_offset]\n"
      "dup v2.4s, %w[shift]\n"
      "ld1r {v3.4s}, [%x[offsets]], #4\n"
      "subs %x[count], %x[count], #3\n"
      "beq 2f\n"

      "1:"
      "subs %x[count], %x[count], #8\n"
      "ld1 {v4.4s, v5.4s}, [%x[source]], #32\n"
      "prfm pldl1keep, [%x[source]]\n"
      "add v4.4s, v4.4s, v3.4s\n"
      "add v5.4s, v5.4s, v3.4s\n"
      "mul v4.4s, v4.4s, v0.4s\n"
      "mul v5.4s, v5.4s, v0.4s\n"
      "add v4.4s, v4.4s, v1.4s\n"
      "add v5.4s, v5.4s, v1.4s\n"
      "sshl v4.4s, v4.4s, v2.4s\n"
      "sshl v5.4s, v5.4s, v2.4s\n"
      "sqxtn v4.4h, v4.4s\n"
      "sqxtn2 v4.8h, v5.4s\n"
      "sqxtun v4.8b, v4.8h\n"
      "st1 {v4.2s}, [%x[destination]], #8\n"

      "bne 1b\n"
      "2:"
      "ld1 {v4.2s}, [%x[source]], #8\n"
      "ld1 {v4.s}[2], [%x[source]], #4\n"
      "add v4.4s, v4.4s, v3.4s\n"
      "mul v4.4s, v4.4s, v0.4s\n"
      "add v4.4s, v4.4s, v1.4s\n"
      "sshl v4.4s, v4.4s, v2.4s\n"
      "sqxtn v4.4h, v4.4s\n"
      "sqxtun v4.8b, v4.8h\n"
      "st1 {v4.h}[0], [%x[destination]], #2\n"
      "st1 {v4.b}[2], [%x[destination]], #1\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [stride] "+r"(stride), [shift] "+r"(shift),
        [destination] "+r"(destination), [offsets] "+r"(offsets),
        [source] "+r"(source), [destination_stride] "+r"(destination_stride),
        [rounding_offset] "+r"(rounding_offset)
      :
      : "v0", "v1", "v2", "v3", "v4", "v5", "cc", "memory");
}

void qnt_1x8_4(const std::int32_t* source, std::int32_t count,
               std::int32_t stride, const std::int32_t* offsets,
               std::uint8_t* destination, std::int32_t destination_stride,
               std::int32_t multiplicative_offset, std::int32_t rounding_offset,
               std::int32_t shift) {
  asm volatile(
      "dup v0.4s, %w[multiplicative_offset]\n"
      "dup v1.4s, %w[rounding_offset]\n"
      "dup v2.4s, %w[shift]\n"
      "ld1r {v3.4s}, [%x[offsets]], #4\n"
      "subs %x[count], %x[count], #4\n"
      "beq 2f\n"

      "1:"
      "subs %x[count], %x[count], #8\n"
      "ld1 {v4.4s, v5.4s}, [%x[source]], #32\n"
      "prfm pldl1keep, [%x[source]]\n"
      "add v4.4s, v4.4s, v3.4s\n"
      "add v5.4s, v5.4s, v3.4s\n"
      "mul v4.4s, v4.4s, v0.4s\n"
      "mul v5.4s, v5.4s, v0.4s\n"
      "add v4.4s, v4.4s, v1.4s\n"
      "add v5.4s, v5.4s, v1.4s\n"
      "sshl v4.4s, v4.4s, v2.4s\n"
      "sshl v5.4s, v5.4s, v2.4s\n"
      "sqxtn v4.4h, v4.4s\n"
      "sqxtn2 v4.8h, v5.4s\n"
      "sqxtun v4.8b, v4.8h\n"
      "st1 {v4.2s}, [%x[destination]], #8\n"

      "bne 1b\n"
      "2:"
      "ld1 {v4.4s}, [%x[source]], #16\n"
      "add v4.4s, v4.4s, v3.4s\n"
      "mul v4.4s, v4.4s, v0.4s\n"
      "add v4.4s, v4.4s, v1.4s\n"
      "sshl v4.4s, v4.4s, v2.4s\n"
      "sqxtn v4.4h, v4.4s\n"
      "sqxtun v4.8b, v4.8h\n"
      "st1 {v4.s}[0], [%x[destination]], #4\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [stride] "+r"(stride), [shift] "+r"(shift),
        [destination] "+r"(destination), [offsets] "+r"(offsets),
        [source] "+r"(source), [destination_stride] "+r"(destination_stride),
        [rounding_offset] "+r"(rounding_offset)
      :
      : "v0", "v1", "v2", "v3", "v4", "v5", "cc", "memory");
}

void qnt_1x8_5(const std::int32_t* source, std::int32_t count,
               std::int32_t stride, const std::int32_t* offsets,
               std::uint8_t* destination, std::int32_t destination_stride,
               std::int32_t multiplicative_offset, std::int32_t rounding_offset,
               std::int32_t shift) {
  asm volatile(
      "dup v0.4s, %w[multiplicative_offset]\n"
      "dup v1.4s, %w[rounding_offset]\n"
      "dup v2.4s, %w[shift]\n"
      "ld1r {v3.4s}, [%x[offsets]], #4\n"
      "subs %x[count], %x[count], #5\n"
      "beq 2f\n"

      "1:"
      "subs %x[count], %x[count], #8\n"
      "ld1 {v4.4s, v5.4s}, [%x[source]], #32\n"
      "prfm pldl1keep, [%x[source]]\n"
      "add v4.4s, v4.4s, v3.4s\n"
      "add v5.4s, v5.4s, v3.4s\n"
      "mul v4.4s, v4.4s, v0.4s\n"
      "mul v5.4s, v5.4s, v0.4s\n"
      "add v4.4s, v4.4s, v1.4s\n"
      "add v5.4s, v5.4s, v1.4s\n"
      "sshl v4.4s, v4.4s, v2.4s\n"
      "sshl v5.4s, v5.4s, v2.4s\n"
      "sqxtn v4.4h, v4.4s\n"
      "sqxtn2 v4.8h, v5.4s\n"
      "sqxtun v4.8b, v4.8h\n"
      "st1 {v4.2s}, [%x[destination]], #8\n"

      "bne 1b\n"
      "2:"
      "ld1 {v4.4s}, [%x[source]], #16\n"
      "ld1 {v5.s}[0], [%x[source]], #4\n"
      "add v4.4s, v4.4s, v3.4s\n"
      "add v5.4s, v5.4s, v3.4s\n"
      "mul v4.4s, v4.4s, v0.4s\n"
      "mul v5.4s, v5.4s, v0.4s\n"
      "add v4.4s, v4.4s, v1.4s\n"
      "add v5.4s, v5.4s, v1.4s\n"
      "sshl v4.4s, v4.4s, v2.4s\n"
      "sshl v5.4s, v5.4s, v2.4s\n"
      "sqxtn v4.4h, v4.4s\n"
      "sqxtn2 v4.8h, v5.4s\n"
      "sqxtun v4.8b, v4.8h\n"
      "st1 {v4.s}[0], [%x[destination]], #4\n"
      "st1 {v4.b}[4], [%x[destination]], #1\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [stride] "+r"(stride), [shift] "+r"(shift),
        [destination] "+r"(destination), [offsets] "+r"(offsets),
        [source] "+r"(source), [destination_stride] "+r"(destination_stride),
        [rounding_offset] "+r"(rounding_offset)
      :
      : "v0", "v1", "v2", "v3", "v4", "v5", "cc", "memory");
}

void qnt_1x8_6(const std::int32_t* source, std::int32_t count,
               std::int32_t stride, const std::int32_t* offsets,
               std::uint8_t* destination, std::int32_t destination_stride,
               std::int32_t multiplicative_offset, std::int32_t rounding_offset,
               std::int32_t shift) {
  asm volatile(
      "dup v0.4s, %w[multiplicative_offset]\n"
      "dup v1.4s, %w[rounding_offset]\n"
      "dup v2.4s, %w[shift]\n"
      "ld1r {v3.4s}, [%x[offsets]], #4\n"
      "subs %x[count], %x[count], #6\n"
      "beq 2f\n"

      "1:"
      "subs %x[count], %x[count], #8\n"
      "ld1 {v4.4s, v5.4s}, [%x[source]], #32\n"
      "prfm pldl1keep, [%x[source]]\n"
      "add v4.4s, v4.4s, v3.4s\n"
      "add v5.4s, v5.4s, v3.4s\n"
      "mul v4.4s, v4.4s, v0.4s\n"
      "mul v5.4s, v5.4s, v0.4s\n"
      "add v4.4s, v4.4s, v1.4s\n"
      "add v5.4s, v5.4s, v1.4s\n"
      "sshl v4.4s, v4.4s, v2.4s\n"
      "sshl v5.4s, v5.4s, v2.4s\n"
      "sqxtn v4.4h, v4.4s\n"
      "sqxtn2 v4.8h, v5.4s\n"
      "sqxtun v4.8b, v4.8h\n"
      "st1 {v4.2s}, [%x[destination]], #8\n"

      "bne 1b\n"
      "2:"
      "ld1 {v4.4s}, [%x[source]], #16\n"
      "ld1 {v5.2s}, [%x[source]], #8\n"
      "add v4.4s, v4.4s, v3.4s\n"
      "add v5.4s, v5.4s, v3.4s\n"
      "mul v4.4s, v4.4s, v0.4s\n"
      "mul v5.4s, v5.4s, v0.4s\n"
      "add v4.4s, v4.4s, v1.4s\n"
      "add v5.4s, v5.4s, v1.4s\n"
      "sshl v4.4s, v4.4s, v2.4s\n"
      "sshl v5.4s, v5.4s, v2.4s\n"
      "sqxtn v4.4h, v4.4s\n"
      "sqxtn2 v4.8h, v5.4s\n"
      "sqxtun v4.8b, v4.8h\n"
      "st1 {v4.s}[0], [%x[destination]], #4\n"
      "st1 {v4.h}[2], [%x[destination]], #2\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [stride] "+r"(stride), [shift] "+r"(shift),
        [destination] "+r"(destination), [offsets] "+r"(offsets),
        [source] "+r"(source), [destination_stride] "+r"(destination_stride),
        [rounding_offset] "+r"(rounding_offset)
      :
      : "v0", "v1", "v2", "v3", "v4", "v5", "cc", "memory");
}

void qnt_1x8_7(const std::int32_t* source, std::int32_t count,
               std::int32_t stride, const std::int32_t* offsets,
               std::uint8_t* destination, std::int32_t destination_stride,
               std::int32_t multiplicative_offset, std::int32_t rounding_offset,
               std::int32_t shift) {
  asm volatile(
      "dup v0.4s, %w[multiplicative_offset]\n"
      "dup v1.4s, %w[rounding_offset]\n"
      "dup v2.4s, %w[shift]\n"
      "ld1r {v3.4s}, [%x[offsets]], #4\n"
      "subs %x[count], %x[count], #7\n"
      "beq 2f\n"

      "1:"
      "subs %x[count], %x[count], #8\n"
      "ld1 {v4.4s, v5.4s}, [%x[source]], #32\n"
      "prfm pldl1keep, [%x[source]]\n"
      "add v4.4s, v4.4s, v3.4s\n"
      "add v5.4s, v5.4s, v3.4s\n"
      "mul v4.4s, v4.4s, v0.4s\n"
      "mul v5.4s, v5.4s, v0.4s\n"
      "add v4.4s, v4.4s, v1.4s\n"
      "add v5.4s, v5.4s, v1.4s\n"
      "sshl v4.4s, v4.4s, v2.4s\n"
      "sshl v5.4s, v5.4s, v2.4s\n"
      "sqxtn v4.4h, v4.4s\n"
      "sqxtn2 v4.8h, v5.4s\n"
      "sqxtun v4.8b, v4.8h\n"
      "st1 {v4.2s}, [%x[destination]], #8\n"

      "bne 1b\n"
      "2:"
      "ld1 {v4.4s}, [%x[source]], #16\n"
      "ld1 {v5.2s}, [%x[source]], #8\n"
      "ld1 {v5.s}[2], [%x[source]], #4\n"
      "add v4.4s, v4.4s, v3.4s\n"
      "add v5.4s, v5.4s, v3.4s\n"
      "mul v4.4s, v4.4s, v0.4s\n"
      "mul v5.4s, v5.4s, v0.4s\n"
      "add v4.4s, v4.4s, v1.4s\n"
      "add v5.4s, v5.4s, v1.4s\n"
      "sshl v4.4s, v4.4s, v2.4s\n"
      "sshl v5.4s, v5.4s, v2.4s\n"
      "sqxtn v4.4h, v4.4s\n"
      "sqxtn2 v4.8h, v5.4s\n"
      "sqxtun v4.8b, v4.8h\n"
      "st1 {v4.s}[0], [%x[destination]], #4\n"
      "st1 {v4.h}[2], [%x[destination]], #2\n"
      "st1 {v4.b}[6], [%x[destination]], #1\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [stride] "+r"(stride), [shift] "+r"(shift),
        [destination] "+r"(destination), [offsets] "+r"(offsets),
        [source] "+r"(source), [destination_stride] "+r"(destination_stride),
        [rounding_offset] "+r"(rounding_offset)
      :
      : "v0", "v1", "v2", "v3", "v4", "v5", "cc", "memory");
}

void qnt_2x8(const std::int32_t* source, std::int32_t count,
             std::int32_t stride, const std::int32_t* offsets,
             std::uint8_t* destination, std::int32_t destination_stride,
             std::int32_t multiplicative_offset, std::int32_t rounding_offset,
             std::int32_t shift) {
  asm volatile(
      "dup v0.4s, %w[multiplicative_offset]\n"
      "dup v1.4s, %w[rounding_offset]\n"
      "dup v2.4s, %w[shift]\n"
      "ld1r {v3.4s}, [%x[offsets]], #4\n"
      "ld1r {v4.4s}, [%x[offsets]], #4\n"
      "add x0, %x[source], %x[stride]\n"
      "add x1, %x[destination], %x[destination_stride]\n"

      "1:"
      "subs %x[count], %x[count], #8\n"
      "ld1 {v5.4s, v6.4s}, [%x[source]], #32\n"
      "ld1 {v7.4s, v8.4s}, [x0], #32\n"
      "prfm pldl1keep, [%x[source]]\n"
      "prfm pldl1keep, [x0]\n"
      "add v5.4s, v5.4s, v3.4s\n"
      "add v6.4s, v6.4s, v3.4s\n"
      "add v7.4s, v7.4s, v4.4s\n"
      "add v8.4s, v8.4s, v4.4s\n"
      "mul v5.4s, v5.4s, v0.4s\n"
      "mul v6.4s, v6.4s, v0.4s\n"
      "mul v7.4s, v7.4s, v0.4s\n"
      "mul v8.4s, v8.4s, v0.4s\n"
      "add v5.4s, v5.4s, v1.4s\n"
      "add v6.4s, v6.4s, v1.4s\n"
      "add v7.4s, v7.4s, v1.4s\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "sshl v5.4s, v5.4s, v2.4s\n"
      "sshl v6.4s, v6.4s, v2.4s\n"
      "sshl v7.4s, v7.4s, v2.4s\n"
      "sshl v8.4s, v8.4s, v2.4s\n"
      "sqxtn v5.4h, v5.4s\n"
      "sqxtn2 v5.8h, v6.4s\n"
      "sqxtn v7.4h, v7.4s\n"
      "sqxtn2 v7.8h, v8.4s\n"
      "sqxtun v5.8b, v5.8h\n"
      "sqxtun v7.8b, v7.8h\n"
      "st1 {v5.2s}, [%x[destination]], #8\n"
      "st1 {v7.2s}, [x1], #8\n"

      "bne 1b\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [stride] "+r"(stride), [shift] "+r"(shift),
        [destination] "+r"(destination), [offsets] "+r"(offsets),
        [source] "+r"(source), [destination_stride] "+r"(destination_stride),
        [rounding_offset] "+r"(rounding_offset)
      :
      : "x0", "x1", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "cc",
        "memory");
}

void qnt_2x8_1(const std::int32_t* source, std::int32_t count,
               std::int32_t stride, const std::int32_t* offsets,
               std::uint8_t* destination, std::int32_t destination_stride,
               std::int32_t multiplicative_offset, std::int32_t rounding_offset,
               std::int32_t shift) {
  asm volatile(
      "dup v0.4s, %w[multiplicative_offset]\n"
      "dup v1.4s, %w[rounding_offset]\n"
      "dup v2.4s, %w[shift]\n"
      "ld1r {v3.4s}, [%x[offsets]], #4\n"
      "ld1r {v4.4s}, [%x[offsets]], #4\n"
      "add x0, %x[source], %x[stride]\n"
      "add x1, %x[destination], %x[destination_stride]\n"
      "subs %x[count], %x[count], #1\n"
      "beq 2f\n"

      "1:"
      "subs %x[count], %x[count], #8\n"
      "ld1 {v5.4s, v6.4s}, [%x[source]], #32\n"
      "ld1 {v7.4s, v8.4s}, [x0], #32\n"
      "prfm pldl1keep, [%x[source]]\n"
      "prfm pldl1keep, [x0]\n"
      "add v5.4s, v5.4s, v3.4s\n"
      "add v6.4s, v6.4s, v3.4s\n"
      "add v7.4s, v7.4s, v4.4s\n"
      "add v8.4s, v8.4s, v4.4s\n"
      "mul v5.4s, v5.4s, v0.4s\n"
      "mul v6.4s, v6.4s, v0.4s\n"
      "mul v7.4s, v7.4s, v0.4s\n"
      "mul v8.4s, v8.4s, v0.4s\n"
      "add v5.4s, v5.4s, v1.4s\n"
      "add v6.4s, v6.4s, v1.4s\n"
      "add v7.4s, v7.4s, v1.4s\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "sshl v5.4s, v5.4s, v2.4s\n"
      "sshl v6.4s, v6.4s, v2.4s\n"
      "sshl v7.4s, v7.4s, v2.4s\n"
      "sshl v8.4s, v8.4s, v2.4s\n"
      "sqxtn v5.4h, v5.4s\n"
      "sqxtn2 v5.8h, v6.4s\n"
      "sqxtn v7.4h, v7.4s\n"
      "sqxtn2 v7.8h, v8.4s\n"
      "sqxtun v5.8b, v5.8h\n"
      "sqxtun v7.8b, v7.8h\n"
      "st1 {v5.2s}, [%x[destination]], #8\n"
      "st1 {v7.2s}, [x1], #8\n"

      "bne 1b\n"
      "2:"
      "ld1 {v5.s}[0], [%x[source]], #4\n"
      "ld1 {v7.s}[0], [x0], #4\n"
      "add v5.4s, v5.4s, v3.4s\n"
      "add v7.4s, v7.4s, v4.4s\n"
      "mul v5.4s, v5.4s, v0.4s\n"
      "mul v7.4s, v7.4s, v0.4s\n"
      "add v5.4s, v5.4s, v1.4s\n"
      "add v7.4s, v7.4s, v1.4s\n"
      "sshl v5.4s, v5.4s, v2.4s\n"
      "sshl v7.4s, v7.4s, v2.4s\n"
      "sqxtn v5.4h, v5.4s\n"
      "sqxtn v7.4h, v7.4s\n"
      "sqxtun v5.8b, v5.8h\n"
      "sqxtun v7.8b, v7.8h\n"
      "st1 {v5.b}[0], [%x[destination]], #1\n"
      "st1 {v7.b}[0], [x1], #1\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [stride] "+r"(stride), [shift] "+r"(shift),
        [destination] "+r"(destination), [offsets] "+r"(offsets),
        [source] "+r"(source), [destination_stride] "+r"(destination_stride),
        [rounding_offset] "+r"(rounding_offset)
      :
      : "x0", "x1", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "cc",
        "memory");
}

void qnt_2x8_2(const std::int32_t* source, std::int32_t count,
               std::int32_t stride, const std::int32_t* offsets,
               std::uint8_t* destination, std::int32_t destination_stride,
               std::int32_t multiplicative_offset, std::int32_t rounding_offset,
               std::int32_t shift) {
  asm volatile(
      "dup v0.4s, %w[multiplicative_offset]\n"
      "dup v1.4s, %w[rounding_offset]\n"
      "dup v2.4s, %w[shift]\n"
      "ld1r {v3.4s}, [%x[offsets]], #4\n"
      "ld1r {v4.4s}, [%x[offsets]], #4\n"
      "add x0, %x[source], %x[stride]\n"
      "add x1, %x[destination], %x[destination_stride]\n"
      "subs %x[count], %x[count], #2\n"
      "beq 2f\n"

      "1:"
      "subs %x[count], %x[count], #8\n"
      "ld1 {v5.4s, v6.4s}, [%x[source]], #32\n"
      "ld1 {v7.4s, v8.4s}, [x0], #32\n"
      "prfm pldl1keep, [%x[source]]\n"
      "prfm pldl1keep, [x0]\n"
      "add v5.4s, v5.4s, v3.4s\n"
      "add v6.4s, v6.4s, v3.4s\n"
      "add v7.4s, v7.4s, v4.4s\n"
      "add v8.4s, v8.4s, v4.4s\n"
      "mul v5.4s, v5.4s, v0.4s\n"
      "mul v6.4s, v6.4s, v0.4s\n"
      "mul v7.4s, v7.4s, v0.4s\n"
      "mul v8.4s, v8.4s, v0.4s\n"
      "add v5.4s, v5.4s, v1.4s\n"
      "add v6.4s, v6.4s, v1.4s\n"
      "add v7.4s, v7.4s, v1.4s\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "sshl v5.4s, v5.4s, v2.4s\n"
      "sshl v6.4s, v6.4s, v2.4s\n"
      "sshl v7.4s, v7.4s, v2.4s\n"
      "sshl v8.4s, v8.4s, v2.4s\n"
      "sqxtn v5.4h, v5.4s\n"
      "sqxtn2 v5.8h, v6.4s\n"
      "sqxtn v7.4h, v7.4s\n"
      "sqxtn2 v7.8h, v8.4s\n"
      "sqxtun v5.8b, v5.8h\n"
      "sqxtun v7.8b, v7.8h\n"
      "st1 {v5.2s}, [%x[destination]], #8\n"
      "st1 {v7.2s}, [x1], #8\n"

      "bne 1b\n"
      "2:"
      "ld1 {v5.2s}, [%x[source]], #8\n"
      "ld1 {v7.2s}, [x0], #8\n"
      "add v5.4s, v5.4s, v3.4s\n"
      "add v7.4s, v7.4s, v4.4s\n"
      "mul v5.4s, v5.4s, v0.4s\n"
      "mul v7.4s, v7.4s, v0.4s\n"
      "add v5.4s, v5.4s, v1.4s\n"
      "add v7.4s, v7.4s, v1.4s\n"
      "sshl v5.4s, v5.4s, v2.4s\n"
      "sshl v7.4s, v7.4s, v2.4s\n"
      "sqxtn v5.4h, v5.4s\n"
      "sqxtn v7.4h, v7.4s\n"
      "sqxtun v5.8b, v5.8h\n"
      "sqxtun v7.8b, v7.8h\n"
      "st1 {v5.h}[0], [%x[destination]], #2\n"
      "st1 {v7.h}[0], [x1], #2\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [stride] "+r"(stride), [shift] "+r"(shift),
        [destination] "+r"(destination), [offsets] "+r"(offsets),
        [source] "+r"(source), [destination_stride] "+r"(destination_stride),
        [rounding_offset] "+r"(rounding_offset)
      :
      : "x0", "x1", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "cc",
        "memory");
}

void qnt_2x8_3(const std::int32_t* source, std::int32_t count,
               std::int32_t stride, const std::int32_t* offsets,
               std::uint8_t* destination, std::int32_t destination_stride,
               std::int32_t multiplicative_offset, std::int32_t rounding_offset,
               std::int32_t shift) {
  asm volatile(
      "dup v0.4s, %w[multiplicative_offset]\n"
      "dup v1.4s, %w[rounding_offset]\n"
      "dup v2.4s, %w[shift]\n"
      "ld1r {v3.4s}, [%x[offsets]], #4\n"
      "ld1r {v4.4s}, [%x[offsets]], #4\n"
      "add x0, %x[source], %x[stride]\n"
      "add x1, %x[destination], %x[destination_stride]\n"
      "subs %x[count], %x[count], #3\n"
      "beq 2f\n"

      "1:"
      "subs %x[count], %x[count], #8\n"
      "ld1 {v5.4s, v6.4s}, [%x[source]], #32\n"
      "ld1 {v7.4s, v8.4s}, [x0], #32\n"
      "prfm pldl1keep, [%x[source]]\n"
      "prfm pldl1keep, [x0]\n"
      "add v5.4s, v5.4s, v3.4s\n"
      "add v6.4s, v6.4s, v3.4s\n"
      "add v7.4s, v7.4s, v4.4s\n"
      "add v8.4s, v8.4s, v4.4s\n"
      "mul v5.4s, v5.4s, v0.4s\n"
      "mul v6.4s, v6.4s, v0.4s\n"
      "mul v7.4s, v7.4s, v0.4s\n"
      "mul v8.4s, v8.4s, v0.4s\n"
      "add v5.4s, v5.4s, v1.4s\n"
      "add v6.4s, v6.4s, v1.4s\n"
      "add v7.4s, v7.4s, v1.4s\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "sshl v5.4s, v5.4s, v2.4s\n"
      "sshl v6.4s, v6.4s, v2.4s\n"
      "sshl v7.4s, v7.4s, v2.4s\n"
      "sshl v8.4s, v8.4s, v2.4s\n"
      "sqxtn v5.4h, v5.4s\n"
      "sqxtn2 v5.8h, v6.4s\n"
      "sqxtn v7.4h, v7.4s\n"
      "sqxtn2 v7.8h, v8.4s\n"
      "sqxtun v5.8b, v5.8h\n"
      "sqxtun v7.8b, v7.8h\n"
      "st1 {v5.2s}, [%x[destination]], #8\n"
      "st1 {v7.2s}, [x1], #8\n"

      "bne 1b\n"
      "2:"
      "ld1 {v5.2s}, [%x[source]], #8\n"
      "ld1 {v5.s}[2], [%x[source]], #4\n"
      "ld1 {v7.2s}, [x0], #8\n"
      "ld1 {v7.s}[2], [x0], #4\n"
      "add v5.4s, v5.4s, v3.4s\n"
      "add v7.4s, v7.4s, v4.4s\n"
      "mul v5.4s, v5.4s, v0.4s\n"
      "mul v7.4s, v7.4s, v0.4s\n"
      "add v5.4s, v5.4s, v1.4s\n"
      "add v7.4s, v7.4s, v1.4s\n"
      "sshl v5.4s, v5.4s, v2.4s\n"
      "sshl v7.4s, v7.4s, v2.4s\n"
      "sqxtn v5.4h, v5.4s\n"
      "sqxtn v7.4h, v7.4s\n"
      "sqxtun v5.8b, v5.8h\n"
      "sqxtun v7.8b, v7.8h\n"
      "st1 {v5.h}[0], [%x[destination]], #2\n"
      "st1 {v5.b}[2], [%x[destination]], #1\n"
      "st1 {v7.h}[0], [x1], #2\n"
      "st1 {v7.b}[2], [x1], #1\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [stride] "+r"(stride), [shift] "+r"(shift),
        [destination] "+r"(destination), [offsets] "+r"(offsets),
        [source] "+r"(source), [destination_stride] "+r"(destination_stride),
        [rounding_offset] "+r"(rounding_offset)
      :
      : "x0", "x1", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "cc",
        "memory");
}

void qnt_2x8_4(const std::int32_t* source, std::int32_t count,
               std::int32_t stride, const std::int32_t* offsets,
               std::uint8_t* destination, std::int32_t destination_stride,
               std::int32_t multiplicative_offset, std::int32_t rounding_offset,
               std::int32_t shift) {
  asm volatile(
      "dup v0.4s, %w[multiplicative_offset]\n"
      "dup v1.4s, %w[rounding_offset]\n"
      "dup v2.4s, %w[shift]\n"
      "ld1r {v3.4s}, [%x[offsets]], #4\n"
      "ld1r {v4.4s}, [%x[offsets]], #4\n"
      "add x0, %x[source], %x[stride]\n"
      "add x1, %x[destination], %x[destination_stride]\n"
      "subs %x[count], %x[count], #4\n"
      "beq 2f\n"

      "1:"
      "subs %x[count], %x[count], #8\n"
      "ld1 {v5.4s, v6.4s}, [%x[source]], #32\n"
      "ld1 {v7.4s, v8.4s}, [x0], #32\n"
      "prfm pldl1keep, [%x[source]]\n"
      "prfm pldl1keep, [x0]\n"
      "add v5.4s, v5.4s, v3.4s\n"
      "add v6.4s, v6.4s, v3.4s\n"
      "add v7.4s, v7.4s, v4.4s\n"
      "add v8.4s, v8.4s, v4.4s\n"
      "mul v5.4s, v5.4s, v0.4s\n"
      "mul v6.4s, v6.4s, v0.4s\n"
      "mul v7.4s, v7.4s, v0.4s\n"
      "mul v8.4s, v8.4s, v0.4s\n"
      "add v5.4s, v5.4s, v1.4s\n"
      "add v6.4s, v6.4s, v1.4s\n"
      "add v7.4s, v7.4s, v1.4s\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "sshl v5.4s, v5.4s, v2.4s\n"
      "sshl v6.4s, v6.4s, v2.4s\n"
      "sshl v7.4s, v7.4s, v2.4s\n"
      "sshl v8.4s, v8.4s, v2.4s\n"
      "sqxtn v5.4h, v5.4s\n"
      "sqxtn2 v5.8h, v6.4s\n"
      "sqxtn v7.4h, v7.4s\n"
      "sqxtn2 v7.8h, v8.4s\n"
      "sqxtun v5.8b, v5.8h\n"
      "sqxtun v7.8b, v7.8h\n"
      "st1 {v5.2s}, [%x[destination]], #8\n"
      "st1 {v7.2s}, [x1], #8\n"

      "bne 1b\n"
      "2:"
      "ld1 {v5.4s}, [%x[source]], #16\n"
      "ld1 {v7.4s}, [x0], #16\n"
      "add v5.4s, v5.4s, v3.4s\n"
      "add v7.4s, v7.4s, v4.4s\n"
      "mul v5.4s, v5.4s, v0.4s\n"
      "mul v7.4s, v7.4s, v0.4s\n"
      "add v5.4s, v5.4s, v1.4s\n"
      "add v7.4s, v7.4s, v1.4s\n"
      "sshl v5.4s, v5.4s, v2.4s\n"
      "sshl v7.4s, v7.4s, v2.4s\n"
      "sqxtn v5.4h, v5.4s\n"
      "sqxtn v7.4h, v7.4s\n"
      "sqxtun v5.8b, v5.8h\n"
      "sqxtun v7.8b, v7.8h\n"
      "st1 {v5.s}[0], [%x[destination]], #4\n"
      "st1 {v7.s}[0], [x1], #4\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [stride] "+r"(stride), [shift] "+r"(shift),
        [destination] "+r"(destination), [offsets] "+r"(offsets),
        [source] "+r"(source), [destination_stride] "+r"(destination_stride),
        [rounding_offset] "+r"(rounding_offset)
      :
      : "x0", "x1", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "cc",
        "memory");
}

void qnt_2x8_5(const std::int32_t* source, std::int32_t count,
               std::int32_t stride, const std::int32_t* offsets,
               std::uint8_t* destination, std::int32_t destination_stride,
               std::int32_t multiplicative_offset, std::int32_t rounding_offset,
               std::int32_t shift) {
  asm volatile(
      "dup v0.4s, %w[multiplicative_offset]\n"
      "dup v1.4s, %w[rounding_offset]\n"
      "dup v2.4s, %w[shift]\n"
      "ld1r {v3.4s}, [%x[offsets]], #4\n"
      "ld1r {v4.4s}, [%x[offsets]], #4\n"
      "add x0, %x[source], %x[stride]\n"
      "add x1, %x[destination], %x[destination_stride]\n"
      "subs %x[count], %x[count], #5\n"
      "beq 2f\n"

      "1:"
      "subs %x[count], %x[count], #8\n"
      "ld1 {v5.4s, v6.4s}, [%x[source]], #32\n"
      "ld1 {v7.4s, v8.4s}, [x0], #32\n"
      "prfm pldl1keep, [%x[source]]\n"
      "prfm pldl1keep, [x0]\n"
      "add v5.4s, v5.4s, v3.4s\n"
      "add v6.4s, v6.4s, v3.4s\n"
      "add v7.4s, v7.4s, v4.4s\n"
      "add v8.4s, v8.4s, v4.4s\n"
      "mul v5.4s, v5.4s, v0.4s\n"
      "mul v6.4s, v6.4s, v0.4s\n"
      "mul v7.4s, v7.4s, v0.4s\n"
      "mul v8.4s, v8.4s, v0.4s\n"
      "add v5.4s, v5.4s, v1.4s\n"
      "add v6.4s, v6.4s, v1.4s\n"
      "add v7.4s, v7.4s, v1.4s\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "sshl v5.4s, v5.4s, v2.4s\n"
      "sshl v6.4s, v6.4s, v2.4s\n"
      "sshl v7.4s, v7.4s, v2.4s\n"
      "sshl v8.4s, v8.4s, v2.4s\n"
      "sqxtn v5.4h, v5.4s\n"
      "sqxtn2 v5.8h, v6.4s\n"
      "sqxtn v7.4h, v7.4s\n"
      "sqxtn2 v7.8h, v8.4s\n"
      "sqxtun v5.8b, v5.8h\n"
      "sqxtun v7.8b, v7.8h\n"
      "st1 {v5.2s}, [%x[destination]], #8\n"
      "st1 {v7.2s}, [x1], #8\n"

      "bne 1b\n"
      "2:"
      "ld1 {v5.4s}, [%x[source]], #16\n"
      "ld1 {v6.s}[0], [%x[source]], #4\n"
      "ld1 {v7.4s}, [x0], #16\n"
      "ld1 {v8.s}[0], [x0], #4\n"
      "add v5.4s, v5.4s, v3.4s\n"
      "add v6.4s, v6.4s, v3.4s\n"
      "add v7.4s, v7.4s, v4.4s\n"
      "add v8.4s, v8.4s, v4.4s\n"
      "mul v5.4s, v5.4s, v0.4s\n"
      "mul v6.4s, v6.4s, v0.4s\n"
      "mul v7.4s, v7.4s, v0.4s\n"
      "mul v8.4s, v8.4s, v0.4s\n"
      "add v5.4s, v5.4s, v1.4s\n"
      "add v6.4s, v6.4s, v1.4s\n"
      "add v7.4s, v7.4s, v1.4s\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "sshl v5.4s, v5.4s, v2.4s\n"
      "sshl v6.4s, v6.4s, v2.4s\n"
      "sshl v7.4s, v7.4s, v2.4s\n"
      "sshl v8.4s, v8.4s, v2.4s\n"
      "sqxtn v5.4h, v5.4s\n"
      "sqxtn2 v5.8h, v6.4s\n"
      "sqxtn v7.4h, v7.4s\n"
      "sqxtn2 v7.8h, v8.4s\n"
      "sqxtun v5.8b, v5.8h\n"
      "sqxtun v7.8b, v7.8h\n"
      "st1 {v5.s}[0], [%x[destination]], #4\n"
      "st1 {v5.b}[4], [%x[destination]], #1\n"
      "st1 {v7.s}[0], [x1], #4\n"
      "st1 {v7.b}[4], [x1], #1\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [stride] "+r"(stride), [shift] "+r"(shift),
        [destination] "+r"(destination), [offsets] "+r"(offsets),
        [source] "+r"(source), [destination_stride] "+r"(destination_stride),
        [rounding_offset] "+r"(rounding_offset)
      :
      : "x0", "x1", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "cc",
        "memory");
}

void qnt_2x8_6(const std::int32_t* source, std::int32_t count,
               std::int32_t stride, const std::int32_t* offsets,
               std::uint8_t* destination, std::int32_t destination_stride,
               std::int32_t multiplicative_offset, std::int32_t rounding_offset,
               std::int32_t shift) {
  asm volatile(
      "dup v0.4s, %w[multiplicative_offset]\n"
      "dup v1.4s, %w[rounding_offset]\n"
      "dup v2.4s, %w[shift]\n"
      "ld1r {v3.4s}, [%x[offsets]], #4\n"
      "ld1r {v4.4s}, [%x[offsets]], #4\n"
      "add x0, %x[source], %x[stride]\n"
      "add x1, %x[destination], %x[destination_stride]\n"
      "subs %x[count], %x[count], #6\n"
      "beq 2f\n"

      "1:"
      "subs %x[count], %x[count], #8\n"
      "ld1 {v5.4s, v6.4s}, [%x[source]], #32\n"
      "ld1 {v7.4s, v8.4s}, [x0], #32\n"
      "prfm pldl1keep, [%x[source]]\n"
      "prfm pldl1keep, [x0]\n"
      "add v5.4s, v5.4s, v3.4s\n"
      "add v6.4s, v6.4s, v3.4s\n"
      "add v7.4s, v7.4s, v4.4s\n"
      "add v8.4s, v8.4s, v4.4s\n"
      "mul v5.4s, v5.4s, v0.4s\n"
      "mul v6.4s, v6.4s, v0.4s\n"
      "mul v7.4s, v7.4s, v0.4s\n"
      "mul v8.4s, v8.4s, v0.4s\n"
      "add v5.4s, v5.4s, v1.4s\n"
      "add v6.4s, v6.4s, v1.4s\n"
      "add v7.4s, v7.4s, v1.4s\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "sshl v5.4s, v5.4s, v2.4s\n"
      "sshl v6.4s, v6.4s, v2.4s\n"
      "sshl v7.4s, v7.4s, v2.4s\n"
      "sshl v8.4s, v8.4s, v2.4s\n"
      "sqxtn v5.4h, v5.4s\n"
      "sqxtn2 v5.8h, v6.4s\n"
      "sqxtn v7.4h, v7.4s\n"
      "sqxtn2 v7.8h, v8.4s\n"
      "sqxtun v5.8b, v5.8h\n"
      "sqxtun v7.8b, v7.8h\n"
      "st1 {v5.2s}, [%x[destination]], #8\n"
      "st1 {v7.2s}, [x1], #8\n"

      "bne 1b\n"
      "2:"
      "ld1 {v5.4s}, [%x[source]], #16\n"
      "ld1 {v6.2s}, [%x[source]], #8\n"
      "ld1 {v7.4s}, [x0], #16\n"
      "ld1 {v8.2s}, [x0], #8\n"
      "add v5.4s, v5.4s, v3.4s\n"
      "add v6.4s, v6.4s, v3.4s\n"
      "add v7.4s, v7.4s, v4.4s\n"
      "add v8.4s, v8.4s, v4.4s\n"
      "mul v5.4s, v5.4s, v0.4s\n"
      "mul v6.4s, v6.4s, v0.4s\n"
      "mul v7.4s, v7.4s, v0.4s\n"
      "mul v8.4s, v8.4s, v0.4s\n"
      "add v5.4s, v5.4s, v1.4s\n"
      "add v6.4s, v6.4s, v1.4s\n"
      "add v7.4s, v7.4s, v1.4s\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "sshl v5.4s, v5.4s, v2.4s\n"
      "sshl v6.4s, v6.4s, v2.4s\n"
      "sshl v7.4s, v7.4s, v2.4s\n"
      "sshl v8.4s, v8.4s, v2.4s\n"
      "sqxtn v5.4h, v5.4s\n"
      "sqxtn2 v5.8h, v6.4s\n"
      "sqxtn v7.4h, v7.4s\n"
      "sqxtn2 v7.8h, v8.4s\n"
      "sqxtun v5.8b, v5.8h\n"
      "sqxtun v7.8b, v7.8h\n"
      "st1 {v5.s}[0], [%x[destination]], #4\n"
      "st1 {v5.h}[2], [%x[destination]], #2\n"
      "st1 {v7.s}[0], [x1], #4\n"
      "st1 {v7.h}[2], [x1], #2\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [stride] "+r"(stride), [shift] "+r"(shift),
        [destination] "+r"(destination), [offsets] "+r"(offsets),
        [source] "+r"(source), [destination_stride] "+r"(destination_stride),
        [rounding_offset] "+r"(rounding_offset)
      :
      : "x0", "x1", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "cc",
        "memory");
}

void qnt_2x8_7(const std::int32_t* source, std::int32_t count,
               std::int32_t stride, const std::int32_t* offsets,
               std::uint8_t* destination, std::int32_t destination_stride,
               std::int32_t multiplicative_offset, std::int32_t rounding_offset,
               std::int32_t shift) {
  asm volatile(
      "dup v0.4s, %w[multiplicative_offset]\n"
      "dup v1.4s, %w[rounding_offset]\n"
      "dup v2.4s, %w[shift]\n"
      "ld1r {v3.4s}, [%x[offsets]], #4\n"
      "ld1r {v4.4s}, [%x[offsets]], #4\n"
      "add x0, %x[source], %x[stride]\n"
      "add x1, %x[destination], %x[destination_stride]\n"
      "subs %x[count], %x[count], #7\n"
      "beq 2f\n"

      "1:"
      "subs %x[count], %x[count], #8\n"
      "ld1 {v5.4s, v6.4s}, [%x[source]], #32\n"
      "ld1 {v7.4s, v8.4s}, [x0], #32\n"
      "prfm pldl1keep, [%x[source]]\n"
      "prfm pldl1keep, [x0]\n"
      "add v5.4s, v5.4s, v3.4s\n"
      "add v6.4s, v6.4s, v3.4s\n"
      "add v7.4s, v7.4s, v4.4s\n"
      "add v8.4s, v8.4s, v4.4s\n"
      "mul v5.4s, v5.4s, v0.4s\n"
      "mul v6.4s, v6.4s, v0.4s\n"
      "mul v7.4s, v7.4s, v0.4s\n"
      "mul v8.4s, v8.4s, v0.4s\n"
      "add v5.4s, v5.4s, v1.4s\n"
      "add v6.4s, v6.4s, v1.4s\n"
      "add v7.4s, v7.4s, v1.4s\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "sshl v5.4s, v5.4s, v2.4s\n"
      "sshl v6.4s, v6.4s, v2.4s\n"
      "sshl v7.4s, v7.4s, v2.4s\n"
      "sshl v8.4s, v8.4s, v2.4s\n"
      "sqxtn v5.4h, v5.4s\n"
      "sqxtn2 v5.8h, v6.4s\n"
      "sqxtn v7.4h, v7.4s\n"
      "sqxtn2 v7.8h, v8.4s\n"
      "sqxtun v5.8b, v5.8h\n"
      "sqxtun v7.8b, v7.8h\n"
      "st1 {v5.2s}, [%x[destination]], #8\n"
      "st1 {v7.2s}, [x1], #8\n"

      "bne 1b\n"
      "2:"
      "ld1 {v5.4s}, [%x[source]], #16\n"
      "ld1 {v6.2s}, [%x[source]], #8\n"
      "ld1 {v6.s}[2], [%x[source]], #4\n"
      "ld1 {v7.4s}, [x0], #16\n"
      "ld1 {v8.2s}, [x0], #8\n"
      "ld1 {v8.s}[2], [x0], #4\n"
      "add v5.4s, v5.4s, v3.4s\n"
      "add v6.4s, v6.4s, v3.4s\n"
      "add v7.4s, v7.4s, v4.4s\n"
      "add v8.4s, v8.4s, v4.4s\n"
      "mul v5.4s, v5.4s, v0.4s\n"
      "mul v6.4s, v6.4s, v0.4s\n"
      "mul v7.4s, v7.4s, v0.4s\n"
      "mul v8.4s, v8.4s, v0.4s\n"
      "add v5.4s, v5.4s, v1.4s\n"
      "add v6.4s, v6.4s, v1.4s\n"
      "add v7.4s, v7.4s, v1.4s\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "sshl v5.4s, v5.4s, v2.4s\n"
      "sshl v6.4s, v6.4s, v2.4s\n"
      "sshl v7.4s, v7.4s, v2.4s\n"
      "sshl v8.4s, v8.4s, v2.4s\n"
      "sqxtn v5.4h, v5.4s\n"
      "sqxtn2 v5.8h, v6.4s\n"
      "sqxtn v7.4h, v7.4s\n"
      "sqxtn2 v7.8h, v8.4s\n"
      "sqxtun v5.8b, v5.8h\n"
      "sqxtun v7.8b, v7.8h\n"
      "st1 {v5.s}[0], [%x[destination]], #4\n"
      "st1 {v5.h}[2], [%x[destination]], #2\n"
      "st1 {v5.b}[6], [%x[destination]], #1\n"
      "st1 {v7.s}[0], [x1], #4\n"
      "st1 {v7.h}[2], [x1], #2\n"
      "st1 {v7.b}[6], [x1], #1\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [stride] "+r"(stride), [shift] "+r"(shift),
        [destination] "+r"(destination), [offsets] "+r"(offsets),
        [source] "+r"(source), [destination_stride] "+r"(destination_stride),
        [rounding_offset] "+r"(rounding_offset)
      :
      : "x0", "x1", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "cc",
        "memory");
}

void qnt_3x8(const std::int32_t* source, std::int32_t count,
             std::int32_t stride, const std::int32_t* offsets,
             std::uint8_t* destination, std::int32_t destination_stride,
             std::int32_t multiplicative_offset, std::int32_t rounding_offset,
             std::int32_t shift) {
  asm volatile(
      "dup v0.4s, %w[multiplicative_offset]\n"
      "dup v1.4s, %w[rounding_offset]\n"
      "dup v2.4s, %w[shift]\n"
      "ld1r {v3.4s}, [%x[offsets]], #4\n"
      "ld1r {v4.4s}, [%x[offsets]], #4\n"
      "ld1r {v5.4s}, [%x[offsets]], #4\n"
      "add x0, %x[source], %x[stride]\n"
      "add x1, %x[destination], %x[destination_stride]\n"
      "add x2, x0, %x[stride]\n"
      "add x3, x1, %x[destination_stride]\n"

      "1:"
      "subs %x[count], %x[count], #8\n"
      "ld1 {v6.4s, v7.4s}, [%x[source]], #32\n"
      "ld1 {v8.4s, v9.4s}, [x0], #32\n"
      "ld1 {v10.4s, v11.4s}, [x2], #32\n"
      "prfm pldl1keep, [%x[source]]\n"
      "prfm pldl1keep, [x0]\n"
      "prfm pldl1keep, [x2]\n"
      "add v6.4s, v6.4s, v3.4s\n"
      "add v7.4s, v7.4s, v3.4s\n"
      "add v8.4s, v8.4s, v4.4s\n"
      "add v9.4s, v9.4s, v4.4s\n"
      "add v10.4s, v10.4s, v5.4s\n"
      "add v11.4s, v11.4s, v5.4s\n"
      "mul v6.4s, v6.4s, v0.4s\n"
      "mul v7.4s, v7.4s, v0.4s\n"
      "mul v8.4s, v8.4s, v0.4s\n"
      "mul v9.4s, v9.4s, v0.4s\n"
      "mul v10.4s, v10.4s, v0.4s\n"
      "mul v11.4s, v11.4s, v0.4s\n"
      "add v6.4s, v6.4s, v1.4s\n"
      "add v7.4s, v7.4s, v1.4s\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "add v9.4s, v9.4s, v1.4s\n"
      "add v10.4s, v10.4s, v1.4s\n"
      "add v11.4s, v11.4s, v1.4s\n"
      "sshl v6.4s, v6.4s, v2.4s\n"
      "sshl v7.4s, v7.4s, v2.4s\n"
      "sshl v8.4s, v8.4s, v2.4s\n"
      "sshl v9.4s, v9.4s, v2.4s\n"
      "sshl v10.4s, v10.4s, v2.4s\n"
      "sshl v11.4s, v11.4s, v2.4s\n"
      "sqxtn v6.4h, v6.4s\n"
      "sqxtn2 v6.8h, v7.4s\n"
      "sqxtn v8.4h, v8.4s\n"
      "sqxtn2 v8.8h, v9.4s\n"
      "sqxtn v10.4h, v10.4s\n"
      "sqxtn2 v10.8h, v11.4s\n"
      "sqxtun v6.8b, v6.8h\n"
      "sqxtun v8.8b, v8.8h\n"
      "sqxtun v10.8b, v10.8h\n"
      "st1 {v6.2s}, [%x[destination]], #8\n"
      "st1 {v8.2s}, [x1], #8\n"
      "st1 {v10.2s}, [x3], #8\n"

      "bne 1b\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [stride] "+r"(stride), [shift] "+r"(shift),
        [destination] "+r"(destination), [offsets] "+r"(offsets),
        [source] "+r"(source), [destination_stride] "+r"(destination_stride),
        [rounding_offset] "+r"(rounding_offset)
      :
      : "x0", "x1", "x2", "x3", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
        "v8", "v9", "v10", "v11", "cc", "memory");
}

void qnt_3x8_1(const std::int32_t* source, std::int32_t count,
               std::int32_t stride, const std::int32_t* offsets,
               std::uint8_t* destination, std::int32_t destination_stride,
               std::int32_t multiplicative_offset, std::int32_t rounding_offset,
               std::int32_t shift) {
  asm volatile(
      "dup v0.4s, %w[multiplicative_offset]\n"
      "dup v1.4s, %w[rounding_offset]\n"
      "dup v2.4s, %w[shift]\n"
      "ld1r {v3.4s}, [%x[offsets]], #4\n"
      "ld1r {v4.4s}, [%x[offsets]], #4\n"
      "ld1r {v5.4s}, [%x[offsets]], #4\n"
      "add x0, %x[source], %x[stride]\n"
      "add x1, %x[destination], %x[destination_stride]\n"
      "add x2, x0, %x[stride]\n"
      "add x3, x1, %x[destination_stride]\n"
      "subs %x[count], %x[count], #1\n"
      "beq 2f\n"

      "1:"
      "subs %x[count], %x[count], #8\n"
      "ld1 {v6.4s, v7.4s}, [%x[source]], #32\n"
      "ld1 {v8.4s, v9.4s}, [x0], #32\n"
      "ld1 {v10.4s, v11.4s}, [x2], #32\n"
      "prfm pldl1keep, [%x[source]]\n"
      "prfm pldl1keep, [x0]\n"
      "prfm pldl1keep, [x2]\n"
      "add v6.4s, v6.4s, v3.4s\n"
      "add v7.4s, v7.4s, v3.4s\n"
      "add v8.4s, v8.4s, v4.4s\n"
      "add v9.4s, v9.4s, v4.4s\n"
      "add v10.4s, v10.4s, v5.4s\n"
      "add v11.4s, v11.4s, v5.4s\n"
      "mul v6.4s, v6.4s, v0.4s\n"
      "mul v7.4s, v7.4s, v0.4s\n"
      "mul v8.4s, v8.4s, v0.4s\n"
      "mul v9.4s, v9.4s, v0.4s\n"
      "mul v10.4s, v10.4s, v0.4s\n"
      "mul v11.4s, v11.4s, v0.4s\n"
      "add v6.4s, v6.4s, v1.4s\n"
      "add v7.4s, v7.4s, v1.4s\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "add v9.4s, v9.4s, v1.4s\n"
      "add v10.4s, v10.4s, v1.4s\n"
      "add v11.4s, v11.4s, v1.4s\n"
      "sshl v6.4s, v6.4s, v2.4s\n"
      "sshl v7.4s, v7.4s, v2.4s\n"
      "sshl v8.4s, v8.4s, v2.4s\n"
      "sshl v9.4s, v9.4s, v2.4s\n"
      "sshl v10.4s, v10.4s, v2.4s\n"
      "sshl v11.4s, v11.4s, v2.4s\n"
      "sqxtn v6.4h, v6.4s\n"
      "sqxtn2 v6.8h, v7.4s\n"
      "sqxtn v8.4h, v8.4s\n"
      "sqxtn2 v8.8h, v9.4s\n"
      "sqxtn v10.4h, v10.4s\n"
      "sqxtn2 v10.8h, v11.4s\n"
      "sqxtun v6.8b, v6.8h\n"
      "sqxtun v8.8b, v8.8h\n"
      "sqxtun v10.8b, v10.8h\n"
      "st1 {v6.2s}, [%x[destination]], #8\n"
      "st1 {v8.2s}, [x1], #8\n"
      "st1 {v10.2s}, [x3], #8\n"

      "bne 1b\n"
      "2:"
      "ld1 {v6.s}[0], [%x[source]], #4\n"
      "ld1 {v8.s}[0], [x0], #4\n"
      "ld1 {v10.s}[0], [x2], #4\n"
      "add v6.4s, v6.4s, v3.4s\n"
      "add v8.4s, v8.4s, v4.4s\n"
      "add v10.4s, v10.4s, v5.4s\n"
      "mul v6.4s, v6.4s, v0.4s\n"
      "mul v8.4s, v8.4s, v0.4s\n"
      "mul v10.4s, v10.4s, v0.4s\n"
      "add v6.4s, v6.4s, v1.4s\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "add v10.4s, v10.4s, v1.4s\n"
      "sshl v6.4s, v6.4s, v2.4s\n"
      "sshl v8.4s, v8.4s, v2.4s\n"
      "sshl v10.4s, v10.4s, v2.4s\n"
      "sqxtn v6.4h, v6.4s\n"
      "sqxtn v8.4h, v8.4s\n"
      "sqxtn v10.4h, v10.4s\n"
      "sqxtun v6.8b, v6.8h\n"
      "sqxtun v8.8b, v8.8h\n"
      "sqxtun v10.8b, v10.8h\n"
      "st1 {v6.b}[0], [%x[destination]], #1\n"
      "st1 {v8.b}[0], [x1], #1\n"
      "st1 {v10.b}[0], [x3], #1\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [stride] "+r"(stride), [shift] "+r"(shift),
        [destination] "+r"(destination), [offsets] "+r"(offsets),
        [source] "+r"(source), [destination_stride] "+r"(destination_stride),
        [rounding_offset] "+r"(rounding_offset)
      :
      : "x0", "x1", "x2", "x3", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
        "v8", "v9", "v10", "v11", "cc", "memory");
}

void qnt_3x8_2(const std::int32_t* source, std::int32_t count,
               std::int32_t stride, const std::int32_t* offsets,
               std::uint8_t* destination, std::int32_t destination_stride,
               std::int32_t multiplicative_offset, std::int32_t rounding_offset,
               std::int32_t shift) {
  asm volatile(
      "dup v0.4s, %w[multiplicative_offset]\n"
      "dup v1.4s, %w[rounding_offset]\n"
      "dup v2.4s, %w[shift]\n"
      "ld1r {v3.4s}, [%x[offsets]], #4\n"
      "ld1r {v4.4s}, [%x[offsets]], #4\n"
      "ld1r {v5.4s}, [%x[offsets]], #4\n"
      "add x0, %x[source], %x[stride]\n"
      "add x1, %x[destination], %x[destination_stride]\n"
      "add x2, x0, %x[stride]\n"
      "add x3, x1, %x[destination_stride]\n"
      "subs %x[count], %x[count], #2\n"
      "beq 2f\n"

      "1:"
      "subs %x[count], %x[count], #8\n"
      "ld1 {v6.4s, v7.4s}, [%x[source]], #32\n"
      "ld1 {v8.4s, v9.4s}, [x0], #32\n"
      "ld1 {v10.4s, v11.4s}, [x2], #32\n"
      "prfm pldl1keep, [%x[source]]\n"
      "prfm pldl1keep, [x0]\n"
      "prfm pldl1keep, [x2]\n"
      "add v6.4s, v6.4s, v3.4s\n"
      "add v7.4s, v7.4s, v3.4s\n"
      "add v8.4s, v8.4s, v4.4s\n"
      "add v9.4s, v9.4s, v4.4s\n"
      "add v10.4s, v10.4s, v5.4s\n"
      "add v11.4s, v11.4s, v5.4s\n"
      "mul v6.4s, v6.4s, v0.4s\n"
      "mul v7.4s, v7.4s, v0.4s\n"
      "mul v8.4s, v8.4s, v0.4s\n"
      "mul v9.4s, v9.4s, v0.4s\n"
      "mul v10.4s, v10.4s, v0.4s\n"
      "mul v11.4s, v11.4s, v0.4s\n"
      "add v6.4s, v6.4s, v1.4s\n"
      "add v7.4s, v7.4s, v1.4s\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "add v9.4s, v9.4s, v1.4s\n"
      "add v10.4s, v10.4s, v1.4s\n"
      "add v11.4s, v11.4s, v1.4s\n"
      "sshl v6.4s, v6.4s, v2.4s\n"
      "sshl v7.4s, v7.4s, v2.4s\n"
      "sshl v8.4s, v8.4s, v2.4s\n"
      "sshl v9.4s, v9.4s, v2.4s\n"
      "sshl v10.4s, v10.4s, v2.4s\n"
      "sshl v11.4s, v11.4s, v2.4s\n"
      "sqxtn v6.4h, v6.4s\n"
      "sqxtn2 v6.8h, v7.4s\n"
      "sqxtn v8.4h, v8.4s\n"
      "sqxtn2 v8.8h, v9.4s\n"
      "sqxtn v10.4h, v10.4s\n"
      "sqxtn2 v10.8h, v11.4s\n"
      "sqxtun v6.8b, v6.8h\n"
      "sqxtun v8.8b, v8.8h\n"
      "sqxtun v10.8b, v10.8h\n"
      "st1 {v6.2s}, [%x[destination]], #8\n"
      "st1 {v8.2s}, [x1], #8\n"
      "st1 {v10.2s}, [x3], #8\n"

      "bne 1b\n"
      "2:"
      "ld1 {v6.2s}, [%x[source]], #8\n"
      "ld1 {v8.2s}, [x0], #8\n"
      "ld1 {v10.2s}, [x2], #8\n"
      "add v6.4s, v6.4s, v3.4s\n"
      "add v8.4s, v8.4s, v4.4s\n"
      "add v10.4s, v10.4s, v5.4s\n"
      "mul v6.4s, v6.4s, v0.4s\n"
      "mul v8.4s, v8.4s, v0.4s\n"
      "mul v10.4s, v10.4s, v0.4s\n"
      "add v6.4s, v6.4s, v1.4s\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "add v10.4s, v10.4s, v1.4s\n"
      "sshl v6.4s, v6.4s, v2.4s\n"
      "sshl v8.4s, v8.4s, v2.4s\n"
      "sshl v10.4s, v10.4s, v2.4s\n"
      "sqxtn v6.4h, v6.4s\n"
      "sqxtn v8.4h, v8.4s\n"
      "sqxtn v10.4h, v10.4s\n"
      "sqxtun v6.8b, v6.8h\n"
      "sqxtun v8.8b, v8.8h\n"
      "sqxtun v10.8b, v10.8h\n"
      "st1 {v6.h}[0], [%x[destination]], #2\n"
      "st1 {v8.h}[0], [x1], #2\n"
      "st1 {v10.h}[0], [x3], #2\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [stride] "+r"(stride), [shift] "+r"(shift),
        [destination] "+r"(destination), [offsets] "+r"(offsets),
        [source] "+r"(source), [destination_stride] "+r"(destination_stride),
        [rounding_offset] "+r"(rounding_offset)
      :
      : "x0", "x1", "x2", "x3", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
        "v8", "v9", "v10", "v11", "cc", "memory");
}

void qnt_3x8_3(const std::int32_t* source, std::int32_t count,
               std::int32_t stride, const std::int32_t* offsets,
               std::uint8_t* destination, std::int32_t destination_stride,
               std::int32_t multiplicative_offset, std::int32_t rounding_offset,
               std::int32_t shift) {
  asm volatile(
      "dup v0.4s, %w[multiplicative_offset]\n"
      "dup v1.4s, %w[rounding_offset]\n"
      "dup v2.4s, %w[shift]\n"
      "ld1r {v3.4s}, [%x[offsets]], #4\n"
      "ld1r {v4.4s}, [%x[offsets]], #4\n"
      "ld1r {v5.4s}, [%x[offsets]], #4\n"
      "add x0, %x[source], %x[stride]\n"
      "add x1, %x[destination], %x[destination_stride]\n"
      "add x2, x0, %x[stride]\n"
      "add x3, x1, %x[destination_stride]\n"
      "subs %x[count], %x[count], #3\n"
      "beq 2f\n"

      "1:"
      "subs %x[count], %x[count], #8\n"
      "ld1 {v6.4s, v7.4s}, [%x[source]], #32\n"
      "ld1 {v8.4s, v9.4s}, [x0], #32\n"
      "ld1 {v10.4s, v11.4s}, [x2], #32\n"
      "prfm pldl1keep, [%x[source]]\n"
      "prfm pldl1keep, [x0]\n"
      "prfm pldl1keep, [x2]\n"
      "add v6.4s, v6.4s, v3.4s\n"
      "add v7.4s, v7.4s, v3.4s\n"
      "add v8.4s, v8.4s, v4.4s\n"
      "add v9.4s, v9.4s, v4.4s\n"
      "add v10.4s, v10.4s, v5.4s\n"
      "add v11.4s, v11.4s, v5.4s\n"
      "mul v6.4s, v6.4s, v0.4s\n"
      "mul v7.4s, v7.4s, v0.4s\n"
      "mul v8.4s, v8.4s, v0.4s\n"
      "mul v9.4s, v9.4s, v0.4s\n"
      "mul v10.4s, v10.4s, v0.4s\n"
      "mul v11.4s, v11.4s, v0.4s\n"
      "add v6.4s, v6.4s, v1.4s\n"
      "add v7.4s, v7.4s, v1.4s\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "add v9.4s, v9.4s, v1.4s\n"
      "add v10.4s, v10.4s, v1.4s\n"
      "add v11.4s, v11.4s, v1.4s\n"
      "sshl v6.4s, v6.4s, v2.4s\n"
      "sshl v7.4s, v7.4s, v2.4s\n"
      "sshl v8.4s, v8.4s, v2.4s\n"
      "sshl v9.4s, v9.4s, v2.4s\n"
      "sshl v10.4s, v10.4s, v2.4s\n"
      "sshl v11.4s, v11.4s, v2.4s\n"
      "sqxtn v6.4h, v6.4s\n"
      "sqxtn2 v6.8h, v7.4s\n"
      "sqxtn v8.4h, v8.4s\n"
      "sqxtn2 v8.8h, v9.4s\n"
      "sqxtn v10.4h, v10.4s\n"
      "sqxtn2 v10.8h, v11.4s\n"
      "sqxtun v6.8b, v6.8h\n"
      "sqxtun v8.8b, v8.8h\n"
      "sqxtun v10.8b, v10.8h\n"
      "st1 {v6.2s}, [%x[destination]], #8\n"
      "st1 {v8.2s}, [x1], #8\n"
      "st1 {v10.2s}, [x3], #8\n"

      "bne 1b\n"
      "2:"
      "ld1 {v6.2s}, [%x[source]], #8\n"
      "ld1 {v6.s}[2], [%x[source]], #4\n"
      "ld1 {v8.2s}, [x0], #8\n"
      "ld1 {v8.s}[2], [x0], #4\n"
      "ld1 {v10.2s}, [x2], #8\n"
      "ld1 {v10.s}[2], [x2], #4\n"
      "add v6.4s, v6.4s, v3.4s\n"
      "add v8.4s, v8.4s, v4.4s\n"
      "add v10.4s, v10.4s, v5.4s\n"
      "mul v6.4s, v6.4s, v0.4s\n"
      "mul v8.4s, v8.4s, v0.4s\n"
      "mul v10.4s, v10.4s, v0.4s\n"
      "add v6.4s, v6.4s, v1.4s\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "add v10.4s, v10.4s, v1.4s\n"
      "sshl v6.4s, v6.4s, v2.4s\n"
      "sshl v8.4s, v8.4s, v2.4s\n"
      "sshl v10.4s, v10.4s, v2.4s\n"
      "sqxtn v6.4h, v6.4s\n"
      "sqxtn v8.4h, v8.4s\n"
      "sqxtn v10.4h, v10.4s\n"
      "sqxtun v6.8b, v6.8h\n"
      "sqxtun v8.8b, v8.8h\n"
      "sqxtun v10.8b, v10.8h\n"
      "st1 {v6.h}[0], [%x[destination]], #2\n"
      "st1 {v6.b}[2], [%x[destination]], #1\n"
      "st1 {v8.h}[0], [x1], #2\n"
      "st1 {v8.b}[2], [x1], #1\n"
      "st1 {v10.h}[0], [x3], #2\n"
      "st1 {v10.b}[2], [x3], #1\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [stride] "+r"(stride), [shift] "+r"(shift),
        [destination] "+r"(destination), [offsets] "+r"(offsets),
        [source] "+r"(source), [destination_stride] "+r"(destination_stride),
        [rounding_offset] "+r"(rounding_offset)
      :
      : "x0", "x1", "x2", "x3", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
        "v8", "v9", "v10", "v11", "cc", "memory");
}

void qnt_3x8_4(const std::int32_t* source, std::int32_t count,
               std::int32_t stride, const std::int32_t* offsets,
               std::uint8_t* destination, std::int32_t destination_stride,
               std::int32_t multiplicative_offset, std::int32_t rounding_offset,
               std::int32_t shift) {
  asm volatile(
      "dup v0.4s, %w[multiplicative_offset]\n"
      "dup v1.4s, %w[rounding_offset]\n"
      "dup v2.4s, %w[shift]\n"
      "ld1r {v3.4s}, [%x[offsets]], #4\n"
      "ld1r {v4.4s}, [%x[offsets]], #4\n"
      "ld1r {v5.4s}, [%x[offsets]], #4\n"
      "add x0, %x[source], %x[stride]\n"
      "add x1, %x[destination], %x[destination_stride]\n"
      "add x2, x0, %x[stride]\n"
      "add x3, x1, %x[destination_stride]\n"
      "subs %x[count], %x[count], #4\n"
      "beq 2f\n"

      "1:"
      "subs %x[count], %x[count], #8\n"
      "ld1 {v6.4s, v7.4s}, [%x[source]], #32\n"
      "ld1 {v8.4s, v9.4s}, [x0], #32\n"
      "ld1 {v10.4s, v11.4s}, [x2], #32\n"
      "prfm pldl1keep, [%x[source]]\n"
      "prfm pldl1keep, [x0]\n"
      "prfm pldl1keep, [x2]\n"
      "add v6.4s, v6.4s, v3.4s\n"
      "add v7.4s, v7.4s, v3.4s\n"
      "add v8.4s, v8.4s, v4.4s\n"
      "add v9.4s, v9.4s, v4.4s\n"
      "add v10.4s, v10.4s, v5.4s\n"
      "add v11.4s, v11.4s, v5.4s\n"
      "mul v6.4s, v6.4s, v0.4s\n"
      "mul v7.4s, v7.4s, v0.4s\n"
      "mul v8.4s, v8.4s, v0.4s\n"
      "mul v9.4s, v9.4s, v0.4s\n"
      "mul v10.4s, v10.4s, v0.4s\n"
      "mul v11.4s, v11.4s, v0.4s\n"
      "add v6.4s, v6.4s, v1.4s\n"
      "add v7.4s, v7.4s, v1.4s\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "add v9.4s, v9.4s, v1.4s\n"
      "add v10.4s, v10.4s, v1.4s\n"
      "add v11.4s, v11.4s, v1.4s\n"
      "sshl v6.4s, v6.4s, v2.4s\n"
      "sshl v7.4s, v7.4s, v2.4s\n"
      "sshl v8.4s, v8.4s, v2.4s\n"
      "sshl v9.4s, v9.4s, v2.4s\n"
      "sshl v10.4s, v10.4s, v2.4s\n"
      "sshl v11.4s, v11.4s, v2.4s\n"
      "sqxtn v6.4h, v6.4s\n"
      "sqxtn2 v6.8h, v7.4s\n"
      "sqxtn v8.4h, v8.4s\n"
      "sqxtn2 v8.8h, v9.4s\n"
      "sqxtn v10.4h, v10.4s\n"
      "sqxtn2 v10.8h, v11.4s\n"
      "sqxtun v6.8b, v6.8h\n"
      "sqxtun v8.8b, v8.8h\n"
      "sqxtun v10.8b, v10.8h\n"
      "st1 {v6.2s}, [%x[destination]], #8\n"
      "st1 {v8.2s}, [x1], #8\n"
      "st1 {v10.2s}, [x3], #8\n"

      "bne 1b\n"
      "2:"
      "ld1 {v6.4s}, [%x[source]], #16\n"
      "ld1 {v8.4s}, [x0], #16\n"
      "ld1 {v10.4s}, [x2], #16\n"
      "add v6.4s, v6.4s, v3.4s\n"
      "add v8.4s, v8.4s, v4.4s\n"
      "add v10.4s, v10.4s, v5.4s\n"
      "mul v6.4s, v6.4s, v0.4s\n"
      "mul v8.4s, v8.4s, v0.4s\n"
      "mul v10.4s, v10.4s, v0.4s\n"
      "add v6.4s, v6.4s, v1.4s\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "add v10.4s, v10.4s, v1.4s\n"
      "sshl v6.4s, v6.4s, v2.4s\n"
      "sshl v8.4s, v8.4s, v2.4s\n"
      "sshl v10.4s, v10.4s, v2.4s\n"
      "sqxtn v6.4h, v6.4s\n"
      "sqxtn v8.4h, v8.4s\n"
      "sqxtn v10.4h, v10.4s\n"
      "sqxtun v6.8b, v6.8h\n"
      "sqxtun v8.8b, v8.8h\n"
      "sqxtun v10.8b, v10.8h\n"
      "st1 {v6.s}[0], [%x[destination]], #4\n"
      "st1 {v8.s}[0], [x1], #4\n"
      "st1 {v10.s}[0], [x3], #4\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [stride] "+r"(stride), [shift] "+r"(shift),
        [destination] "+r"(destination), [offsets] "+r"(offsets),
        [source] "+r"(source), [destination_stride] "+r"(destination_stride),
        [rounding_offset] "+r"(rounding_offset)
      :
      : "x0", "x1", "x2", "x3", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
        "v8", "v9", "v10", "v11", "cc", "memory");
}

void qnt_3x8_5(const std::int32_t* source, std::int32_t count,
               std::int32_t stride, const std::int32_t* offsets,
               std::uint8_t* destination, std::int32_t destination_stride,
               std::int32_t multiplicative_offset, std::int32_t rounding_offset,
               std::int32_t shift) {
  asm volatile(
      "dup v0.4s, %w[multiplicative_offset]\n"
      "dup v1.4s, %w[rounding_offset]\n"
      "dup v2.4s, %w[shift]\n"
      "ld1r {v3.4s}, [%x[offsets]], #4\n"
      "ld1r {v4.4s}, [%x[offsets]], #4\n"
      "ld1r {v5.4s}, [%x[offsets]], #4\n"
      "add x0, %x[source], %x[stride]\n"
      "add x1, %x[destination], %x[destination_stride]\n"
      "add x2, x0, %x[stride]\n"
      "add x3, x1, %x[destination_stride]\n"
      "subs %x[count], %x[count], #5\n"
      "beq 2f\n"

      "1:"
      "subs %x[count], %x[count], #8\n"
      "ld1 {v6.4s, v7.4s}, [%x[source]], #32\n"
      "ld1 {v8.4s, v9.4s}, [x0], #32\n"
      "ld1 {v10.4s, v11.4s}, [x2], #32\n"
      "prfm pldl1keep, [%x[source]]\n"
      "prfm pldl1keep, [x0]\n"
      "prfm pldl1keep, [x2]\n"
      "add v6.4s, v6.4s, v3.4s\n"
      "add v7.4s, v7.4s, v3.4s\n"
      "add v8.4s, v8.4s, v4.4s\n"
      "add v9.4s, v9.4s, v4.4s\n"
      "add v10.4s, v10.4s, v5.4s\n"
      "add v11.4s, v11.4s, v5.4s\n"
      "mul v6.4s, v6.4s, v0.4s\n"
      "mul v7.4s, v7.4s, v0.4s\n"
      "mul v8.4s, v8.4s, v0.4s\n"
      "mul v9.4s, v9.4s, v0.4s\n"
      "mul v10.4s, v10.4s, v0.4s\n"
      "mul v11.4s, v11.4s, v0.4s\n"
      "add v6.4s, v6.4s, v1.4s\n"
      "add v7.4s, v7.4s, v1.4s\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "add v9.4s, v9.4s, v1.4s\n"
      "add v10.4s, v10.4s, v1.4s\n"
      "add v11.4s, v11.4s, v1.4s\n"
      "sshl v6.4s, v6.4s, v2.4s\n"
      "sshl v7.4s, v7.4s, v2.4s\n"
      "sshl v8.4s, v8.4s, v2.4s\n"
      "sshl v9.4s, v9.4s, v2.4s\n"
      "sshl v10.4s, v10.4s, v2.4s\n"
      "sshl v11.4s, v11.4s, v2.4s\n"
      "sqxtn v6.4h, v6.4s\n"
      "sqxtn2 v6.8h, v7.4s\n"
      "sqxtn v8.4h, v8.4s\n"
      "sqxtn2 v8.8h, v9.4s\n"
      "sqxtn v10.4h, v10.4s\n"
      "sqxtn2 v10.8h, v11.4s\n"
      "sqxtun v6.8b, v6.8h\n"
      "sqxtun v8.8b, v8.8h\n"
      "sqxtun v10.8b, v10.8h\n"
      "st1 {v6.2s}, [%x[destination]], #8\n"
      "st1 {v8.2s}, [x1], #8\n"
      "st1 {v10.2s}, [x3], #8\n"

      "bne 1b\n"
      "2:"
      "ld1 {v6.4s}, [%x[source]], #16\n"
      "ld1 {v7.s}[0], [%x[source]], #4\n"
      "ld1 {v8.4s}, [x0], #16\n"
      "ld1 {v9.s}[0], [x0], #4\n"
      "ld1 {v10.4s}, [x2], #16\n"
      "ld1 {v11.s}[0], [x2], #4\n"
      "add v6.4s, v6.4s, v3.4s\n"
      "add v7.4s, v7.4s, v3.4s\n"
      "add v8.4s, v8.4s, v4.4s\n"
      "add v9.4s, v9.4s, v4.4s\n"
      "add v10.4s, v10.4s, v5.4s\n"
      "add v11.4s, v11.4s, v5.4s\n"
      "mul v6.4s, v6.4s, v0.4s\n"
      "mul v7.4s, v7.4s, v0.4s\n"
      "mul v8.4s, v8.4s, v0.4s\n"
      "mul v9.4s, v9.4s, v0.4s\n"
      "mul v10.4s, v10.4s, v0.4s\n"
      "mul v11.4s, v11.4s, v0.4s\n"
      "add v6.4s, v6.4s, v1.4s\n"
      "add v7.4s, v7.4s, v1.4s\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "add v9.4s, v9.4s, v1.4s\n"
      "add v10.4s, v10.4s, v1.4s\n"
      "add v11.4s, v11.4s, v1.4s\n"
      "sshl v6.4s, v6.4s, v2.4s\n"
      "sshl v7.4s, v7.4s, v2.4s\n"
      "sshl v8.4s, v8.4s, v2.4s\n"
      "sshl v9.4s, v9.4s, v2.4s\n"
      "sshl v10.4s, v10.4s, v2.4s\n"
      "sshl v11.4s, v11.4s, v2.4s\n"
      "sqxtn v6.4h, v6.4s\n"
      "sqxtn2 v6.8h, v7.4s\n"
      "sqxtn v8.4h, v8.4s\n"
      "sqxtn2 v8.8h, v9.4s\n"
      "sqxtn v10.4h, v10.4s\n"
      "sqxtn2 v10.8h, v11.4s\n"
      "sqxtun v6.8b, v6.8h\n"
      "sqxtun v8.8b, v8.8h\n"
      "sqxtun v10.8b, v10.8h\n"
      "st1 {v6.s}[0], [%x[destination]], #4\n"
      "st1 {v6.b}[4], [%x[destination]], #1\n"
      "st1 {v8.s}[0], [x1], #4\n"
      "st1 {v8.b}[4], [x1], #1\n"
      "st1 {v10.s}[0], [x3], #4\n"
      "st1 {v10.b}[4], [x3], #1\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [stride] "+r"(stride), [shift] "+r"(shift),
        [destination] "+r"(destination), [offsets] "+r"(offsets),
        [source] "+r"(source), [destination_stride] "+r"(destination_stride),
        [rounding_offset] "+r"(rounding_offset)
      :
      : "x0", "x1", "x2", "x3", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
        "v8", "v9", "v10", "v11", "cc", "memory");
}

void qnt_3x8_6(const std::int32_t* source, std::int32_t count,
               std::int32_t stride, const std::int32_t* offsets,
               std::uint8_t* destination, std::int32_t destination_stride,
               std::int32_t multiplicative_offset, std::int32_t rounding_offset,
               std::int32_t shift) {
  asm volatile(
      "dup v0.4s, %w[multiplicative_offset]\n"
      "dup v1.4s, %w[rounding_offset]\n"
      "dup v2.4s, %w[shift]\n"
      "ld1r {v3.4s}, [%x[offsets]], #4\n"
      "ld1r {v4.4s}, [%x[offsets]], #4\n"
      "ld1r {v5.4s}, [%x[offsets]], #4\n"
      "add x0, %x[source], %x[stride]\n"
      "add x1, %x[destination], %x[destination_stride]\n"
      "add x2, x0, %x[stride]\n"
      "add x3, x1, %x[destination_stride]\n"
      "subs %x[count], %x[count], #6\n"
      "beq 2f\n"

      "1:"
      "subs %x[count], %x[count], #8\n"
      "ld1 {v6.4s, v7.4s}, [%x[source]], #32\n"
      "ld1 {v8.4s, v9.4s}, [x0], #32\n"
      "ld1 {v10.4s, v11.4s}, [x2], #32\n"
      "prfm pldl1keep, [%x[source]]\n"
      "prfm pldl1keep, [x0]\n"
      "prfm pldl1keep, [x2]\n"
      "add v6.4s, v6.4s, v3.4s\n"
      "add v7.4s, v7.4s, v3.4s\n"
      "add v8.4s, v8.4s, v4.4s\n"
      "add v9.4s, v9.4s, v4.4s\n"
      "add v10.4s, v10.4s, v5.4s\n"
      "add v11.4s, v11.4s, v5.4s\n"
      "mul v6.4s, v6.4s, v0.4s\n"
      "mul v7.4s, v7.4s, v0.4s\n"
      "mul v8.4s, v8.4s, v0.4s\n"
      "mul v9.4s, v9.4s, v0.4s\n"
      "mul v10.4s, v10.4s, v0.4s\n"
      "mul v11.4s, v11.4s, v0.4s\n"
      "add v6.4s, v6.4s, v1.4s\n"
      "add v7.4s, v7.4s, v1.4s\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "add v9.4s, v9.4s, v1.4s\n"
      "add v10.4s, v10.4s, v1.4s\n"
      "add v11.4s, v11.4s, v1.4s\n"
      "sshl v6.4s, v6.4s, v2.4s\n"
      "sshl v7.4s, v7.4s, v2.4s\n"
      "sshl v8.4s, v8.4s, v2.4s\n"
      "sshl v9.4s, v9.4s, v2.4s\n"
      "sshl v10.4s, v10.4s, v2.4s\n"
      "sshl v11.4s, v11.4s, v2.4s\n"
      "sqxtn v6.4h, v6.4s\n"
      "sqxtn2 v6.8h, v7.4s\n"
      "sqxtn v8.4h, v8.4s\n"
      "sqxtn2 v8.8h, v9.4s\n"
      "sqxtn v10.4h, v10.4s\n"
      "sqxtn2 v10.8h, v11.4s\n"
      "sqxtun v6.8b, v6.8h\n"
      "sqxtun v8.8b, v8.8h\n"
      "sqxtun v10.8b, v10.8h\n"
      "st1 {v6.2s}, [%x[destination]], #8\n"
      "st1 {v8.2s}, [x1], #8\n"
      "st1 {v10.2s}, [x3], #8\n"

      "bne 1b\n"
      "2:"
      "ld1 {v6.4s}, [%x[source]], #16\n"
      "ld1 {v7.2s}, [%x[source]], #8\n"
      "ld1 {v8.4s}, [x0], #16\n"
      "ld1 {v9.2s}, [x0], #8\n"
      "ld1 {v10.4s}, [x2], #16\n"
      "ld1 {v11.2s}, [x2], #8\n"
      "add v6.4s, v6.4s, v3.4s\n"
      "add v7.4s, v7.4s, v3.4s\n"
      "add v8.4s, v8.4s, v4.4s\n"
      "add v9.4s, v9.4s, v4.4s\n"
      "add v10.4s, v10.4s, v5.4s\n"
      "add v11.4s, v11.4s, v5.4s\n"
      "mul v6.4s, v6.4s, v0.4s\n"
      "mul v7.4s, v7.4s, v0.4s\n"
      "mul v8.4s, v8.4s, v0.4s\n"
      "mul v9.4s, v9.4s, v0.4s\n"
      "mul v10.4s, v10.4s, v0.4s\n"
      "mul v11.4s, v11.4s, v0.4s\n"
      "add v6.4s, v6.4s, v1.4s\n"
      "add v7.4s, v7.4s, v1.4s\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "add v9.4s, v9.4s, v1.4s\n"
      "add v10.4s, v10.4s, v1.4s\n"
      "add v11.4s, v11.4s, v1.4s\n"
      "sshl v6.4s, v6.4s, v2.4s\n"
      "sshl v7.4s, v7.4s, v2.4s\n"
      "sshl v8.4s, v8.4s, v2.4s\n"
      "sshl v9.4s, v9.4s, v2.4s\n"
      "sshl v10.4s, v10.4s, v2.4s\n"
      "sshl v11.4s, v11.4s, v2.4s\n"
      "sqxtn v6.4h, v6.4s\n"
      "sqxtn2 v6.8h, v7.4s\n"
      "sqxtn v8.4h, v8.4s\n"
      "sqxtn2 v8.8h, v9.4s\n"
      "sqxtn v10.4h, v10.4s\n"
      "sqxtn2 v10.8h, v11.4s\n"
      "sqxtun v6.8b, v6.8h\n"
      "sqxtun v8.8b, v8.8h\n"
      "sqxtun v10.8b, v10.8h\n"
      "st1 {v6.s}[0], [%x[destination]], #4\n"
      "st1 {v6.h}[2], [%x[destination]], #2\n"
      "st1 {v8.s}[0], [x1], #4\n"
      "st1 {v8.h}[2], [x1], #2\n"
      "st1 {v10.s}[0], [x3], #4\n"
      "st1 {v10.h}[2], [x3], #2\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [stride] "+r"(stride), [shift] "+r"(shift),
        [destination] "+r"(destination), [offsets] "+r"(offsets),
        [source] "+r"(source), [destination_stride] "+r"(destination_stride),
        [rounding_offset] "+r"(rounding_offset)
      :
      : "x0", "x1", "x2", "x3", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
        "v8", "v9", "v10", "v11", "cc", "memory");
}

void qnt_3x8_7(const std::int32_t* source, std::int32_t count,
               std::int32_t stride, const std::int32_t* offsets,
               std::uint8_t* destination, std::int32_t destination_stride,
               std::int32_t multiplicative_offset, std::int32_t rounding_offset,
               std::int32_t shift) {
  asm volatile(
      "dup v0.4s, %w[multiplicative_offset]\n"
      "dup v1.4s, %w[rounding_offset]\n"
      "dup v2.4s, %w[shift]\n"
      "ld1r {v3.4s}, [%x[offsets]], #4\n"
      "ld1r {v4.4s}, [%x[offsets]], #4\n"
      "ld1r {v5.4s}, [%x[offsets]], #4\n"
      "add x0, %x[source], %x[stride]\n"
      "add x1, %x[destination], %x[destination_stride]\n"
      "add x2, x0, %x[stride]\n"
      "add x3, x1, %x[destination_stride]\n"
      "subs %x[count], %x[count], #7\n"
      "beq 2f\n"

      "1:"
      "subs %x[count], %x[count], #8\n"
      "ld1 {v6.4s, v7.4s}, [%x[source]], #32\n"
      "ld1 {v8.4s, v9.4s}, [x0], #32\n"
      "ld1 {v10.4s, v11.4s}, [x2], #32\n"
      "prfm pldl1keep, [%x[source]]\n"
      "prfm pldl1keep, [x0]\n"
      "prfm pldl1keep, [x2]\n"
      "add v6.4s, v6.4s, v3.4s\n"
      "add v7.4s, v7.4s, v3.4s\n"
      "add v8.4s, v8.4s, v4.4s\n"
      "add v9.4s, v9.4s, v4.4s\n"
      "add v10.4s, v10.4s, v5.4s\n"
      "add v11.4s, v11.4s, v5.4s\n"
      "mul v6.4s, v6.4s, v0.4s\n"
      "mul v7.4s, v7.4s, v0.4s\n"
      "mul v8.4s, v8.4s, v0.4s\n"
      "mul v9.4s, v9.4s, v0.4s\n"
      "mul v10.4s, v10.4s, v0.4s\n"
      "mul v11.4s, v11.4s, v0.4s\n"
      "add v6.4s, v6.4s, v1.4s\n"
      "add v7.4s, v7.4s, v1.4s\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "add v9.4s, v9.4s, v1.4s\n"
      "add v10.4s, v10.4s, v1.4s\n"
      "add v11.4s, v11.4s, v1.4s\n"
      "sshl v6.4s, v6.4s, v2.4s\n"
      "sshl v7.4s, v7.4s, v2.4s\n"
      "sshl v8.4s, v8.4s, v2.4s\n"
      "sshl v9.4s, v9.4s, v2.4s\n"
      "sshl v10.4s, v10.4s, v2.4s\n"
      "sshl v11.4s, v11.4s, v2.4s\n"
      "sqxtn v6.4h, v6.4s\n"
      "sqxtn2 v6.8h, v7.4s\n"
      "sqxtn v8.4h, v8.4s\n"
      "sqxtn2 v8.8h, v9.4s\n"
      "sqxtn v10.4h, v10.4s\n"
      "sqxtn2 v10.8h, v11.4s\n"
      "sqxtun v6.8b, v6.8h\n"
      "sqxtun v8.8b, v8.8h\n"
      "sqxtun v10.8b, v10.8h\n"
      "st1 {v6.2s}, [%x[destination]], #8\n"
      "st1 {v8.2s}, [x1], #8\n"
      "st1 {v10.2s}, [x3], #8\n"

      "bne 1b\n"
      "2:"
      "ld1 {v6.4s}, [%x[source]], #16\n"
      "ld1 {v7.2s}, [%x[source]], #8\n"
      "ld1 {v7.s}[2], [%x[source]], #4\n"
      "ld1 {v8.4s}, [x0], #16\n"
      "ld1 {v9.2s}, [x0], #8\n"
      "ld1 {v9.s}[2], [x0], #4\n"
      "ld1 {v10.4s}, [x2], #16\n"
      "ld1 {v11.2s}, [x2], #8\n"
      "ld1 {v11.s}[2], [x2], #4\n"
      "add v6.4s, v6.4s, v3.4s\n"
      "add v7.4s, v7.4s, v3.4s\n"
      "add v8.4s, v8.4s, v4.4s\n"
      "add v9.4s, v9.4s, v4.4s\n"
      "add v10.4s, v10.4s, v5.4s\n"
      "add v11.4s, v11.4s, v5.4s\n"
      "mul v6.4s, v6.4s, v0.4s\n"
      "mul v7.4s, v7.4s, v0.4s\n"
      "mul v8.4s, v8.4s, v0.4s\n"
      "mul v9.4s, v9.4s, v0.4s\n"
      "mul v10.4s, v10.4s, v0.4s\n"
      "mul v11.4s, v11.4s, v0.4s\n"
      "add v6.4s, v6.4s, v1.4s\n"
      "add v7.4s, v7.4s, v1.4s\n"
      "add v8.4s, v8.4s, v1.4s\n"
      "add v9.4s, v9.4s, v1.4s\n"
      "add v10.4s, v10.4s, v1.4s\n"
      "add v11.4s, v11.4s, v1.4s\n"
      "sshl v6.4s, v6.4s, v2.4s\n"
      "sshl v7.4s, v7.4s, v2.4s\n"
      "sshl v8.4s, v8.4s, v2.4s\n"
      "sshl v9.4s, v9.4s, v2.4s\n"
      "sshl v10.4s, v10.4s, v2.4s\n"
      "sshl v11.4s, v11.4s, v2.4s\n"
      "sqxtn v6.4h, v6.4s\n"
      "sqxtn2 v6.8h, v7.4s\n"
      "sqxtn v8.4h, v8.4s\n"
      "sqxtn2 v8.8h, v9.4s\n"
      "sqxtn v10.4h, v10.4s\n"
      "sqxtn2 v10.8h, v11.4s\n"
      "sqxtun v6.8b, v6.8h\n"
      "sqxtun v8.8b, v8.8h\n"
      "sqxtun v10.8b, v10.8h\n"
      "st1 {v6.s}[0], [%x[destination]], #4\n"
      "st1 {v6.h}[2], [%x[destination]], #2\n"
      "st1 {v6.b}[6], [%x[destination]], #1\n"
      "st1 {v8.s}[0], [x1], #4\n"
      "st1 {v8.h}[2], [x1], #2\n"
      "st1 {v8.b}[6], [x1], #1\n"
      "st1 {v10.s}[0], [x3], #4\n"
      "st1 {v10.h}[2], [x3], #2\n"
      "st1 {v10.b}[6], [x3], #1\n"
      : [count] "+r"(count),
        [multiplicative_offset] "+r"(multiplicative_offset),
        [stride] "+r"(stride), [shift] "+r"(shift),
        [destination] "+r"(destination), [offsets] "+r"(offsets),
        [source] "+r"(source), [destination_stride] "+r"(destination_stride),
        [rounding_offset] "+r"(rounding_offset)
      :
      : "x0", "x1", "x2", "x3", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
        "v8", "v9", "v10", "v11", "cc", "memory");
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
#warning "Meta gemm for arm64 requires: GEMMLOWP_NEON_64!"
#endif

#endif  // GEMMLOWP_META_SINGLE_THREAD_GEMM_ARM64_H_
