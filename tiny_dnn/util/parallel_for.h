/*
    Copyright (c) 2016, Taiga Nomi
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include <cassert>
#include <cstdio>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

#include "aligned_allocator.h"
#include "nn_error.h"
#include "tiny_dnn/config.h"

#ifdef CNN_USE_TBB
#ifndef NOMINMAX
#define NOMINMAX  // tbb includes windows.h in tbb/machine/windows_api.h
#endif
#include <tbb/task_group.h>
#include <tbb/tbb.h>
#endif

#if !defined(CNN_USE_OMP) && !defined(CNN_SINGLE_THREAD)
#include <future>
#include <thread>
#endif

#if defined(CNN_USE_GCD) && !defined(CNN_SINGLE_THREAD)
#include <dispatch/dispatch.h>
#endif

namespace tiny_dnn {

#ifdef CNN_USE_TBB

static tbb::task_scheduler_init tbbScheduler(
  tbb::task_scheduler_init::automatic);  // tbb::task_scheduler_init::deferred);

typedef tbb::blocked_range<int> blocked_range;

template <typename Func>
void parallel_for(int begin, int end, const Func &f, int grainsize) {
  tbb::parallel_for(
    blocked_range(begin, end, end - begin > grainsize ? grainsize : 1), f);
}

template <typename Func>
void xparallel_for(int begin, int end, const Func &f) {
  f(blocked_range(begin, end, 100));
}

#else

struct blocked_range {
  typedef int const_iterator;

  blocked_range(int begin, int end) : begin_(begin), end_(end) {}
  blocked_range(size_t begin, size_t end)
    : begin_(static_cast<int>(begin)), end_(static_cast<int>(end)) {}

  const_iterator begin() const { return begin_; }
  const_iterator end() const { return end_; }

 private:
  int begin_;
  int end_;
};

template <typename Func>
void xparallel_for(size_t begin, size_t end, const Func &f) {
  blocked_range r(begin, end);
  f(r);
}

#if defined(CNN_USE_OMP)

template <typename Func>
void parallel_for(int begin, int end, const Func &f, int /*grainsize*/) {
#pragma omp parallel for
  for (int i = begin; i < end; ++i) f(blocked_range(i, i + 1));
}

#elif defined(CNN_USE_GCD)

template <typename Func>
void parallel_for(int begin, int end, const Func &f, int grainsize) {
  int count     = end - begin;
  int blockSize = grainsize;
  if (count < blockSize || blockSize == 0) {
    blockSize = 1;
  }
  int blockCount = (count + blockSize - 1) / blockSize;
  assert(blockCount > 0);

  dispatch_apply(blockCount, dispatch_get_global_queue(QOS_CLASS_DEFAULT, 0),
                 ^(size_t block) {
                   int blockStart = static_cast<int>(block * blockSize);
                   int blockEnd   = blockStart + blockSize;
                   if (blockEnd > end) {
                     blockEnd = end;
                   }
                   assert(blockStart < blockEnd);

                   f(blocked_range(blockStart, blockEnd));
                 });
}

#elif defined(CNN_SINGLE_THREAD)

template <typename Func>
void parallel_for(int begin, int end, const Func &f, int /*grainsize*/) {
  xparallel_for(static_cast<size_t>(begin), static_cast<size_t>(end), f);
}

#else

template <typename Func>
void parallel_for(int start, int end, const Func &f, int /*grainsize*/) {
  int nthreads  = std::thread::hardware_concurrency();
  int blockSize = (end - start) / nthreads;
  if (blockSize * nthreads < end - start) blockSize++;

  std::vector<std::future<void> > futures;

  int blockStart               = start;
  int blockEnd                 = blockStart + blockSize;
  if (blockEnd > end) blockEnd = end;

  for (int i = 0; i < nthreads; i++) {
    futures.push_back(
      std::move(std::async(std::launch::async, [blockStart, blockEnd, &f] {
        f(blocked_range(blockStart, blockEnd));
      })));

    blockStart += blockSize;
    blockEnd = blockStart + blockSize;
    if (blockStart >= end) break;
    if (blockEnd > end) blockEnd = end;
  }

  for (auto &future : futures) future.wait();
}

#endif

#endif  // CNN_USE_TBB

template <typename T, typename U>
bool value_representation(U const &value) {
  return static_cast<U>(static_cast<T>(value)) == value;
}

template <typename T, typename Func>
inline void for_(std::true_type,
                 bool parallelize,
                 int begin,
                 T end,
                 Func f,
                 int grainsize = 100) {
  parallelize = parallelize && value_representation<int>(end);
  parallelize ? parallel_for(begin, static_cast<int>(end), f, grainsize)
              : xparallel_for(begin, static_cast<int>(end), f);
}

template <typename T, typename Func>
inline void for_(std::false_type,
                 bool parallelize,
                 int begin,
                 T end,
                 Func f,
                 int grainsize = 100) {
  parallelize ? parallel_for(begin, static_cast<int>(end), f, grainsize)
              : xparallel_for(begin, end, f);
}

template <typename T, typename Func>
inline void for_(
  bool parallelize, int begin, T end, Func f, int grainsize = 100) {
  static_assert(std::is_integral<T>::value, "end must be integral type");
  for_(typename std::is_unsigned<T>::type(), parallelize, begin, end, f,
       grainsize);
}

template <typename T, typename Func>
void for_i(bool parallelize, T size, Func f, int grainsize = 100) {
#ifdef CNN_SINGLE_THREAD
  parallelize = false;
#endif
  for_(parallelize, 0, size,
       [&](const blocked_range &r) {
#ifdef CNN_USE_OMP
#pragma omp parallel for
#endif
         for (int i = r.begin(); i < r.end(); i++) f(i);
       },
       grainsize);
}

template <typename T, typename Func>
void for_i(T size, Func f, int grainsize = 100) {
  for_i(true, size, f, grainsize);
}

}  // namespace tiny_dnn
