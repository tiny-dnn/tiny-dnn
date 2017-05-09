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

typedef tbb::blocked_range<size_t> blocked_range;

template <typename Func>
void parallel_for(size_t begin, size_t end, const Func &f, size_t grainsize) {
  assert(end >= begin);
  tbb::parallel_for(
    blocked_range(begin, end, end - begin > grainsize ? grainsize : 1), f);
}

template <typename Func>
void xparallel_for(size_t begin, size_t end, const Func &f) {
  f(blocked_range(begin, end, 100));
}

#else

struct blocked_range {
  typedef size_t const_iterator;

  blocked_range(size_t begin, size_t end) : begin_(begin), end_(end) {}
  blocked_range(int begin, int end)
    : begin_(static_cast<size_t>(begin)), end_(static_cast<size_t>(end)) {}

  const_iterator begin() const { return begin_; }
  const_iterator end() const { return end_; }

 private:
  size_t begin_;
  size_t end_;
};

template <typename Func>
void xparallel_for(size_t begin, size_t end, const Func &f) {
  blocked_range r(begin, end);
  f(r);
}

#if defined(CNN_USE_OMP)

template <typename Func>
void parallel_for(size_t begin,
                  size_t end,
                  const Func &f,
                  size_t /*grainsize*/) {
  assert(end >= begin);
#pragma omp parallel for
  for (size_t i = begin; i < end; ++i) f(blocked_range(i, i + 1));
}

#elif defined(CNN_USE_GCD)

template <typename Func>
void parallel_for(size_t begin, size_t end, const Func &f, size_t grainsize) {
  assert(end >= begin);
  size_t count     = end - begin;
  size_t blockSize = grainsize;
  if (count < blockSize || blockSize == 0) {
    blockSize = 1;
  }
  size_t blockCount = (count + blockSize - 1) / blockSize;
  assert(blockCount > 0);

  dispatch_apply(blockCount, dispatch_get_global_queue(QOS_CLASS_DEFAULT, 0),
                 ^(size_t block) {
                   size_t blockStart = block * blockSize;
                   size_t blockEnd   = blockStart + blockSize;
                   if (blockEnd > end) {
                     blockEnd = end;
                   }
                   assert(blockStart < blockEnd);

                   f(blocked_range(blockStart, blockEnd));
                 });
}

#elif defined(CNN_SINGLE_THREAD)

template <typename Func>
void parallel_for(size_t begin,
                  size_t end,
                  const Func &f,
                  size_t /*grainsize*/) {
  xparallel_for(begin, end, f);
}

#else

template <typename Func>
void parallel_for(size_t begin,
                  size_t end,
                  const Func &f,
                  size_t /*grainsize*/) {
  assert(end >= begin);
  size_t nthreads  = std::thread::hardware_concurrency();
  size_t blockSize = (end - begin) / nthreads;
  if (blockSize * nthreads < end - begin) blockSize++;

  std::vector<std::future<void> > futures;

  size_t blockBegin            = begin;
  size_t blockEnd              = blockBegin + blockSize;
  if (blockEnd > end) blockEnd = end;

  for (size_t i = 0; i < nthreads; i++) {
    futures.push_back(
      std::move(std::async(std::launch::async, [blockBegin, blockEnd, &f] {
        f(blocked_range(blockBegin, blockEnd));
      })));

    blockBegin += blockSize;
    blockEnd = blockBegin + blockSize;
    if (blockBegin >= end) break;
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
                 size_t begin,
                 T end,
                 Func f,
                 int grainsize = 100) {
  parallelize = parallelize && value_representation<size_t>(end);
  parallelize ? parallel_for(begin, static_cast<size_t>(end), f, grainsize)
              : xparallel_for(begin, static_cast<size_t>(end), f);
}

template <typename T, typename Func>
inline void for_(std::false_type,
                 bool parallelize,
                 size_t begin,
                 T end,
                 Func f,
                 int grainsize = 100) {
  parallelize ? parallel_for(begin, static_cast<size_t>(end), f, grainsize)
              : xparallel_for(begin, end, f);
}

template <typename T, typename Func>
inline void for_(
  bool parallelize, size_t begin, T end, Func f, size_t grainsize = 100) {
  static_assert(std::is_integral<T>::value, "end must be integral type");
  for_(typename std::is_unsigned<T>::type(), parallelize, begin, end, f,
       grainsize);
}

template <typename T, typename Func>
void for_i(bool parallelize, T size, Func f, size_t grainsize = 100) {
#ifdef CNN_SINGLE_THREAD
  parallelize = false;
#endif
  for_(parallelize, 0, size,
       [&](const blocked_range &r) {
#ifdef CNN_USE_OMP
#pragma omp parallel for
#endif
         for (size_t i = r.begin(); i < r.end(); i++) f(i);
       },
       grainsize);
}

template <typename T, typename Func>
void for_i(T size, Func f, size_t grainsize = 100) {
  for_i(true, size, f, grainsize);
}

}  // namespace tiny_dnn
