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
#include <queue>
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

#define STRINGIFY(s) XSTRINGIFY(s)
#define XSTRINGIFY(s) #s
#pragma message("THREAD_POOL_KIND=" STRINGIFY(THREAD_POOL_KIND))

#if THREAD_POOL_KIND == 1
#include <third_party/thread_pool_lib_candidates/beru_thread_pool_4_parallel_for/thread_pool_4_parallel_for.h>
#elif THREAD_POOL_KIND == 2
#include <third_party/thread_pool_lib_candidates/bloomen_cxxpool/src/cxxpool.h>
#elif THREAD_POOL_KIND == 3
#include <third_party/thread_pool_lib_candidates/inkooboo_thread-pool-cpp/include/thread_pool.hpp>
#elif THREAD_POOL_KIND == 4
#include <third_party/thread_pool_lib_candidates/nbsdx_ThreadPool/ThreadPool.h>
#elif THREAD_POOL_KIND == 5
#include <third_party/thread_pool_lib_candidates/progschj_ThreadPool/ThreadPool.h>
#elif THREAD_POOL_KIND == 6
#include <third_party/thread_pool_lib_candidates/vit-vit_CTPL/ctpl_stl.h>
#else
#endif

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

namespace detail {

template <typename T>
class singleton {
 private:
  singleton()  = default;
  ~singleton() = default;

 public:
  singleton(const singleton&) = delete;
  singleton& operator=(const singleton&) = delete;
  singleton(singleton&&)                 = delete;
  singleton& operator=(singleton&&) = delete;
  static T& get_instance() {
    static T inst;
    return inst;
  }
};

}  // namespace detail

template <typename Func>
void parallel_for(int begin, int end, const Func& f, int /*grainsize*/) {

#if THREAD_POOL_KIND == 1
  detail::singleton<
    detail::thread_pool_4_parallel_for<blocked_range> >::get_instance()
    .run(begin, end, f);
#elif THREAD_POOL_KIND == 2
  auto& tp    = detail::singleton<cxxpool::thread_pool>::get_instance();
  size_t diff = std::thread::hardware_concurrency() - tp.n_threads();
  if (diff) {
    tp.add_threads(diff);
  }
  size_t nthreads = tp.n_threads();
  static std::vector<std::future<void> > futures;
  futures.clear();
  int total_size     = end - begin;
  int block_size     = std::max<int>(1, (total_size + nthreads - 1) / nthreads);
  int nthrads_to_use = (total_size + block_size - 1) / block_size;
  int block_begin    = begin;
  for (int i = 0; i < nthrads_to_use; ++i, block_begin += block_size) {
    int block_end = std::min(end, block_begin + block_size);
    futures.emplace_back(tp.push(f, blocked_range(block_begin, block_end)));
  }
  for (auto& f : futures) {
    f.wait();
  }
#elif THREAD_POOL_KIND == 3
  auto& pool      = detail::singleton<tp::ThreadPool<> >::get_instance();
  size_t nthreads = std::thread::hardware_concurrency();
  static std::vector<std::future<void> > futures;
  futures.clear();
  int total_size     = end - begin;
  int block_size     = std::max<int>(1, (total_size + nthreads - 1) / nthreads);
  int nthrads_to_use = (total_size + block_size - 1) / block_size;
  int block_begin    = begin;
  for (int i = 0; i < nthrads_to_use; ++i, block_begin += block_size) {
    int block_end = std::min(end, block_begin + block_size);
    std::packaged_task<void()> t([&, block_begin, block_end]() {
      f(blocked_range(block_begin, block_end));
    });
    futures.emplace_back(t.get_future());
    pool.post(t);
  }
  for (auto& f : futures) {
    f.wait();
  }
#elif THREAD_POOL_KIND == 4
  nbsdx::concurrent::ThreadPool<> tp;
  size_t nthreads    = tp.Size();
  int total_size     = end - begin;
  int block_size     = std::max<int>(1, (total_size + nthreads - 1) / nthreads);
  int nthrads_to_use = (total_size + block_size - 1) / block_size;
  int block_begin    = begin;
  // printf("before\n");
  for (int i = 0; i < nthrads_to_use; ++i, block_begin += block_size) {
    int block_end = std::min(end, block_begin + block_size);
    tp.AddJob([&, block_begin, block_end]() {
      f(blocked_range(block_begin, block_end));
      // printf("%d %d\n", block_begin, block_end);
    });
  }
  tp.WaitAll();
// printf("after\n");
#elif THREAD_POOL_KIND == 5
  static int nthreads = std::thread::hardware_concurrency();
  static ThreadPool pool(nthreads);
  std::vector<std::future<void> > futures;
  int total_size     = end - begin;
  int block_size     = std::max<int>(1, (total_size + nthreads - 1) / nthreads);
  int nthrads_to_use = (total_size + block_size - 1) / block_size;
  int block_begin    = begin;
  for (int i = 0; i < nthrads_to_use; ++i, block_begin += block_size) {
    int block_end = std::min(end, block_begin + block_size);
    futures.emplace_back(pool.enqueue([block_begin, block_end, &f] {
      f(blocked_range(block_begin, block_end));
    }));
  }
  for (auto& future : futures) future.wait();
#elif THREAD_POOL_KIND == 6
  static int nthreads = std::thread::hardware_concurrency();
  static ctpl::thread_pool pool(nthreads);
  std::vector<std::future<void> > futures;
  int total_size     = end - begin;
  int block_size     = std::max<int>(1, (total_size + nthreads - 1) / nthreads);
  int nthrads_to_use = (total_size + block_size - 1) / block_size;
  int block_begin    = begin;
  for (int i = 0; i < nthrads_to_use; ++i, block_begin += block_size) {
    int block_end = std::min(end, block_begin + block_size);
    futures.emplace_back(pool.push([&, block_begin, block_end](int id) {
      f(blocked_range(block_begin, block_end));
    }));
  }
  for (auto& future : futures) future.wait();
#endif
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
