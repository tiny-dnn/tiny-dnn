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
#include <thread>
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

#elif defined(CNN_SINGLE_THREAD)

template <typename Func>
void parallel_for(int begin, int end, const Func &f, int /*grainsize*/) {
  xparallel_for(static_cast<size_t>(begin), static_cast<size_t>(end), f);
}

#else

namespace detail {

struct thread_pool_4_parallel_for {
  thread_pool_4_parallel_for(
    size_t nthreads = std::thread::hardware_concurrency())
    : nthreads(nthreads), stop(false) {
    packaged_tasks.resize(nthreads);
    futures.reserve(nthreads);
    for (size_t i = 0; i < nthreads; ++i) {
      workers.emplace_back([this] {
        std::function<void()> task;
        for (;;) {
          {
            std::unique_lock<std::mutex> lock(mutex);
            condition.wait(lock,
                           [this] { return !tasks.empty() || this->stop; });
            if (stop) return;
            task = std::move(tasks.front());
            tasks.pop();
          }
          task();
        }
      });
    }
  }
  ~thread_pool_4_parallel_for() {
    {
      std::unique_lock<std::mutex> lock(mutex);
      stop = true;
    }
    condition.notify_all();
    for (auto& w : workers) w.join();
  }
  inline void run(int begin,
                  int end,
                  const std::function<void(const blocked_range& r)>& f) {
    if (stop || end <= begin) {
      return;
    }
    futures.clear();
    int total_size = end - begin;
    int block_size = std::max<int>(1, (total_size + nthreads - 1) / nthreads);
    int nthrads_to_use = (total_size + block_size - 1) / block_size;
    int block_begin    = begin;
    for (int i = 0; i < nthrads_to_use; ++i, block_begin += block_size) {
      packaged_tasks[i] = std::packaged_task<void(const blocked_range& r)>(f);
      auto& pt          = packaged_tasks[i];
      futures.emplace_back(pt.get_future());
      {
        int block_end = std::min(end, block_begin + block_size);
        std::unique_lock<std::mutex> lock(mutex);
        tasks.emplace([&pt, block_begin, block_end]() {
          pt(blocked_range(block_begin, block_end));
        });
      }
      condition.notify_one();
    }
    for (auto& f : futures) {
      f.wait();
    }
  }

 private:
  size_t nthreads;
  std::vector<std::thread> workers;
  std::vector<std::packaged_task<void(const blocked_range& r)> > packaged_tasks;
  std::vector<std::future<void> > futures;
  std::queue<std::function<void()> > tasks;
  std::mutex mutex;
  std::condition_variable condition;
  bool stop;
};

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
  detail::singleton<detail::thread_pool_4_parallel_for>::get_instance().run(
    begin, end, f);
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
