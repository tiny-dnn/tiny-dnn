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

struct thread_pool {
  thread_pool(size_t nthreads = std::thread::hardware_concurrency())
    : nthreads(nthreads), workers(nthreads) {
    done = -1;
    for (size_t i = 0; i < nthreads; ++i) {
      workers[i].thread = std::thread([&, i] {
        std::pair<int, int> range;
        for (;;) {
          {
            std::unique_lock<std::mutex> lock(mutex);
            condition.wait(lock,
                           [&, i] { return !(done & (1 << i)) || this->stop; });
            if (stop) return;
            range = workers[i].range;
          }
          f(blocked_range(range.first, range.second));
          {
            std::unique_lock<std::mutex> lock(mutex);
            done |= (1 << i);
          }
          // printf("%d %d done\n", begin, end);
          condition2.notify_one();
        }
      });
    }
  }
  ~thread_pool() {
    {
      std::unique_lock<std::mutex> lock(mutex);
      stop = true;
    }
    condition.notify_all();
    for (auto& w : workers) w.thread.join();
    condition2.notify_one();
  }
  inline void run(int begin,
                  int end,
                  std::function<void(const blocked_range& r)> f) {
    if (stop) {
      return;
    }
    {
      std::unique_lock<std::mutex> lock(mutex);
      this->f        = f;
      int total_size = end - begin;
      if (total_size <= nthreads) {
        for (int i = 0; i < total_size; i++) {
          auto& r  = workers[i].range;
          r.first  = begin + i;
          r.second = begin + i + 1;
        }
        done = (-1) << total_size;
      } else {
        int block_size        = (total_size + nthreads - 1) / nthreads;
        int block_begin       = begin;
        size_t nthrads_to_use = total_size / block_size;
        for (size_t i = 0; i < nthrads_to_use; ++i, block_begin += block_size) {
          int block_end = std::min(end, block_begin + block_size);
          auto& r       = workers[i].range;
          r.first       = block_begin;
          r.second      = block_end;
        }
        done = (-1) << nthrads_to_use;
      }
    }
    condition.notify_all();
    {
      std::unique_lock<std::mutex> lock(mutex);
      condition2.wait(lock, [&] { return (done == -1) || stop; });
    }
  }

 private:
  size_t nthreads;
  std::function<void(const blocked_range& r)> f;
  struct thread_info {
    std::thread thread;
    std::pair<int, int> range;
  };
  std::vector<thread_info> workers;
  std::mutex mutex;
  std::condition_variable condition;
  std::condition_variable condition2;
  int64_t done;
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

template <typename Func>
void parallel_for(int begin, int end, const Func& f, int /*grainsize*/) {
  singleton<thread_pool>::get_instance().run(begin, end, f);
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
