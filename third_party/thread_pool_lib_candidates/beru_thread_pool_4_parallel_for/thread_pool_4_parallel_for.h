#pragma once

#include <cassert>
#include <limits>
#include <queue>
#include <string>
#include <type_traits>
#include <vector>
#include <future>
#include <thread>

namespace tiny_dnn {

namespace detail {

template <typename blocked_range>
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
  
} // namespace detail
} // namespace tiny_dnn

