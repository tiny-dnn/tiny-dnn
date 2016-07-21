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

// instrumentation.h: contains the definitions needed to
// instrument code for profiling:
//   ScopedProfilingLabel, RegisterCurrentThreadForProfiling.
//
// profiler.h is only needed to drive the profiler:
//   StartProfiling, FinishProfiling.
//
// See the usage example in profiler.h.

#ifndef GEMMLOWP_PROFILING_INSTRUMENTATION_H_
#define GEMMLOWP_PROFILING_INSTRUMENTATION_H_

#include <thread>
#include <cstdio>

#ifndef GEMMLOWP_USE_STLPORT
#include <cstdint>
#else
#include <stdint.h>
namespace std {
using ::uint8_t;
using ::uint16_t;
using ::uint32_t;
using ::int8_t;
using ::int16_t;
using ::int32_t;
using ::size_t;
using ::uintptr_t;
}
#endif

#include <algorithm>
#include <cassert>
#include <cstdlib>

#ifdef GEMMLOWP_PROFILING
#include <cstring>
#include <set>
#endif

// We should always use C++11 thread_local; unfortunately that
// isn't fully supported on Apple yet.
#ifdef __APPLE__
#define GEMMLOWP_THREAD_LOCAL static __thread
#define GEMMLOWP_USING_OLD_THREAD_LOCAL
#else
#define GEMMLOWP_THREAD_LOCAL thread_local
#endif

namespace gemmlowp {

inline void ReleaseBuildAssertion(bool condition, const char* msg) {
  if (!condition) {
    fprintf(stderr, "gemmlowp error: %s\n", msg);
    abort();
  }
}

// To be used as template parameter for GlobalLock.
// GlobalLock<ProfilerLockId> is the profiler global lock:
// registering threads, starting profiling, finishing profiling, and
// the profiler itself as it samples threads, all need to lock it.
struct ProfilerLockId;

// A very plain global lock. Templated in LockId so we can have multiple
// locks, one for each LockId type.
template <typename LockId>
class GlobalLock {
  static pthread_mutex_t* Mutex() {
    static pthread_mutex_t m = PTHREAD_MUTEX_INITIALIZER;
    return &m;
  }

 public:
  static void Lock() { pthread_mutex_lock(Mutex()); }
  static void Unlock() { pthread_mutex_unlock(Mutex()); }
};

// A very simple RAII helper to lock and unlock a GlobalLock
template <typename LockId>
struct AutoGlobalLock {
  AutoGlobalLock() { GlobalLock<LockId>::Lock(); }
  ~AutoGlobalLock() { GlobalLock<LockId>::Unlock(); }
};

// MemoryBarrier is purely a compile-time thing; it tells two things
// to the compiler:
//   1) It prevents reordering code across it
//     (thanks to the 'volatile' after 'asm')
//   2) It requires the compiler to assume that any value previously
//     read from memory, may have changed. Thus it offers an alternative
//     to using 'volatile' variables.
inline void MemoryBarrier() { asm volatile("" ::: "memory"); }

// Profiling definitions. Two paths: when profiling is enabled,
// and when profiling is disabled.
#ifdef GEMMLOWP_PROFILING
// This code path is when profiling is enabled.

// A pseudo-call-stack. Contrary to a real call-stack, this only
// contains pointers to literal strings that were manually entered
// in the instrumented code (see ScopedProfilingLabel).
struct ProfilingStack {
  static const std::size_t kMaxSize = 15;
  typedef const char* LabelsArrayType[kMaxSize];
  LabelsArrayType labels;
  std::size_t size;

  ProfilingStack() { memset(this, 0, sizeof(ProfilingStack)); }

  void Push(const char* label) {
    MemoryBarrier();
    ReleaseBuildAssertion(size < kMaxSize, "ProfilingStack overflow");
    labels[size] = label;
    MemoryBarrier();
    size++;
    MemoryBarrier();
  }

  void Pop() {
    MemoryBarrier();
    ReleaseBuildAssertion(size > 0, "ProfilingStack underflow");
    size--;
    MemoryBarrier();
  }

  void UpdateTop(const char* new_label) {
    MemoryBarrier();
    assert(size);
    labels[size - 1] = new_label;
    MemoryBarrier();
  }

  ProfilingStack& operator=(const ProfilingStack& other) {
    memcpy(this, &other, sizeof(ProfilingStack));
    return *this;
  }

  bool operator==(const ProfilingStack& other) const {
    return !memcmp(this, &other, sizeof(ProfilingStack));
  }
};

static_assert(
    !(sizeof(ProfilingStack) & (sizeof(ProfilingStack) - 1)),
    "ProfilingStack should have power-of-two size to fit in cache lines");

struct ThreadInfo;

// The global set of threads being profiled.
inline std::set<ThreadInfo*>& ThreadsUnderProfiling() {
  static std::set<ThreadInfo*> v;
  return v;
}

struct ThreadInfo {
  pthread_key_t key;  // used only to get a callback at thread exit.
  ProfilingStack stack;

  ThreadInfo() {
    pthread_key_create(&key, ThreadExitCallback);
    pthread_setspecific(key, this);
  }

  static void ThreadExitCallback(void* ptr) {
    AutoGlobalLock<ProfilerLockId> lock;
    ThreadInfo* self = static_cast<ThreadInfo*>(ptr);
    ThreadsUnderProfiling().erase(self);
    pthread_key_delete(self->key);
  }
};

inline ThreadInfo& ThreadLocalThreadInfo() {
#ifdef GEMMLOWP_USING_OLD_THREAD_LOCAL
  // We're leaking this ThreadInfo structure, because Apple doesn't support
  // non-trivial constructors or destructors for their __thread type modifier.
  GEMMLOWP_THREAD_LOCAL ThreadInfo* i = nullptr;
  if (i == nullptr) {
    i = new ThreadInfo();
  }
  return *i;
#else
  GEMMLOWP_THREAD_LOCAL ThreadInfo i;
  return i;
#endif
}

// ScopedProfilingLabel is how one instruments code for profiling
// with this profiler. Construct local ScopedProfilingLabel variables,
// passing a literal string describing the local code. Profile
// samples will then be annotated with this label, while it is in scope
// (whence the name --- also known as RAII).
// See the example in profiler.h.
class ScopedProfilingLabel {
  ProfilingStack* profiling_stack_;

 public:
  explicit ScopedProfilingLabel(const char* label)
      : profiling_stack_(&ThreadLocalThreadInfo().stack) {
    profiling_stack_->Push(label);
  }

  ~ScopedProfilingLabel() { profiling_stack_->Pop(); }

  void Update(const char* new_label) { profiling_stack_->UpdateTop(new_label); }
};

// To be called once on each thread to be profiled.
inline void RegisterCurrentThreadForProfiling() {
  AutoGlobalLock<ProfilerLockId> lock;
  ThreadsUnderProfiling().insert(&ThreadLocalThreadInfo());
}

#else  // not GEMMLOWP_PROFILING
// This code path is when profiling is disabled.

// This empty definition of ScopedProfilingLabel ensures that
// it has zero runtime overhead when profiling is disabled.
struct ScopedProfilingLabel {
  explicit ScopedProfilingLabel(const char*) {}
  void Update(const char*) {}
};

inline void RegisterCurrentThreadForProfiling() {}

#endif

}  // end namespace gemmlowp

#endif  // GEMMLOWP_PROFILING_INSTRUMENTATION_H_
