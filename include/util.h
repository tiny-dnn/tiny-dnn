/*
    Copyright (c) 2013, Taiga Nomi
    All rights reserved.
    
    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in the
    documentation and/or other materials provided with the distribution.
    * Neither the name of the <organization> nor the
    names of its contributors may be used to endorse or promote products
    derived from this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY 
    EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED 
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE 
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY 
    DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES 
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; 
    LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND 
    ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT 
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS 
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
#pragma once
#include <vector>
#include <functional>
#include <random>
#include <type_traits>
#include <limits>

#ifdef CNN_USE_TBB
#ifndef NOMINMAX
#define NOMINMAX // tbb includes windows.h in tbb/machine/windows_api.h
#endif
#include <tbb/tbb.h>
#include <tbb/task_group.h>
#endif
#include "fixed_point.h"

#define CNN_UNREFERENCED_PARAMETER(x) (void)(x)

namespace tiny_cnn {

typedef double float_t;
typedef unsigned short layer_size_t;
typedef size_t label_t;
typedef std::vector<float_t> vec_t;

class nn_error : public std::exception {
public:
    explicit nn_error(const std::string& msg) : msg_(msg) {}
    const char* what() const throw() override { return msg_.c_str(); }
private:
    std::string msg_;
};

template <typename T>
struct index3d {
    index3d(T width, T height, T depth) : width_(width), height_(height), depth_(depth) {}

    T get_index(T x, T y, T channel) const {
        return (height_ * channel + y) * width_ + x;
    }

    T size() const {
        return width_ * height_ * depth_;
    }

    T width_;
    T height_;
    T depth_;
};

template<int Q>
inline fixed_point<Q> uniform_rand(fixed_point<Q> min, fixed_point<Q> max) {
    // avoid gen(0) for MSVC known issue
    // https://connect.microsoft.com/VisualStudio/feedback/details/776456
    static std::mt19937 gen(1);
    std::uniform_real_distribution<double> dst(min.to_real(), max.to_real());
    return dst(gen);
}

template<typename T> inline
typename std::enable_if<std::is_integral<T>::value, T>::type
uniform_rand(T min, T max) {
    static std::mt19937 gen(1);
    std::uniform_int_distribution<T> dst(min, max);
    return dst(gen);
}

template<typename T> inline
typename std::enable_if<std::is_floating_point<T>::value, T>::type
uniform_rand(T min, T max) {
    static std::mt19937 gen(1);
    std::uniform_real_distribution<T> dst(min, max);
    return dst(gen);
}

inline bool bernoulli(double p) {
    return uniform_rand(0.0, 1.0) <= p;
}

template<typename Iter>
void uniform_rand(Iter begin, Iter end, float_t min, float_t max) {
    for (Iter it = begin; it != end; ++it) 
        *it = uniform_rand(min, max);
}

template<typename T>
T* reverse_endian(T* p) {
    std::reverse(reinterpret_cast<char*>(p), reinterpret_cast<char*>(p) + sizeof(T));
    return p;
}

template<typename T>
int max_index(const T& vec) {
    typename T::value_type max_val = std::numeric_limits<typename T::value_type>::lowest();
    int max_index = -1;

    for (size_t i = 0; i < vec.size(); i++) {
        if (vec[i] > max_val) {
            max_index = i;
            max_val = vec[i];
        }
    }
    return max_index;
}

template<typename T, typename U>
U rescale(T x, T src_min, T src_max, U dst_min, U dst_max) {
    U value =  static_cast<U>(((x - src_min) * (dst_max - dst_min)) / (src_max - src_min) + dst_min);
    return std::min(dst_max, std::max(value, dst_min));
}

inline void nop() {
    // do nothing
}

#ifdef CNN_USE_TBB

typedef tbb::blocked_range<int> blocked_range;
typedef tbb::task_group task_group;

template<typename Func>
void parallel_for(int begin, int end, const Func& f) {
    tbb::parallel_for(blocked_range(begin, end, 100), f);
}
template<typename Func>
void xparallel_for(int begin, int end, const Func& f) {
    f(blocked_range(begin, end, 100));
}

template<typename Func>
void for_(bool parallelize, int begin, int end, Func f) {
    parallelize ? parallel_for(begin, end, f) : xparallel_for(begin, end, f);
}

#else

struct blocked_range {
    typedef int const_iterator;

    blocked_range(int begin, int end) : begin_(begin), end_(end) {}

    const_iterator begin() const { return begin_; }
    const_iterator end() const { return end_; }
private:
    int begin_;
    int end_;
};

template<typename Func>
void xparallel_for(size_t begin, size_t end, const Func& f) {
    blocked_range r(begin, end);
    f(r);
}

#ifdef CNN_USE_OMP

template<typename Func>
void parallel_for(int begin, int end, const Func& f) {
    #pragma omp parallel for
    for (int i=begin; i<end; ++i)
        f(blocked_range(i,i+1));
}

template<typename T, typename U>
bool const value_representation(U const &value)
{
    return static_cast<U>(static_cast<T>(value)) == value;
}

template<typename Func>
void for_(bool parallelize, size_t begin, size_t end, Func f) {
    parallelize = parallelize && value_representation<int>(begin);
    parallelize = parallelize && value_representation<int>(end);
    parallelize? parallel_for(static_cast<int>(begin), static_cast<int>(end), f) : xparallel_for(begin, end, f);
}

#else

template<typename Func>
void for_(bool /*parallelize*/, size_t begin, size_t end, Func f) { // ignore parallelize if you don't define CNN_USE_TBB
    xparallel_for(begin, end, f);
}

#endif

class task_group {
public:
    template<typename Func>
    void run(Func f) {
        functions_.push_back(f);
    }

    void wait() {
        for (auto f : functions_)
            f();
    }
private:
    std::vector<std::function<void()>> functions_;
};

#endif // CNN_USE_TBB

template <typename Func>
void for_i(int size, Func f)
{
    for_(true, 0, size, [&](const blocked_range& r) {
#ifdef CNN_USE_OMP
#pragma omp parallel for
#endif
        for (int i = r.begin(); i < r.end(); i++)
            f(i);
    });
}

template <typename T> inline T sqr(T value) { return value*value; }

} // namespace tiny_cnn
