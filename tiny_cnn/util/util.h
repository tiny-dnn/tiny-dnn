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
#include <cassert>
#include <cstdio>
#include <cstdarg>
#include <string>
#include "aligned_allocator.h"
#include "nn_error.h"
#include "tiny_cnn/config.h"

#ifdef CNN_USE_TBB
#ifndef NOMINMAX
#define NOMINMAX // tbb includes windows.h in tbb/machine/windows_api.h
#endif
#include <tbb/tbb.h>
#include <tbb/task_group.h>
#endif

#ifndef CNN_USE_OMP
#include <thread>
#include <future>
#endif

#define CNN_UNREFERENCED_PARAMETER(x) (void)(x)

namespace tiny_cnn {

///< output label(class-index) for classification
///< must be equal to cnn_size_t, because size of last layer is equal to num. of classes
typedef cnn_size_t label_t;

typedef cnn_size_t layer_size_t; // for backward compatibility

typedef std::vector<float_t, aligned_allocator<float_t, 64>> vec_t;

enum class net_phase {
    train,
    test
};

template<typename T> inline
typename std::enable_if<std::is_integral<T>::value, T>::type
uniform_rand(T min, T max) {
    // avoid gen(0) for MSVC known issue
    // https://connect.microsoft.com/VisualStudio/feedback/details/776456
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

template<typename T> inline
typename std::enable_if<std::is_floating_point<T>::value, T>::type
gaussian_rand(T mean, T sigma) {
    static std::mt19937 gen(1);
    std::normal_distribution<T> dst(mean, sigma);
    return dst(gen);
}

template<typename Container>
inline int uniform_idx(const Container& t) {
    return uniform_rand(0, int(t.size() - 1));
}

inline bool bernoulli(float_t p) {
    return uniform_rand(float_t(0), float_t(1)) <= p;
}

template<typename Iter>
void uniform_rand(Iter begin, Iter end, float_t min, float_t max) {
    for (Iter it = begin; it != end; ++it) 
        *it = uniform_rand(min, max);
}

template<typename Iter>
void gaussian_rand(Iter begin, Iter end, float_t mean, float_t sigma) {
    for (Iter it = begin; it != end; ++it)
        *it = gaussian_rand(mean, sigma);
}

template<typename T>
T* reverse_endian(T* p) {
    std::reverse(reinterpret_cast<char*>(p), reinterpret_cast<char*>(p) + sizeof(T));
    return p;
}

inline bool is_little_endian() {
    int x = 1;
    return *(char*) &x != 0;
}


template<typename T>
size_t max_index(const T& vec) {
    auto begin_iterator = std::begin(vec);
    return std::max_element(begin_iterator, std::end(vec)) - begin_iterator;
}

template<typename T, typename U>
U rescale(T x, T src_min, T src_max, U dst_min, U dst_max) {
    U value =  static_cast<U>(((x - src_min) * (dst_max - dst_min)) / (src_max - src_min) + dst_min);
    return std::min(dst_max, std::max(value, dst_min));
}

inline void nop() 
{
    // do nothing
}


#ifdef CNN_USE_TBB

static tbb::task_scheduler_init tbbScheduler(tbb::task_scheduler_init::automatic);//tbb::task_scheduler_init::deferred);

typedef tbb::blocked_range<int> blocked_range;

template<typename Func>
void parallel_for(int begin, int end, const Func& f, int grainsize) {
    tbb::parallel_for(blocked_range(begin, end, end - begin > grainsize ? grainsize : 1), f);
}
template<typename Func>
void xparallel_for(int begin, int end, const Func& f) {
    f(blocked_range(begin, end, 100));
}

#else

struct blocked_range {
    typedef int const_iterator;

    blocked_range(int begin, int end) : begin_(begin), end_(end) {}
    blocked_range(size_t begin, size_t end) : begin_(static_cast<int>(begin)), end_(static_cast<int>(end)) {}

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
void parallel_for(int begin, int end, const Func& f, int /*grainsize*/) {
    #pragma omp parallel for
    for (int i=begin; i<end; ++i)
        f(blocked_range(i,i+1));
}

#else

template<typename Func>
void parallel_for(int start, int end, const Func &f, int /*grainsize*/) {
    int nthreads = std::thread::hardware_concurrency();
    int blockSize = (end - start) / nthreads;
    if (blockSize*nthreads < end - start)
        blockSize++;

    std::vector<std::future<void>> futures;

    int blockStart = start;
    int blockEnd = blockStart + blockSize;
    if (blockEnd > end) blockEnd = end;

    for (int i = 0; i < nthreads; i++) {
        futures.push_back(std::move(std::async(std::launch::async, [blockStart, blockEnd, &f] {
            f(blocked_range(blockStart, blockEnd));
        })));

        blockStart += blockSize;
        blockEnd = blockStart + blockSize;
        if (blockStart >= end) break;
        if (blockEnd > end) blockEnd = end;
    }

    for (auto &future : futures)
        future.wait();
}

#endif

#endif // CNN_USE_TBB

template<typename T, typename U>
bool value_representation(U const &value) {
    return static_cast<U>(static_cast<T>(value)) == value;
}

template<typename T, typename Func>
inline
void for_(std::true_type, bool parallelize, int begin, T end, Func f, int grainsize = 100){
    parallelize = parallelize && value_representation<int>(end);
    parallelize ? parallel_for(begin, static_cast<int>(end), f, grainsize) :
                  xparallel_for(begin, static_cast<int>(end), f);
}

template<typename T, typename Func>
inline
void for_(std::false_type, bool parallelize, int begin, T end, Func f, int grainsize = 100){
    parallelize ? parallel_for(begin, static_cast<int>(end), f, grainsize) : xparallel_for(begin, end, f);
}

template<typename T, typename Func>
inline
void for_(bool parallelize, int begin, T end, Func f, int grainsize = 100) {
    static_assert(std::is_integral<T>::value, "end must be integral type");
    for_(typename std::is_unsigned<T>::type(), parallelize, begin, end, f, grainsize);
}

template <typename T, typename Func>
void for_i(bool parallelize, T size, Func f, int grainsize = 100)
{
    for_(parallelize, 0, size, [&](const blocked_range& r) {
#ifdef CNN_USE_OMP
#pragma omp parallel for
#endif
        for (int i = r.begin(); i < r.end(); i++)
            f(i);
    }, grainsize);
}

template <typename T, typename Func>
void for_i(T size, Func f, int grainsize = 100) {
    for_i(true, size, f, grainsize);
}

template <typename T> inline T sqr(T value) { return value*value; }

inline bool isfinite(float_t x) {
    return x == x;
}

template <typename Container> inline bool has_infinite(const Container& c) {
    for (auto v : c)
        if (!isfinite(v)) return true;
    return false;
}

template <typename Container>
size_t max_size(const Container& c) {
    typedef typename Container::value_type value_t;
    return std::max_element(c.begin(), c.end(),
        [](const value_t& left, const value_t& right) { return left.size() < right.size(); })->size();
}

inline std::string format_str(const char *fmt, ...) {
    static char buf[2048];

#ifdef _MSC_VER
#pragma warning(disable:4996)
#endif
    va_list args;
    va_start(args, fmt);
    vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);
#ifdef _MSC_VER
#pragma warning(default:4996)
#endif
    return std::string(buf);
}

template <typename T>
struct index3d {
    index3d(T width, T height, T depth) {
        reshape(width, height, depth);
    }

    index3d() : width_(0), height_(0), depth_(0) {}

    void reshape(T width, T height, T depth) {
        width_ = width;
        height_ = height;
        depth_ = depth;

        if ((long long) width * height * depth > std::numeric_limits<T>::max())
            throw nn_error(
            format_str("error while constructing layer: layer size too large for tiny-cnn\nWidthxHeightxChannels=%dx%dx%d >= max size of [%s](=%d)",
            width, height, depth, typeid(T).name(), std::numeric_limits<T>::max()));
    }

    T get_index(T x, T y, T channel) const {
        assert(x >= 0 && x < width_);
        assert(y >= 0 && y < height_);
        assert(channel >= 0 && channel < depth_);
        return (height_ * channel + y) * width_ + x; 
    }

    T area() const {
        return width_ * height_;
    }

    T size() const {
        return width_ * height_ * depth_;
    }

    T width_;
    T height_;
    T depth_;
};

template <typename T>
bool operator == (const index3d<T>& lhs, const index3d<T>& rhs) {
    return (lhs.width_ == rhs.width_) && (lhs.height_ == rhs.height_) && (lhs.depth_ == rhs.depth_);
}

template <typename T>
bool operator != (const index3d<T>& lhs, const index3d<T>& rhs) {
    return !(lhs == rhs);
}

typedef index3d<cnn_size_t> layer_shape_t;

template <typename Stream, typename T>
Stream& operator << (Stream& s, const index3d<T>& d) {
    s << d.width_ << "x" << d.height_ << "x" << d.depth_;
    return s;
}


// boilerplate to resolve dependent name
#define CNN_USE_LAYER_MEMBERS using layer_base::in_size_;\
    using layer_base::out_size_; \
    using layer_base::parallelize_; \
    using layer_base::next_; \
    using layer_base::prev_; \
    using layer_base::a_; \
    using layer_base::output_; \
    using layer_base::prev_delta_; \
    using layer_base::W_; \
    using layer_base::b_; \
    using layer_base::dW_; \
    using layer_base::db_; \
    using layer_base::Whessian_; \
    using layer_base::bhessian_; \
    using layer_base::prev_delta2_; \
    using layer<Activation>::h_


#define CNN_LOG_VECTOR(vec, name)
/*
void CNN_LOG_VECTOR(const vec_t& vec, const std::string& name) {
    std::cout << name << ",";

    if (vec.empty()) {
        std::cout << "(empty)" << std::endl;
    }
    else {
        for (size_t i = 0; i < vec.size(); i++) {
            std::cout << vec[i] << ",";
        }
    }

    std::cout << std::endl;
}
*/

} // namespace tiny_cnn

#if defined(_MSC_VER) && (_MSC_VER <= 1800)
#define CNN_DEFAULT_MOVE_CONSTRUCTOR_UNAVAILABLE
#define CNN_DEFAULT_ASSIGNMENT_OPERATOR_UNAVAILABLE
#endif
