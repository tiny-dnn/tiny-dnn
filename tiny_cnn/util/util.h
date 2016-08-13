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
#include <sstream>
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
typedef std::vector<vec_t> tensor_t;

enum class net_phase {
    train,
    test
};

class random_generator {
public:
    static random_generator& get_instance() {
        static random_generator instance;
        return instance;
    }

    std::mt19937& operator()() {
        return gen_;
    }

    void set_seed(unsigned int seed) {
        gen_.seed(seed);
    }
private:
    // avoid gen_(0) for MSVC known issue
    // https://connect.microsoft.com/VisualStudio/feedback/details/776456
    random_generator() : gen_(1) {}
    std::mt19937 gen_;
};

template<typename T> inline
typename std::enable_if<std::is_integral<T>::value, T>::type
uniform_rand(T min, T max) {
    std::uniform_int_distribution<T> dst(min, max);
    return dst(random_generator::get_instance()());
}

template<typename T> inline
typename std::enable_if<std::is_floating_point<T>::value, T>::type
uniform_rand(T min, T max) {
    std::uniform_real_distribution<T> dst(min, max);
    return dst(random_generator::get_instance()());
}

template<typename T> inline
typename std::enable_if<std::is_floating_point<T>::value, T>::type
gaussian_rand(T mean, T sigma) {
    std::normal_distribution<T> dst(mean, sigma);
    return dst(random_generator::get_instance()());
}

inline void set_random_seed(unsigned int seed) {
    random_generator::get_instance().set_seed(seed);
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

#if defined(CNN_USE_OMP)

template<typename Func>
void parallel_for(int begin, int end, const Func& f, int /*grainsize*/) {
    #pragma omp parallel for
    for (int i=begin; i<end; ++i)
        f(blocked_range(i,i+1));
}

#elif defined(CNN_SINGLE_THREAD)

template<typename Func>
void parallel_for(int begin, int end, const Func& f, int /*grainsize*/) {
    xparallel_for(static_cast<size_t>(begin), static_cast<size_t>(end), f);
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

typedef index3d<cnn_size_t> shape3d;

template <typename T>
bool operator == (const index3d<T>& lhs, const index3d<T>& rhs) {
    return (lhs.width_ == rhs.width_) && (lhs.height_ == rhs.height_) && (lhs.depth_ == rhs.depth_);
}

template <typename T>
bool operator != (const index3d<T>& lhs, const index3d<T>& rhs) {
    return !(lhs == rhs);
}

template <typename Stream, typename T>
Stream& operator << (Stream& s, const index3d<T>& d) {
    s << d.width_ << "x" << d.height_ << "x" << d.depth_;
    return s;
}

template <typename Stream, typename T>
Stream& operator << (Stream& s, const std::vector<index3d<T>>& d) {
    s << "[";
    for (cnn_size_t i = 0; i < d.size(); i++) {
        if (i) s << ",";
        s << "[" << d[i] << "]";
    }
    s << "]";
    return s;
}

// equivalent to std::to_string, which android NDK doesn't support
template <typename T>
std::string to_string(T value) {
    std::ostringstream os;
    os << value;
    return os.str();
}

// boilerplate to resolve dependent name
#define CNN_USE_LAYER_MEMBERS using layer::parallelize_; \
    using feedforward_layer<Activation>::h_


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


template <typename T, typename Pred, typename Sum>
cnn_size_t sumif(const std::vector<T>& vec, Pred p, Sum s) {
    size_t sum = 0;
    for (size_t i = 0; i < vec.size(); i++) {
        if (p(i)) sum += s(vec[i]);
    }
    return sum;
}

template <typename T, typename Pred>
std::vector<T> filter(const std::vector<T>& vec, Pred p) {
    std::vector<T> res;
    for (size_t i = 0; i < vec.size(); i++) {
        if (p(i)) res.push_back(vec[i]);
    }
    return res;
}

template <typename Result, typename T, typename Pred>
std::vector<Result> map_(const std::vector<T>& vec, Pred p) {
    std::vector<Result> res;
    for (auto& v : vec) {
        res.push_back(p(v));
    }
    return res;
}

enum class vector_type : int32_t {
    // 0x0001XXX : in/out data
    data = 0x0001000, // input/output data, fed by other layer or input channel

    // 0x0002XXX : trainable parameters, updated for each back propagation
    weight = 0x0002000,
    bias = 0x0002001,

    label = 0x0004000,
    aux = 0x0010000 // layer-specific storage
};

inline std::string to_string(vector_type vtype) {
    switch (vtype)
    {
    case tiny_cnn::vector_type::data:
        return "data";
    case tiny_cnn::vector_type::weight:
        return "weight";
    case tiny_cnn::vector_type::bias:
        return "bias";
    case tiny_cnn::vector_type::label:
        return "label";
    case tiny_cnn::vector_type::aux:
        return "aux";
    default:
        return "unknown";
    }
}

inline std::ostream& operator << (std::ostream& os, vector_type vtype) {
    os << to_string(vtype);
    return os;
}

inline vector_type operator & (vector_type lhs, vector_type rhs) {
    return (vector_type)(static_cast<int32_t>(lhs) & static_cast<int32_t>(rhs));
}

inline bool is_trainable_weight(vector_type vtype) {
    return (vtype & vector_type::weight) == vector_type::weight;
}

inline std::vector<vector_type> std_input_order(bool has_bias) {
    if (has_bias) {
        return{ vector_type::data, vector_type::weight, vector_type::bias };
    }
    else {
        return{ vector_type::data, vector_type::weight };
    }
}

inline std::vector<vector_type> std_output_order(bool has_activation) {
    if (has_activation) {
        return{ vector_type::data, vector_type::aux };
    }
    else {
        return{ vector_type::data };
    }
}


} // namespace tiny_cnn

#if defined(_MSC_VER) && (_MSC_VER <= 1800)
#define CNN_DEFAULT_MOVE_CONSTRUCTOR_UNAVAILABLE
#define CNN_DEFAULT_ASSIGNMENT_OPERATOR_UNAVAILABLE
#endif
