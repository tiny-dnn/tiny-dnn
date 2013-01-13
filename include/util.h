#pragma once
#include <vector>
#include <limits>
#include <boost/random.hpp>
#ifdef CNN_USE_TBB
#define NOMINMAX // tbb includes windows.h in tbb/machine/windows_api.h
#include <tbb/tbb.h>
#endif
#include "fixed_point.h"

namespace tiny_cnn {

typedef double float_t;
typedef int label_t;
typedef std::vector<float_t> vec_t;

class nn_error : public std::exception {
public:
    nn_error(const std::string& msg) : msg_(msg){}
    ~nn_error() throw() {}
    const char* what() const throw() { return msg_.c_str(); }
private:
    std::string msg_;
};

struct tensor3d {
    tensor3d(int width, int height, int depth) : width_(width), height_(height), depth_(depth) {
        if (width <= 0 || height <= 0 || depth <= 0)
            throw nn_error("invalid tensor size");
    }

    int get_index(int x, int y, int channel) const {
        return (width_ * height_) * channel + width_ * y + x;
    }

    int size() const {
        return width_ * height_ * depth_;
    }

    int width_;
    int height_;
    int depth_;
};

template<int Q>
inline fixed_point<Q> uniform_rand(fixed_point<Q> min, fixed_point<Q> max) {
    static boost::mt19937 gen(0);
    boost::uniform_real<double> dst(min.to_real(), max.to_real());
    return dst(gen);
}

template<typename T>
inline double uniform_rand(T min, T max) {
    static boost::mt19937 gen(0);
    boost::uniform_real<T> dst(min, max);
    return dst(gen);
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
int max_index(const std::vector<T>& vec) {
    T max_val = -1;
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

template<typename Func>
void parallel_for(int begin, int end, Func f) {
    tbb::parallel_for(tbb::blocked_range<int>(begin, end, 100), f);
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
void parallel_for(int begin, int end, Func f) {
    blocked_range r(begin, end);
    f(r);
}

#endif // CNN_USE_TBB


}
