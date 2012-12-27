#pragma once
#include <vector>
#include <limits>
#include <boost/random.hpp>

namespace nn {

typedef double float_t;
typedef int label_t;
typedef std::vector<double> vec_t;
typedef std::vector<double*> pvec_t;
typedef std::vector<vec_t> mat_t;

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

template<typename T, typename U>
T& operator << (T& os, const std::vector<U>& vec) {

    for (size_t i = 0; i < vec.size(); i++) 
        os << vec[i] << (i == vec.size() - 1 ? ',' : ' ');
    return os;
}

inline vec_t operator - (const vec_t& v1, const vec_t& v2) {
    const int dim = v1.size();
    vec_t v(dim);

    for (int i = 0; i < dim; i++)
        v[i] = v1[i] - v2[i];
    return v;
}

inline vec_t operator + (const vec_t& v1, const vec_t& v2) {
    const int dim = v1.size();
    vec_t v(dim);

    for (int i = 0; i < dim; i++)
        v[i] = v1[i] + v2[i];
    return v;
}

inline float_t uniform_rand(float_t min, float_t max) {
    static boost::mt19937 gen(0);
    boost::uniform_real<float_t> dst(min, max);
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

}
