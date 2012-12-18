#pragma once
#include <vector>
#include <boost/random.hpp>

namespace nn {

typedef double float_t;
typedef std::vector<double> vec_t;
typedef std::vector<double*> pvec_t;
typedef std::vector<vec_t> mat_t;

template<typename T, typename U>
T& operator << (T& os, const std::vector<U>& vec) {

    for (size_t i = 0; i < vec.size(); i++) 
        os << vec[i] << (i == vec.size() - 1 ? ',' : ' ');
    return os;
}

vec_t operator - (const vec_t& v1, const vec_t& v2) {
    const int dim = v1.size();
    vec_t v(dim);

    for (int i = 0; i < dim; i++)
        v[i] = v1[i] - v2[i];
    return v;
}

vec_t operator + (const vec_t& v1, const vec_t& v2) {
    const int dim = v1.size();
    vec_t v(dim);

    for (int i = 0; i < dim; i++)
        v[i] = v1[i] + v2[i];
    return v;
}

float_t sum_square(const vec_t& vec) {
    float_t sum = 0.0;
    for (auto v : vec)
        sum += v * v;
    return sum;
}

float_t sum_square(const pvec_t& vec) {
    float_t sum = 0.0;
    for (auto v : vec)
        sum += *v * *v;
    return sum;
}

float_t uniform_rand(float_t min, float_t max) {
    static boost::mt19937 gen(0);
    boost::uniform_real<float_t> dst(min, max);
    boost::variate_generator<boost::mt19937&, boost::uniform_real<float_t> > rand(gen, dst);
    return rand();
}

template<typename Iter>
void uniform_rand(Iter begin, Iter end, float_t min, float_t max) {
    for (Iter it = begin; it != end; ++it) 
        *it = uniform_rand(min, max);
}

}
