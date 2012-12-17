#pragma once
#include <vector>

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

}
