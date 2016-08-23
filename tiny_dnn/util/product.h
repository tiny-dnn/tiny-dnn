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
#if defined(CNN_USE_SSE) || defined(CNN_USE_AVX)
#include <immintrin.h>
#endif
#include <cstdint>
#include <cassert>
#include <numeric>

#if defined(_MSC_VER)
#define VECTORIZE_ALIGN(x) __declspec(align(x))
#elif defined(__GNUC__)
#define VECTORIZE_ALIGN(x) __attribute__((aligned(x)))
#else
#define VECTORIZE_ALIGN(x) __attribute__((aligned(x)))
#endif

namespace vectorize {
namespace detail {


template<typename T>
inline bool is_aligned(T, const typename T::value_type* /*p*/) {
    return true;
}

template<typename T>
inline bool is_aligned(T, const typename T::value_type* p1, const typename T::value_type* p2) {
    return is_aligned(T(), p1) && is_aligned(T(), p2);
}

// traits

template <typename T>
struct generic_vec_type {
    typedef T register_type;
    typedef T value_type;
    enum {
        unroll_size = 1
    };
    static register_type set1(const value_type& x) { return x; }
    static register_type zero() { return register_type(0); }
    static register_type mul(const register_type& v1, const register_type& v2) { return v1 * v2; }
    static register_type add(const register_type& v1, const register_type& v2) { return v1 + v2; }
    static register_type load(const value_type* px) { return *px; }
    static register_type loadu(const value_type* px) { return *px; }
    static void store(value_type* px, const register_type& v) { *px = v; }
    static void storeu(value_type* px, const register_type& v) { *px = v; }
    static value_type resemble(const register_type& x) { return x; }
};

#ifdef CNN_USE_SSE

struct float_sse {
    typedef __m128 register_type;
    typedef float value_type;
    enum {
        unroll_size = 4
    };
    static register_type set1(const value_type& x) { return _mm_set1_ps(x); }
    static register_type zero() { register_type v = {}; return v; }
    static register_type mul(const register_type& v1, const register_type& v2) { return _mm_mul_ps(v1, v2); }
    static register_type add(const register_type& v1, const register_type& v2) { return _mm_add_ps(v1, v2); }
    static register_type load(const value_type* px) { return _mm_load_ps(px); }
    static register_type loadu(const value_type* px) { return _mm_loadu_ps(px); }
    static void store(value_type* px, const register_type& v) { _mm_store_ps(px, v); }
    static void storeu(value_type* px, const register_type& v) { _mm_storeu_ps(px, v); }
    static value_type resemble(const register_type& x) {
        VECTORIZE_ALIGN(16) float tmp[4];
        _mm_store_ps(tmp, x);
        return tmp[0] + tmp[1] + tmp[2] + tmp[3];
    }
};

struct double_sse {
    typedef __m128d register_type;
    typedef double value_type;
    enum {
        unroll_size = 2
    };
    static register_type set1(const value_type& x) { return _mm_set1_pd(x); }
    static register_type zero() { register_type v = {}; return v; }
    static register_type mul(const register_type& v1, const register_type& v2) { return _mm_mul_pd(v1, v2); }
    static register_type add(const register_type& v1, const register_type& v2) { return _mm_add_pd(v1, v2); }
    static register_type load(const value_type* px) { return _mm_load_pd(px); }
    static register_type loadu(const value_type* px) { return _mm_loadu_pd(px); }
    static void store(value_type* px, const register_type& v) { _mm_store_pd(px, v); }
    static void storeu(value_type* px, const register_type& v) { _mm_storeu_pd(px, v); }
    static value_type resemble(const register_type& x) {
        VECTORIZE_ALIGN(16) double tmp[2];
        _mm_store_pd(tmp, x);
        return tmp[0] + tmp[1];
    }
};

template<typename T>
struct sse {};
template<>
struct sse<float> : public float_sse {};
template<>
struct sse<double> : public double_sse {};

template<typename T>
inline bool is_aligned(sse<T>, const typename sse<T>::value_type* p) {
    return reinterpret_cast<std::size_t>(p) % 16 == 0;
}

#endif // CNN_USE_SSE

#ifdef CNN_USE_AVX

struct float_avx {
    typedef __m256 register_type;
    typedef float value_type;
    enum {
        unroll_size = 8
    };
    static register_type set1(const value_type& x) { return _mm256_set1_ps(x); }
    static register_type zero() { register_type v = {}; return v; }
    static register_type mul(const register_type& v1, const register_type& v2) { return _mm256_mul_ps(v1, v2); }
    static register_type add(const register_type& v1, const register_type& v2) { return _mm256_add_ps(v1, v2); }
    static register_type load(const value_type* px) { return _mm256_load_ps(px); }
    static register_type loadu(const value_type* px) { return _mm256_loadu_ps(px); }
    static void store(value_type* px, const register_type& v) { _mm256_store_ps(px, v); }
    static void storeu(value_type* px, const register_type& v) { _mm256_storeu_ps(px, v); }
    static value_type resemble(const register_type& x) { 
        VECTORIZE_ALIGN(32) float tmp[8];
        _mm256_store_ps(tmp, x);
        return std::accumulate(tmp, tmp + 8, 0.0f);
    }
};

struct double_avx {
    typedef __m256d register_type;
    typedef double value_type;
    enum {
        unroll_size = 4
    };
    static register_type set1(const value_type& x) { return _mm256_set1_pd(x); }
    static register_type zero() { register_type v = {}; return v; }
    static register_type mul(const register_type& v1, const register_type& v2) { return _mm256_mul_pd(v1, v2); }
    static register_type add(const register_type& v1, const register_type& v2) { return _mm256_add_pd(v1, v2); }
    static register_type load(const value_type* px) { return _mm256_load_pd(px); }
    static register_type loadu(const value_type* px) { return _mm256_loadu_pd(px); }
    static void store(value_type* px, const register_type& v) { _mm256_store_pd(px, v); }
    static void storeu(value_type* px, const register_type& v) { _mm256_storeu_pd(px, v); }
    static value_type resemble(const register_type& x) {
        VECTORIZE_ALIGN(32) double tmp[4];
        _mm256_store_pd(tmp, x);
        return std::accumulate(tmp, tmp + 4, 0.0);  
    }
};

template<typename T>
struct avx {};
template<>
struct avx<float> : public float_avx {};
template<>
struct avx<double> : public double_avx {};

template<typename T>
inline bool is_aligned(avx<T>, const typename avx<T>::value_type* p) {
    return reinterpret_cast<std::size_t>(p) % 32 == 0;
}

#endif // CNN_USE_AVX

// generic dot-product
template<typename T>
inline typename T::value_type dot_product_nonaligned(const typename T::value_type* f1, const typename T::value_type* f2, std::size_t  size) {
    typename T::register_type result = T::zero();

    for (std::size_t i = 0; i < size/T::unroll_size; i++) 
        result = T::add(result, T::mul(T::loadu(&f1[i*T::unroll_size]), T::loadu(&f2[i*T::unroll_size])));  

    typename T::value_type sum = T::resemble(result);

    for (std::size_t i = (size/T::unroll_size)*T::unroll_size; i < size; i++)
        sum += f1[i] * f2[i];

    return sum;
}

// generic dot-product(aligned)
template<typename T>
inline typename T::value_type dot_product_aligned(const typename T::value_type* f1, const typename T::value_type* f2, std::size_t  size) {
    typename T::register_type result = T::zero();

    assert(is_aligned(T(), f1));
    assert(is_aligned(T(), f2));

    for (std::size_t i = 0; i < size/T::unroll_size; i++) 
        result = T::add(result, T::mul(T::load(&f1[i*T::unroll_size]), T::load(&f2[i*T::unroll_size])));  
    
    typename T::value_type sum = T::resemble(result);

    for (std::size_t i = (size/T::unroll_size)*T::unroll_size; i < size; i++)
        sum += f1[i] * f2[i];

    return sum;
}

template<typename T>
inline void muladd_aligned(const typename T::value_type* src, typename T::value_type c, std::size_t  size, typename T::value_type* dst) {
    typename T::register_type factor = T::set1(c);

    for (std::size_t i = 0; i < size/T::unroll_size; i++) {
        typename T::register_type d = T::load(&dst[i*T::unroll_size]);
        typename T::register_type s = T::load(&src[i*T::unroll_size]);
        T::store(&dst[i*T::unroll_size], T::add(d, T::mul(s, factor)));
    }

    for (std::size_t i = (size/T::unroll_size)*T::unroll_size; i < size; i++)
        dst[i] += src[i] * c;
}


template<typename T>
inline void muladd_nonaligned(const typename T::value_type* src, typename T::value_type c, std::size_t  size, typename T::value_type* dst) {
    typename T::register_type factor = T::set1(c);

    for (std::size_t i = 0; i < size/T::unroll_size; i++) {
        typename T::register_type d = T::loadu(&dst[i*T::unroll_size]);
        typename T::register_type s = T::loadu(&src[i*T::unroll_size]);
        T::storeu(&dst[i*T::unroll_size], T::add(d, T::mul(s, factor)));
    }

    for (std::size_t i = (size/T::unroll_size)*T::unroll_size; i < size; i++)
        dst[i] += src[i] * c;
}

template<typename T>
inline void reduce_nonaligned(const typename T::value_type* src, std::size_t  size, typename T::value_type* dst) {
    for (std::size_t i = 0; i < size/T::unroll_size; i++) {
        typename T::register_type d = T::loadu(&dst[i*T::unroll_size]);
        typename T::register_type s = T::loadu(&src[i*T::unroll_size]);
        T::storeu(&dst[i*T::unroll_size], T::add(d, s));
    }

    for (std::size_t i = (size/T::unroll_size)*T::unroll_size; i < size; i++)
        dst[i] += src[i];
}

template<typename T>
inline void reduce_aligned(const typename T::value_type* src, std::size_t  size, typename T::value_type* dst) {
    for (std::size_t i = 0; i < size/T::unroll_size; i++) {
        typename T::register_type d = T::loadu(&dst[i*T::unroll_size]);
        typename T::register_type s = T::loadu(&src[i*T::unroll_size]);
        T::storeu(&dst[i*T::unroll_size], T::add(d, s));
    }

    for (std::size_t i = (size/T::unroll_size)*T::unroll_size; i < size; i++)
        dst[i] += src[i];
}

} // namespace detail

#if defined(CNN_USE_AVX)
#define VECTORIZE_TYPE(T) detail::avx<T>
#elif defined(CNN_USE_SSE)
#define VECTORIZE_TYPE(T) detail::sse<T>
#else
#define VECTORIZE_TYPE(T) detail::generic_vec_type<T>
#endif

// dst[i] += c * src[i]
template<typename T>
void muladd(const T* src, T c, std::size_t  size, T* dst) {
    if (detail::is_aligned(VECTORIZE_TYPE(T)(), src, dst))
        detail::muladd_aligned<VECTORIZE_TYPE(T)>(src, c, size, dst);
    else
        detail::muladd_nonaligned<VECTORIZE_TYPE(T)>(src, c, size, dst);
}

// sum(s1[i] * s2[i])
template<typename T>
T dot(const T* s1, const T* s2, std::size_t  size) {
    if (detail::is_aligned(VECTORIZE_TYPE(T)(), s1, s2))
        return detail::dot_product_aligned<VECTORIZE_TYPE(T)>(s1, s2, size);
    else
        return detail::dot_product_nonaligned<VECTORIZE_TYPE(T)>(s1, s2, size);
}

/// dst[i] += src[i]
template<typename T>
void reduce(const T* src, std::size_t  size, T* dst) {
    if (detail::is_aligned(VECTORIZE_TYPE(T)(), src, dst))
        return detail::reduce_aligned<VECTORIZE_TYPE(T)>(src, size, dst);
    else
        return detail::reduce_nonaligned<VECTORIZE_TYPE(T)>(src, size, dst);
}

} // namespace vectorize
