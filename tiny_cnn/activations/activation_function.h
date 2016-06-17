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
#include "tiny_cnn/util/util.h"
#include <algorithm>

#ifdef CNN_USE_HERUMI_FMATH
#ifdef CNN_USE_AVX2
#ifndef __AVX2__
#define __AVX2__
#endif
#endif
#include "fmath/fmath.hpp"
#endif // #ifdef CNN_USE_HERUMI_FMATH

namespace tiny_cnn {
namespace activation {

class function {
public:
    function() = default;
    function(const function &) = default;
#ifndef CNN_DEFAULT_MOVE_CONSTRUCTOR_UNAVAILABLE
    function(function &&) = default;
#endif
    function &operator =(const function &) = default;
#ifndef CNN_DEFAULT_ASSIGNMENT_OPERATOR_UNAVAILABLE
    function &operator =(function &&) = default;
#endif
    virtual ~function() = default;

    virtual float f(const fvec_t& v, cnn_size_t index) const = 0;
    virtual double f(const dvec_t& v, cnn_size_t index) const = 0;
	virtual void f(fvec_t& dst, const fvec_t& v) const {
		for (size_t i=0; i<v.size(); ++i) {
			dst[i] = f(v, i);
		}
	}
	virtual void f(dvec_t& dst, const dvec_t& v) const {
		for (size_t i=0; i<v.size(); ++i) {
			dst[i] = f(v, i);
		}
	}

    // dfi/dyi
    virtual float_t df(float_t y) const = 0;

    // dfi/dyk (k=0,1,..n)
    virtual vec_t df(const vec_t& y, cnn_size_t i) const { vec_t v(y.size(), 0); v[i] = df(y[i]); return v; }

    // return if dfi/dyk is one-hot vector
    virtual bool one_hot() const { return true; }

    // target value range for learning
    virtual std::pair<float_t, float_t> scale() const = 0;
};

class identity : public function {
public:
    using function::df;
    float f(const fvec_t& v, cnn_size_t i) const override { return v[i]; }
    double f(const dvec_t& v, cnn_size_t i) const override { return v[i]; }

#if 0
	// TODO: vectorize
	virtual void f(fvec_t& dst, const fvec_t& v) const {
		for (size_t i=0; i<v.size(); ++i) {
			dst[i] = f(v, i);
		}
	}
	virtual void f(dvec_t& dst, const dvec_t& v) const {
		for (size_t i=0; i<v.size(); ++i) {
			dst[i] = f(v, i);
		}
	}
#endif

    float_t df(float_t /*y*/) const override {
		return float_t(1);
	}
#ifdef CNN_USE_AVX
	inline __m256 df(__m256 /*y*/) const {
		return _mm256_set1_ps(1.0f);
	}
	inline __m256d df(__m256d /*y*/) const {
		return _mm256_set1_pd(1.0);
	}
#endif

    std::pair<float_t, float_t> scale() const override { return std::make_pair(float_t(0.1), float_t(0.9)); }
};

class sigmoid : public function {
public:
    using function::df;
    float f(const fvec_t& v, cnn_size_t i) const override { return float(1) / (float(1) + std::exp(-v[i])); }
    double f(const dvec_t& v, cnn_size_t i) const override { return double(1) / (double(1) + std::exp(-v[i])); }
    
#if 0
	// TODO: vectorize
	virtual void f(fvec_t& dst, const fvec_t& v) const {
		for (size_t i=0; i<v.size(); ++i) {
			dst[i] = f(v, i);
		}
	}
	virtual void f(dvec_t& dst, const dvec_t& v) const {
		for (size_t i=0; i<v.size(); ++i) {
			dst[i] = f(v, i);
		}
	}
#endif

	float_t df(float_t y) const override {
		return y * (float_t(1) - y);
	}
#ifdef CNN_USE_AVX
	inline __m256 df(__m256 y) const {
		__m256 one = _mm256_set1_ps(1.0f);
		return _mm256_mul_ps(y, _mm256_sub_ps(one, y));
	}
	inline __m256d df(__m256d y) const {
		__m256d one = _mm256_set1_pd(1.0f);
		return _mm256_mul_pd(y, _mm256_sub_pd(one, y));
	}
#endif
    std::pair<float_t, float_t> scale() const override { return std::make_pair(float_t(0.1), float_t(0.9)); }
};

class relu : public function {
public:
    using function::df;
    float f(const fvec_t& v, cnn_size_t i) const override { return std::max(float(0), v[i]); }
    double f(const dvec_t& v, cnn_size_t i) const override { return std::max(double(0), v[i]); }

#if 0
	// TODO: vectorize
	virtual void f(fvec_t& dst, const fvec_t& v) const {
		for (size_t i=0; i<v.size(); ++i) {
			dst[i] = f(v, i);
		}
	}
	virtual void f(dvec_t& dst, const dvec_t& v) const {
		for (size_t i=0; i<v.size(); ++i) {
			dst[i] = f(v, i);
		}
	}
#endif

	float_t df(float_t y) const override {
		return y > float_t(0) ? float_t(1) : float_t(0);
	}
#ifdef CNN_USE_AVX
	inline __m256 df(__m256 y) const {
		__m256 mask = _mm256_cmp_ps(y, _mm256_setzero_ps(), _CMP_GT_OS);
		__m256 result = _mm256_blendv_ps(_mm256_set1_ps(1.0f), _mm256_set1_ps(0.0f), mask);
		return result;
	}
	inline __m256d df(__m256d y) const {
		__m256d mask = _mm256_cmp_pd(y, _mm256_setzero_pd(), _CMP_GT_OS);
		__m256d result = _mm256_blendv_pd(_mm256_set1_pd(1.0), _mm256_set1_pd(0.0), mask);
		return result;
	}
#endif

    std::pair<float_t, float_t> scale() const override { return std::make_pair(float_t(0.1), float_t(0.9)); }
};

typedef relu rectified_linear; // for compatibility

class leaky_relu : public function {
public:
    using function::df;
    float_t f(const vec_t& v, cnn_size_t i) const override { return (v[i] > float_t(0)) ? v[i] : float_t(0.01) * v[i]; }

#if 0
	// TODO: vectorize
	virtual void f(fvec_t& dst, const fvec_t& v) const {
		for (size_t i=0; i<v.size(); ++i) {
			dst[i] = f(v, i);
		}
	}
	virtual void f(dvec_t& dst, const dvec_t& v) const {
		for (size_t i=0; i<v.size(); ++i) {
			dst[i] = f(v, i);
		}
	}
#endif

    float_t df(float_t y) const override {
		return y > float_t(0) ? float_t(1) : float_t(0.01);
	}
#ifdef CNN_USE_AVX
	inline __m256 df(__m256 y) const {
		__m256 mask = _mm256_cmp_ps(y, _mm256_setzero_ps(), _CMP_GT_OS);
		__m256 result = _mm256_blendv_ps(_mm256_set1_ps(1.0f), _mm256_set1_ps(0.01f), mask);
		return result;
	}
	inline __m256d df(__m256d y) const {
		__m256d mask = _mm256_cmp_pd(y, _mm256_setzero_pd(), _CMP_GT_OS);
		__m256d result = _mm256_blendv_pd(_mm256_set1_pd(1.0), _mm256_set1_pd(0.01), mask);
		return result;
	}
#endif

    std::pair<float_t, float_t> scale() const override { return std::make_pair(float_t(0.1), float_t(0.9)); }
};

class elu : public function {
public:
    using function::df;
    float_t f(const vec_t& v, cnn_size_t i) const override { return (v[i]<float_t(0) ? (exp(v[i])- float_t(1)) : v[i]); }

#if 0
	// TODO: vectorize
	virtual void f(fvec_t& dst, const fvec_t& v) const {
		for (size_t i=0; i<v.size(); ++i) {
			dst[i] = f(v, i);
		}
	}
	virtual void f(dvec_t& dst, const dvec_t& v) const {
		for (size_t i=0; i<v.size(); ++i) {
			dst[i] = f(v, i);
		}
	}
#endif

    float_t df(float_t y) const override {
		return (y > float_t(0) ? float_t(1) : (float_t(1)+y));
	}
#ifdef CNN_USE_AVX
	inline __m256 df(__m256 y) const {
		__m256 mask = _mm256_cmp_ps(y, _mm256_setzero_ps(), _CMP_GT_OS);
		__m256 one = _mm256_set1_ps(1.0f);
		__m256 result = _mm256_blendv_ps(one, _mm256_add_ps(one, y), mask);
		return result;
	}
	inline __m256d df(__m256d y) const {
		__m256d mask = _mm256_cmp_pd(y, _mm256_setzero_pd(), _CMP_GT_OS);
		__m256d one = _mm256_set1_pd(1.0);
		__m256d result = _mm256_blendv_pd(one, _mm256_add_pd(one, y), mask);
		return result;
	}
#endif
    std::pair<float_t, float_t> scale() const override { return std::make_pair(float_t(0.1), float_t(0.9)); }
};

class softmax : public function {
public:
	template <typename Float, typename Vec>
	Float fimpl(const Vec& v, cnn_size_t i) const {
        Float alpha = *std::max_element(v.begin(), v.end());
        Float numer = std::exp(v[i] - alpha);
        Float denom = Float(0);
        for (auto x : v)
            denom += std::exp(x - alpha);
        return numer / denom;
    }

    float f(const fvec_t& v, cnn_size_t i) const override {
        return fimpl<float>(v, i);
    }
    double f(const dvec_t& v, cnn_size_t i) const override {
        return fimpl<double>(v, i);
    }

#if 0
	// TODO: vectorize
	virtual void f(fvec_t& dst, const fvec_t& v) const {
		for (size_t i=0; i<v.size(); ++i) {
			dst[i] = f(v, i);
		}
	}
	virtual void f(dvec_t& dst, const dvec_t& v) const {
		for (size_t i=0; i<v.size(); ++i) {
			dst[i] = f(v, i);
		}
	}
#endif

    float_t df(float_t y) const override {
        return y * (float_t(1) - y);
    }

#ifdef CNN_USE_AVX
	inline __m256 df(__m256 y) const {
		__m256 one = _mm256_set1_ps(1.0f);
		return _mm256_mul_ps(y, _mm256_sub_ps(one, y));
	}
	inline __m256d df(__m256d y) const {
		__m256d one = _mm256_set1_pd(1.0f);
		return _mm256_mul_pd(y, _mm256_sub_pd(one, y));
	}
#endif

    virtual vec_t df(const vec_t& y, cnn_size_t index) const override {
        vec_t v(y.size(), 0);
        for (cnn_size_t i = 0; i < y.size(); i++)
            v[i] = (i == index) ? df(y[index]) : -y[i] * y[index];

        return v;
    }

    virtual bool one_hot() const override { return false; }

    std::pair<float_t, float_t> scale() const override { return std::make_pair(float_t(0), float_t(1)); }
};

#ifdef CNN_USE_AVX

static inline __m256 exp_ps(__m256 x)
{
#ifdef CNN_USE_HERUMI_FMATH

#ifdef CNN_USE_AVX2
	return fmath::exp_ps256(x);
#else
	__m128 lo = fmath::exp_ps(_mm256_castps256_ps128(x));
	__m128 hi = fmath::exp_ps(_mm256_extractf128_ps(x, 1));
	return _mm256_setr_m128(lo, hi);
#endif // #ifdef CNN_USE_AVX2

#else // #ifdef CNN_USE_HERUMI_FMATH
	//_mm_extract_ps(
	//_MM_EXTRACT_FLOAT(
	union {
		__m256 y;
		float s[8];
	};
	y = x;
	for (size_t i=0; i<8; ++i) {
		s[i] = std::exp(s[i]);
	}
	return y;
#endif // #ifdef CNN_USE_HERUMI_FMATH #else
}

static inline __m256d exp_pd(__m256d x)
{
#if defined(CNN_USE_HERUMI_FMATH) && defined(CNN_USE_AVX2)
	return fmath::exp_pd256(x);
#else
	VECTORIZE_ALIGN(16) double doubles[4];
	doubles[0] = std::exp(x.m256d_f64[0]);
	doubles[1] = std::exp(x.m256d_f64[1]);
	doubles[2] = std::exp(x.m256d_f64[2]);
	doubles[3] = std::exp(x.m256d_f64[3]);
	return _mm256_load_pd(doubles);
#endif
}

#endif // #ifdef CNN_USE_AVX

class tan_h : public function {
public:
    using function::df;

#ifdef CNN_USE_HERUMI_FMATH
	float f(float v) const {
        const float ep = fmath::exp(v);
        const float em = fmath::exp(-v); 
		float ret = (ep - em) / (ep + em);
        return ret;
	}

	double f(double v) const {
        const double ep = fmath::expd(v);
        const double em = fmath::expd(-v); 
		double ret = (ep - em) / (ep + em);
        return ret;
	}
#else // #ifdef CNN_USE_HERUMI_FMATH
	template <typename T>
	inline T fimpl(T v) const {
        const T ep = std::exp(v);
        const T em = std::exp(-v);
		T ret = (ep - em) / (ep + em);
        return ret;
	}

	float f(float v) const {
        return fimpl(v);
	}
	double f(double v) const {
        return fimpl(v);
	}
#endif // #ifdef CNN_USE_HERUMI_FMATH #else

    float f(const fvec_t& v, cnn_size_t i) const override {
		return f(v[i]);
    }
    double f(const dvec_t& v, cnn_size_t i) const override {
		return f(v[i]);
    }

#ifdef CNN_USE_AVX

	void f(fvec_t& dst, const fvec_t& v) const override {
		assert(dst.size() == v.size());
		size_t sz = v.size();
		size_t nblocks = sz >> 3;
		for (size_t i = 0; i<nblocks; ++i) {
			__m256 x = _mm256_load_ps(&v[i*8]);
			__m256 ep = exp_ps(x);
			__m256 mx = _mm256_sub_ps(_mm256_setzero_ps(), x);
			__m256 em = exp_ps(mx);
			__m256 ep_minus_em = _mm256_sub_ps(ep, em);
			__m256 ep_plus_em = _mm256_add_ps(ep, em);
#if 1
			__m256 ret = _mm256_div_ps(ep_minus_em, ep_plus_em);
#else
			__m256 rcp_ep_plus_em = _mm256_rcp_ps(ep_plus_em);
			// TODO: perform NR iteration to improve numerical precision.
			__m256 ret = _mm256_mul_ps(ep_minus_em, rcp_ep_plus_em);
#endif
			_mm256_store_ps(&dst[i*8], ret);
		}
		for (size_t i=(nblocks << 3); i<sz; ++i) {
			dst[i] = f(v[i]);
		}
	}

	void f(dvec_t& dst, const dvec_t& v) const override {
		assert(dst.size() == v.size());
		size_t sz = v.size();
		size_t nblocks = sz >> 2;
		for (size_t i = 0; i<nblocks; ++i) {
			__m256d x = _mm256_load_pd(&v[i*4]);
			__m256d ep = exp_pd(x);
			__m256d mx = _mm256_sub_pd(_mm256_setzero_pd(), x);
			__m256d em = exp_pd(mx);
			__m256d ep_minus_em = _mm256_sub_pd(ep, em);
			__m256d ep_plus_em = _mm256_add_pd(ep, em);
#if 1
			__m256d ret = _mm256_div_pd(ep_minus_em, ep_plus_em);
#else
			__m256d rcp_ep_plus_em = _mm256_rcp_pd(ep_plus_em);
			// TODO: perform NR iteration to improve numerical precision.
			__m256d ret = _mm256_mul_pd(ep_minus_em, rcp_ep_plus_em);
#endif
			_mm256_store_pd(&dst[i*4], ret);
		}
		for (size_t i=(nblocks << 2); i<sz; ++i) {
			dst[i] = f(v[i]);
		}
	}

#else // #ifdef CNN_USE_AVX

	void f(fvec_t& dst, const fvec_t& v) const {
		assert(dst.size() == v.size());
		for (size_t i=0; i<v.size(); ++i) {
			dst[i] = f(v[i]);
		}
	}

	void f(dvec_t& dst, const dvec_t& v) const {
		assert(dst.size() == v.size());
		for (size_t i=0; i<v.size(); ++i) {
			dst[i] = f(v[i]);
		}
	}

#endif // #ifdef CNN_USE_AVX #else

    // fast approximation of tanh (improve 2-3% speed in LeNet-5)
    /*float_t f(float_t x) const {
        const float_t x2 = x * x;
        x *= 1.0 + x2 * (0.1653 + x2 * 0.0097);
        return x / std::sqrt(1.0 + x * x);// invsqrt(static_cast<float>(1.0 + x * x));
    }*/

    float_t df(float_t y) const override {
		return float_t(1) - sqr(y);
	}
#ifdef CNN_USE_AVX
	inline __m256 df(__m256 y) const {
		__m256 one = _mm256_set1_ps(1.0f);
		return _mm256_sub_ps(one, _mm256_mul_ps(y, y));
	}
	inline __m256d df(__m256d y) const {
		__m256d one = _mm256_set1_pd(1.0);
		return _mm256_sub_pd(one, _mm256_mul_pd(y, y));
	}
#endif

    std::pair<float_t, float_t> scale() const override { return std::make_pair(float_t(-0.8), float_t(0.8)); }

private:
    /*float invsqrt(float x) const {
        float x2 = x * 0.5f;
        long i = *reinterpret_cast<long*>(&x);

        i = 0x5f3759df - (i >> 1);
        x = *reinterpret_cast<float*>(&i);
        x = x * (1.5f - (x2 * x * x));
        return x;
    }*/
};

// s tan_h, but scaled to match the other functions
class tan_hp1m2 : public function {
public:
    using function::df;
    float_t f(const vec_t& v, cnn_size_t i) const override {
        const float_t ep = std::exp(v[i]);
        return ep / (ep + std::exp(-v[i]));
    }

#if 0
	// TODO: vectorize
	virtual void f(fvec_t& dst, const fvec_t& v) const {
		for (size_t i=0; i<v.size(); ++i) {
			dst[i] = f(v, i);
		}
	}
	virtual void f(dvec_t& dst, const dvec_t& v) const {
		for (size_t i=0; i<v.size(); ++i) {
			dst[i] = f(v, i);
		}
	}
#endif

    float_t df(float_t y) const override {
		return 2 * y *(float_t(1) - y);
	}
#ifdef CNN_USE_AVX
	inline __m256 df(__m256 y) const {
		__m256 one = _mm256_set1_ps(1.0f);
		__m256 result = _mm256_mul_ps(y, _mm256_sub_ps(one, y));
		result = _mm256_add_ps(result, result);
		return result;
	}
	inline __m256d df(__m256d y) const {
		__m256d one = _mm256_set1_pd(1.0);
		__m256d result = _mm256_mul_pd(y, _mm256_sub_pd(one, y));
		result = _mm256_add_pd(result, result);
		return result;
	}
#endif

    std::pair<float_t, float_t> scale() const override { return std::make_pair(float_t(0.1), float_t(0.9)); }
};

} // namespace activation
} // namespace tiny_cnn
