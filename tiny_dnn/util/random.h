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
#if !defined(CNN_TR1_RANDOM)
#include <random>
#else
#include <tr1/random>
#endif
#include <type_traits>
#include <limits>
#include "nn_error.h"
#include "tiny_dnn/config.h"

#if !defined(CNN_TR1_RANDOM)
#define STD_RANDOM_NS std
#define STD_RANDOM_UNI_INT uniform_int_distribution
#define STD_RANDOM_UNI_REAL uniform_real_distribution
#else
#define STD_RANDOM_NS std::tr1
#define STD_RANDOM_UNI_INT uniform_int
#define STD_RANDOM_UNI_REAL uniform_real
#endif

namespace tiny_dnn {

class random_generator {
public:
    static random_generator& get_instance() {
        static random_generator instance;
        return instance;
    }

    STD_RANDOM_NS::mt19937& operator()() {
        return gen_;
    }

    void set_seed(unsigned int seed) {
        gen_.seed(seed);
    }
private:
    // avoid gen_(0) for MSVC known issue
    // https://connect.microsoft.com/VisualStudio/feedback/details/776456
    random_generator() : gen_(1) {}
    STD_RANDOM_NS::mt19937 gen_;
};

template<typename T> inline
typename std::enable_if<std::is_integral<T>::value, T>::type
uniform_rand(T min, T max) {
    STD_RANDOM_NS::STD_RANDOM_UNI_INT<T> dst(min, max);
    return dst(random_generator::get_instance()());
}

template<typename T> inline
typename std::enable_if<std::is_floating_point<T>::value, T>::type
uniform_rand(T min, T max) {
    STD_RANDOM_NS::STD_RANDOM_UNI_REAL<T> dst(min, max);
    return dst(random_generator::get_instance()());
}

template<typename T> inline
typename std::enable_if<std::is_floating_point<T>::value, T>::type
gaussian_rand(T mean, T sigma) {
    STD_RANDOM_NS::normal_distribution<T> dst(mean, sigma);
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

} // namespace tiny_dnn
