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

#include <cstdint>
#include <limits>

namespace tiny_cnn {

template<int Q>
class fixed_point {
public:
    typedef int32_t in_type;
    typedef int64_t sin_type; 

    fixed_point(){}
    template<typename T>
    fixed_point(T v) : v_(to_fixed(v)) {}
    fixed_point(const fixed_point<Q>& rhs) : v_(rhs.v_) {}
    fixed_point& operator = (const fixed_point<Q>& rhs) { v_ = rhs.v_; return *this; }
    
    void operator += (const fixed_point<Q>& rhs) { 
        assert(rhs.v_ < 0 || (v_ <= std::numeric_limits<in_type>::max() - rhs.v_));
        v_ += rhs.v_; 
    }

    void operator -= (const fixed_point<Q>& rhs) { 
        assert(rhs.v_ < 0 || (v_ >= std::numeric_limits<in_type>::min() + rhs.v_));
        v_ -= rhs.v_; 
    }

    void operator *= (const fixed_point<Q>& rhs) {
        v_ = static_cast<in_type>(((sin_type)rhs.v_ * v_) >> Q);
    }

    void operator /= (const fixed_point<Q>& rhs) {
        v_ = static_cast<in_type>(((sin_type)v_ << Q) / rhs.v_);
    }

    fixed_point<Q> operator - () const {
        fixed_point<Q> tmp;
        tmp.v_ = -v_;
        return tmp;
    }

    //operator double() const { return to_real(); }

    double to_real() const { return static_cast<double>(v_) / base(); }

    bool operator < (const fixed_point<Q>& rhs) const {
        return v_ < rhs.v_;
    }

    bool operator > (const fixed_point<Q>& rhs) const {
        return v_ > rhs.v_;
    }

    bool operator == (const fixed_point<Q>& rhs) const {
        return v_ == rhs.v_;
    }

    bool operator <= (const fixed_point<Q>& rhs) const {
        return !(*this > rhs);
    }

    bool operator >= (const fixed_point<Q>& rhs) const {
        return !(*this < rhs);
    }

private:
    in_type base() const {
        return 1 << Q;
    }

    template<typename T>
    in_type to_fixed(const T& v) const { 
        assert(v * base() <= std::numeric_limits<in_type>::max());
        return static_cast<in_type>(v * base()); 
    }

    in_type v_;
};

template<int Q>
fixed_point<Q> operator + (const fixed_point<Q>& lhs, const fixed_point<Q>& rhs) {
    fixed_point<Q> tmp(lhs);
    tmp += rhs;
    return tmp;
}

template<int Q, typename T>
fixed_point<Q> operator + (const fixed_point<Q>& lhs, const T& rhs) {
    fixed_point<Q> tmp(rhs);
    tmp += lhs;
    return tmp;
}

template<int Q, typename T>
fixed_point<Q> operator + (const T& lhs, const fixed_point<Q>& rhs) {
    fixed_point<Q> tmp(lhs);
    tmp += rhs;
    return tmp;
}

template<int Q>
fixed_point<Q> operator - (const fixed_point<Q>& lhs, const fixed_point<Q>& rhs) {
    fixed_point<Q> tmp(lhs);
    tmp -= rhs;
    return tmp;
}

template<int Q, typename T>
fixed_point<Q> operator - (const fixed_point<Q>& lhs, const T& rhs) {
    fixed_point<Q> tmp(rhs);
    tmp -= lhs;
    return tmp;
}

template<int Q, typename T>
fixed_point<Q> operator - (const T& lhs, const fixed_point<Q>& rhs) {
    fixed_point<Q> tmp(lhs);
    tmp -= rhs;
    return tmp;
}

template<int Q>
fixed_point<Q> operator * (const fixed_point<Q>& lhs, const fixed_point<Q>& rhs) {
    fixed_point<Q> tmp(lhs);
    tmp *= rhs;
    return tmp;
}

template<int Q, typename T>
fixed_point<Q> operator * (const fixed_point<Q>& lhs, const T& rhs) {
    fixed_point<Q> tmp(rhs);
    tmp *= lhs;
    return tmp;
}

template<int Q, typename T>
fixed_point<Q> operator * (const T& lhs, const fixed_point<Q>& rhs) {
    fixed_point<Q> tmp(lhs);
    tmp *= rhs;
    return tmp;
}

template<int Q>
fixed_point<Q> operator / (const fixed_point<Q>& lhs, const fixed_point<Q>& rhs) {
    fixed_point<Q> tmp(lhs);
    tmp /= rhs;
    return tmp;
}

template<int Q, typename T>
fixed_point<Q> operator / (const fixed_point<Q>& lhs, const T& rhs) {
    fixed_point<Q> tmp(rhs);
    tmp /= lhs;
    return tmp;
}

template<int Q, typename T>
fixed_point<Q> operator / (const T& lhs, const fixed_point<Q>& rhs) {
    fixed_point<Q> tmp(lhs);
    tmp /= rhs;
    return tmp;
}

}

namespace std {

template<int Q>
tiny_cnn::fixed_point<Q> exp(const tiny_cnn::fixed_point<Q>& f) {
    return std::exp(f.to_real());
}

template<int Q>
tiny_cnn::fixed_point<Q> sqrt(const tiny_cnn::fixed_point<Q>& f) {
    return std::sqrt(f.to_real());
}

} // namespace tiny_cnn
