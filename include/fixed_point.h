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

}
