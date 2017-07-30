/***************************************************************************
* Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

/**
 * @brief standard mathematical functions for xexpressions
 */

#ifndef XMATH_HPP
#define XMATH_HPP

#include <cmath>
#include <complex>
#include <type_traits>

#include "xoperation.hpp"
#include "xreducer.hpp"

namespace xt
{
    template <class T>
    struct numeric_constants
    {
        static constexpr T PI = 3.141592653589793238463;
        static constexpr T PI_2 = 1.57079632679489661923;
        static constexpr T PI_4 = 0.785398163397448309616;
        static constexpr T D_1_PI = 0.318309886183790671538;
        static constexpr T D_2_PI = 0.636619772367581343076;
        static constexpr T D_2_SQRTPI = 1.12837916709551257390;
        static constexpr T SQRT2 = 1.41421356237309504880;
        static constexpr T SQRT1_2 = 0.707106781186547524401;
        static constexpr T E = 2.71828182845904523536;
        static constexpr T LOG2E = 1.44269504088896340736;
        static constexpr T LOG10E = 0.434294481903251827651;
        static constexpr T LN2 = 0.693147180559945309417;
    };

    /***********
     * Helpers *
     ***********/

    namespace detail
    {
        template <class T>
        struct bool_functor_return_type
        {
            using type = bool;
        };
    }
#define UNARY_MATH_FUNCTOR(NAME)                   \
    template <class T>                             \
    struct NAME##_fun                              \
    {                                              \
        using argument_type = T;                   \
        using result_type = T;                     \
        constexpr T operator()(const T& arg) const \
        {                                          \
            using std::NAME;                       \
            return NAME(arg);                      \
        }                                          \
    }

#define UNARY_MATH_FUNCTOR_COMPLEX_REDUCING(NAME)            \
    template <class T>                                       \
    struct NAME##_fun                                        \
    {                                                        \
        using argument_type = T;                             \
        using result_type = complex_value_type_t<T>;         \
        constexpr result_type operator()(const T& arg) const \
        {                                                    \
            using std::NAME;                                 \
            return NAME(arg);                                \
        }                                                    \
    }

#define BINARY_MATH_FUNCTOR(NAME)                                  \
    template <class T>                                             \
    struct NAME##_fun                                              \
    {                                                              \
        using first_argument_type = T;                             \
        using second_argument_type = T;                            \
        using result_type = T;                                     \
        constexpr T operator()(const T& arg1, const T& arg2) const \
        {                                                          \
            using std::NAME;                                       \
            return NAME(arg1, arg2);                               \
        }                                                          \
    }

#define TERNARY_MATH_FUNCTOR(NAME)                                                \
    template <class T>                                                            \
    struct NAME##_fun                                                             \
    {                                                                             \
        using first_argument_type = T;                                            \
        using second_argument_type = T;                                           \
        using third_argument_type = T;                                            \
        using result_type = T;                                                    \
        constexpr T operator()(const T& arg1, const T& arg2, const T& arg3) const \
        {                                                                         \
            using std::NAME;                                                      \
            return NAME(arg1, arg2, arg3);                                        \
        }                                                                         \
    }

#define UNARY_BOOL_FUNCTOR(NAME)                                                    \
    template <class T>                                                              \
    struct NAME##_fun                                                               \
    {                                                                               \
        using argument_type = T;                                                    \
        using result_type = typename xt::detail::bool_functor_return_type<T>::type; \
        constexpr result_type operator()(const T& arg) const                        \
        {                                                                           \
            using std::NAME;                                                        \
            return NAME(arg);                                                       \
        }                                                                           \
    }

    namespace math
    {
        UNARY_MATH_FUNCTOR_COMPLEX_REDUCING(abs);
        UNARY_MATH_FUNCTOR(fabs);
        BINARY_MATH_FUNCTOR(fmod);
        BINARY_MATH_FUNCTOR(remainder);
        TERNARY_MATH_FUNCTOR(fma);
        BINARY_MATH_FUNCTOR(fmax);
        BINARY_MATH_FUNCTOR(fmin);
        BINARY_MATH_FUNCTOR(fdim);
        UNARY_MATH_FUNCTOR(exp);
        UNARY_MATH_FUNCTOR(exp2);
        UNARY_MATH_FUNCTOR(expm1);
        UNARY_MATH_FUNCTOR(log);
        UNARY_MATH_FUNCTOR(log10);
        UNARY_MATH_FUNCTOR(log2);
        UNARY_MATH_FUNCTOR(log1p);
        BINARY_MATH_FUNCTOR(pow);
        UNARY_MATH_FUNCTOR(sqrt);
        UNARY_MATH_FUNCTOR(cbrt);
        BINARY_MATH_FUNCTOR(hypot);
        UNARY_MATH_FUNCTOR(sin);
        UNARY_MATH_FUNCTOR(cos);
        UNARY_MATH_FUNCTOR(tan);
        UNARY_MATH_FUNCTOR(asin);
        UNARY_MATH_FUNCTOR(acos);
        UNARY_MATH_FUNCTOR(atan);
        BINARY_MATH_FUNCTOR(atan2);
        UNARY_MATH_FUNCTOR(sinh);
        UNARY_MATH_FUNCTOR(cosh);
        UNARY_MATH_FUNCTOR(tanh);
        UNARY_MATH_FUNCTOR(asinh);
        UNARY_MATH_FUNCTOR(acosh);
        UNARY_MATH_FUNCTOR(atanh);
        UNARY_MATH_FUNCTOR(erf);
        UNARY_MATH_FUNCTOR(erfc);
        UNARY_MATH_FUNCTOR(tgamma);
        UNARY_MATH_FUNCTOR(lgamma);
        UNARY_MATH_FUNCTOR(ceil);
        UNARY_MATH_FUNCTOR(floor);
        UNARY_MATH_FUNCTOR(trunc);
        UNARY_MATH_FUNCTOR(round);
        UNARY_MATH_FUNCTOR(nearbyint);
        UNARY_MATH_FUNCTOR(rint);
        UNARY_BOOL_FUNCTOR(isfinite);
        UNARY_BOOL_FUNCTOR(isinf);
        UNARY_BOOL_FUNCTOR(isnan);
    }

#undef UNARY_BOOL_FUNCTOR
#undef TERNARY_MATH_FUNCTOR
#undef BINARY_MATH_FUNCTOR
#undef UNARY_MATH_FUNCTOR
#undef UNARY_MATH_FUNCTOR_COMPLEX_REDUCING

    /*******************
     * basic functions *
     *******************/

    /**
     * @defgroup basic_functions Basic functions
     */

    /**
     * @ingroup basic_functions
     * @brief Absolute value function.
     * 
     * Returns an \ref xfunction for the element-wise absolute value
     * of \em e.
     * @param e an \ref xexpression
     * @return an \ref xfunction
     */
    template <class E>
    inline auto abs(E&& e) noexcept
        -> detail::xfunction_type_t<math::abs_fun, E>
    {
        return detail::make_xfunction<math::abs_fun>(std::forward<E>(e));
    }

    /**
     * @ingroup basic_functions
     * @brief Absolute value function.
     * 
     * Returns an \ref xfunction for the element-wise absolute value
     * of \em e.
     * @param e an \ref xexpression
     * @return an \ref xfunction
     */
    template <class E>
    inline auto fabs(E&& e) noexcept
        -> detail::xfunction_type_t<math::fabs_fun, E>
    {
        return detail::make_xfunction<math::fabs_fun>(std::forward<E>(e));
    }

    /**
     * @ingroup basic_functions
     * @brief Remainder of the floating point division operation.
     * 
     * Returns an \ref xfunction for the element-wise remainder of
     * the floating point division operation <em>e1 / e2</em>.
     * @param e1 an \ref xexpression or a scalar
     * @param e2 an \ref xexpression or a scalar
     * @return an \ref xfunction
     * @note e1 and e2 can't be both scalars.
     */
    template <class E1, class E2>
    inline auto fmod(E1&& e1, E2&& e2) noexcept
        -> detail::xfunction_type_t<math::fmod_fun, E1, E2>
    {
        return detail::make_xfunction<math::fmod_fun>(std::forward<E1>(e1), std::forward<E2>(e2));
    }

    /**
     * @ingroup basic_functions
     * @brief Signed remainder of the division operation.
     * 
     * Returns an \ref xfunction for the element-wise signed remainder
     * of the floating point division operation <em>e1 / e2</em>.
     * @param e1 an \ref xexpression or a scalar
     * @param e2 an \ref xexpression or a scalar
     * @return an \ref xfunction
     * @note e1 and e2 can't be both scalars.
     */
    template <class E1, class E2>
    inline auto remainder(E1&& e1, E2&& e2) noexcept
        -> detail::xfunction_type_t<math::remainder_fun, E1, E2>
    {
        return detail::make_xfunction<math::remainder_fun>(std::forward<E1>(e1), std::forward<E2>(e2));
    }

    /**
     * @ingroup basic_functions
     * @brief Fused multiply-add operation.
     *
     * Returns an \ref xfunction for <em>e1 * e2 + e3</em> as if
     * to infinite precision and rounded only once to fit the result type.
     * @param e1 an \ref xfunction or a scalar
     * @param e2 an \ref xfunction or a scalar
     * @param e3 an \ref xfunction or a scalar
     * @return an \ref xfunction
     * @note e1, e2 and e3 can't be scalars every three.
     */
    template <class E1, class E2, class E3>
    inline auto fma(E1&& e1, E2&& e2, E3&& e3) noexcept
        -> detail::xfunction_type_t<math::fma_fun, E1, E2, E3>
    {
        return detail::make_xfunction<math::fma_fun>(std::forward<E1>(e1), std::forward<E2>(e2), std::forward<E3>(e3));
    }

    /**
     * @ingroup basic_functions
     * @brief Maximum function.
     *
     * Returns an \ref xfunction for the element-wise maximum
     * of \a e1 and \a e2.
     * @param e1 an \ref xexpression or a scalar
     * @param e2 an \ref xexpression or a scalar
     * @return an \ref xfunction
     * @note e1 and e2 can't be both scalars.
     */
    template <class E1, class E2>
    inline auto fmax(E1&& e1, E2&& e2) noexcept
        -> detail::xfunction_type_t<math::fmax_fun, E1, E2>
    {
        return detail::make_xfunction<math::fmax_fun>(std::forward<E1>(e1), std::forward<E2>(e2));
    }

    /**
     * @ingroup basic_functions
     * @brief Minimum function.
     *
     * Returns an \ref xfunction for the element-wise minimum
     * of \a e1 and \a e2.
     * @param e1 an \ref xexpression or a scalar
     * @param e2 an \ref xexpression or a scalar
     * @return an \ref xfunction
     * @note e1 and e2 can't be both scalars.
     */
    template <class E1, class E2>
    inline auto fmin(E1&& e1, E2&& e2) noexcept
        -> detail::xfunction_type_t<math::fmin_fun, E1, E2>
    {
        return detail::make_xfunction<math::fmin_fun>(std::forward<E1>(e1), std::forward<E2>(e2));
    }

    /**
     * @ingroup basic_functions
     * @brief Positive difference function.
     *
     * Returns an \ref xfunction for the element-wise positive
     * difference of \a e1 and \a e2.
     * @param e1 an \ref xexpression or a scalar
     * @param e2 an \ref xexpression or a scalar
     * @return an \ref xfunction
     * @note e1 and e2 can't be both scalars.
     */
    template <class E1, class E2>
    inline auto fdim(E1&& e1, E2&& e2) noexcept
        -> detail::xfunction_type_t<math::fdim_fun, E1, E2>
    {
        return detail::make_xfunction<math::fdim_fun>(std::forward<E1>(e1), std::forward<E2>(e2));
    }

    namespace math
    {
        template <class T>
        struct minimum
        {
            using result_type = T;

            constexpr result_type operator()(const T& t1, const T& t2) const noexcept
            {
                return (t1 < t2) ? t1 : t2;
            }
        };

        template <class T>
        struct maximum
        {
            using result_type = T;

            constexpr result_type operator()(const T& t1, const T& t2) const noexcept
            {
                return (t1 > t2) ? t1 : t2;
            }
        };

        template <class T>
        struct clamp_fun
        {
            using first_argument_type = T;
            using second_argument_type = T;
            using third_argument_type = T;
            using result_type = T;
            constexpr T operator()(const T& v, const T& lo, const T& hi) const
            {
                return v < lo ? lo : hi < v ? hi : v;
            }
        };
    }

    /**
     * @ingroup basic_functions
     * @brief Elementwise maximum
     *
     * Returns an \ref xfunction for the element-wise
     * maximum between e1 and e2.
     * @param e1 an \ref xexpression
     * @param e2 an \ref xexpression
     * @return an \ref xfunction
     */
    template <class E1, class E2>
    inline auto maximum(E1&& e1, E2&& e2) noexcept
        -> detail::xfunction_type_t<math::maximum, E1, E2>
    {
        return detail::make_xfunction<math::maximum>(std::forward<E1>(e1), std::forward<E2>(e2));
    }

    /**
     * @ingroup basic_functions
     * @brief Elementwise minimum
     *
     * Returns an \ref xfunction for the element-wise
     * minimum between e1 and e2.
     * @param e1 an \ref xexpression
     * @param e2 an \ref xexpression
     * @return an \ref xfunction
     */
    template <class E1, class E2>
    inline auto minimum(E1&& e1, E2&& e2) noexcept
        -> detail::xfunction_type_t<math::minimum, E1, E2>
    {
        return detail::make_xfunction<math::minimum>(std::forward<E1>(e1), std::forward<E2>(e2));
    }

    /**
     * @ingroup basic_functions
     * @brief Maximum element along given axis.
     *
     * Returns an \ref xreducer for the maximum of elements over given
     * \em axes.
     * @param e an \ref xexpression
     * @param axes the axes along which the maximum is found (optional)
     * @return an \ref xreducer
     */
    template <class E, class X>
    inline auto amax(E&& e, X&& axes) noexcept
    {
        using functor_type = math::maximum<typename std::decay_t<E>::value_type>;
        return reduce(functor_type(), std::forward<E>(e), std::forward<X>(axes));
    }

    template <class E>
    inline auto amax(E&& e) noexcept
    {
        using functor_type = math::maximum<typename std::decay_t<E>::value_type>;
        return reduce(functor_type(), std::forward<E>(e));
    }

#ifdef X_OLD_CLANG
    template <class E, class I>
    inline auto amax(E&& e, std::initializer_list<I> axes) noexcept
    {
        using functor_type = math::maximum<typename std::decay_t<E>::value_type>;
        return reduce(functor_type(), std::forward<E>(e), axes);
    }
#else
    template <class E, class I, std::size_t N>
    inline auto amax(E&& e, const I(&axes)[N]) noexcept
    {
        using functor_type = math::maximum<typename std::decay_t<E>::value_type>;
        return reduce(functor_type(), std::forward<E>(e), axes);
    }
#endif

    /**
     * @ingroup basic_functions
     * @brief Minimum element along given axis.
     *
     * Returns an \ref xreducer for the minimum of elements over given
     * \em axes.
     * @param e an \ref xexpression
     * @param axes the axes along which the minimum is found (optional)
     * @return an \ref xreducer
     */
    template <class E, class X>
    inline auto amin(E&& e, X&& axes) noexcept
    {
        using functor_type = math::minimum<typename std::decay_t<E>::value_type>;
        return reduce(functor_type(), std::forward<E>(e), std::forward<X>(axes));
    }

    template <class E>
    inline auto amin(E&& e) noexcept
    {
        using functor_type = math::minimum<typename std::decay_t<E>::value_type>;
        return reduce(functor_type(), std::forward<E>(e));
    }

#ifdef X_OLD_CLANG
    template <class E, class I>
    inline auto amin(E&& e, std::initializer_list<I> axes) noexcept
    {
        using functor_type = math::minimum<typename std::decay_t<E>::value_type>;
        return reduce(functor_type(), std::forward<E>(e), axes);
    }
#else
    template <class E, class I, std::size_t N>
    inline auto amin(E&& e, const I(&axes)[N]) noexcept
    {
        using functor_type = math::minimum<typename std::decay_t<E>::value_type>;
        return reduce(functor_type(), std::forward<E>(e), axes);
    }
#endif

    /**
     * @ingroup basic_functions
     * @brief Clip values between hi and lo
     * 
     * Returns an \ref xfunction for the element-wise clipped 
     * values between lo and hi
     * @param e1 an \ref xexpression or a scalar
     * @param lo a scalar
     * @param hi a scalar
     *
     * @return a \ref xfunction
     */
    template <class E1, class E2, class E3>
    inline auto clip(E1&& e1, E2&& lo, E3&& hi) noexcept
        -> detail::xfunction_type_t<math::clamp_fun, E1, E2, E3>
    {
        return detail::make_xfunction<math::clamp_fun>(std::forward<E1>(e1), std::forward<E2>(lo), std::forward<E3>(hi));
    }

    namespace math
    {
        namespace detail
        {
            template <typename T>
            constexpr std::enable_if_t<std::is_signed<T>::value, T>
            sign_impl(T x)
            {
                return std::isnan(x) ? std::numeric_limits<T>::quiet_NaN() : x == 0 ? (T)copysign(T(0), x) : (T)copysign(T(1), x);
            }

            template <typename T>
            inline std::enable_if_t<xt::detail::is_complex<T>::value, T>
            sign_impl(T x)
            {
                typename T::value_type e = x.real() ? x.real() : x.imag();
                return T(sign_impl(e), 0);
            }

            template <typename T>
            constexpr std::enable_if_t<std::is_unsigned<T>::value, T>
            sign_impl(T x)
            {
                return T(x > T(0));
            }
        }

        template <class T>
        struct sign_fun
        {
            using argument_type = T;
            using result_type = T;

            constexpr T operator()(const T& x) const
            {
                return detail::sign_impl(x);
            }
        };
    }

    /**
     * @ingroup basic_functions
     * @brief Returns an element-wise indication of the sign of a number
     *
     * If the number is positive, returns +1. If negative, -1. If the number
     * is zero, returns 0.
     *
     * @param e an \ref xexpression
     * @return an \ref xfunction
     */
    template <class E>
    inline auto sign(E&& e) noexcept
        -> detail::xfunction_type_t<math::sign_fun, E>
    {
        return detail::make_xfunction<math::sign_fun>(std::forward<E>(e));
    }

    /*************************
     * exponential functions *
     *************************/

    /**
     * @defgroup exp_functions Exponential functions
     */

    /**
     * @ingroup exp_functions
     * @brief Natural exponential function.
     *
     * Returns an \ref xfunction for the element-wise natural
     * exponential of \em e.
     * @param e an \ref xexpression
     * @return an \ref xfunction
     */
    template <class E>
    inline auto exp(E&& e) noexcept
        -> detail::xfunction_type_t<math::exp_fun, E>
    {
        return detail::make_xfunction<math::exp_fun>(std::forward<E>(e));
    }

    /**
     * @ingroup exp_functions
     * @brief Base 2 exponential function.
     *
     * Returns an \ref xfunction for the element-wise base 2
     * exponential of \em e.
     * @param e an \ref xexpression
     * @return an \ref xfunction
     */
    template <class E>
    inline auto exp2(E&& e) noexcept
        -> detail::xfunction_type_t<math::exp2_fun, E>
    {
        return detail::make_xfunction<math::exp2_fun>(std::forward<E>(e));
    }

    /**
     * @ingroup exp_functions
     * @brief Natural exponential minus one function.
     *
     * Returns an \ref xfunction for the element-wise natural
     * exponential of \em e, minus 1.
     * @param e an \ref xexpression
     * @return an \ref xfunction
     */
    template <class E>
    inline auto expm1(E&& e) noexcept
        -> detail::xfunction_type_t<math::expm1_fun, E>
    {
        return detail::make_xfunction<math::expm1_fun>(std::forward<E>(e));
    }

    /**
     * @ingroup exp_functions
     * @brief Natural logarithm function.
     *
     * Returns an \ref xfunction for the element-wise natural
     * logarithm of \em e.
     * @param e an \ref xexpression
     * @return an \ref xfunction
     */
    template <class E>
    inline auto log(E&& e) noexcept
        -> detail::xfunction_type_t<math::log_fun, E>
    {
        return detail::make_xfunction<math::log_fun>(std::forward<E>(e));
    }

    /**
     * @ingroup exp_functions
     * @brief Base 10 logarithm function.
     *
     * Returns an \ref xfunction for the element-wise base 10
     * logarithm of \em e.
     * @param e an \ref xexpression
     * @return an \ref xfunction
     */
    template <class E>
    inline auto log10(E&& e) noexcept
        -> detail::xfunction_type_t<math::log10_fun, E>
    {
        return detail::make_xfunction<math::log10_fun>(std::forward<E>(e));
    }

    /**
     * @ingroup exp_functions
     * @brief Base 2 logarithm function.
     *
     * Returns an \ref xfunction for the element-wise base 2
     * logarithm of \em e.
     * @param e an \ref xexpression
     * @return an \ref xfunction
     */
    template <class E>
    inline auto log2(E&& e) noexcept
        -> detail::xfunction_type_t<math::log2_fun, E>
    {
        return detail::make_xfunction<math::log2_fun>(std::forward<E>(e));
    }

    /**
     * @ingroup exp_functions
     * @brief Natural logarithm of one plus function.
     *
     * Returns an \ref xfunction for the element-wise natural
     * logarithm of \em e, plus 1.
     * @param e an \ref xexpression
     * @return an \ref xfunction
     */
    template <class E>
    inline auto log1p(E&& e) noexcept
        -> detail::xfunction_type_t<math::log1p_fun, E>
    {
        return detail::make_xfunction<math::log1p_fun>(std::forward<E>(e));
    }

    /*******************
     * power functions *
     *******************/

    /**
     * @defgroup pow_functions Power functions
     */

    /**
     * @ingroup pow_functions
     * @brief Power function.
     *
     * Returns an \ref xfunction for the element-wise value of
     * of \em e1 raised to the power \em e2.
     * @param e1 an \ref xexpression or a scalar
     * @param e2 an \ref xexpression or a scalar
     * @return an \ref xfunction
     * @note e1 and e2 can't be both scalars.
     */
    template <class E1, class E2>
    inline auto pow(E1&& e1, E2&& e2) noexcept
        -> detail::xfunction_type_t<math::pow_fun, E1, E2>
    {
        return detail::make_xfunction<math::pow_fun>(std::forward<E1>(e1), std::forward<E2>(e2));
    }

    /**
     * @ingroup pow_functions
     * @brief Square root function.
     *
     * Returns an \ref xfunction for the element-wise square 
     * root of \em e.
     * @param e an \ref xexpression
     * @return an \ref xfunction
     */
    template <class E>
    inline auto sqrt(E&& e) noexcept
        -> detail::xfunction_type_t<math::sqrt_fun, E>
    {
        return detail::make_xfunction<math::sqrt_fun>(std::forward<E>(e));
    }

    /**
     * @ingroup pow_functions
     * @brief Cubic root function.
     *
     * Returns an \ref xfunction for the element-wise cubic
     * root of \em e.
     * @param e an \ref xexpression
     * @return an \ref xfunction
     */
    template <class E>
    inline auto cbrt(E&& e) noexcept
        -> detail::xfunction_type_t<math::cbrt_fun, E>
    {
        return detail::make_xfunction<math::cbrt_fun>(std::forward<E>(e));
    }

    /**
     * @ingroup pow_functions
     * @brief Hypotenuse function.
     *
     * Returns an \ref xfunction for the element-wise square
     * root of the sum of the square of \em e1 and \em e2, avoiding
     * overflow and underflow at intermediate stages of computation.
     * @param e1 an \ref xexpression or a scalar
     * @param e2 an \ref xexpression or a scalar
     * @return an \ref xfunction
     * @note e1 and e2 can't be both scalars.
     */
    template <class E1, class E2>
    inline auto hypot(E1&& e1, E2&& e2) noexcept
        -> detail::xfunction_type_t<math::hypot_fun, E1, E2>
    {
        return detail::make_xfunction<math::hypot_fun>(std::forward<E1>(e1), std::forward<E2>(e2));
    }

    /***************************
     * trigonometric functions *
     ***************************/

    /**
     * @defgroup trigo_functions Trigonometric function
     */

    /**
     * @ingroup trigo_functions
     * @brief Sine function.
     *
     * Returns an \ref xfunction for the element-wise sine
     * of \em e (measured in radians).
     * @param e an \ref xexpression
     * @return an \ref xfunction
     */
    template <class E>
    inline auto sin(E&& e) noexcept
        -> detail::xfunction_type_t<math::sin_fun, E>
    {
        return detail::make_xfunction<math::sin_fun>(std::forward<E>(e));
    }

    /**
     * @ingroup trigo_functions
     * @brief Cosine function.
     *
     * Returns an \ref xfunction for the element-wise cosine
     * of \em e (measured in radians).
     * @param e an \ref xexpression
     * @return an \ref xfunction
     */
    template <class E>
    inline auto cos(E&& e) noexcept
        -> detail::xfunction_type_t<math::cos_fun, E>
    {
        return detail::make_xfunction<math::cos_fun>(std::forward<E>(e));
    }

    /**
     * @ingroup trigo_functions
     * @brief Tangent function.
     *
     * Returns an \ref xfunction for the element-wise tangent
     * of \em e (measured in radians).
     * @param e an \ref xexpression
     * @return an \ref xfunction
     */
    template <class E>
    inline auto tan(E&& e) noexcept
        -> detail::xfunction_type_t<math::tan_fun, E>
    {
        return detail::make_xfunction<math::tan_fun>(std::forward<E>(e));
    }

    /**
     * @ingroup trigo_functions
     * @brief Arcsine function.
     *
     * Returns an \ref xfunction for the element-wise arcsine
     * of \em e.
     * @param e an \ref xexpression
     * @return an \ref xfunction
     */
    template <class E>
    inline auto asin(E&& e) noexcept
        -> detail::xfunction_type_t<math::asin_fun, E>
    {
        return detail::make_xfunction<math::asin_fun>(std::forward<E>(e));
    }

    /**
     * @ingroup trigo_functions
     * @brief Arccosine function.
     *
     * Returns an \ref xfunction for the element-wise arccosine
     * of \em e.
     * @param e an \ref xexpression
     * @return an \ref xfunction
     */
    template <class E>
    inline auto acos(E&& e) noexcept
        -> detail::xfunction_type_t<math::acos_fun, E>
    {
        return detail::make_xfunction<math::acos_fun>(std::forward<E>(e));
    }

    /**
     * @ingroup trigo_functions
     * @brief Arctangent function.
     *
     * Returns an \ref xfunction for the element-wise arctangent
     * of \em e.
     * @param e an \ref xexpression
     * @return an \ref xfunction
     */
    template <class E>
    inline auto atan(E&& e) noexcept
        -> detail::xfunction_type_t<math::atan_fun, E>
    {
        return detail::make_xfunction<math::atan_fun>(std::forward<E>(e));
    }

    /**
     * @ingroup trigo_functions
     * @brief Artangent function, using signs to determine quadrants.
     *
     * Returns an \ref xfunction for the element-wise arctangent
     * of <em>e1 / e2</em>, using the signs of arguments to determine the
     * correct quadrant.
     * @param e1 an \ref xexpression or a scalar
     * @param e2 an \ref xexpression or a scalar
     * @return an \ref xfunction
     * @note e1 and e2 can't be both scalars.
     */
    template <class E1, class E2>
    inline auto atan2(E1&& e1, E2&& e2) noexcept
        -> detail::xfunction_type_t<math::atan2_fun, E1, E2>
    {
        return detail::make_xfunction<math::atan2_fun>(std::forward<E1>(e1), std::forward<E2>(e2));
    }

    /************************
     * hyperbolic functions *
     ************************/

    /**
     * @defgroup hyper_functions Hyperbolic functions
     */

    /**
     * @ingroup hyper_functions
     * @brief Hyperbolic sine function.
     *
     * Returns an \ref xfunction for the element-wise hyperbolic
     * sine of \em e.
     * @param e an \ref xexpression
     * @return an \ref xfunction
     */
    template <class E>
    inline auto sinh(E&& e) noexcept
        -> detail::xfunction_type_t<math::sinh_fun, E>
    {
        return detail::make_xfunction<math::sinh_fun>(std::forward<E>(e));
    }

    /**
     * @ingroup hyper_functions
     * @brief Hyperbolic cosine function.
     *
     * Returns an \ref xfunction for the element-wise hyperbolic
     * cosine of \em e.
     * @param e an \ref xexpression
     * @return an \ref xfunction
     */
    template <class E>
    inline auto cosh(E&& e) noexcept
        -> detail::xfunction_type_t<math::cosh_fun, E>
    {
        return detail::make_xfunction<math::cosh_fun>(std::forward<E>(e));
    }

    /**
     * @ingroup hyper_functions
     * @brief Hyperbolic tangent function.
     *
     * Returns an \ref xfunction for the element-wise hyperbolic
     * tangent of \em e.
     * @param e an \ref xexpression
     * @return an \ref xfunction
     */
    template <class E>
    inline auto tanh(E&& e) noexcept
        -> detail::xfunction_type_t<math::tanh_fun, E>
    {
        return detail::make_xfunction<math::tanh_fun>(std::forward<E>(e));
    }

    /**
     * @ingroup hyper_functions
     * @brief Inverse hyperbolic sine function.
     *
     * Returns an \ref xfunction for the element-wise inverse hyperbolic
     * sine of \em e.
     * @param e an \ref xexpression
     * @return an \ref xfunction
     */
    template <class E>
    inline auto asinh(E&& e) noexcept
        -> detail::xfunction_type_t<math::asinh_fun, E>
    {
        return detail::make_xfunction<math::asinh_fun>(std::forward<E>(e));
    }

    /**
     * @ingroup hyper_functions
     * @brief Inverse hyperbolic cosine function.
     *
     * Returns an \ref xfunction for the element-wise inverse hyperbolic
     * cosine of \em e.
     * @param e an \ref xexpression
     * @return an \ref xfunction
     */
    template <class E>
    inline auto acosh(E&& e) noexcept
        -> detail::xfunction_type_t<math::acosh_fun, E>
    {
        return detail::make_xfunction<math::acosh_fun>(std::forward<E>(e));
    }

    /**
     * @ingroup hyper_functions
     * @brief Inverse hyperbolic tangent function.
     *
     * Returns an \ref xfunction for the element-wise inverse hyperbolic
     * tangent of \em e.
     * @param e an \ref xexpression
     * @return an \ref xfunction
     */
    template <class E>
    inline auto atanh(E&& e) noexcept
        -> detail::xfunction_type_t<math::atanh_fun, E>
    {
        return detail::make_xfunction<math::atanh_fun>(std::forward<E>(e));
    }

    /*****************************
     * error and gamma functions *
     *****************************/

    /**
     * @defgroup err_functions Error and gamma functions
     */

    /**
     * @ingroup err_functions
     * @brief Error function.
     *
     * Returns an \ref xfunction for the element-wise error function
     * of \em e.
     * @param e an \ref xexpression
     * @return an \ref xfunction
     */
    template <class E>
    inline auto erf(E&& e) noexcept
        -> detail::xfunction_type_t<math::erf_fun, E>
    {
        return detail::make_xfunction<math::erf_fun>(std::forward<E>(e));
    }

    /**
     * @ingroup err_functions
     * @brief Complementary error function.
     *
     * Returns an \ref xfunction for the element-wise complementary
     * error function of \em e, whithout loss of precision for large argument.
     * @param e an \ref xexpression
     * @return an \ref xfunction
     */
    template <class E>
    inline auto erfc(E&& e) noexcept
        -> detail::xfunction_type_t<math::erfc_fun, E>
    {
        return detail::make_xfunction<math::erfc_fun>(std::forward<E>(e));
    }

    /**
     * @ingroup err_functions
     * @brief Gamma function.
     *
     * Returns an \ref xfunction for the element-wise gamma function
     * of \em e.
     * @param e an \ref xexpression
     * @return an \ref xfunction
     */
    template <class E>
    inline auto tgamma(E&& e) noexcept
        -> detail::xfunction_type_t<math::tgamma_fun, E>
    {
        return detail::make_xfunction<math::tgamma_fun>(std::forward<E>(e));
    }

    /**
     * @ingroup err_functions
     * @brief Natural logarithm of the gamma function.
     *
     * Returns an \ref xfunction for the element-wise logarithm of
     * the asbolute value fo the gamma function of \em e.
     * @param e an \ref xexpression
     * @return an \ref xfunction
     */
    template <class E>
    inline auto lgamma(E&& e) noexcept
        -> detail::xfunction_type_t<math::lgamma_fun, E>
    {
        return detail::make_xfunction<math::lgamma_fun>(std::forward<E>(e));
    }

    /*********************************************
     * nearest integer floating point operations *
     *********************************************/

    /**
     * @defgroup nearint_functions Nearest integer floating point operations
     */

    /**
     * @ingroup nearint_functions
     * @brief ceil function.
     *
     * Returns an \ref xfunction for the element-wise smallest integer value
     * not less than \em e.
     * @param e an \ref xexpression
     * @return an \ref xfunction
     */
    template <class E>
    inline auto ceil(E&& e) noexcept
        -> detail::xfunction_type_t<math::ceil_fun, E>
    {
        return detail::make_xfunction<math::ceil_fun>(std::forward<E>(e));
    }

    /**
    * @ingroup nearint_functions
    * @brief floor function.
    *
    * Returns an \ref xfunction for the element-wise smallest integer value
    * not greater than \em e.
    * @param e an \ref xexpression
    * @return an \ref xfunction
    */
    template <class E>
    inline auto floor(E&& e) noexcept
        -> detail::xfunction_type_t<math::floor_fun, E>
    {
        return detail::make_xfunction<math::floor_fun>(std::forward<E>(e));
    }

    /**
    * @ingroup nearint_functions
    * @brief trunc function.
    *
    * Returns an \ref xfunction for the element-wise nearest integer not greater
    * in magnitude than \em e.
    * @param e an \ref xexpression
    * @return an \ref xfunction
    */
    template <class E>
    inline auto trunc(E&& e) noexcept
        -> detail::xfunction_type_t<math::trunc_fun, E>
    {
        return detail::make_xfunction<math::trunc_fun>(std::forward<E>(e));
    }

    /**
     * @ingroup nearint_functions
     * @brief round function.
     *
     * Returns an \ref xfunction for the element-wise nearest integer value
     * to \em e, rounding halfway cases away from zero, regardless of the
     * current rounding mode.
     * @param e an \ref xexpression
     * @return an \ref xfunction
     */
    template <class E>
    inline auto round(E&& e) noexcept
        -> detail::xfunction_type_t<math::round_fun, E>
    {
        return detail::make_xfunction<math::round_fun>(std::forward<E>(e));
    }

    /**
     * @ingroup nearint_functions
     * @brief nearbyint function.
     *
     * Returns an \ref xfunction for the element-wise rounding of \em e to integer
     * values in floating point format, using the current rounding mode. nearbyint
     * never raises FE_INEXACT error.
     * @param e an \ref xexpression
     * @return an \ref xfunction
     */
    template <class E>
    inline auto nearbyint(E&& e) noexcept
        -> detail::xfunction_type_t<math::nearbyint_fun, E>
    {
        return detail::make_xfunction<math::nearbyint_fun>(std::forward<E>(e));
    }

    /**
     * @ingroup nearint_functions
     * @brief rint function.
     *
     * Returns an \ref xfunction for the element-wise rounding of \em e to integer
     * values in floating point format, using the current rounding mode. Contrary
     * to nearbyint, rint may raise FE_INEXACT error.
     * @param e an \ref xexpression
     * @return an \ref xfunction
     */
    template <class E>
    inline auto rint(E&& e) noexcept
        -> detail::xfunction_type_t<math::rint_fun, E>
    {
        return detail::make_xfunction<math::rint_fun>(std::forward<E>(e));
    }

    /****************************
     * classification functions *
     ****************************/

    /**
     * @defgroup classif_functions Classification functions
     */

    /**
     * @ingroup classif_functions
     * @brief finite value check
     *
     * Returns an \ref xfunction for the element-wise finite value check
     * tangent of \em e.
     * @param e an \ref xexpression
     * @return an \ref xfunction
     */
    template <class E>
    inline auto isfinite(E&& e) noexcept
        -> detail::xfunction_type_t<math::isfinite_fun, E>
    {
        return detail::make_xfunction<math::isfinite_fun>(std::forward<E>(e));
    }

    /**
     * @ingroup classif_functions
     * @brief infinity check
     *
     * Returns an \ref xfunction for the element-wise infinity check
     * tangent of \em e.
     * @param e an \ref xexpression
     * @return an \ref xfunction
     */
    template <class E>
    inline auto isinf(E&& e) noexcept
        -> detail::xfunction_type_t<math::isinf_fun, E>
    {
        return detail::make_xfunction<math::isinf_fun>(std::forward<E>(e));
    }

    /**
     * @ingroup classif_functions
     * @brief NaN check
     *
     * Returns an \ref xfunction for the element-wise NaN check
     * tangent of \em e.
     * @param e an \ref xexpression
     * @return an \ref xfunction
     */
    template <class E>
    inline auto isnan(E&& e) noexcept
        -> detail::xfunction_type_t<math::isnan_fun, E>
    {
        return detail::make_xfunction<math::isnan_fun>(std::forward<E>(e));
    }

    namespace detail
    {
        template <class FUNCTOR, class T, std::size_t... Is>
        inline auto get_functor(T&& args, std::index_sequence<Is...>)
        {
            return FUNCTOR(std::get<Is>(args)...);
        }

        template <template <class...> class F, class... A, class... E>
        inline auto make_xfunction(std::tuple<A...>&& f_args, E&&... e) noexcept
        {
            using functor_type = F<common_value_type_t<std::decay_t<E>...>>;
            using result_type = typename functor_type::result_type;
            using type = xfunction<functor_type, result_type, const_xclosure_t<E>...>;
            auto functor = get_functor<functor_type>(
                std::forward<std::tuple<A...>>(f_args),
                std::make_index_sequence<sizeof...(A)>{}
            );
            return type(std::move(functor), std::forward<E>(e)...);
        }

        template <class T>
        struct isclose
        {
            using result_type = bool;
            isclose(double rtol, double atol, bool equal_nan)
                : m_rtol(rtol), m_atol(atol), m_equal_nan(equal_nan)
            {
            }

            bool operator()(const T& a, const T& b) const
            {
                if (m_equal_nan && std::isnan(a) && std::isnan(b))
                {
                    return true;
                }
                return std::abs(a - b) <= (m_atol + m_rtol * std::abs(b));
            }

        private:
            double m_rtol;
            double m_atol;
            bool m_equal_nan;
        };
    }

    /**
     * @ingroup classif_functions
     * @brief Element-wise closeness detection
     *
     * Returns an \ref xfunction that evaluates to
     * true if the elements in ``e1`` and ``e2`` are close to each other
     * according to parameters ``atol`` and ``rtol``.
     * The equation is: ``std::abs(a - b) <= (m_atol + m_rtol * std::abs(b))``.
     * @param e1 input array to compare
     * @param e2 input array to compare
     * @param rtol the relative tolerance parameter (default 1e-05)
     * @param atol the absolute tolerance parameter (default 1e-08)
     * @param equal_nan if true, isclose returns true if both elements of e1 and e2 are NaN
     * @return an \ref xfunction
     */
    template <class E1, class E2>
    inline auto isclose(E1&& e1, E2&& e2, double rtol = 1e-05, double atol = 1e-08, bool equal_nan = false) noexcept
    {
        return detail::make_xfunction<detail::isclose>(std::make_tuple(rtol, atol, equal_nan),
                                                       std::forward<E1>(e1), std::forward<E2>(e2));
    }

    /**
     * @ingroup classif_functions
     * @brief Check if all elements in \em e1 are close to the
     * corresponding elements in \em e2.
     *
     * Returns true if all elements in ``e1`` and ``e2`` are close to each other
     * according to parameters ``atol`` and ``rtol``.
     * @param e1 input array to compare
     * @param e2 input arrays to compare
     * @param rtol the relative tolerance parameter (default 1e-05)
     * @param atol the absolute tolerance parameter (default 1e-08)
     * @return a boolean
     */
    template <class E1, class E2>
    inline auto allclose(E1&& e1, E2&& e2, double rtol = 1e-05, double atol = 1e-08) noexcept
    {
        return xt::all(isclose(std::forward<E1>(e1), std::forward<E2>(e2), rtol, atol));
    }

    /**********************
     * Reducing functions *
     **********************/

    /**
     * @defgroup  red_functions reducing functions
     */

    /**
     * @ingroup red_functions
     * @brief Sum of elements over given axes.
     *
     * Returns an \ref xreducer for the sum of elements over given
     * \em axes.
     * @param e an \ref xexpression
     * @param axes the axes along which the sum is performed (optional)
     * @return an \ref xreducer
     */
    template <class E, class X>
    inline auto sum(E&& e, X&& axes) noexcept
    {
        using functor_type = std::plus<typename std::decay_t<E>::value_type>;
        return reduce(functor_type(), std::forward<E>(e), std::forward<X>(axes));
    }

    template <class E>
    inline auto sum(E&& e) noexcept
    {
        using functor_type = std::plus<typename std::decay_t<E>::value_type>;
        return reduce(functor_type(), std::forward<E>(e));
    }

#ifdef X_OLD_CLANG
    template <class E, class I>
    inline auto sum(E&& e, std::initializer_list<I> axes) noexcept
    {
        using functor_type = std::plus<typename std::decay_t<E>::value_type>;
        return reduce(functor_type(), std::forward<E>(e), axes);
    }
#else
    template <class E, class I, std::size_t N>
    inline auto sum(E&& e, const I (&axes)[N]) noexcept
    {
        using functor_type = std::plus<typename std::decay_t<E>::value_type>;
        return reduce(functor_type(), std::forward<E>(e), axes);
    }
#endif

    /**
     * @ingroup red_functions
     * @brief Product of elements over given axes.
     *
     * Returns an \ref xreducer for the product of elements over given
     * \em axes.
     * @param e an \ref xexpression
     * @param axes the axes along which the product is computed (optional)
     * @return an \ref xreducer
     */
    template <class E, class X>
    inline auto prod(E&& e, X&& axes) noexcept
    {
        using functor_type = std::multiplies<typename std::decay_t<E>::value_type>;
        return reduce(functor_type(), std::forward<E>(e), std::forward<X>(axes));
    }

    template <class E>
    inline auto prod(E&& e) noexcept
    {
        using functor_type = std::multiplies<typename std::decay_t<E>::value_type>;
        return reduce(functor_type(), std::forward<E>(e));
    }

#ifdef X_OLD_CLANG
    template <class E, class I>
    inline auto prod(E&& e, std::initializer_list<I> axes) noexcept
    {
        using functor_type = std::multiplies<typename std::decay_t<E>::value_type>;
        return reduce(functor_type(), std::forward<E>(e), axes);
    }
#else
    template <class E, class I, std::size_t N>
    inline auto prod(E&& e, const I (&axes)[N]) noexcept
    {
        using functor_type = std::multiplies<typename std::decay_t<E>::value_type>;
        return reduce(functor_type(), std::forward<E>(e), axes);
    }
#endif

    /**
     * @ingroup red_functions
     * @brief Mean of elements over given axes.
     *
     * Returns an \ref xreducer for the mean of elements over given
     * \em axes.
     * @param e an \ref xexpression
     * @param axes the axes along which the mean is computed (optional)
     * @return an \ref xexpression
     */
    template <class E, class X>
    inline auto mean(E&& e, X&& axes) noexcept
    {
        using value_type = typename std::decay_t<E>::value_type;
        auto size = e.size();
        auto s = sum(std::forward<E>(e), std::forward<X>(axes));
        return std::move(s) / value_type(size / s.size());
    }

    template <class E>
    inline auto mean(E&& e) noexcept
    {
        using value_type = typename std::decay_t<E>::value_type;
        auto size = e.size();
        return sum(std::forward<E>(e)) / value_type(size);
    }

#ifdef X_OLD_CLANG
    template <class E, class I>
    inline auto mean(E&& e, std::initializer_list<I> axes) noexcept
    {
        using value_type = typename std::decay_t<E>::value_type;
        auto size = e.size();
        auto s = sum(std::forward<E>(e), axes);
        return std::move(s) / value_type(size / s.size());
    }
#else
    template <class E, class I, std::size_t N>
    inline auto mean(E&& e, const I (&axes)[N]) noexcept
    {
        using value_type = typename std::decay_t<E>::value_type;
        auto size = e.size();
        auto s = sum(std::forward<E>(e), axes);
        return std::move(s) / value_type(size / s.size());
    }
#endif
}

#endif
