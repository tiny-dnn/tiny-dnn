/***************************************************************************
* Copyright (c) 2017, Ullrich Koethe                                       *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XTENSOR_NORM_HPP
#define XTENSOR_NORM_HPP

#include <cmath>
// std::abs(int) prior to C++ 17
#include <complex>
#include <cstdlib>

#include "xconcepts.hpp"
#include "xmath.hpp"
#include "xoperation.hpp"
#include "xutils.hpp"

namespace xt
{
/*************************************
     * norm functions for built-in types *
     *************************************/

///@cond DOXYGEN_INCLUDE_SFINAE
#define XTENSOR_DEFINE_SIGNED_NORMS(T)                          \
    inline auto                                                 \
    norm_lp(T t, double p) noexcept                             \
    {                                                           \
        using rt = decltype(std::abs(t));                       \
        return p == 0.0                                         \
            ? static_cast<rt>(t != 0)                           \
            : std::abs(t);                                      \
    }                                                           \
    inline auto                                                 \
    norm_lp_to_p(T t, double p) noexcept                        \
    {                                                           \
        using rt = real_promote_type_t<T>;                      \
        return p == 0.0                                         \
            ? static_cast<rt>(t != 0)                           \
            : std::pow(static_cast<rt>(std::abs(t)),            \
                       static_cast<rt>(p));                     \
    }                                                           \
    inline size_t norm_l0(T t) noexcept { return (t != 0); }    \
    inline auto norm_l1(T t) noexcept { return std::abs(t); }   \
    inline auto norm_l2(T t) noexcept { return std::abs(t); }   \
    inline auto norm_linf(T t) noexcept { return std::abs(t); } \
    inline auto norm_sq(T t) noexcept { return t * t; }

    XTENSOR_DEFINE_SIGNED_NORMS(signed char)
    XTENSOR_DEFINE_SIGNED_NORMS(short)
    XTENSOR_DEFINE_SIGNED_NORMS(int)
    XTENSOR_DEFINE_SIGNED_NORMS(long)
    XTENSOR_DEFINE_SIGNED_NORMS(long long)
    XTENSOR_DEFINE_SIGNED_NORMS(float)
    XTENSOR_DEFINE_SIGNED_NORMS(double)
    XTENSOR_DEFINE_SIGNED_NORMS(long double)

#undef XTENSOR_DEFINE_SIGNED_NORMS

#define XTENSOR_DEFINE_UNSIGNED_NORMS(T)                      \
    inline T norm_lp(T t, double p) noexcept                  \
    {                                                         \
        return p == 0.0                                       \
            ? (t != 0)                                        \
            : t;                                              \
    }                                                         \
    inline auto                                               \
    norm_lp_to_p(T t, double p) noexcept                      \
    {                                                         \
        using rt = real_promote_type_t<T>;                    \
        return p == 0.0                                       \
            ? static_cast<rt>(t != 0)                         \
            : std::pow(static_cast<rt>(t),                    \
                       static_cast<rt>(p));                   \
    }                                                         \
    inline T norm_l0(T t) noexcept { return t != 0 ? 1 : 0; } \
    inline T norm_l1(T t) noexcept { return t; }              \
    inline T norm_l2(T t) noexcept { return t; }              \
    inline T norm_linf(T t) noexcept { return t; }            \
    inline auto norm_sq(T t) noexcept { return t * t; }

    XTENSOR_DEFINE_UNSIGNED_NORMS(unsigned char)
    XTENSOR_DEFINE_UNSIGNED_NORMS(unsigned short)
    XTENSOR_DEFINE_UNSIGNED_NORMS(unsigned int)
    XTENSOR_DEFINE_UNSIGNED_NORMS(unsigned long)
    XTENSOR_DEFINE_UNSIGNED_NORMS(unsigned long long)

#undef XTENSOR_DEFINE_UNSIGNED_NORMS

    /***********************************
     * norm functions for std::complex *
     ***********************************/

    /**
     * \brief L0 pseudo-norm of a complex number.
     * Equivalent to <tt>t != 0</tt>.
     */
    template <class T>
    inline uint64_t norm_l0(const std::complex<T>& t) noexcept
    {
        return t.real() != 0 || t.imag() != 0;
    }

    /**
     * \brief L1 norm of a complex number.
     */
    template <class T>
    inline auto norm_l1(const std::complex<T>& t) noexcept
    {
        return std::abs(t.real()) + std::abs(t.imag());
    }

    /**
     * \brief L2 norm of a complex number.
     * Equivalent to <tt>std::abs(t)</tt>.
     */
    template <class T>
    inline auto norm_l2(const std::complex<T>& t) noexcept
    {
        return std::abs(t);
    }

    /**
     * \brief Squared norm of a complex number.
     * Equivalent to <tt>std::norm(t)</tt> (yes, the C++ standard really defines
     * <tt>norm()</tt> to compute the squared norm).
     */
    template <class T>
    inline auto norm_sq(const std::complex<T>& t) noexcept
    {
        return std::norm(t);
    }

    /**
     * \brief L-infinity norm of a complex number.
     */
    template <class T>
    inline auto norm_linf(const std::complex<T>& t) noexcept
    {
        return std::max(std::abs(t.real()), std::abs(t.imag()));
    }

    /**
     * \brief p-th power of the Lp norm of a complex number.
     */
    template <class T>
    inline auto norm_lp_to_p(const std::complex<T>& t, double p) noexcept
    {
        using rt = decltype(std::pow(std::abs(t.real()), static_cast<T>(p)));
        return p == 0
            ? static_cast<rt>(t.real() != 0 || t.imag() != 0)
            : std::pow(std::abs(t.real()), static_cast<T>(p)) +
                std::pow(std::abs(t.imag()), static_cast<T>(p));
    }

    /**
     * \brief Lp norm of a complex number.
     */
    template <class T>
    inline auto norm_lp(const std::complex<T>& t, double p) noexcept
    {
        return p == 0
            ? norm_lp_to_p(t, p)
            : std::pow(norm_lp_to_p(t, p), 1.0 / p);
    }

    /***********************************
     * norm functions for xexpressions *
     ***********************************/

#ifdef X_OLD_CLANG
#define XTENSOR_NORM_FUNCTION_AXES(NAME)                                         \
    template <class E, class I>                                                  \
    inline auto NAME(E&& e, std::initializer_list<I> axes) noexcept              \
    {                                                                            \
        using axes_type = std::vector<typename std::decay_t<E>::size_type>;      \
        return NAME(std::forward<E>(e), xtl::forward_sequence<axes_type>(axes)); \
    }

#else
#define XTENSOR_NORM_FUNCTION_AXES(NAME)                                         \
    template <class E, class I, std::size_t N>                                   \
    inline auto NAME(E&& e, const I(&axes)[N]) noexcept                          \
    {                                                                            \
        using axes_type = std::array<typename std::decay_t<E>::size_type, N>;    \
        return NAME(std::forward<E>(e), xtl::forward_sequence<axes_type>(axes)); \
    }
#endif


#define XTENSOR_EMPTY
#define XTENSOR_COMMA ,
#define XTENSOR_NORM_FUNCTION(NAME, RESULT_TYPE, REDUCE_EXPR, REDUCE_OP, MERGE_FUNC) \
    template <class E, class X>                                                      \
    inline auto NAME(E&& e, X&& axes) noexcept                                       \
    {                                                                                \
        using value_type = typename std::decay_t<E>::value_type;                     \
        using result_type = RESULT_TYPE;                                             \
                                                                                     \
        auto reduce_func = [](result_type const& r, value_type const& v) {           \
            return REDUCE_EXPR(r REDUCE_OP NAME(v));                                 \
        };                                                                           \
        auto init_func = [](value_type const& v) {                                   \
            return NAME(v);                                                          \
        };                                                                           \
        return reduce(make_xreducer_functor(std::move(reduce_func),                  \
                                            std::move(init_func),                    \
                                            MERGE_FUNC<result_type>()),              \
                      std::forward<E>(e), std::forward<X>(axes));                    \
    }                                                                                \
                                                                                     \
    template <class E, XTENSOR_REQUIRE<is_xexpression<E>::value>>                    \
    inline auto NAME(E&& e) noexcept                                                 \
    {                                                                                \
        return NAME(std::forward<E>(e), arange(e.dimension()));                      \
    }                                                                                \
    XTENSOR_NORM_FUNCTION_AXES(NAME)

    XTENSOR_NORM_FUNCTION(norm_l0, unsigned long long, XTENSOR_EMPTY, +, std::plus)
    XTENSOR_NORM_FUNCTION(norm_l1, big_promote_type_t<value_type>, XTENSOR_EMPTY, +, std::plus)
    XTENSOR_NORM_FUNCTION(norm_sq, big_promote_type_t<value_type>, XTENSOR_EMPTY, +, std::plus)
    XTENSOR_NORM_FUNCTION(norm_linf, decltype(norm_linf(std::declval<value_type>())), std::max<result_type>, XTENSOR_COMMA, math::maximum)

#undef XTENSOR_EMPTY
#undef XTENSOR_COMMA
#undef XTENSOR_NORM_FUNCTION
#undef XTENSOR_NORM_FUNCTION_AXES
    /// @endcond
    /**
     * @ingroup red_functions
     * @brief L0 (count) pseudo-norm of an array-like argument over given axes.
     *
     * Returns an \ref xreducer for the L0 pseudo-norm of the elements across given \em axes.
     * @param e an \ref xexpression
     * @param axes the axes along which the norm is computed (optional)
     * @return an \ref xreducer
     * When no axes are provided, the norm is calculated over the entire array. In this case,
     * the reducer represents a scalar result, otherwise an array of appropriate dimension.
     */
    template <class E, class X>
    auto norm_l0(E&& e, X&& axes) noexcept;

    /**
     * @ingroup red_functions
     * @brief L1 norm of an array-like argument over given axes.
     *
     * Returns an \ref xreducer for the L1 norm of the elements across given \em axes.
     * @param e an \ref xexpression
     * @param axes the axes along which the norm is computed (optional)
     * @return an \ref xreducer
     * When no axes are provided, the norm is calculated over the entire array. In this case,
     * the reducer represents a scalar result, otherwise an array of appropriate dimension.
     */
    template <class E, class X>
    auto norm_l1(E&& e, X&& axes) noexcept;

    /**
     * @ingroup red_functions
     * @brief Squared L2 norm of an array-like argument over given axes.
     *
     * Returns an \ref xreducer for the squared L2 norm of the elements across given \em axes.
     * @param e an \ref xexpression
     * @param axes the axes along which the norm is computed (optional)
     * @return an \ref xreducer
     * When no axes are provided, the norm is calculated over the entire array. In this case,
     * the reducer represents a scalar result, otherwise an array of appropriate dimension.
     */
    template <class E, class X>
    auto norm_sq(E&& e, X&& axes) noexcept;

    /**
     * @ingroup red_functions
     * @brief L2 norm of a scalar or array-like argument.
     *
     *  For scalar types: implemented as <tt>abs(t)</tt><br>
     *  otherwise: implemented as <tt>sqrt(norm_sq(t))</tt>.
    */
    template <class E>
    inline auto norm_l2(E&& e) noexcept
    {
        using std::sqrt;
        return sqrt(norm_sq(std::forward<E>(e)));
    }

    /**
     * @ingroup red_functions
     * @brief L2 norm of an array-like argument over given axes.
     *
     * Returns an \ref xreducer for the L2 norm of the elements across given \em axes.
     * @param e an \ref xexpression
     * @param axes the axes along which the norm is computed
     * @return an \ref xreducer (specifically: <tt>sqrt(norm_sq(e, axes))</tt>)
    */
    template <class E, class X>
    inline auto norm_l2(E&& e, X&& axes) noexcept
    {
        return sqrt(norm_sq(std::forward<E>(e), std::forward<X>(axes)));
    }

    /**
     * @ingroup red_functions
     * @brief Infinity (maximum) norm of an array-like argument over given axes.
     *
     * Returns an \ref xreducer for the infinity norm of the elements across given \em axes.
     * @param e an \ref xexpression
     * @param axes the axes along which the norm is computed (optional)
     * @return an \ref xreducer
     * When no axes are provided, the norm is calculated over the entire array. In this case,
     * the reducer represents a scalar result, otherwise an array of appropriate dimension.
     */
    template <class E, class X>
    auto norm_linf(E&& e, X&& axes) noexcept;

    /**
     * @ingroup red_functions
     * @brief p-th power of the Lp norm of an array-like argument over given axes.
     *
     * Returns an \ref xreducer for the p-th power of the Lp norm of the elements across given \em axes.
     * @param e an \ref xexpression
     * @param p
     * @param axes the axes along which the norm is computed (optional)
     * @return an \ref xreducer
     * When no axes are provided, the norm is calculated over the entire array. In this case,
     * the reducer represents a scalar result, otherwise an array of appropriate dimension.
     */
    template <class E, class X>
    inline auto norm_lp_to_p(E&& e, double p, X&& axes) noexcept
    {
        using value_type = typename std::decay_t<E>::value_type;
        using result_type = norm_type_t<std::decay_t<E>>;

        auto reduce_func = [p](result_type const& r, value_type const& v) {
            return r + norm_lp_to_p(v, p);
        };

        auto init_func = [p](value_type const& v) {
            return norm_lp_to_p(v, p);
        };
        return reduce(make_xreducer_functor(std::move(reduce_func), std::move(init_func), std::plus<result_type>()),
                      std::forward<E>(e), std::forward<X>(axes));
    }

    template <class E, XTENSOR_REQUIRE<is_xexpression<E>::value>>
    inline auto norm_lp_to_p(E&& e, double p) noexcept
    {
        return norm_lp_to_p(std::forward<E>(e), p, arange(e.dimension()));
    }

#ifdef X_OLD_CLANG
    template <class E, class I>
    inline auto norm_lp_to_p(E&& e, double p, std::initializer_list<I> axes) noexcept
    {
        using axes_type = std::vector<typename std::decay_t<E>::size_type>;
        return norm_lp_to_p(std::forward<E>(e), p, xtl::forward_sequence<axes_type>(axes));
    }
#else
    template <class E, class I, std::size_t N>
    inline auto norm_lp_to_p(E&& e, double p, const I (&axes)[N]) noexcept
    {
        using axes_type = std::array<typename std::decay_t<E>::size_type, N>;
        return norm_lp_to_p(std::forward<E>(e), p, xtl::forward_sequence<axes_type>(axes));
    }
#endif

    /**
     * @ingroup red_functions
     * @brief Lp norm of an array-like argument over given axes.
     *
     * Returns an \ref xreducer for the Lp norm (p != 0) of the elements across given \em axes.
     * @param e an \ref xexpression
     * @param p
     * @param axes the axes along which the norm is computed (optional)
     * @return an \ref xreducer
     * When no axes are provided, the norm is calculated over the entire array. In this case,
     * the reducer represents a scalar result, otherwise an array of appropriate dimension.
     */
    template <class E, class X>
    inline auto norm_lp(E&& e, double p, X&& axes)
    {
        XTENSOR_PRECONDITION(p != 0,
                             "norm_lp(): p must be nonzero, use norm_l0() instead.");
        return pow(norm_lp_to_p(std::forward<E>(e), p, std::forward<X>(axes)), 1.0 / p);
    }

    template <class E, XTENSOR_REQUIRE<is_xexpression<E>::value>>
    inline auto norm_lp(E&& e, double p) noexcept
    {
        return norm_lp(std::forward<E>(e), p, arange(e.dimension()));
    }

#ifdef X_OLD_CLANG
    template <class E, class I>
    inline auto norm_lp(E&& e, double p, std::initializer_list<I> axes) noexcept
    {
        using axes_type = std::vector<typename std::decay_t<E>::size_type>;
        return norm_lp(std::forward<E>(e), p, xtl::forward_sequence<axes_type>(axes));
    }
#else
    template <class E, class I, std::size_t N>
    inline auto norm_lp(E&& e, double p, const I (&axes)[N]) noexcept
    {
        using axes_type = std::array<typename std::decay_t<E>::size_type, N>;
        return norm_lp(std::forward<E>(e), p, xtl::forward_sequence<axes_type>(axes));
    }
#endif

    /**
     * @ingroup red_functions
     * @brief Induced L1 norm of a matrix.
     *
     * Returns an \ref xreducer for the induced L1 norm (i.e. the maximum of the L1 norms of e's columns).
     * @param e a 2D \ref xexpression
     * @return an \ref xreducer
     */
    template <class E, XTENSOR_REQUIRE<is_xexpression<E>::value>>
    inline auto norm_induced_l1(E&& e)
    {
        XTENSOR_PRECONDITION(e.dimension() == 2,
                             "norm_induced_l1(): only applicable to matrices (e.dimension() must be 2).");
        return norm_linf(norm_l1(std::forward<E>(e), {0}));
    }

    /**
     * @ingroup red_functions
     * @brief Induced L-infinity norm of a matrix.
     *
     * Returns an \ref xreducer for the induced L-infinity norm (i.e. the maximum of the L1 norms of e's rows).
     * @param e a 2D \ref xexpression
     * @return an \ref xreducer
     */
    template <class E, XTENSOR_REQUIRE<is_xexpression<E>::value>>
    inline auto norm_induced_linf(E&& e)
    {
        XTENSOR_PRECONDITION(e.dimension() == 2,
                             "norm_induced_linf(): only applicable to matrices (e.dimension() must be 2).");
        return norm_linf(norm_l1(std::forward<E>(e), {1}));
    }

}  // namespace xt

#endif
