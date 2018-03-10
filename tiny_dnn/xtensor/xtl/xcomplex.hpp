/***************************************************************************
* Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XTL_COMPLEX_HPP
#define XTL_COMPLEX_HPP

#include <complex>
#include <cstddef>
#include <type_traits>
#include <utility>

#include "xtl_config.hpp"
#include "xtype_traits.hpp"

namespace xtl
{
    /******************************
     * real and imag declarations *
     ******************************/

    template <class E>
    decltype(auto) real(E&& e) noexcept;

    template <class E>
    decltype(auto) imag(E&& e) noexcept;

    /*****************************
     * is_complex implementation *
     *****************************/

    namespace detail
    {
        template <class T>
        struct is_complex : public std::false_type
        {
        };

        template <class T>
        struct is_complex<std::complex<T>> : public std::true_type
        {
        };
    }

    template <class T>
    struct is_complex
    {
        static constexpr bool value = detail::is_complex<std::decay_t<T>>::value;
    };

    /*************************************
     * complex_value_type implementation *
     *************************************/

    template <class T>
    struct complex_value_type
    {
        using type = T;
    };

    template <class T>
    struct complex_value_type<std::complex<T>>
    {
        using type = T;
    };

    template <class T>
    using complex_value_type_t = typename complex_value_type<T>::type;

    /*********************************
     * forward_offset implementation *
     *********************************/

    namespace detail
    {

        template <class T, class M>
        struct forward_type
        {
            using type = apply_cv_t<T, M>&&;
        };

        template <class T, class M>
        struct forward_type<T&, M>
        {
            using type = apply_cv_t<T, M>&;
        };

        template <class T, class M>
        using forward_type_t = typename forward_type<T, M>::type;
    }

    template <class M, std::size_t I, class T>
    constexpr detail::forward_type_t<T, M> forward_offset(T&& v) noexcept
    {
        using forward_type = detail::forward_type_t<T, M>;
        using cv_value_type = std::remove_reference_t<forward_type>;
        using byte_type = apply_cv_t<std::remove_reference_t<T>, char>;

        return static_cast<forward_type>(
            *reinterpret_cast<cv_value_type*>(
                reinterpret_cast<byte_type*>(&v) + I
            )
        );
    }

    /**********************************************
     * forward_real & forward_imag implementation *
     **********************************************/

    // forward_real

    template <class T>
    auto forward_real(T&& v)
        -> std::enable_if_t<!is_complex<T>::value, detail::forward_type_t<T, T>>  // real case -> forward
    {
        return static_cast<detail::forward_type_t<T, T>>(v);
    }

    template <class T>
    auto forward_real(T&& v)
        -> std::enable_if_t<is_complex<T>::value, detail::forward_type_t<T, typename std::decay_t<T>::value_type>>  // complex case -> forward the real part
    {
        return forward_offset<typename std::decay_t<T>::value_type, 0>(v);
    }

    // forward_imag

    template <class T>
    auto forward_imag(T &&)
        -> std::enable_if_t<!is_complex<T>::value, std::decay_t<T>>  // real case -> always return 0 by value
    {
        return 0;
    }

    template <class T>
    auto forward_imag(T&& v)
        -> std::enable_if_t<is_complex<T>::value, detail::forward_type_t<T, typename std::decay_t<T>::value_type>>  // complex case -> forwards the imaginary part
    {
        using real_type = typename std::decay_t<T>::value_type;
        return forward_offset<real_type, sizeof(real_type)>(v);
    }

    /******************************
     * real & imag implementation *
     ******************************/

    template <class E>
    inline decltype(auto) real(E&& e) noexcept
    {
        return forward_real(std::forward<E>(e));
    }

    template <class E>
    inline decltype(auto) imag(E&& e) noexcept
    {
        return forward_imag(std::forward<E>(e));
    }

    /******************************************************
     * operator overloads for complex and closure wrapper *
     *****************************************************/

    template <class C, class T, std::enable_if_t<!xtl::is_complex<T>::value, int> = 0>
    std::complex<C> operator+(const std::complex<C>& c, const T& t)
    {
        std::complex<C> result(c);
        result += t;
        return result;
    }

    template <class C, class T, std::enable_if_t<!xtl::is_complex<T>::value, int> = 0>
    std::complex<C> operator+(const T& t, const std::complex<C>& c)
    {
        std::complex<C> result(t);
        result += c;
        return result;
    }

    template <class C, class T, std::enable_if_t<!xtl::is_complex<T>::value, int> = 0>
    std::complex<C> operator-(const std::complex<C>& c, const T& t)
    {
        std::complex<C> result(c);
        result -= t;
        return result;
    }

    template <class C, class T, std::enable_if_t<!xtl::is_complex<T>::value, int> = 0>
    std::complex<C> operator-(const T& t, const std::complex<C>& c)
    {
        std::complex<C> result(t);
        result -= c;
        return result;
    }

    template <class C, class T, std::enable_if_t<!xtl::is_complex<T>::value, int> = 0>
    std::complex<C> operator*(const std::complex<C>& c, const T& t)
    {
        std::complex<C> result(c);
        result *= t;
        return result;
    }

    template <class C, class T, std::enable_if_t<!xtl::is_complex<T>::value, int> = 0>
    std::complex<C> operator*(const T& t, const std::complex<C>& c)
    {
        std::complex<C> result(t);
        result *= c;
        return result;
    }

    template <class C, class T, std::enable_if_t<!xtl::is_complex<T>::value, int> = 0>
    std::complex<C> operator/(const std::complex<C>& c, const T& t)
    {
        std::complex<C> result(c);
        result /= t;
        return result;
    }

    template <class C, class T, std::enable_if_t<!xtl::is_complex<T>::value, int> = 0>
    std::complex<C> operator/(const T& t, const std::complex<C>& c)
    {
        std::complex<C> result(t);
        result /= c;
        return result;
    }
}

#endif
