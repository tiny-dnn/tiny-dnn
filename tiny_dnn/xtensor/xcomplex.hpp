/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XCOMPLEX_HPP
#define XCOMPLEX_HPP

#include <type_traits>
#include <utility>

#include "xtensor/xbuilder.hpp"
#include "xtensor/xexpression.hpp"
#include "xtensor/xoffsetview.hpp"

namespace xt
{

    /******************************
     * real and imag declarations *
     ******************************/

    template <class E>
    decltype(auto) real(E&& e) noexcept;

    template <class E>
    decltype(auto) imag(E&& e) noexcept;

    /********************************
     * real and imag implementation *
     ********************************/

    namespace detail
    {
        template <bool iscomplex = true>
        struct complex_helper
        {
            template <class E>
            static inline auto real(E&& e) noexcept
            {
                using real_type = typename std::decay_t<E>::value_type::value_type;
                return xoffsetview<xclosure_t<E>, real_type, 0>(std::forward<E>(e));
            }

            template <class E>
            static inline auto imag(E&& e) noexcept
            {
                using real_type = typename std::decay_t<E>::value_type::value_type;
                return xoffsetview<xclosure_t<E>, real_type, sizeof(real_type)>(std::forward<E>(e));
            }
        };

        template <>
        struct complex_helper<false>
        {
            template <class E>
            static inline decltype(auto) real(E&& e) noexcept
            {
                return e;
            }

            template <class E>
            static inline auto imag(E&& e) noexcept
            {
                return zeros<typename std::decay_t<E>::value_type>(e.shape());
            }
        };

        template <bool isexpression = true>
        struct complex_expression_helper
        {
            template <class E>
            static inline auto real(E&& e) noexcept
            {
                return detail::complex_helper<is_complex<typename std::decay_t<E>::value_type>::value>::real(e);
            }

            template <class E>
            static inline auto imag(E&& e) noexcept
            {
                return detail::complex_helper<is_complex<typename std::decay_t<E>::value_type>::value>::imag(e);
            }
        };

        template <>
        struct complex_expression_helper<false>
        {
            template <class E>
            static inline decltype(auto) real(E&& e) noexcept
            {
                return forward_real(std::forward<E>(e));
            }

            template <class E>
            static inline decltype(auto) imag(E&& e) noexcept
            {
                return forward_imag(std::forward<E>(e));
            }
        };
    }

    /**
     * @brief Returns an \ref xexpression representing the real part of the given expression.
     *
     * @tparam e the \ref xexpression
     *
     * The returned expression either hold a const reference to \p e or a copy
     * depending on whether \p e is an lvalue or an rvalue.
     */
    template <class E>
    inline decltype(auto) real(E&& e) noexcept
    {
        return detail::complex_expression_helper<is_xexpression<std::decay_t<E>>::value>::real(std::forward<E>(e));
    }

    /**
     * @brief Returns an \ref xexpression representing the imaginary part of the given expression.
     *
     * @tparam e the \ref xexpression
     *
     * The returned expression either hold a const reference to \p e or a copy
     * depending on whether \p e is an lvalue or an rvalue.
     */
    template <class E>
    inline decltype(auto) imag(E&& e) noexcept
    {
        return detail::complex_expression_helper<is_xexpression<std::decay_t<E>>::value>::imag(std::forward<E>(e));
    }
}
#endif
