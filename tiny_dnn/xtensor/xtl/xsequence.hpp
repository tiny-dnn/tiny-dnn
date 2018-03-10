/***************************************************************************
* Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XTL_SEQUENCE_HPP
#define XTL_SEQUENCE_HPP

#include <array>
#include <cstddef>
#include <type_traits>
#include <utility>
#include <vector>

#include "xtl_config.hpp"

namespace xtl
{
    template <class S>
    S make_sequence(typename S::size_type size, typename S::value_type v);

    template <class R, class A>
    decltype(auto) forward_sequence(A&& s);

    // equivalent to std::size(c) in c++17
    template <class C>
    constexpr auto sequence_size(const C& c) -> decltype(c.size());

    // equivalent to std::size(a) in c++17
    template <class T, std::size_t N>
    constexpr std::size_t sequence_size(const T (&a)[N]);

    /********************************
     * make_sequence implementation *
     ********************************/

    namespace detail
    {
        template <class S>
        struct sequence_builder
        {
            using value_type = typename S::value_type;
            using size_type = typename S::size_type;

            inline static S make(size_type size, value_type v)
            {
                return S(size, v);
            }
        };

        template <class T, std::size_t N>
        struct sequence_builder<std::array<T, N>>
        {
            using sequence_type = std::array<T, N>;
            using value_type = typename sequence_type::value_type;
            using size_type = typename sequence_type::size_type;

            inline static sequence_type make(size_type /*size*/, value_type v)
            {
                sequence_type s;
                s.fill(v);
                return s;
            }
        };
    }

    template <class S>
    inline S make_sequence(typename S::size_type size, typename S::value_type v)
    {
        return detail::sequence_builder<S>::make(size, v);
    }

    /***********************************
     * forward_sequence implementation *
     ***********************************/

    namespace detail
    {
        template <class R, class A, class E = void>
        struct sequence_forwarder
        {
            template <class T>
            static inline R forward(const T& r)
            {
                return R(std::begin(r), std::end(r));
            }
        };

        template <class I, std::size_t L, class A>
        struct sequence_forwarder<std::array<I, L>, A,
                                  std::enable_if_t<!std::is_same<std::array<I, L>, A>::value>>
        {
            using R = std::array<I, L>;

            template <class T>
            static inline R forward(const T& r)
            {
                R ret;
                std::copy(std::begin(r), std::end(r), std::begin(ret));
                return ret;
            }
        };

        template <class R>
        struct sequence_forwarder<R, R>
        {
            template <class T>
            static inline T&& forward(typename std::remove_reference<T>::type& t) noexcept
            {
                return static_cast<T&&>(t);
            }

            template <class T>
            static inline T&& forward(typename std::remove_reference<T>::type&& t) noexcept
            {
                return static_cast<T&&>(t);
            }
        };
    }

    template <class R, class A>
    inline decltype(auto) forward_sequence(A&& s)
    {
        using forwarder = detail::sequence_forwarder<
            std::decay_t<R>,
            std::remove_cv_t<std::remove_reference_t<A>>
        >;
        return forwarder::template forward<A>(s);
    }

    /********************************
     * sequence_size implementation *
     ********************************/

    // equivalent to std::size(c) in c++17
    template <class C>
    constexpr auto sequence_size(const C& c) -> decltype(c.size())
    {
        return c.size();
    }

    // equivalent to std::size(a) in c++17
    template <class T, std::size_t N>
    constexpr std::size_t sequence_size(const T (&)[N])
    {
        return N;
    }
}

#endif
