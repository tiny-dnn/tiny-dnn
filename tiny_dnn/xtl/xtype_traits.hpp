/***************************************************************************
* Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XTL_TYPE_TRAITS_HPP
#define XTL_TYPE_TRAITS_HPP

#include <type_traits>

#include "xtl_config.hpp"

namespace xtl
{
    /************
     * apply_cv *
     ************/

    namespace detail
    {
        template <class T, class U, bool = std::is_const<std::remove_reference_t<T>>::value,
                  bool = std::is_volatile<std::remove_reference_t<T>>::value>
        struct apply_cv_impl
        {
            using type = U;
        };

        template <class T, class U>
        struct apply_cv_impl<T, U, true, false>
        {
            using type = const U;
        };

        template <class T, class U>
        struct apply_cv_impl<T, U, false, true>
        {
            using type = volatile U;
        };

        template <class T, class U>
        struct apply_cv_impl<T, U, true, true>
        {
            using type = const volatile U;
        };

        template <class T, class U>
        struct apply_cv_impl<T&, U, false, false>
        {
            using type = U&;
        };

        template <class T, class U>
        struct apply_cv_impl<T&, U, true, false>
        {
            using type = const U&;
        };

        template <class T, class U>
        struct apply_cv_impl<T&, U, false, true>
        {
            using type = volatile U&;
        };

        template <class T, class U>
        struct apply_cv_impl<T&, U, true, true>
        {
            using type = const volatile U&;
        };
    }

    template <class T, class U>
    struct apply_cv
    {
        using type = typename detail::apply_cv_impl<T, U>::type;
    };

    template <class T, class U>
    using apply_cv_t = typename apply_cv<T, U>::type;

    /****************************************************************
     * C++17 logical operators (disjunction, conjunction, negation) *
     ****************************************************************/

    /********************
     * disjunction - or *
     ********************/

    template <class...>
    struct disjunction;

    template <>
    struct disjunction<> : std::false_type
    {
    };

    template <class Arg>
    struct disjunction<Arg> : Arg
    {
    };

    template <class Arg1, class Arg2, class... Args>
    struct disjunction<Arg1, Arg2, Args...> : std::conditional_t<Arg1::value, Arg1, disjunction<Arg2, Args...>>
    {
    };

    /*********************
     * conjunction - and *
     *********************/

    template <class...>
    struct conjunction;

    template <>
    struct conjunction<> : std::true_type
    {
    };

    template <class Arg1>
    struct conjunction<Arg1> : Arg1
    {
    };

    template <class Arg1, class Arg2, class... Args>
    struct conjunction<Arg1, Arg2, Args...> : std::conditional_t<Arg1::value, conjunction<Arg2, Args...>, Arg1>
    {
    };

    /******************
     * negation - and *
     ******************/

    template <class Arg>
    struct negation : std::integral_constant<bool, !Arg::value>
    {
    };
}

#endif
