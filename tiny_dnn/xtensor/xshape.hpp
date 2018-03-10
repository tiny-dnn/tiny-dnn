/***************************************************************************
* Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XTENSOR_XSHAPE_HPP
#define XTENSOR_XSHAPE_HPP

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <iterator>
#include <memory>

#include "xstorage.hpp"
#include "xexception.hpp"

namespace xt
{
    template <class T>
    using dynamic_shape = svector<T, 4>;

    template <class T, std::size_t N>
    using static_shape = std::array<T, N>;

    template <std::size_t... X>
    class fixed_shape;

    using xindex = dynamic_shape<std::size_t>;

    /*************************************
     * promote_shape and promote_strides *
     *************************************/

    namespace detail
    {
        template <class T1, class T2>
        constexpr std::common_type_t<T1, T2> imax(const T1& a, const T2& b)
        {
            return a > b ? a : b;
        }

        // Variadic meta-function returning the maximal size of std::arrays.
        template <class... T>
        struct max_array_size;

        template <>
        struct max_array_size<>
        {
            static constexpr std::size_t value = 0;
        };

        template <class T, class... Ts>
        struct max_array_size<T, Ts...> : std::integral_constant<std::size_t, imax(std::tuple_size<T>::value, max_array_size<Ts...>::value)>
        {
        };

        // Simple is_array and only_array meta-functions
        template <class S>
        struct is_array
        {
            static constexpr bool value = false;
        };

        template <class T, std::size_t N>
        struct is_array<std::array<T, N>>
        {
            static constexpr bool value = true;
        };

        template <class... S>
        using only_array = xtl::conjunction<is_array<S>...>;

        // The promote_index meta-function returns std::vector<promoted_value_type> in the
        // general case and an array of the promoted value type and maximal size if all
        // arguments are of type std::array

        template <bool A, class... S>
        struct promote_index_impl;

        template <class... S>
        struct promote_index_impl<false, S...>
        {
            using type = dynamic_shape<typename std::common_type<typename S::value_type...>::type>;
        };

        template <class... S>
        struct promote_index_impl<true, S...>
        {
            using type = std::array<typename std::common_type<typename S::value_type...>::type, max_array_size<S...>::value>;
        };

        template <>
        struct promote_index_impl<true>
        {
            using type = std::array<std::size_t, 0>;
        };

        template <class... S>
        struct promote_index
        {
            using type = typename promote_index_impl<only_array<S...>::value, S...>::type;
        };
    }

    template <class... S>
    using promote_shape_t = typename detail::promote_index<S...>::type;

    template <class... S>
    using promote_strides_t = typename detail::promote_index<S...>::type;

}

#endif
