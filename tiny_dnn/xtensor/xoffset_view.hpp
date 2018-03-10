/***************************************************************************
* Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XTENSOR_OFFSET_VIEW_HPP
#define XTENSOR_OFFSET_VIEW_HPP

#include "xtl/xcomplex.hpp"

#include "xtensor/xfunctor_view.hpp"

namespace xt
{
    namespace detail
    {
        template <class M, std::size_t I>
        struct offset_forwarder
        {
            using value_type = M;
            using reference = M&;
            using const_reference = const M&;
            using pointer = M*;
            using const_pointer = const M*;

            template <class T>
            decltype(auto) operator()(T&& t) const
            {
                return xtl::forward_offset<M, I>(t);
            }
        };
    }

    template <class CT, class M, std::size_t I>
    using xoffset_view = xfunctor_view<detail::offset_forwarder<M, I>, CT>;
}

#endif
