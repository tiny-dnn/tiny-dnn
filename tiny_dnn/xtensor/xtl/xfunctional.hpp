/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XTL_FUNCTIONAL_HPP
#define XTL_FUNCTIONAL_HPP

#include <utility>

#include "xtl_config.hpp"

namespace xtl
{
    struct identity
    {
        template <class T>
        T&& operator()(T&& x) const
        {
            return std::forward<T>(x);
        }
    };
}
#endif

