/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XTL_CRTP_HPP
#define XTL_CRTP_HPP

#include "xtl_config.hpp"

namespace xtl
{
    template <class T>
    auto derived_cast(T&& value)
    {
        return (std::forward<T>(value)).derived_cast();
    }
}

#endif
