/***************************************************************************
* Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XTENSOR_CONFIG_HPP
#define XTENSOR_CONFIG_HPP

#define XTENSOR_VERSION_MAJOR 0
#define XTENSOR_VERSION_MINOR 15
#define XTENSOR_VERSION_PATCH 4

// DETECT 3.6 <= clang < 3.8 for compiler bug workaround.
#ifdef __clang__
    #if __clang_major__ == 3 && __clang_minor__ < 8
        #define X_OLD_CLANG
        #include <initializer_list>
        #include <vector>
    #endif
#endif

#ifndef DEFAULT_DATA_CONTAINER
#define DEFAULT_DATA_CONTAINER(T, A) uvector<T, A>
#endif

#ifndef DEFAULT_SHAPE_CONTAINER
#define DEFAULT_SHAPE_CONTAINER(T, EA, SA) \
    xt::svector<typename DEFAULT_DATA_CONTAINER(T, EA)::size_type, 4, SA>
#endif

#ifndef DEFAULT_ALLOCATOR
#ifdef XTENSOR_USE_XSIMD
#include "xsimd/xsimd.hpp"
#define DEFAULT_ALLOCATOR(T) \
    xsimd::aligned_allocator<T, XSIMD_DEFAULT_ALIGNMENT>
#else
#define DEFAULT_ALLOCATOR(T) \
    std::allocator<T>
#endif
#endif

#ifndef DEFAULT_LAYOUT
#define DEFAULT_LAYOUT layout_type::row_major
#endif

#endif
