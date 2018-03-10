/***************************************************************************
* Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XTENSOR_SIMD_HPP
#define XTENSOR_SIMD_HPP

#include <vector>

#include "xstorage.hpp"

#ifdef XTENSOR_USE_XSIMD

#include "xsimd/xsimd.hpp"

#else  // XTENSOR_USE_XSIMD

namespace xsimd
{
    struct aligned_mode
    {
    };

    struct unaligned_mode
    {
    };

    template <class A>
    struct allocator_alignment
    {
        using type = unaligned_mode;
    };

    template <class A>
    using allocator_alignment_t = typename allocator_alignment<A>::type;

    template <class C>
    struct container_alignment
    {
        using type = unaligned_mode;
    };

    template <class C>
    using container_alignment_t = typename container_alignment<C>::type;

    template <class T>
    struct simd_traits
    {
        using type = T;
        using bool_type = bool;
        static constexpr size_t size = 1;
    };

    template <class T>
    struct revert_simd_traits
    {
        using type = T;
        static constexpr size_t size = simd_traits<type>::size;
    };

    template <class T>
    using simd_type = typename simd_traits<T>::type;

    template <class T>
    using simd_bool_type = typename simd_traits<T>::bool_type;

    template <class T>
    using revert_simd_type = typename revert_simd_traits<T>::type;

    template <class T>
    inline simd_type<T> set_simd(const T& value)
    {
        return value;
    }

    template <class T>
    inline simd_type<T> load_simd(const T* src, aligned_mode)
    {
        return *src;
    }

    template <class T>
    inline simd_type<T> load_simd(const T* src, unaligned_mode)
    {
        return *src;
    }

    template <class T>
    inline void store_simd(T* dst, const simd_type<T>& src, aligned_mode)
    {
        *dst = src;
    }

    template <class T>
    inline void store_simd(T* dst, const simd_type<T>& src, unaligned_mode)
    {
        *dst = src;
    }

    template <class T>
    inline T select(bool cond, const T& t1, const T& t2)
    {
        return cond ? t1 : t2;
    }

    template <class T>
    inline std::size_t get_alignment_offset(const T* /*p*/, std::size_t size, std::size_t /*block_size*/)
    {
        return size;
    }
}

#endif  // XTENSOR_USE_XSIMD

namespace xt
{
    using xsimd::aligned_mode;
    using xsimd::unaligned_mode;

    struct inner_aligned_mode
    {
    };

    namespace detail
    {
        template <class A1, class A2>
        struct driven_align_mode_impl
        {
            using type = std::conditional_t<std::is_same<A1, A2>::value, A1, ::xsimd::unaligned_mode>;
        };

        template <class A>
        struct driven_align_mode_impl<inner_aligned_mode, A>
        {
            using type = A;
        };
    }

    template <class A1, class A2>
    struct driven_align_mode
    {
        using type = typename detail::driven_align_mode_impl<A1, A2>::type;
    };

    template <class A1, class A2>
    using driven_align_mode_t = typename detail::driven_align_mode_impl<A1, A2>::type;
}

#endif
