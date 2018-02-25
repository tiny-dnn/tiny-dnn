/***************************************************************************
* Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XTENSOR_ADAPT_HPP
#define XTENSOR_ADAPT_HPP

#include <array>
#include <cstddef>
#include <memory>
#include <type_traits>

#include "xtl/xsequence.hpp"

#include "xarray.hpp"
#include "xtensor.hpp"

namespace xt
{

    namespace detail
    {
        template <class>
        struct array_size_impl;

        template <class T, std::size_t N>
        struct array_size_impl<std::array<T, N>>
        {
            static constexpr std::size_t value = N;
        };

        template <class C>
        using array_size = array_size_impl<std::decay_t<C>>;
    }

    /**************************
     * xarray_adaptor builder *
     **************************/

    /**
     * Constructs an xarray_adaptor of the given stl-like container,
     * with the specified shape and layout.
     * @param container the container to adapt
     * @param shape the shape of the xarray_adaptor
     * @param l the layout_type of the xarray_adaptor
     */
    template <class C, class SC, layout_type L = DEFAULT_LAYOUT, typename std::enable_if_t<!detail::is_array<std::decay_t<SC>>::value, int> = 0>
    xarray_adaptor<xtl::closure_type_t<C>, L, std::decay_t<SC>>
    adapt(C&& container, const SC& shape, layout_type l = L);

    /**
     * Constructs an xarray_adaptor of the given stl-like container,
     * with the specified shape and strides.
     * @param container the container to adapt
     * @param shape the shape of the xarray_adaptor
     * @param strides the strides of the xarray_adaptor
     */
    template <class C, class SC, class SS, typename std::enable_if_t<!detail::is_array<std::decay_t<SC>>::value, int> = 0>
    xarray_adaptor<xtl::closure_type_t<C>, layout_type::dynamic, std::decay_t<SC>>
    adapt(C&& container, SC&& shape, SS&& strides);

    /**
     * Constructs an xarray_adaptor of the given dynamically allocated C array,
     * with the specified shape and layout.
     * @param pointer the pointer to the beginning of the dynamic array
     * @param size the size of the dynamic array
     * @param ownership indicates whether the adaptor takes ownership of the array.
     *        Possible values are ``no_ownerhsip()`` or ``acquire_ownership()``
     * @param shape the shape of the xarray_adaptor
     * @param l the layout_type of the xarray_adaptor
     * @param alloc the allocator used for allocating / deallocating the dynamic array
     */
    template <class P, class O, class SC, layout_type L = DEFAULT_LAYOUT, class A = std::allocator<std::remove_pointer_t<std::remove_reference_t<P>>>, typename std::enable_if_t<!detail::is_array<std::decay_t<SC>>::value, int> = 0>
    xarray_adaptor<xbuffer_adaptor<xtl::closure_type_t<P>, O, A>, L, SC>
    adapt(P&& pointer, typename A::size_type size, O ownership, const SC& shape, layout_type l = L, const A& alloc = A());

    /**
     * Constructs an xarray_adaptor of the given dynamically allocated C array,
     * with the specified shape and layout.
     * @param pointer the pointer to the beginning of the dynamic array
     * @param size the size of the dynamic array
     * @param ownership indicates whether the adaptor takes ownership of the array.
     *        Possible values are ``no_ownerhsip()`` or ``acquire_ownership()``
     * @param shape the shape of the xarray_adaptor
     * @param strides the strides of the xarray_adaptor
     * @param alloc the allocator used for allocating / deallocating the dynamic array
    */
    template <class P, class O, class SC, class SS, class A = std::allocator<std::remove_pointer_t<std::remove_reference_t<P>>>, typename std::enable_if_t<!detail::is_array<std::decay_t<SC>>::value, int> = 0>
    xarray_adaptor<xbuffer_adaptor<xtl::closure_type_t<P>, O, A>, layout_type::dynamic, std::decay_t<SC>>
    adapt(P&& pointer, typename A::size_type size, O ownership, SC&& shape, SS&& strides, const A& alloc = A());

    /***************************
     * xtensor_adaptor builder *
     ***************************/

    /**
     * Constructs a 1D xtensor_adaptor of the given stl-like container,
     * with the specified layout_type.
     * @param container the container to adapt
     * @param l the layout_type of the xtensor_adaptor
     */
    template <class C, layout_type L = DEFAULT_LAYOUT>
    xtensor_adaptor<C, 1, L>
    adapt(C&& container, layout_type l = L);
    
    /**
     * Constructs an xtensor_adaptor of the given stl-like container,
     * with the specified shape and layout_type.
     * @param container the container to adapt
     * @param shape the shape of the xtensor_adaptor
     * @param l the layout_type of the xtensor_adaptor
     */
    template <class C, class SC, layout_type L = DEFAULT_LAYOUT, typename std::enable_if_t<detail::is_array<std::decay_t<SC>>::value, int> = 0>
    xtensor_adaptor<C, detail::array_size<SC>::value, L>
    adapt(C&& container, const SC& shape, layout_type l = L);

    /**
     * Constructs an xtensor_adaptor of the given stl-like container,
     * with the specified shape and strides.
     * @param container the container to adapt
     * @param shape the shape of the xtensor_adaptor
     * @param strides the strides of the xtensor_adaptor
     */
    template <class C, class SC, class SS, typename std::enable_if_t<detail::is_array<std::decay_t<SC>>::value, int> = 0>
    xtensor_adaptor<C, detail::array_size<SC>::value, layout_type::dynamic>
    adapt(C&& container, SC&& shape, SS&& strides);

    /**
     * Constructs a 1D xtensor_adaptor of the given dynamically allocated C array,
     * with the specified layout.
     * @param pointer the pointer to the beginning of the dynamic array
     * @param size the size of the dynamic array
     * @param ownership indicates whether the adaptor takes ownership of the array.
     *        Possible values are ``no_ownerhsip()`` or ``acquire_ownership()``
     * @param l the layout_type of the xtensor_adaptor
     * @param alloc the allocator used for allocating / deallocating the dynamic array
     */
    template <class P, class O, layout_type L = DEFAULT_LAYOUT, class A = std::allocator<std::remove_pointer_t<std::remove_reference_t<P>>>>
    xtensor_adaptor<xbuffer_adaptor<xtl::closure_type_t<P>, O, A>, 1, L>
    adapt(P&& pointer, typename A::size_type size, O ownership, layout_type l = L, const A& alloc = A());

    /**
     * Constructs an xtensor_adaptor of the given dynamically allocated C array,
     * with the specified shape and layout.
     * @param pointer the pointer to the beginning of the dynamic array
     * @param size the size of the dynamic array
     * @param ownership indicates whether the adaptor takes ownership of the array.
     *        Possible values are ``no_ownerhsip()`` or ``acquire_ownership()``
     * @param shape the shape of the xtensor_adaptor
     * @param l the layout_type of the xtensor_adaptor
     * @param alloc the allocator used for allocating / deallocating the dynamic array
     */
    template <class P, class O, class SC, layout_type L = DEFAULT_LAYOUT, class A = std::allocator<std::remove_pointer_t<std::remove_reference_t<P>>>, typename std::enable_if_t<detail::is_array<std::decay_t<SC>>::value, int> = 0>
    xtensor_adaptor<xbuffer_adaptor<xtl::closure_type_t<P>, O, A>, detail::array_size<SC>::value, L>
    adapt(P&& pointer, typename A::size_type size, O ownership, const SC& shape, layout_type l = L, const A& alloc = A());

    /**
     * Constructs an xtensor_adaptor of the given dynamically allocated C array,
     * with the specified shape and strides.
     * @param pointer the pointer to the beginning of the dynamic array
     * @param size the size of the dynamic array
     * @param ownership indicates whether the adaptor takes ownership of the array.
     *        Possible values are ``no_ownerhsip()`` or ``acquire_ownership()``
     * @param shape the shape of the xtensor_adaptor
     * @param strides the strides of the xtensor_adaptor
     * @param alloc the allocator used for allocating / deallocating the dynamic array
     */
    template <class P, class O, class SC, class SS, class A = std::allocator<std::remove_pointer_t<std::remove_reference_t<P>>>, typename std::enable_if_t<detail::is_array<std::decay_t<SC>>::value, int> = 0>
    xtensor_adaptor<xbuffer_adaptor<xtl::closure_type_t<P>, O, A>, detail::array_size<SC>::value, layout_type::dynamic>
    adapt(P&& pointer, typename A::size_type size, O ownership, SC&& shape, SS&& strides, const A& alloc = A());

    /*****************************************
     * xarray_adaptor builder implementation *
     *****************************************/

    // shape only - container version
    template <class C, class SC, layout_type L, typename std::enable_if_t<!detail::is_array<std::decay_t<SC>>::value, int>>
    inline xarray_adaptor<xtl::closure_type_t<C>, L, std::decay_t<SC>>
    adapt(C&& container, const SC& shape, layout_type l)
    {
        using return_type = xarray_adaptor<xtl::closure_type_t<C>, L, std::decay_t<SC>>;
        return return_type(std::forward<C>(container), shape, l);
    }

    // shape and strides - container version
    template <class C, class SC, class SS, typename std::enable_if_t<!detail::is_array<std::decay_t<SC>>::value, int>>
    inline xarray_adaptor<xtl::closure_type_t<C>, layout_type::dynamic, std::decay_t<SC>>
    adapt(C&& container, SC&& shape, SS&& strides)
    {
        using return_type = xarray_adaptor<xtl::closure_type_t<C>, layout_type::dynamic, std::decay_t<SC>>;
        return return_type(std::forward<C>(container),
                           xtl::forward_sequence<typename return_type::inner_shape_type>(shape),
                           xtl::forward_sequence<typename return_type::inner_strides_type>(strides));
    }

    // shape only - buffer version
    template <class P, class O, class SC, layout_type L, class A, typename std::enable_if_t<!detail::is_array<std::decay_t<SC>>::value, int>>
    inline xarray_adaptor<xbuffer_adaptor<xtl::closure_type_t<P>, O, A>, L, SC>
    adapt(P&& pointer, typename A::size_type size, O, const SC& shape, layout_type l, const A& alloc)
    {
        using buffer_type = xbuffer_adaptor<xtl::closure_type_t<P>, O, A>;
        using return_type = xarray_adaptor<buffer_type, L, SC>;
        buffer_type buf(std::forward<P>(pointer), size, alloc);
        return return_type(std::move(buf), shape, l);
    }

    // shape and strides - buffer version
    template <class P, class O, class SC, class SS, class A, typename std::enable_if_t<!detail::is_array<std::decay_t<SC>>::value, int>>
    inline xarray_adaptor<xbuffer_adaptor<xtl::closure_type_t<P>, O, A>, layout_type::dynamic, std::decay_t<SC>>
    adapt(P&& pointer, typename A::size_type size, O, SC&& shape, SS&& strides, const A& alloc)
    {
        using buffer_type = xbuffer_adaptor<xtl::closure_type_t<P>, O, A>;
        using return_type = xarray_adaptor<buffer_type, layout_type::dynamic, std::decay_t<SC>>;
        buffer_type buf(std::forward<P>(pointer), size, alloc);
        return return_type(std::move(buf),
                           xtl::forward_sequence<typename return_type::inner_shape_type>(shape),
                           xtl::forward_sequence<typename return_type::inner_strides_type>(strides));
    }

    /******************************************
     * xtensor_adaptor builder implementation *
     ******************************************/

    // 1-D case - container version
    template <class C, layout_type L>
    inline xtensor_adaptor<C, 1, L>
    adapt(C&& container, layout_type l)
    {
        const std::array<typename std::decay_t<C>::size_type, 1> shape{container.size()};
        using return_type = xtensor_adaptor<xtl::closure_type_t<C>, 1, L>;
        return return_type(std::forward<C>(container), shape, l);
    }

    // shape only - container version
    template <class C, class SC, layout_type L, typename std::enable_if_t<detail::is_array<std::decay_t<SC>>::value, int>>
    inline xtensor_adaptor<C, detail::array_size<SC>::value, L>
    adapt(C&& container, const SC& shape, layout_type l)
    {
        constexpr std::size_t N = detail::array_size<SC>::value;
        using return_type = xtensor_adaptor<xtl::closure_type_t<C>, N, L>;
        return return_type(std::forward<C>(container), shape, l);
    }

    // shape and strides - container version
    template <class C, class SC, class SS, typename std::enable_if_t<detail::is_array<std::decay_t<SC>>::value, int>>
    inline xtensor_adaptor<C, detail::array_size<SC>::value, layout_type::dynamic>
    adapt(C&& container, SC&& shape, SS&& strides)
    {
        constexpr std::size_t N = detail::array_size<SC>::value;
        using return_type = xtensor_adaptor<xtl::closure_type_t<C>, N, layout_type::dynamic>;
        return return_type(std::forward<C>(container),
                           xtl::forward_sequence<typename return_type::inner_shape_type>(shape),
                           xtl::forward_sequence<typename return_type::inner_strides_type>(strides));
    }

    // 1-D case - buffer version
    template <class P, class O, layout_type L, class A>
    inline xtensor_adaptor<xbuffer_adaptor<xtl::closure_type_t<P>, O, A>, 1, L>
    adapt(P&& pointer, typename A::size_type size, O, layout_type l, const A& alloc)
    {
        using buffer_type = xbuffer_adaptor<xtl::closure_type_t<P>, O, A>;
        using return_type = xtensor_adaptor<buffer_type, 1, L>;
        buffer_type buf(std::forward<P>(pointer), size, alloc);
        const std::array<typename A::size_type, 1> shape{size};
        return return_type(std::move(buf), shape, l);
    }

    // shape only - buffer version
    template <class P, class O, class SC, layout_type L, class A, typename std::enable_if_t<detail::is_array<std::decay_t<SC>>::value, int>>
    inline xtensor_adaptor<xbuffer_adaptor<xtl::closure_type_t<P>, O, A>, detail::array_size<SC>::value, L>
    adapt(P&& pointer, typename A::size_type size, O, const SC& shape, layout_type l, const A& alloc)
    {
        using buffer_type = xbuffer_adaptor<xtl::closure_type_t<P>, O, A>;
        constexpr std::size_t N = detail::array_size<SC>::value;
        using return_type = xtensor_adaptor<buffer_type, N, L>;
        buffer_type buf(std::forward<P>(pointer), size, alloc);
        return return_type(std::move(buf), shape, l);
    }

    // shape and strides - buffer version
    template <class P, class O, class SC, class SS, class A, typename std::enable_if_t<detail::is_array<std::decay_t<SC>>::value, int>>
    inline xtensor_adaptor<xbuffer_adaptor<xtl::closure_type_t<P>, O, A>, detail::array_size<SC>::value, layout_type::dynamic>
    adapt(P&& pointer, typename A::size_type size, O, SC&& shape, SS&& strides, const A& alloc)
    {
        using buffer_type = xbuffer_adaptor<xtl::closure_type_t<P>, O, A>;
        constexpr std::size_t N = detail::array_size<SC>::value;
        using return_type = xtensor_adaptor<buffer_type, N, layout_type::dynamic>;
        buffer_type buf(std::forward<P>(pointer), size, alloc);
        return return_type(std::move(buf),
                           xtl::forward_sequence<typename return_type::inner_shape_type>(shape),
                           xtl::forward_sequence<typename return_type::inner_strides_type>(strides));
    }
}

#endif
