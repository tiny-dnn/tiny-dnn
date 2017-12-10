/***************************************************************************
* Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XADAPT_HPP
#define XADAPT_HPP

#include <array>
#include <cstddef>
#include <memory>
#include <type_traits>

#include "xarray.hpp"
#include "xtensor.hpp"

namespace xt {

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
template <class C, class SC, layout_type L = DEFAULT_LAYOUT>
std::enable_if_t<!detail::is_array<SC>::value, xarray_adaptor<C, L, SC>> xadapt(
  C& container, const SC& shape, layout_type l = L);

/**
 * Constructs an xarray_adaptor of the given stl-like container,
 * with the specified shape and strides.
 * @param container the container to adapt
 * @param shape the shape of the xarray_adaptor
 * @param strides the strides of the xarray_adaptor
 */
template <class C, class SC>
std::enable_if_t<!detail::is_array<SC>::value,
                 xarray_adaptor<C, layout_type::dynamic, SC>>
xadapt(C& container, const SC& shape, const SC& strides);

/**
 * Constructs an xarray_adaptor of the given dynamically allocated C array,
 * with the specified shape and layout.
 * @param pointer the pointer to the beginning of the dynamic array
 * @param size the size of the dynamic array
 * @param ownership indicates whether the adaptor takes ownership of the array.
 *        Possible values are ``no_ownerhsip()`` or ``accept_ownership()``
 * @param shape the shape of the xarray_adaptor
 * @param l the layout_type of the xarray_adaptor
 * @param alloc the allocator used for allocating / deallocating the dynamic
 * array
 */
template <class P,
          class O,
          class SC,
          layout_type L = DEFAULT_LAYOUT,
          class A       = std::allocator<std::remove_pointer_t<P>>>
std::enable_if_t<
  !detail::is_array<SC>::value,
  xarray_adaptor<xbuffer_adaptor<std::remove_pointer_t<P>, O, A>, L, SC>>
xadapt(P& pointer,
       typename A::size_type size,
       O ownership,
       const SC& shape,
       layout_type l  = L,
       const A& alloc = A());

/**
 * Constructs an xarray_adaptor of the given dynamically allocated C array,
 * with the specified shape and layout.
 * @param pointer the pointer to the beginning of the dynamic array
 * @param size the size of the dynamic array
 * @param ownership indicates whether the adaptor takes ownership of the array.
 *        Possible values are ``no_ownerhsip()`` or ``accept_ownership()``
 * @param shape the shape of the xarray_adaptor
 * @param strides the strides of the xarray_adaptor
 * @param alloc the allocator used for allocating / deallocating the dynamic
 * array
*/
template <class P,
          class O,
          class SC,
          class A = std::allocator<std::remove_pointer_t<P>>>
std::enable_if_t<!detail::is_array<SC>::value,
                 xarray_adaptor<xbuffer_adaptor<std::remove_pointer_t<P>, O, A>,
                                layout_type::dynamic,
                                SC>>
xadapt(P& pointer,
       typename A::size_type size,
       O ownership,
       const SC& shape,
       const SC& strides,
       const A& alloc = A());

/***************************
 * xtensor_adaptor builder *
 ***************************/

/**
 * Constructs an xtensor_adaptor of the given stl-like container,
 * with the specified shape and layout_type.
 * @param container the container to adapt
 * @param shape the shape of the xtensor_adaptor
 * @param l the layout_type of the xtensor_adaptor
 */
template <class C, std::size_t N, layout_type L = DEFAULT_LAYOUT>
xtensor_adaptor<C, N, L> xadapt(
  C& container,
  const std::array<typename C::size_type, N>& shape,
  layout_type l = L);

/**
 * Constructs an xtensor_adaptor of the given stl-like container,
 * with the specified shape and strides.
 * @param container the container to adapt
 * @param shape the shape of the xtensor_adaptor
 * @param strides the strides of the xtensor_adaptor
 */
template <class C, std::size_t N>
xtensor_adaptor<C, N, layout_type::dynamic> xadapt(
  C& container,
  const std::array<typename C::size_type, N>& shape,
  const std::array<typename C::size_type, N>& strides);

/**
 * Constructs an xtensor_adaptor of the given dynamically allocated C array,
 * with the specified shape and layout.
 * @param pointer the pointer to the beginning of the dynamic array
 * @param size the size of the dynamic array
 * @param ownership indicates whether the adaptor takes ownership of the array.
 *        Possible values are ``no_ownerhsip()`` or ``accept_ownership()``
 * @param shape the shape of the xtensor_adaptor
 * @param l the layout_type of the xtensor_adaptor
 * @param alloc the allocator used for allocating / deallocating the dynamic
 * array
 */
template <class P,
          std::size_t N,
          class O,
          layout_type L = DEFAULT_LAYOUT,
          class A       = std::allocator<std::remove_pointer_t<P>>>
xtensor_adaptor<xbuffer_adaptor<std::remove_pointer_t<P>, O, A>, N, L> xadapt(
  P& pointer,
  typename A::size_type size,
  O ownership,
  const std::array<typename A::size_type, N>& shape,
  layout_type l  = L,
  const A& alloc = A());

/**
 * Constructs an xtensor_adaptor of the given dynamically allocated C array,
 * with the specified shape and layout.
 * @param pointer the pointer to the beginning of the dynamic array
 * @param size the size of the dynamic array
 * @param ownership indicates whether the adaptor takes ownership of the array.
 *        Possible values are ``no_ownerhsip()`` or ``accept_ownership()``
 * @param shape the shape of the xtensor_adaptor
 * @param strides the strides of the xtensor_adaptor
 * @param alloc the allocator used for allocating / deallocating the dynamic
 * array
 */
template <class P,
          std::size_t N,
          class O,
          class A = std::allocator<std::remove_pointer_t<P>>>
xtensor_adaptor<xbuffer_adaptor<std::remove_pointer_t<P>, O, A>,
                N,
                layout_type::dynamic>
xadapt(P& pointer,
       typename A::size_type size,
       O ownership,
       const std::array<typename A::size_type, N>& shape,
       const std::array<typename A::size_type, N>& strides,
       const A& alloc = A());

/*****************************************
 * xarray_adaptor builder implementation *
 *****************************************/

template <class C, class SC, layout_type L>
inline std::enable_if_t<!detail::is_array<SC>::value, xarray_adaptor<C, L, SC>>
xadapt(C& container, const SC& shape, layout_type l) {
  return xarray_adaptor<C, L, SC>(container, shape, l);
}

template <class C, class SC>
inline std::enable_if_t<!detail::is_array<SC>::value,
                        xarray_adaptor<C, layout_type::dynamic, SC>>
xadapt(C& container, const SC& shape, const SC& strides) {
  return xarray_adaptor<C, layout_type::dynamic, SC>(container, shape, strides);
}

template <class P, class O, class SC, layout_type L, class A>
inline std::enable_if_t<
  !detail::is_array<SC>::value,
  xarray_adaptor<xbuffer_adaptor<std::remove_pointer_t<P>, O, A>, L, SC>>
xadapt(P& pointer,
       typename A::size_type size,
       O,
       const SC& shape,
       layout_type l,
       const A& alloc) {
  using buffer_type = xbuffer_adaptor<std::remove_pointer_t<P>, O, A>;
  buffer_type buf(pointer, size, alloc);
  return xarray_adaptor<buffer_type, L, SC>(std::move(buf), shape, l);
}

template <class P, class O, class SC, class A>
inline std::enable_if_t<
  !detail::is_array<SC>::value,
  xarray_adaptor<xbuffer_adaptor<std::remove_pointer_t<P>, O, A>,
                 layout_type::dynamic,
                 SC>>
xadapt(P& pointer,
       typename A::size_type size,
       O,
       const SC& shape,
       const SC& strides,
       const A& alloc) {
  using buffer_type = xbuffer_adaptor<std::remove_pointer_t<P>, O, A>;
  buffer_type buf(pointer, size, alloc);
  return xarray_adaptor<buffer_type, layout_type::dynamic, SC>(std::move(buf),
                                                               shape, strides);
}

/******************************************
 * xtensor_adaptor builder implementation *
 ******************************************/

template <class C, std::size_t N, layout_type L>
inline xtensor_adaptor<C, N, L> xadapt(
  C& container,
  const std::array<typename C::size_type, N>& shape,
  layout_type l) {
  return xtensor_adaptor<C, N, L>(container, shape, l);
}

template <class C, std::size_t N>
inline xtensor_adaptor<C, N, layout_type::dynamic> xadapt(
  C& container,
  const std::array<typename C::size_type, N>& shape,
  const std::array<typename C::size_type, N>& strides) {
  return xtensor_adaptor<C, N, layout_type::dynamic>(container, shape, strides);
}

template <class P, std::size_t N, class O, layout_type L, class A>
inline xtensor_adaptor<xbuffer_adaptor<std::remove_pointer_t<P>, O, A>, N, L>
xadapt(P& pointer,
       typename A::size_type size,
       O,
       const std::array<typename A::size_type, N>& shape,
       layout_type l,
       const A& alloc) {
  using buffer_type = xbuffer_adaptor<std::remove_pointer_t<P>, O, A>;
  buffer_type buf(pointer, size, alloc);
  return xtensor_adaptor<buffer_type, N, L>(std::move(buf), shape, l);
}

template <class P, std::size_t N, class O, class A>
inline xtensor_adaptor<xbuffer_adaptor<std::remove_pointer_t<P>, O, A>,
                       N,
                       layout_type::dynamic>
xadapt(P& pointer,
       typename A::size_type size,
       O,
       const std::array<typename A::size_type, N>& shape,
       const std::array<typename A::size_type, N>& strides,
       const A& alloc) {
  using buffer_type = xbuffer_adaptor<std::remove_pointer_t<P>, O, A>;
  buffer_type buf(pointer, size, alloc);
  return xtensor_adaptor<buffer_type, N, layout_type::dynamic>(std::move(buf),
                                                               shape, strides);
}
}

#endif
