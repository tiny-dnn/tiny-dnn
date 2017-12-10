/***************************************************************************
* Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XBROADCAST_HPP
#define XBROADCAST_HPP

#include <algorithm>
#include <array>
#include <cstddef>
#include <iterator>
#include <numeric>
#include <type_traits>
#include <utility>

#include "xexpression.hpp"
#include "xiterable.hpp"
#include "xstrides.hpp"
#include "xutils.hpp"

namespace xt {

/*************
 * broadcast *
 *************/

template <class E, class S>
auto broadcast(E&& e, const S& s) noexcept;

#ifdef X_OLD_CLANG
template <class E, class I>
auto broadcast(E&& e, std::initializer_list<I> s) noexcept;
#else
template <class E, class I, std::size_t L>
auto broadcast(E&& e, const I (&s)[L]) noexcept;
#endif

/**************
 * xbroadcast *
 **************/

template <class CT, class X>
class xbroadcast;

template <class CT, class X>
struct xiterable_inner_types<xbroadcast<CT, X>> {
  using xexpression_type = std::decay_t<CT>;
  using inner_shape_type =
    promote_shape_t<typename xexpression_type::shape_type, X>;
  using const_stepper = typename xexpression_type::const_stepper;
  using stepper       = const_stepper;
};

/**
 * @class xbroadcast
 * @brief Broadcasted xexpression to a specified shape.
 *
 * The xbroadcast class implements the broadcasting of an \ref xexpression
 * to a specified shape. xbroadcast is not meant to be used directly, but
 * only with the \ref broadcast helper functions.
 *
 * @tparam CT the closure type of the \ref xexpression to broadcast
 * @tparam X the type of the specified shape.
 *
 * @sa broadcast
 */
template <class CT, class X>
class xbroadcast : public xexpression<xbroadcast<CT, X>>,
                   public xconst_iterable<xbroadcast<CT, X>> {
 public:
  using self_type        = xbroadcast<CT, X>;
  using xexpression_type = std::decay_t<CT>;

  using value_type      = typename xexpression_type::value_type;
  using reference       = typename xexpression_type::reference;
  using const_reference = typename xexpression_type::const_reference;
  using pointer         = typename xexpression_type::pointer;
  using const_pointer   = typename xexpression_type::const_pointer;
  using size_type       = typename xexpression_type::size_type;
  using difference_type = typename xexpression_type::difference_type;

  using iterable_base    = xconst_iterable<self_type>;
  using inner_shape_type = typename iterable_base::inner_shape_type;
  using shape_type       = inner_shape_type;

  using stepper       = typename iterable_base::stepper;
  using const_stepper = typename iterable_base::const_stepper;

  static constexpr layout_type static_layout = xexpression_type::static_layout;
  // static constexpr bool contiguous_layout =
  // xexpression_type::contiguous_layout;
  static constexpr bool contiguous_layout = false;

  template <class CTA, class S>
  xbroadcast(CTA&& e, S&& s) noexcept;

  size_type size() const noexcept;
  size_type dimension() const noexcept;
  const inner_shape_type& shape() const noexcept;
  layout_type layout() const noexcept;

  template <class... Args>
  const_reference operator()(Args... args) const;
  const_reference operator[](const xindex& index) const;
  const_reference operator[](size_type i) const;

  template <class It>
  const_reference element(It, It last) const;

  template <class S>
  bool broadcast_shape(S& shape) const;

  template <class S>
  bool is_trivial_broadcast(const S& strides) const noexcept;

  template <class S>
  const_stepper stepper_begin(const S& shape) const noexcept;
  template <class S>
  const_stepper stepper_end(const S& shape, layout_type l) const noexcept;

 private:
  CT m_e;
  inner_shape_type m_shape;
};

/****************************
 * broadcast implementation *
 ****************************/

/**
 * @brief Returns an \ref xexpression broadcasting the given expression to
 * a specified shape.
 *
 * @tparam e the \ref xexpression to broadcast
 * @tparam s the specified shape to broadcast.
 *
 * The returned expression either hold a const reference to \p e or a copy
 * depending on whether \p e is an lvalue or an rvalue.
 */
template <class E, class S>
inline auto broadcast(E&& e, const S& s) noexcept {
  using broadcast_type = xbroadcast<const_xclosure_t<E>, S>;
  using shape_type     = typename broadcast_type::shape_type;
  return broadcast_type(std::forward<E>(e), forward_sequence<shape_type>(s));
}

#ifdef X_OLD_CLANG
template <class E, class I>
inline auto broadcast(E&& e, std::initializer_list<I> s) noexcept {
  using broadcast_type =
    xbroadcast<const_xclosure_t<E>, std::vector<std::size_t>>;
  using shape_type = typename broadcast_type::shape_type;
  return broadcast_type(std::forward<E>(e), forward_sequence<shape_type>(s));
}
#else
template <class E, class I, std::size_t L>
inline auto broadcast(E&& e, const I (&s)[L]) noexcept {
  using broadcast_type =
    xbroadcast<const_xclosure_t<E>, std::array<std::size_t, L>>;
  using shape_type = typename broadcast_type::shape_type;
  return broadcast_type(std::forward<E>(e), forward_sequence<shape_type>(s));
}
#endif

/*****************************
 * xbroadcast implementation *
 *****************************/

/**
 * @name Constructor
 */
//@{
/**
 * Constructs an xbroadcast expression broadcasting the specified
 * \ref xexpression to the given shape
 *
 * @param e the expression to broadcast
 * @param s the shape to apply
 */
template <class CT, class X>
template <class CTA, class S>
inline xbroadcast<CT, X>::xbroadcast(CTA&& e, S&& s) noexcept
  : m_e(std::forward<CTA>(e)), m_shape(std::forward<S>(s)) {
  xt::broadcast_shape(m_e.shape(), m_shape);
}
//@}

/**
 * @name Size and shape
 */
/**
 * Returns the size of the expression.
 */
template <class CT, class X>
inline auto xbroadcast<CT, X>::size() const noexcept -> size_type {
  return compute_size(shape());
}

/**
 * Returns the number of dimensions of the expression.
 */
template <class CT, class X>
inline auto xbroadcast<CT, X>::dimension() const noexcept -> size_type {
  return m_shape.size();
}

/**
 * Returns the shape of the expression.
 */
template <class CT, class X>
inline auto xbroadcast<CT, X>::shape() const noexcept
  -> const inner_shape_type& {
  return m_shape;
}

/**
 * Returns the layout_type of the expression.
 */
template <class CT, class X>
inline layout_type xbroadcast<CT, X>::layout() const noexcept {
  return m_e.layout();
}
//@}

/**
 * @name Data
 */
/**
 * Returns a constant reference to the element at the specified position in the
 * expression.
 * @param args a list of indices specifying the position in the function.
 * Indices
 * must be unsigned integers, the number of indices should be equal or greater
 * than
 * the number of dimensions of the expression.
 */
template <class CT, class X>
template <class... Args>
inline auto xbroadcast<CT, X>::operator()(Args... args) const
  -> const_reference {
  return detail::get_element(m_e, args...);
}

/**
 * Returns a constant reference to the element at the specified position in the
 * expression.
 * @param index a sequence of indices specifying the position in the function.
 * Indices
 * must be unsigned integers, the number of indices in the sequence should be
 * equal or greater
 * than the number of dimensions of the container.
 */
template <class CT, class X>
inline auto xbroadcast<CT, X>::operator[](const xindex& index) const
  -> const_reference {
  return element(index.cbegin(), index.cend());
}

template <class CT, class X>
inline auto xbroadcast<CT, X>::operator[](size_type i) const
  -> const_reference {
  return operator()(i);
}

/**
 * Returns a constant reference to the element at the specified position in the
 * expression.
 * @param first iterator starting the sequence of indices
 * @param last iterator ending the sequence of indices
 * The number of indices in the sequence should be equal to or greater
 * than the number of dimensions of the function.
 */
template <class CT, class X>
template <class It>
inline auto xbroadcast<CT, X>::element(It, It last) const -> const_reference {
  return m_e.element(last - dimension(), last);
}
//@}

/**
 * @name Broadcasting
 */
//@{
/**
 * Broadcast the shape of the function to the specified parameter.
 * @param shape the result shape
 * @return a boolean indicating whether the broadcasting is trivial
 */
template <class CT, class X>
template <class S>
inline bool xbroadcast<CT, X>::broadcast_shape(S& shape) const {
  return xt::broadcast_shape(m_shape, shape);
}

/**
 * Compares the specified strides with those of the container to see whether
 * the broadcasting is trivial.
 * @return a boolean indicating whether the broadcasting is trivial
 */
template <class CT, class X>
template <class S>
inline bool xbroadcast<CT, X>::is_trivial_broadcast(const S& strides) const
  noexcept {
  return dimension() == m_e.dimension() &&
         std::equal(m_shape.cbegin(), m_shape.cend(), m_e.shape().cbegin()) &&
         m_e.is_trivial_broadcast(strides);
}
//@}

template <class CT, class X>
template <class S>
inline auto xbroadcast<CT, X>::stepper_begin(const S& shape) const noexcept
  -> const_stepper {
  // Could check if (broadcastable(shape, m_shape)
  return m_e.stepper_begin(shape);
}

template <class CT, class X>
template <class S>
inline auto xbroadcast<CT, X>::stepper_end(const S& shape, layout_type l) const
  noexcept -> const_stepper {
  // Could check if (broadcastable(shape, m_shape)
  return m_e.stepper_end(shape, l);
}
}

#endif
