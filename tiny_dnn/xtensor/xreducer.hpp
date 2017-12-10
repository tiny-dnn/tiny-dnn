/***************************************************************************
* Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XREDUCER_HPP
#define XREDUCER_HPP

#include <algorithm>
#include <cstddef>
#include <initializer_list>
#include <iterator>
#include <stdexcept>
#include <type_traits>
#include <utility>
#ifdef X_OLD_CLANG
#include <vector>
#endif

#include "xbuilder.hpp"
#include "xexpression.hpp"
#include "xgenerator.hpp"
#include "xiterable.hpp"
#include "xreducer.hpp"
#include "xutils.hpp"

namespace xt {

/**********
 * reduce *
 **********/

template <class F, class E, class X>
auto reduce(F&& f, E&& e, X&& axes) noexcept;

template <class F, class E>
auto reduce(F&& f, E&& e) noexcept;

#ifdef X_OLD_CLANG
template <class F, class E, class I>
auto reduce(F&& f, E&& e, std::initializer_list<I> axes) noexcept;
#else
template <class F, class E, class I, std::size_t N>
auto reduce(F&& f, E&& e, const I (&axes)[N]) noexcept;
#endif

/*************
 * xreducer  *
 *************/

template <class ST, class X>
struct xreducer_shape_type;

template <class F, class CT, class X>
class xreducer;

template <class F, class CT, class X>
class xreducer_stepper;

template <class F, class CT, class X>
struct xiterable_inner_types<xreducer<F, CT, X>> {
  using xexpression_type = std::decay_t<CT>;
  using inner_shape_type =
    typename xreducer_shape_type<typename xexpression_type::shape_type,
                                 X>::type;
  using const_stepper = xreducer_stepper<F, CT, X>;
  using stepper       = const_stepper;
};

/**
 * @class xreducer
 * @brief Reducing function operating over specified axes.
 *
 * The xreducer class implements an \ref xexpression applying
 * a reducing function to an \ref xexpression over the specified
 * axes.
 *
 * @tparam F the function type
 * @tparam CT the closure type of the \ref xexpression to reduce
 * @tparam X the list of axes
 *
 * @sa reduce
 */
template <class F, class CT, class X>
class xreducer : public xexpression<xreducer<F, CT, X>>,
                 public xconst_iterable<xreducer<F, CT, X>> {
 public:
  using self_type        = xreducer<F, CT, X>;
  using functor_type     = typename std::remove_reference<F>::type;
  using xexpression_type = std::decay_t<CT>;
  using axes_type        = X;

  using value_type      = typename xexpression_type::value_type;
  using reference       = value_type;
  using const_reference = value_type;
  using pointer         = value_type*;
  using const_pointer   = const value_type*;

  using size_type       = typename xexpression_type::size_type;
  using difference_type = typename xexpression_type::difference_type;

  using iterable_base    = xconst_iterable<self_type>;
  using inner_shape_type = typename iterable_base::inner_shape_type;
  using shape_type       = inner_shape_type;

  using stepper       = typename iterable_base::stepper;
  using const_stepper = typename iterable_base::const_stepper;

  static constexpr layout_type static_layout = layout_type::dynamic;
  static constexpr bool contiguous_layout    = false;

  template <class Func, class CTA, class AX>
  xreducer(Func&& func, CTA&& e, AX&& axes);

  size_type size() const noexcept;
  size_type dimension() const noexcept;
  const inner_shape_type& shape() const noexcept;
  layout_type layout() const noexcept;

  template <class... Args>
  const_reference operator()(Args... args) const;
  const_reference operator[](const xindex& index) const;
  const_reference operator[](size_type i) const;

  template <class It>
  const_reference element(It first, It last) const;

  template <class S>
  bool broadcast_shape(S& shape) const;

  template <class S>
  bool is_trivial_broadcast(const S& strides) const noexcept;

  template <class S>
  const_stepper stepper_begin(const S& shape) const noexcept;
  template <class S>
  const_stepper stepper_end(const S& shape, layout_type) const noexcept;

 private:
  CT m_e;
  functor_type m_f;
  axes_type m_axes;
  inner_shape_type m_shape;
  shape_type m_dim_mapping;

  friend class xreducer_stepper<F, CT, X>;
};

/*************************
 * reduce implementation *
 *************************/

/**
 * @brief Returns an \ref xexpression applying the speficied reducing
 * function to an expresssion over the given axes.
 *
 * @param f the reducing function to apply.
 * @param e the \ref xexpression to reduce.
 * @param axes the list of axes.
 *
 * The returned expression either hold a const reference to \p e or a copy
 * depending on whether \p e is an lvalue or an rvalue.
 */

template <class F, class E, class X>
inline auto reduce(F&& f, E&& e, X&& axes) noexcept {
  using reducer_type = xreducer<F, const_xclosure_t<E>, const_closure_t<X>>;
  return reducer_type(std::forward<F>(f), std::forward<E>(e),
                      std::forward<X>(axes));
}

template <class F, class E>
inline auto reduce(F&& f, E&& e) noexcept {
  auto ar            = arange(e.dimension());
  using AR           = decltype(ar);
  using reducer_type = xreducer<F, const_xclosure_t<E>, AR>;
  return reducer_type(std::forward<F>(f), std::forward<E>(e), std::move(ar));
}

#ifdef X_OLD_CLANG
template <class F, class E, class I>
inline auto reduce(F&& f, E&& e, std::initializer_list<I> axes) noexcept {
  using axes_type    = std::vector<typename std::decay_t<E>::size_type>;
  using reducer_type = xreducer<F, const_xclosure_t<E>, axes_type>;
  return reducer_type(std::forward<F>(f), std::forward<E>(e),
                      forward_sequence<axes_type>(axes));
}
#else
template <class F, class E, class I, std::size_t N>
inline auto reduce(F&& f, E&& e, const I (&axes)[N]) noexcept {
  using axes_type    = std::array<typename std::decay_t<E>::size_type, N>;
  using reducer_type = xreducer<F, const_xclosure_t<E>, axes_type>;
  return reducer_type(std::forward<F>(f), std::forward<E>(e),
                      forward_sequence<axes_type>(axes));
}
#endif

/********************
 * xreducer_stepper *
 ********************/

template <class F, class CT, class X>
class xreducer_stepper {
 public:
  using self_type     = xreducer_stepper<F, CT, X>;
  using xreducer_type = xreducer<F, CT, X>;

  using value_type      = typename xreducer_type::value_type;
  using reference       = typename xreducer_type::value_type;
  using pointer         = typename xreducer_type::const_pointer;
  using size_type       = typename xreducer_type::size_type;
  using difference_type = typename xreducer_type::difference_type;

  using xexpression_type = typename xreducer_type::xexpression_type;
  using substepper_type  = typename xexpression_type::const_stepper;
  using shape_type       = typename xreducer_type::shape_type;

  xreducer_stepper(const xreducer_type& red,
                   size_type offset,
                   bool end      = false,
                   layout_type l = layout_type::row_major);

  reference operator*() const;

  void step(size_type dim, size_type n = 1);
  void step_back(size_type dim, size_type n = 1);
  void reset(size_type dim);
  void reset_back(size_type dim);

  void to_begin();
  void to_end(layout_type l);

  bool equal(const self_type& rhs) const;

 private:
  reference aggregate(size_type dim) const;

  substepper_type get_substepper_begin() const;
  size_type get_dim(size_type dim) const noexcept;
  size_type shape(size_type i) const noexcept;
  size_type axis(size_type i) const noexcept;

  const xreducer_type& m_reducer;
  size_type m_offset;
  mutable substepper_type m_stepper;
};

template <class F, class CT, class X>
bool operator==(const xreducer_stepper<F, CT, X>& lhs,
                const xreducer_stepper<F, CT, X>& rhs);

template <class F, class CT, class X>
bool operator!=(const xreducer_stepper<F, CT, X>& lhs,
                const xreducer_stepper<F, CT, X>& rhs);

/******************
 * xreducer utils *
 ******************/

// meta-function returning the shape type for an xreducer
template <class ST, class X>
struct xreducer_shape_type {
  using type = promote_shape_t<ST, std::decay_t<X>>;
};

template <class I1, std::size_t N1, class I2, std::size_t N2>
struct xreducer_shape_type<std::array<I1, N1>, std::array<I2, N2>> {
  using type = std::array<I2, N1 - N2>;
};

namespace detail {
template <class InputIt, class ExcludeIt, class OutputIt>
inline void excluding_copy(InputIt first,
                           InputIt last,
                           ExcludeIt e_first,
                           ExcludeIt e_last,
                           OutputIt d_first,
                           OutputIt map_first) {
  using difference_type =
    typename std::iterator_traits<InputIt>::difference_type;
  using value_type = typename std::iterator_traits<OutputIt>::value_type;
  InputIt iter     = first;
  while (iter != last && e_first != e_last) {
    auto diff = std::distance(first, iter);
    if (diff != difference_type(*e_first)) {
      *d_first++   = *iter++;
      *map_first++ = value_type(diff);
    } else {
      ++iter;
      ++e_first;
    }
  }
  auto diff = std::distance(first, iter);
  auto end  = std::distance(iter, last);
  std::iota(map_first, map_first + end, diff);
  std::copy(iter, last, d_first);
}
}

/***************************
 * xreducer implementation *
 ***************************/

/**
 * @name Constructor
 */
//@{
/**
 * Constructs an xreducer expression applying the specified
 * function to the given expression over the given axes.
 *
 * @param func the function to apply
 * @param e the expression to reduce
 * @param axes the axes along which the reduction is performed
 */
template <class F, class CT, class X>
template <class Func, class CTA, class AX>
inline xreducer<F, CT, X>::xreducer(Func&& func, CTA&& e, AX&& axes)
  : m_e(std::forward<CTA>(e)),
    m_f(std::forward<Func>(func)),
    m_axes(std::forward<AX>(axes)),
    m_shape(make_sequence<shape_type>(m_e.dimension() - m_axes.size(), 0)),
    m_dim_mapping(
      make_sequence<shape_type>(m_e.dimension() - m_axes.size(), 0)) {
  if (!std::is_sorted(m_axes.cbegin(), m_axes.cend())) {
    throw std::runtime_error("Reducing axes should be sorted");
  }
  detail::excluding_copy(m_e.shape().begin(), m_e.shape().end(), m_axes.begin(),
                         m_axes.end(), m_shape.begin(), m_dim_mapping.begin());
}
//@}

/**
 * @name Size and shape
 */
/**
 * Returns the size of the expression.
 */
template <class F, class CT, class X>
inline auto xreducer<F, CT, X>::size() const noexcept -> size_type {
  return compute_size(shape());
}

/**
 * Returns the number of dimensions of the expression.
 */
template <class F, class CT, class X>
inline auto xreducer<F, CT, X>::dimension() const noexcept -> size_type {
  return m_shape.size();
}

/**
 * Returns the shape of the expression.
 */
template <class F, class CT, class X>
inline auto xreducer<F, CT, X>::shape() const noexcept
  -> const inner_shape_type& {
  return m_shape;
}

/**
 * Returns the shape of the expression.
 */
template <class F, class CT, class X>
inline layout_type xreducer<F, CT, X>::layout() const noexcept {
  return static_layout;
}
//@}

/**
 * @name Data
 */
/**
 * Returns a constant reference to the element at the specified position in the
 * reducer.
 * @param args a list of indices specifying the position in the reducer. Indices
 * must be unsigned integers, the number of indices should be equal or greater
 * than
 * the number of dimensions of the reducer.
 */
template <class F, class CT, class X>
template <class... Args>
inline auto xreducer<F, CT, X>::operator()(Args... args) const
  -> const_reference {
  std::array<std::size_t, sizeof...(Args)> arg_array = {
    {static_cast<std::size_t>(args)...}};
  return element(arg_array.cbegin(), arg_array.cend());
}

/**
 * Returns a constant reference to the element at the specified position in the
 * reducer.
 * @param index a sequence of indices specifying the position in the reducer.
 * Indices
 * must be unsigned integers, the number of indices in the sequence should be
 * equal or greater
 * than the number of dimensions of the reducer.
 */
template <class F, class CT, class X>
inline auto xreducer<F, CT, X>::operator[](const xindex& index) const
  -> const_reference {
  return element(index.cbegin(), index.cend());
}

template <class F, class CT, class X>
inline auto xreducer<F, CT, X>::operator[](size_type i) const
  -> const_reference {
  return operator()(i);
}

/**
 * Returns a constant reference to the element at the specified position in the
 * reducer.
 * @param first iterator starting the sequence of indices
 * @param last iterator ending the sequence of indices
 * The number of indices in the sequence should be equal to or greater
 * than the number of dimensions of the reducer.
 */
template <class F, class CT, class X>
template <class It>
inline auto xreducer<F, CT, X>::element(It first, It last) const
  -> const_reference {
  auto stepper  = const_stepper(*this, 0);
  size_type dim = 0;
  while (first != last) {
    stepper.step(dim++, *first++);
  }
  return *stepper;
}
//@}

/**
 * @name Broadcasting
 */
//@{
/**
 * Broadcast the shape of the reducer to the specified parameter.
 * @param shape the result shape
 * @return a boolean indicating whether the broadcasting is trivial
 */
template <class F, class CT, class X>
template <class S>
inline bool xreducer<F, CT, X>::broadcast_shape(S& shape) const {
  return xt::broadcast_shape(m_shape, shape);
}

/**
 * Compares the specified strides with those of the container to see whether
 * the broadcasting is trivial.
 * @return a boolean indicating whether the broadcasting is trivial
 */
template <class F, class CT, class X>
template <class S>
inline bool xreducer<F, CT, X>::is_trivial_broadcast(const S& /*strides*/) const
  noexcept {
  return false;
}
//@}

template <class F, class CT, class X>
template <class S>
inline auto xreducer<F, CT, X>::stepper_begin(const S& shape) const noexcept
  -> const_stepper {
  size_type offset = shape.size() - dimension();
  return const_stepper(*this, offset);
}

template <class F, class CT, class X>
template <class S>
inline auto xreducer<F, CT, X>::stepper_end(const S& shape, layout_type l) const
  noexcept -> const_stepper {
  size_type offset = shape.size() - dimension();
  return const_stepper(*this, offset, true, l);
}

/***********************************
 * xreducer_stepper implementation *
 ***********************************/

template <class F, class CT, class X>
inline xreducer_stepper<F, CT, X>::xreducer_stepper(const xreducer_type& red,
                                                    size_type offset,
                                                    bool end,
                                                    layout_type l)
  : m_reducer(red), m_offset(offset), m_stepper(get_substepper_begin()) {
  if (end) {
    to_end(l);
  }
}

template <class F, class CT, class X>
inline auto xreducer_stepper<F, CT, X>::operator*() const -> reference {
  reference r = aggregate(0);
  return r;
}

template <class F, class CT, class X>
inline void xreducer_stepper<F, CT, X>::step(size_type dim, size_type n) {
  if (dim >= m_offset) {
    m_stepper.step(get_dim(dim), n);
  }
}

template <class F, class CT, class X>
inline void xreducer_stepper<F, CT, X>::step_back(size_type dim, size_type n) {
  if (dim >= m_offset) {
    m_stepper.step_back(get_dim(dim), n);
  }
}

template <class F, class CT, class X>
inline void xreducer_stepper<F, CT, X>::reset(size_type dim) {
  if (dim >= m_offset) {
    m_stepper.reset(get_dim(dim));
  }
}

template <class F, class CT, class X>
inline void xreducer_stepper<F, CT, X>::reset_back(size_type dim) {
  if (dim >= m_offset) {
    m_stepper.reset_back(get_dim(dim));
  }
}

template <class F, class CT, class X>
inline void xreducer_stepper<F, CT, X>::to_begin() {
  m_stepper.to_begin();
}

template <class F, class CT, class X>
inline void xreducer_stepper<F, CT, X>::to_end(layout_type l) {
  m_stepper.to_end(l);
}

template <class F, class CT, class X>
inline bool xreducer_stepper<F, CT, X>::equal(const self_type& rhs) const {
  return &m_reducer == &(rhs.m_reducer) && m_stepper.equal(rhs.m_stepper);
}

template <class F, class CT, class X>
inline auto xreducer_stepper<F, CT, X>::aggregate(size_type dim) const
  -> reference {
  size_type index = axis(dim);
  size_type size  = shape(index);
  reference res;
  if (dim != m_reducer.m_axes.size() - 1) {
    res = aggregate(dim + 1);
    for (size_type i = 1; i != size; ++i) {
      m_stepper.step(index);
      res = m_reducer.m_f(res, aggregate(dim + 1));
    }
  } else {
    res = *m_stepper;
    for (size_type i = 1; i != size; ++i) {
      m_stepper.step(index);
      res = m_reducer.m_f(res, *m_stepper);
    }
  }
  m_stepper.reset(index);
  return res;
}

template <class F, class CT, class X>
inline auto xreducer_stepper<F, CT, X>::get_substepper_begin() const
  -> substepper_type {
  return m_reducer.m_e.stepper_begin(m_reducer.m_e.shape());
}

template <class F, class CT, class X>
inline auto xreducer_stepper<F, CT, X>::get_dim(size_type dim) const noexcept
  -> size_type {
  return m_reducer.m_dim_mapping[dim];
}

template <class F, class CT, class X>
inline auto xreducer_stepper<F, CT, X>::shape(size_type i) const noexcept
  -> size_type {
  return m_reducer.m_e.shape()[i];
}

template <class F, class CT, class X>
inline auto xreducer_stepper<F, CT, X>::axis(size_type i) const noexcept
  -> size_type {
  return m_reducer.m_axes[i];
}

template <class F, class CT, class X>
inline bool operator==(const xreducer_stepper<F, CT, X>& lhs,
                       const xreducer_stepper<F, CT, X>& rhs) {
  return lhs.equal(rhs);
}

template <class F, class CT, class X>
inline bool operator!=(const xreducer_stepper<F, CT, X>& lhs,
                       const xreducer_stepper<F, CT, X>& rhs) {
  return !lhs.equal(rhs);
}
}

#endif
