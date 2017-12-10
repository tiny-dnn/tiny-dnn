/***************************************************************************
* Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XFUNCTORVIEW_HPP
#define XFUNCTORVIEW_HPP

#include <algorithm>
#include <array>
#include <cstddef>
#include <type_traits>
#include <utility>

#include "xtensor/xexpression.hpp"
#include "xtensor/xiterator.hpp"
#include "xtensor/xsemantic.hpp"
#include "xtensor/xutils.hpp"

#include "xtensor/xarray.hpp"
#include "xtensor/xtensor.hpp"

namespace xt {

/****************************
 * xfunctorview declaration *
 ****************************/

template <class F, class IT>
class xfunctor_iterator;

template <class F, class ST>
class xfunctor_stepper;

template <class F, class CT>
class xfunctorview;

/*******************************
 * xfunctorview_temporary_type *
 *******************************/

namespace detail {
template <class F, class S, layout_type L>
struct functorview_temporary_type_impl {
  using type = xarray<typename F::value_type, L>;
};

template <class F, class T, std::size_t N, layout_type L>
struct functorview_temporary_type_impl<F, std::array<T, N>, L> {
  using type = xtensor<typename F::value_type, N, L>;
};
}

template <class F, class E>
struct xfunctorview_temporary_type {
  using type =
    typename detail::functorview_temporary_type_impl<F,
                                                     typename E::shape_type,
                                                     E::static_layout>::type;
};

template <class F, class CT>
struct xcontainer_inner_types<xfunctorview<F, CT>> {
  using xexpression_type = std::decay_t<CT>;
  using temporary_type =
    typename xfunctorview_temporary_type<F, xexpression_type>::type;
};

#define DL DEFAULT_LAYOUT
/**
 * @class xfunctorview
 * @brief View of an xexpression .
 *
 * The xfunctorview class is an expression addressing its elements by applying a
 * functor to the
 * corresponding element of an underlying expression. Unlike e.g. xgenerator, an
 * xfunctorview is
 * an lvalue. It is used e.g. to access real and imaginary parts of complex
 * expressions.
 *
 * xfunctorview is not meant to be used directly, but through helper functions
 * such
 * as \ref real or \ref imag.
 *
 * @tparam F the functor type to be applied to the elements of specified
 * expression.
 * @tparam CT the closure type of the \ref xexpression type underlying this view
 *
 * @sa real, imag
 */
template <class F, class CT>
class xfunctorview : public xview_semantic<xfunctorview<F, CT>> {
 public:
  using self_type        = xfunctorview<F, CT>;
  using xexpression_type = std::decay_t<CT>;
  using semantic_base    = xview_semantic<self_type>;
  using functor_type     = typename std::decay_t<F>;

  using value_type      = typename functor_type::value_type;
  using reference       = typename functor_type::reference;
  using const_reference = typename functor_type::const_reference;
  using pointer         = typename functor_type::pointer;
  using const_pointer   = typename functor_type::const_pointer;
  using size_type       = typename xexpression_type::size_type;
  using difference_type = typename xexpression_type::difference_type;

  using shape_type = typename xexpression_type::shape_type;

  static constexpr layout_type static_layout = xexpression_type::static_layout;
  static constexpr bool contiguous_layout    = false;

  using stepper =
    xfunctor_stepper<functor_type, typename xexpression_type::stepper>;
  using const_stepper =
    xfunctor_stepper<functor_type, typename xexpression_type::const_stepper>;

  template <layout_type L>
  using layout_iterator =
    xfunctor_iterator<functor_type,
                      typename xexpression_type::template layout_iterator<L>>;
  template <layout_type L>
  using const_layout_iterator = xfunctor_iterator<
    functor_type,
    typename xexpression_type::template const_layout_iterator<L>>;

  template <layout_type L>
  using reverse_layout_iterator = xfunctor_iterator<
    functor_type,
    typename xexpression_type::template reverse_layout_iterator<L>>;
  template <layout_type L>
  using const_reverse_layout_iterator = xfunctor_iterator<
    functor_type,
    typename xexpression_type::template const_reverse_layout_iterator<L>>;

  template <class S, layout_type L>
  using broadcast_iterator =
    xfunctor_iterator<functor_type,
                      xiterator<typename xexpression_type::stepper, S, L>>;
  template <class S, layout_type L>
  using const_broadcast_iterator = xfunctor_iterator<
    functor_type,
    xiterator<typename xexpression_type::const_stepper, S, L>>;

  template <class S, layout_type L>
  using reverse_broadcast_iterator = xfunctor_iterator<
    functor_type,
    typename xexpression_type::template reverse_broadcast_iterator<S, L>>;
  template <class S, layout_type L>
  using const_reverse_broadcast_iterator = xfunctor_iterator<
    functor_type,
    typename xexpression_type::template const_reverse_broadcast_iterator<S, L>>;

  using storage_iterator =
    xfunctor_iterator<functor_type,
                      typename xexpression_type::storage_iterator>;
  using const_storage_iterator =
    xfunctor_iterator<functor_type,
                      typename xexpression_type::const_storage_iterator>;
  using reverse_storage_iterator =
    xfunctor_iterator<functor_type,
                      typename xexpression_type::reverse_storage_iterator>;
  using const_reverse_storage_iterator = xfunctor_iterator<
    functor_type,
    typename xexpression_type::const_reverse_storage_iterator>;

  using iterator =
    xfunctor_iterator<functor_type, typename xexpression_type::iterator>;
  using const_iterator =
    xfunctor_iterator<functor_type, typename xexpression_type::const_iterator>;
  using reverse_iterator =
    xfunctor_iterator<functor_type,
                      typename xexpression_type::reverse_iterator>;
  using const_reverse_iterator =
    xfunctor_iterator<functor_type,
                      typename xexpression_type::const_reverse_iterator>;

  xfunctorview(CT) noexcept;

  template <class Func, class E>
  xfunctorview(Func&&, E&&) noexcept;

  template <class E>
  self_type& operator=(const xexpression<E>& e);

  template <class E>
  disable_xexpression<E, self_type>& operator=(const E& e);

  size_type size() const noexcept;
  size_type dimension() const noexcept;
  const shape_type& shape() const noexcept;
  layout_type layout() const noexcept;

  template <class... Args>
  reference operator()(Args... args);
  reference operator[](const xindex& index);
  reference operator[](size_type i);

  template <class IT>
  reference element(IT first, IT last);

  template <class... Args>
  const_reference operator()(Args... args) const;
  const_reference operator[](const xindex& index) const;
  const_reference operator[](size_type i) const;

  template <class IT>
  const_reference element(IT first, IT last) const;

  template <class S>
  bool broadcast_shape(S& shape) const;

  template <class S>
  bool is_trivial_broadcast(const S& strides) const;

  template <layout_type L = DL>
  auto begin() noexcept;
  template <layout_type L = DL>
  auto end() noexcept;

  template <layout_type L = DL>
  auto begin() const noexcept;
  template <layout_type L = DL>
  auto end() const noexcept;
  template <layout_type L = DL>
  auto cbegin() const noexcept;
  template <layout_type L = DL>
  auto cend() const noexcept;

  template <layout_type L = DL>
  auto rbegin() noexcept;
  template <layout_type L = DL>
  auto rend() noexcept;

  template <layout_type L = DL>
  auto rbegin() const noexcept;
  template <layout_type L = DL>
  auto rend() const noexcept;
  template <layout_type L = DL>
  auto crbegin() const noexcept;
  template <layout_type L = DL>
  auto crend() const noexcept;

  template <class S, layout_type L = DL>
  broadcast_iterator<S, L> begin(const S& shape) noexcept;
  template <class S, layout_type L = DL>
  broadcast_iterator<S, L> end(const S& shape) noexcept;

  template <class S, layout_type L = DL>
  const_broadcast_iterator<S, L> begin(const S& shape) const noexcept;
  template <class S, layout_type L = DL>
  const_broadcast_iterator<S, L> end(const S& shape) const noexcept;
  template <class S, layout_type L = DL>
  const_broadcast_iterator<S, L> cbegin(const S& shape) const noexcept;
  template <class S, layout_type L = DL>
  const_broadcast_iterator<S, L> cend(const S& shape) const noexcept;

  template <class S, layout_type L = DL>
  reverse_broadcast_iterator<S, L> rbegin(const S& shape) noexcept;
  template <class S, layout_type L = DL>
  reverse_broadcast_iterator<S, L> rend(const S& shape) noexcept;

  template <class S, layout_type L = DL>
  const_reverse_broadcast_iterator<S, L> rbegin(const S& shape) const noexcept;
  template <class S, layout_type L = DL>
  const_reverse_broadcast_iterator<S, L> rend(const S& shape) const noexcept;
  template <class S, layout_type L = DL>
  const_reverse_broadcast_iterator<S, L> crbegin(const S& shape) const noexcept;
  template <class S, layout_type L = DL>
  const_reverse_broadcast_iterator<S, L> crend(const S& shape) const noexcept;

  template <layout_type L = DL>
  storage_iterator storage_begin() noexcept;
  template <layout_type L = DL>
  storage_iterator storage_end() noexcept;

  template <layout_type L = DL>
  const_storage_iterator storage_begin() const noexcept;
  template <layout_type L = DL>
  const_storage_iterator storage_end() const noexcept;
  template <layout_type L = DL>
  const_storage_iterator storage_cbegin() const noexcept;
  template <layout_type L = DL>
  const_storage_iterator storage_cend() const noexcept;

  template <layout_type L = DL>
  reverse_storage_iterator storage_rbegin() noexcept;
  template <layout_type L = DL>
  reverse_storage_iterator storage_rend() noexcept;

  template <layout_type L = DL>
  const_reverse_storage_iterator storage_rbegin() const noexcept;
  template <layout_type L = DL>
  const_reverse_storage_iterator storage_rend() const noexcept;
  template <layout_type L = DL>
  const_reverse_storage_iterator storage_crbegin() const noexcept;
  template <layout_type L = DL>
  const_reverse_storage_iterator storage_crend() const noexcept;

  template <class S>
  stepper stepper_begin(const S& shape) noexcept;
  template <class S>
  stepper stepper_end(const S& shape, layout_type l) noexcept;
  template <class S>
  const_stepper stepper_begin(const S& shape) const noexcept;
  template <class S>
  const_stepper stepper_end(const S& shape, layout_type l) const noexcept;

 private:
  CT m_e;
  functor_type m_functor;

  using temporary_type =
    typename xcontainer_inner_types<self_type>::temporary_type;
  void assign_temporary_impl(temporary_type&& tmp);
  friend class xview_semantic<xfunctorview<F, CT>>;
};

#undef DL

/*********************************
 * xfunctor_iterator declaration *
 *********************************/

template <class F, class IT>
class xfunctor_iterator {
 public:
  using functor_type = std::decay_t<F>;
  using value_type   = typename functor_type::value_type;

  using subiterator_traits = std::iterator_traits<IT>;

  using reference =
    apply_cv_t<typename subiterator_traits::reference, value_type>;
  using pointer           = std::remove_reference_t<reference>*;
  using difference_type   = typename subiterator_traits::difference_type;
  using iterator_category = typename subiterator_traits::iterator_category;

  using self_type = xfunctor_iterator<F, IT>;

  xfunctor_iterator(const IT&, const functor_type*);

  self_type& operator++();
  self_type operator++(int);

  reference operator*() const;
  pointer operator->() const;

  bool equal(const xfunctor_iterator& rhs) const;

 private:
  IT m_it;
  const functor_type* p_functor;

  template <class F_, class IT_>
  friend xfunctor_iterator<F_, IT_> operator+(xfunctor_iterator<F_, IT_>,
                                              xfunctor_iterator<F_, IT_>);

  template <class F_, class IT_>
  friend typename xfunctor_iterator<F_, IT_>::difference_type operator-(
    xfunctor_iterator<F_, IT_>, xfunctor_iterator<F_, IT_>);
};

template <class F, class IT>
bool operator==(const xfunctor_iterator<F, IT>& lhs,
                const xfunctor_iterator<F, IT>& rhs);

template <class F, class IT>
bool operator!=(const xfunctor_iterator<F, IT>& lhs,
                const xfunctor_iterator<F, IT>& rhs);

template <class F, class IT>
xfunctor_iterator<F, IT> operator+(xfunctor_iterator<F, IT> it1,
                                   xfunctor_iterator<F, IT> it2) {
  return xfunctor_iterator<F, IT>(it1.m_it + it2.m_it);
}

template <class F, class IT>
typename xfunctor_iterator<F, IT>::difference_type operator-(
  xfunctor_iterator<F, IT> it1, xfunctor_iterator<F, IT> it2) {
  return it1.m_it - it2.m_it;
}

/********************************
 * xfunctor_stepper declaration *
 ********************************/

template <class F, class ST>
class xfunctor_stepper {
 public:
  using functor_type = std::decay_t<F>;

  using value_type      = typename functor_type::value_type;
  using reference       = apply_cv_t<typename ST::reference, value_type>;
  using pointer         = std::remove_reference_t<reference>*;
  using size_type       = typename ST::size_type;
  using difference_type = typename ST::difference_type;

  xfunctor_stepper() = default;
  xfunctor_stepper(const ST&, const functor_type*);

  reference operator*() const;

  void step(size_type dim, size_type n = 1);
  void step_back(size_type dim, size_type n = 1);
  void reset(size_type dim);
  void reset_back(size_type dim);

  void to_begin();
  void to_end(layout_type);

  bool equal(const xfunctor_stepper& rhs) const;

 private:
  ST m_stepper;
  const functor_type* p_functor;
};

template <class F, class ST>
bool operator==(const xfunctor_stepper<F, ST>& lhs,
                const xfunctor_stepper<F, ST>& rhs);

template <class F, class ST>
bool operator!=(const xfunctor_stepper<F, ST>& lhs,
                const xfunctor_stepper<F, ST>& rhs);

/*******************************
 * xfunctorview implementation *
 *******************************/

/**
 * @name Constructors
 */
//@{

/**
 * Constructs an xfunctorview expression wrappering the specified \ref
 * xexpression.
 *
 * @param e the underlying expression
 */
template <class F, class CT>
inline xfunctorview<F, CT>::xfunctorview(CT e) noexcept
  : m_e(e), m_functor(functor_type()) {}

/**
* Constructs an xfunctorview expression wrappering the specified \ref
* xexpression.
*
* @param func the functor to be applied to the elements of the underlying
* expression.
* @param e the underlying expression
*/
template <class F, class CT>
template <class Func, class E>
inline xfunctorview<F, CT>::xfunctorview(Func&& func, E&& e) noexcept
  : m_e(std::forward<E>(e)), m_functor(std::forward<Func>(func)) {}
//@}

/**
 * @name Extended copy semantic
 */
//@{
/**
 * The extended assignment operator.
 */
template <class F, class CT>
template <class E>
inline auto xfunctorview<F, CT>::operator=(const xexpression<E>& e)
  -> self_type& {
  bool cond = (e.derived_cast().shape().size() == dimension()) &&
              std::equal(shape().begin(), shape().end(),
                         e.derived_cast().shape().begin());
  if (!cond) {
    semantic_base::operator=(broadcast(e.derived_cast(), shape()));
  } else {
    semantic_base::operator=(e);
  }
  return *this;
}
//@}

template <class F, class CT>
template <class E>
inline auto xfunctorview<F, CT>::operator=(const E& e)
  -> disable_xexpression<E, self_type>& {
  std::fill(begin(), end(), e);
  return *this;
}

template <class F, class CT>
inline void xfunctorview<F, CT>::assign_temporary_impl(temporary_type&& tmp) {
  std::copy(tmp.cbegin(), tmp.cend(), begin());
}

/**
 * @name Size and shape
 */
/**
 * Returns the size of the expression.
 */
template <class F, class CT>
inline auto xfunctorview<F, CT>::size() const noexcept -> size_type {
  return m_e.size();
}

/**
 * Returns the number of dimensions of the expression.
 */
template <class F, class CT>
inline auto xfunctorview<F, CT>::dimension() const noexcept -> size_type {
  return m_e.dimension();
}

/**
 * Returns the shape of the expression.
 */
template <class F, class CT>
inline auto xfunctorview<F, CT>::shape() const noexcept -> const shape_type& {
  return m_e.shape();
}

/**
 * Returns the layout_type of the expression.
 */
template <class F, class CT>
inline layout_type xfunctorview<F, CT>::layout() const noexcept {
  return m_e.layout();
}
//@}

/**
 * @name Data
 */
/**
 * Returns a reference to the element at the specified position in the
 * expression.
 * @param args a list of indices specifying the position in the function.
 * Indices
 * must be unsigned integers, the number of indices should be equal or greater
 * than
 * the number of dimensions of the expression.
 */
template <class F, class CT>
template <class... Args>
inline auto xfunctorview<F, CT>::operator()(Args... args) -> reference {
  return m_functor(m_e(args...));
}

/**
 * Returns a reference to the element at the specified position in the
 * expression.
 * @param index a sequence of indices specifying the position in the function.
 * Indices
 * must be unsigned integers, the number of indices in the sequence should be
 * equal or greater
 * than the number of dimensions of the container.
 */
template <class F, class CT>
inline auto xfunctorview<F, CT>::operator[](const xindex& index) -> reference {
  return m_functor(m_e[index]);
}

template <class F, class CT>
inline auto xfunctorview<F, CT>::operator[](size_type i) -> reference {
  return operator()(i);
}

/**
 * Returns a reference to the element at the specified position in the
 * expression.
 * @param first iterator starting the sequence of indices
 * @param last iterator ending the sequence of indices
 * The number of indices in the sequence should be equal to or greater
 * than the number of dimensions of the function.
 */
template <class F, class CT>
template <class IT>
inline auto xfunctorview<F, CT>::element(IT first, IT last) -> reference {
  return m_functor(m_e.element(first, last));
}

/**
 * Returns a constant reference to the element at the specified position in the
 * expression.
 * @param args a list of indices specifying the position in the function.
 * Indices
 * must be unsigned integers, the number of indices should be equal or greater
 * than
 * the number of dimensions of the expression.
 */
template <class F, class CT>
template <class... Args>
inline auto xfunctorview<F, CT>::operator()(Args... args) const
  -> const_reference {
  return m_functor(m_e(args...));
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
template <class F, class CT>
inline auto xfunctorview<F, CT>::operator[](const xindex& index) const
  -> const_reference {
  return m_functor(m_e[index]);
}

template <class F, class CT>
inline auto xfunctorview<F, CT>::operator[](size_type i) const
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
template <class F, class CT>
template <class IT>
inline auto xfunctorview<F, CT>::element(IT first, IT last) const
  -> const_reference {
  return m_functor(m_e.element(first, last));
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
template <class F, class CT>
template <class S>
inline bool xfunctorview<F, CT>::broadcast_shape(S& shape) const {
  return m_e.broadcast_shape(shape);
}

/**
 * Compares the specified strides with those of the container to see whether
 * the broadcasting is trivial.
 * @return a boolean indicating whether the broadcasting is trivial
 */
template <class F, class CT>
template <class S>
inline bool xfunctorview<F, CT>::is_trivial_broadcast(const S& strides) const {
  return m_e.is_trivial_broadcast(strides);
}
//@}

/**
 * @name Iterators
 */
//@{
/**
 * Returns an iterator to the first element of the expression.
 * @tparam L layout used for the traversal. Default value is \c DEFAULT_LAYOUT.
 */
template <class F, class CT>
template <layout_type L>
inline auto xfunctorview<F, CT>::begin() noexcept {
  return xfunctor_iterator<functor_type, decltype(m_e.template begin<L>())>(
    m_e.template begin<L>(), &m_functor);
}

/**
 * Returns an iterator to the element following the last element
 * of the expression.
 * @tparam L layout used for the traversal. Default value is \c DEFAULT_LAYOUT.
 */
template <class F, class CT>
template <layout_type L>
inline auto xfunctorview<F, CT>::end() noexcept {
  return xfunctor_iterator<functor_type, decltype(m_e.template end<L>())>(
    m_e.template end<L>(), &m_functor);
}

/**
 * Returns a constant iterator to the first element of the expression.
 * @tparam L layout used for the traversal. Default value is \c DEFAULT_LAYOUT.
 */
template <class F, class CT>
template <layout_type L>
inline auto xfunctorview<F, CT>::begin() const noexcept {
  return cbegin<L>();
}

/**
 * Returns a constant iterator to the element following the last element
 * of the expression.
 * @tparam L layout used for the traversal. Default value is \c DEFAULT_LAYOUT.
 */
template <class F, class CT>
template <layout_type L>
inline auto xfunctorview<F, CT>::end() const noexcept {
  return cend<L>();
}

/**
 * Returns a constant iterator to the first element of the expression.
 * @tparam L layout used for the traversal. Default value is \c DEFAULT_LAYOUT.
 */
template <class F, class CT>
template <layout_type L>
inline auto xfunctorview<F, CT>::cbegin() const noexcept {
  return xfunctor_iterator<functor_type, decltype(m_e.template cbegin<L>())>(
    m_e.template cbegin<L>(), &m_functor);
}

/**
 * Returns a constant iterator to the element following the last element
 * of the expression.
 * @tparam L layout used for the traversal. Default value is \c DEFAULT_LAYOUT.
 */
template <class F, class CT>
template <layout_type L>
inline auto xfunctorview<F, CT>::cend() const noexcept {
  return xfunctor_iterator<functor_type, decltype(m_e.template cend<L>())>(
    m_e.template cend<L>(), &m_functor);
}
//@}

/**
 * @name Broadcast iterators
 */
//@{
/**
 * Returns a constant iterator to the first element of the expression. The
 * iteration is broadcasted to the specified shape.
 * @param shape the shape used for broadcasting
 * @tparam S type of the \c shape parameter.
 * @tparam L layout used for the traversal. Default value is \c DEFAULT_LAYOUT.
 */
template <class F, class CT>
template <class S, layout_type L>
inline auto xfunctorview<F, CT>::begin(const S& shape) noexcept
  -> broadcast_iterator<S, L> {
  return broadcast_iterator<S, L>(m_e.template begin<S, L>(shape), &m_functor);
}

/**
 * Returns a constant iterator to the element following the last element of the
 * expression. The iteration is broadcasted to the specified shape.
 * @param shape the shape used for broadcasting
 * @tparam S type of the \c shape parameter.
 * @tparam L layout used for the traversal. Default value is \c DEFAULT_LAYOUT.
 */
template <class F, class CT>
template <class S, layout_type L>
inline auto xfunctorview<F, CT>::end(const S& shape) noexcept
  -> broadcast_iterator<S, L> {
  return broadcast_iterator<S, L>(m_e.template end<S, L>(shape), &m_functor);
}

/**
 * Returns a constant iterator to the first element of the expression. The
 * iteration is broadcasted to the specified shape.
 * @param shape the shape used for broadcasting
 * @tparam S type of the \c shape parameter.
 * @tparam L layout used for the traversal. Default value is \c DEFAULT_LAYOUT.
 */
template <class F, class CT>
template <class S, layout_type L>
inline auto xfunctorview<F, CT>::begin(const S& shape) const noexcept
  -> const_broadcast_iterator<S, L> {
  return cbegin<S, L>(shape);
}

/**
 * Returns a constant iterator to the element following the last element of the
 * expression. The iteration is broadcasted to the specified shape.
 * @param shape the shape used for broadcasting
 * @tparam S type of the \c shape parameter.
 * @tparam L layout used for the traversal. Default value is \c DEFAULT_LAYOUT.
 */
template <class F, class CT>
template <class S, layout_type L>
inline auto xfunctorview<F, CT>::end(const S& shape) const noexcept
  -> const_broadcast_iterator<S, L> {
  return cend<S, L>(shape);
}

/**
 * Returns a constant iterator to the first element of the expression. The
 * iteration is broadcasted to the specified shape.
 * @param shape the shape used for broadcasting
 * @tparam S type of the \c shape parameter.
 * @tparam L layout used for the traversal. Default value is \c DEFAULT_LAYOUT.
 */
template <class F, class CT>
template <class S, layout_type L>
inline auto xfunctorview<F, CT>::cbegin(const S& shape) const noexcept
  -> const_broadcast_iterator<S, L> {
  return const_broadcast_iterator<S, L>(m_e.template cxbegin<S, L>(shape),
                                        &m_functor);
}

/**
 * Returns a constant iterator to the element following the last element of the
 * expression. The iteration is broadcasted to the specified shape.
 * @param shape the shape used for broadcasting
 * @tparam S type of the \c shape parameter.
 * @tparam L layout used for the traversal. Default value is \c DEFAULT_LAYOUT.
 */
template <class F, class CT>
template <class S, layout_type L>
inline auto xfunctorview<F, CT>::cend(const S& shape) const noexcept
  -> const_broadcast_iterator<S, L> {
  return const_broadcast_iterator<S, L>(m_e.template cxend<S, L>(shape),
                                        &m_functor);
}
//@}

/**
 * @name Reverse iterators
 */
//@{
/**
 * Returns an iterator to the first element of the reversed expression.
 * @tparam L layout used for the traversal. Default value is \c DEFAULT_LAYOUT.
 */
template <class F, class CT>
template <layout_type L>
inline auto xfunctorview<F, CT>::rbegin() noexcept {
  return xfunctor_iterator<functor_type, decltype(m_e.template rbegin<L>())>(
    m_e.template rbegin<L>(), &m_functor);
}

/**
 * Returns an iterator to the element following the last element
 * of the reversed expression.
 * @tparam L layout used for the traversal. Default value is \c DEFAULT_LAYOUT.
 */
template <class F, class CT>
template <layout_type L>
inline auto xfunctorview<F, CT>::rend() noexcept {
  return xfunctor_iterator<functor_type, decltype(m_e.template rend<L>())>(
    m_e.template rend<L>(), &m_functor);
}

/**
 * Returns a constant iterator to the first element of the reversed expression.
 * @tparam L layout used for the traversal. Default value is \c DEFAULT_LAYOUT.
 */
template <class F, class CT>
template <layout_type L>
inline auto xfunctorview<F, CT>::rbegin() const noexcept {
  return crbegin<L>();
}

/**
 * Returns a constant iterator to the element following the last element
 * of the reversed expression.
 * @tparam L layout used for the traversal. Default value is \c DEFAULT_LAYOUT.
 */
template <class F, class CT>
template <layout_type L>
inline auto xfunctorview<F, CT>::rend() const noexcept {
  return crend<L>();
}

/**
 * Returns a constant iterator to the first element of the reversed expression.
 * @tparam L layout used for the traversal. Default value is \c DEFAULT_LAYOUT.
 */
template <class F, class CT>
template <layout_type L>
inline auto xfunctorview<F, CT>::crbegin() const noexcept {
  return xfunctor_iterator<functor_type, decltype(m_e.template rbegin<L>())>(
    m_e.template rbegin<L>(), &m_functor);
}

/**
 * Returns a constant iterator to the element following the last element
 * of the reversed expression.
 * @tparam L layout used for the traversal. Default value is \c DEFAULT_LAYOUT.
 */
template <class F, class CT>
template <layout_type L>
inline auto xfunctorview<F, CT>::crend() const noexcept {
  return xfunctor_iterator<functor_type, decltype(m_e.template rend<L>())>(
    m_e.template rend<L>(), &m_functor);
}
//@}

/**
 * @name Reverse broadcast iterators
 */
/**
 * Returns an iterator to the first element of the expression. The
 * iteration is broadcasted to the specified shape.
 * @param shape the shape used for broadcasting
 * @tparam S type of the \c shape parameter.
 * @tparam L layout used for the traversal. Default value is \c DEFAULT_LAYOUT.
 */
template <class F, class CT>
template <class S, layout_type L>
inline auto xfunctorview<F, CT>::rbegin(const S& shape) noexcept
  -> reverse_broadcast_iterator<S, L> {
  return reverse_broadcast_iterator<S, L>(m_e.template xrbegin<S, L>(shape),
                                          &m_functor);
}

/**
 * Returns an iterator to the element following the last element of the
 * reversed expression. The iteration is broadcasted to the specified shape.
 * @param shape the shape used for broadcasting
 * @tparam S type of the \c shape parameter.
 * @tparam L layout used for the traversal. Default value is \c DEFAULT_LAYOUT.
 */
template <class F, class CT>
template <class S, layout_type L>
inline auto xfunctorview<F, CT>::rend(const S& shape) noexcept
  -> reverse_broadcast_iterator<S, L> {
  return reverse_broadcast_iterator<S, L>(m_e.template xrend<S, L>(shape),
                                          &m_functor);
}

/**
 * Returns a constant iterator to the first element of the reversed expression.
 * The iteration is broadcasted to the specified shape.
 * @param shape the shape used for broadcasting
 * @tparam S type of the \c shape parameter.
 * @tparam L layout used for the traversal. Default value is \c DEFAULT_LAYOUT.
 */
template <class F, class CT>
template <class S, layout_type L>
inline auto xfunctorview<F, CT>::rbegin(const S& shape) const noexcept
  -> const_reverse_broadcast_iterator<S, L> {
  return crbegin<S, L>(shape);
}

/**
 * Returns a constant iterator to the element following the last element
 * of the reversed expression.
 * @param shape the shape used for broadcasting
 * @tparam S type of the \c shape parameter.
 * @tparam L layout used for the traversal. Default value is \c DEFAULT_LAYOUT.
 */
template <class F, class CT>
template <class S, layout_type L>
inline auto xfunctorview<F, CT>::rend(const S& /*shape*/) const noexcept
  -> const_reverse_broadcast_iterator<S, L> {
  return crend<S, L>();
}

/**
 * Returns a constant iterator to the first element of the reversed expression.
 * The iteration is broadcasted to the specified shape.
 * @param shape the shape used for broadcasting
 * @tparam S type of the \c shape parameter.
 * @tparam L layout used for the traversal. Default value is \c DEFAULT_LAYOUT.
 */
template <class F, class CT>
template <class S, layout_type L>
inline auto xfunctorview<F, CT>::crbegin(const S& /*shape*/) const noexcept
  -> const_reverse_broadcast_iterator<S, L> {
  return const_reverse_broadcast_iterator<S, L>(m_e.template crbegin<S, L>(),
                                                &m_functor);
}

/**
 * Returns a constant iterator to the element following the last element
 * of the reversed expression.
 * @param shape the shape used for broadcasting
 * @tparam S type of the \c shape parameter.
 * @tparam L layout used for the traversal. Default value is \c DEFAULT_LAYOUT.
 */
template <class F, class CT>
template <class S, layout_type L>
inline auto xfunctorview<F, CT>::crend(const S& shape) const noexcept
  -> const_reverse_broadcast_iterator<S, L> {
  return const_reverse_broadcast_iterator<S, L>(m_e.template crend<S, L>(shape),
                                                &m_functor);
}
//@}

template <class F, class CT>
template <layout_type L>
inline auto xfunctorview<F, CT>::storage_begin() noexcept -> storage_iterator {
  return storage_iterator(m_e.template storage_begin<L>(), &m_functor);
}

template <class F, class CT>
template <layout_type L>
inline auto xfunctorview<F, CT>::storage_end() noexcept -> storage_iterator {
  return storage_iterator(m_e.template storage_end<L>(), &m_functor);
}

template <class F, class CT>
template <layout_type L>
inline auto xfunctorview<F, CT>::storage_begin() const noexcept
  -> const_storage_iterator {
  return const_storage_iterator(m_e.template storage_begin<L>(), &m_functor);
}

template <class F, class CT>
template <layout_type L>
inline auto xfunctorview<F, CT>::storage_end() const noexcept
  -> const_storage_iterator {
  return const_storage_iterator(m_e.template storage_end<L>(), &m_functor);
}

template <class F, class CT>
template <layout_type L>
inline auto xfunctorview<F, CT>::storage_cbegin() const noexcept
  -> const_storage_iterator {
  return const_storage_iterator(m_e.template storage_cbegin<L>(), &m_functor);
}

template <class F, class CT>
template <layout_type L>
inline auto xfunctorview<F, CT>::storage_cend() const noexcept
  -> const_storage_iterator {
  return const_storage_iterator(m_e.template storage_cend<L>(), &m_functor);
}

template <class F, class CT>
template <layout_type L>
inline auto xfunctorview<F, CT>::storage_rbegin() noexcept
  -> reverse_storage_iterator {
  return reverse_storage_iterator(m_e.template storage_rbegin<L>(), &m_functor);
}

template <class F, class CT>
template <layout_type L>
inline auto xfunctorview<F, CT>::storage_rend() noexcept
  -> reverse_storage_iterator {
  return reverse_storage_iterator(m_e.template storage_rend<L>(), &m_functor);
}

template <class F, class CT>
template <layout_type L>
inline auto xfunctorview<F, CT>::storage_rbegin() const noexcept
  -> const_reverse_storage_iterator {
  return const_reverse_storage_iterator(m_e.template storage_rbegin<L>(),
                                        &m_functor);
}

template <class F, class CT>
template <layout_type L>
inline auto xfunctorview<F, CT>::storage_rend() const noexcept
  -> const_reverse_storage_iterator {
  return const_reverse_storage_iterator(m_e.template storage_rend<L>(),
                                        &m_functor);
}

template <class F, class CT>
template <layout_type L>
inline auto xfunctorview<F, CT>::storage_crbegin() const noexcept
  -> const_reverse_storage_iterator {
  return const_reverse_storage_iterator(m_e.template storage_crbegin<L>(),
                                        &m_functor);
}

template <class F, class CT>
template <layout_type L>
inline auto xfunctorview<F, CT>::storage_crend() const noexcept
  -> const_reverse_storage_iterator {
  return const_reverse_storage_iterator(m_e.template storage_crend<L>(),
                                        &m_functor);
}

/***************
 * stepper api *
 ***************/

template <class F, class CT>
template <class S>
inline auto xfunctorview<F, CT>::stepper_begin(const S& shape) noexcept
  -> stepper {
  return stepper(m_e.stepper_begin(shape), &m_functor);
}

template <class F, class CT>
template <class S>
inline auto xfunctorview<F, CT>::stepper_end(const S& shape,
                                             layout_type l) noexcept
  -> stepper {
  return stepper(m_e.stepper_end(shape, l), &m_functor);
}

template <class F, class CT>
template <class S>
inline auto xfunctorview<F, CT>::stepper_begin(const S& shape) const noexcept
  -> const_stepper {
  const xexpression_type& const_m_e = m_e;
  return const_stepper(const_m_e.stepper_begin(shape), &m_functor);
}

template <class F, class CT>
template <class S>
inline auto xfunctorview<F, CT>::stepper_end(const S& shape,
                                             layout_type l) const noexcept
  -> const_stepper {
  const xexpression_type& const_m_e = m_e;
  return const_stepper(const_m_e.stepper_end(shape, l), &m_functor);
}

/************************************
 * xfunctor_iterator implementation *
 ************************************/

template <class F, class IT>
xfunctor_iterator<F, IT>::xfunctor_iterator(const IT& it,
                                            const functor_type* pf)
  : m_it(it), p_functor(pf) {}

template <class F, class IT>
auto xfunctor_iterator<F, IT>::operator++() -> self_type& {
  ++m_it;
  return *this;
}

template <class F, class IT>
auto xfunctor_iterator<F, IT>::operator++(int) -> self_type {
  self_type tmp(*this);
  ++m_it;
  return tmp;
}

template <class F, class IT>
auto xfunctor_iterator<F, IT>::operator*() const -> reference {
  return (*p_functor)(*m_it);
}

template <class F, class IT>
auto xfunctor_iterator<F, IT>::operator-> () const -> pointer {
  // Returning the address of a temporary
  return &((*p_functor)(*m_it));
}

template <class F, class IT>
auto xfunctor_iterator<F, IT>::equal(const xfunctor_iterator& rhs) const
  -> bool {
  return m_it == rhs.m_it;
}

template <class F, class IT>
bool operator==(const xfunctor_iterator<F, IT>& lhs,
                const xfunctor_iterator<F, IT>& rhs) {
  return lhs.equal(rhs);
}

template <class F, class IT>
bool operator!=(const xfunctor_iterator<F, IT>& lhs,
                const xfunctor_iterator<F, IT>& rhs) {
  return !lhs.equal(rhs);
}

/***********************************
 * xfunctor_stepper implementation *
 ***********************************/

template <class F, class ST>
xfunctor_stepper<F, ST>::xfunctor_stepper(const ST& stepper,
                                          const functor_type* pf)
  : m_stepper(stepper), p_functor(pf) {}

template <class F, class ST>
auto xfunctor_stepper<F, ST>::operator*() const -> reference {
  return (*p_functor)(*m_stepper);
}

template <class F, class ST>
void xfunctor_stepper<F, ST>::step(size_type dim, size_type n) {
  m_stepper.step(dim, n);
}

template <class F, class ST>
void xfunctor_stepper<F, ST>::step_back(size_type dim, size_type n) {
  m_stepper.step_back(dim, n);
}

template <class F, class ST>
void xfunctor_stepper<F, ST>::reset(size_type dim) {
  m_stepper.reset(dim);
}

template <class F, class ST>
void xfunctor_stepper<F, ST>::reset_back(size_type dim) {
  m_stepper.reset_back(dim);
}

template <class F, class ST>
void xfunctor_stepper<F, ST>::to_begin() {
  m_stepper.to_begin();
}

template <class F, class ST>
void xfunctor_stepper<F, ST>::to_end(layout_type l) {
  m_stepper.to_end(l);
}

template <class F, class ST>
auto xfunctor_stepper<F, ST>::equal(const xfunctor_stepper& rhs) const -> bool {
  return m_stepper == rhs.m_stepper;
}

template <class F, class ST>
bool operator==(const xfunctor_stepper<F, ST>& lhs,
                const xfunctor_stepper<F, ST>& rhs) {
  return lhs.equal(rhs);
}

template <class F, class ST>
bool operator!=(const xfunctor_stepper<F, ST>& lhs,
                const xfunctor_stepper<F, ST>& rhs) {
  return !lhs.equal(rhs);
}
}
#endif
