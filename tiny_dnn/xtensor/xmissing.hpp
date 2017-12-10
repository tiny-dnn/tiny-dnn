/***************************************************************************
* Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XMISSING_HPP
#define XMISSING_HPP

#include <array>
#include <cstddef>
#include <iterator>
#include <memory>
#include <utility>
#include <vector>

#include "xarray.hpp"
#include "xfunctorview.hpp"
#include "xoptional.hpp"
#include "xtensor.hpp"
#include "xtensor_forward.hpp"
#include "xutils.hpp"

namespace xt {

/**************************************
 * Optimized 1-D xoptional containers *
 **************************************/

template <class T>
struct xoptional_sequence_inner_types;

template <class ITV, class ITB>
class xoptional_iterator;

template <class D>
class xoptional_sequence {
 public:
  // Internal typedefs
  using inner_types = xoptional_sequence_inner_types<D>;

  using base_container_type  = typename inner_types::base_container_type;
  using base_value_type      = typename base_container_type::value_type;
  using base_reference       = typename base_container_type::reference;
  using base_const_reference = typename base_container_type::const_reference;

  using flag_container_type  = typename inner_types::flag_container_type;
  using flag_type            = typename flag_container_type::value_type;
  using flag_reference       = typename flag_container_type::reference;
  using flag_const_reference = typename flag_container_type::const_reference;

  // Container typedefs
  using value_type      = xoptional<base_value_type, flag_type>;
  using reference       = xoptional<base_reference, flag_reference>;
  using const_reference = xoptional<base_const_reference, flag_const_reference>;
  using pointer         = std::nullptr_t;
  using const_pointer   = std::nullptr_t;

  // Other typedefs
  using size_type       = typename base_container_type::size_type;
  using difference_type = typename base_container_type::difference_type;
  using iterator = xoptional_iterator<typename base_container_type::iterator,
                                      typename flag_container_type::iterator>;
  using const_iterator =
    xoptional_iterator<typename base_container_type::const_iterator,
                       typename flag_container_type::const_iterator>;

  using reverse_iterator =
    xoptional_iterator<typename base_container_type::reverse_iterator,
                       typename flag_container_type::reverse_iterator>;
  using const_reverse_iterator =
    xoptional_iterator<typename base_container_type::const_reverse_iterator,
                       typename flag_container_type::const_reverse_iterator>;

  xoptional_sequence() = default;
  xoptional_sequence(size_type s, const base_value_type& v);

  template <class CTO, class CBO>
  xoptional_sequence(size_type s, const xoptional<CTO, CBO>& v);

  bool empty() const noexcept;
  size_type size() const noexcept;

  reference operator[](size_type i);
  const_reference operator[](size_type i) const;

  reference front();
  const_reference front() const;

  reference back();
  const_reference back() const;

  iterator begin() noexcept;
  iterator end() noexcept;

  const_iterator begin() const noexcept;
  const_iterator end() const noexcept;
  const_iterator cbegin() const noexcept;
  const_iterator cend() const noexcept;

  reverse_iterator rbegin() noexcept;
  reverse_iterator rend() noexcept;

  const_reverse_iterator rbegin() const noexcept;
  const_reverse_iterator rend() const noexcept;
  const_reverse_iterator crbegin() const noexcept;
  const_reverse_iterator crend() const noexcept;

 protected:
  base_container_type m_values;
  flag_container_type m_flags;
};

/****************************************************
 * xoptional_vector and xoptional_array inner types *
 ****************************************************/

template <class T, std::size_t I>
class xoptional_array;

template <class T, std::size_t I>
struct xoptional_sequence_inner_types<xoptional_array<T, I>> {
  using base_container_type = std::array<T, I>;
  using flag_container_type = std::array<bool, I>;
};

template <class T, class A, class BA>
class xoptional_vector;

template <class T, class A, class BA>
struct xoptional_sequence_inner_types<xoptional_vector<T, A, BA>> {
  using base_container_type = std::vector<T, A>;
  using flag_container_type = std::vector<bool, BA>;
};

/*****************************************************
 * xoptional_vector and xoptional_array declarations *
 *****************************************************/

template <class T, std::size_t I>
class xoptional_array : public xoptional_sequence<xoptional_array<T, I>> {
 public:
  using self_type       = xoptional_array;
  using base_type       = xoptional_sequence<self_type>;
  using base_value_type = typename base_type::base_value_type;
  using size_type       = typename base_type::size_type;

  xoptional_array() = default;
  xoptional_array(size_type s, const base_value_type& v);

  template <class CTO, class CBO>
  xoptional_array(size_type s, const xoptional<CTO, CBO>& v);
};

template <class T, class A = std::allocator<T>, class BA = std::allocator<bool>>
class xoptional_vector : public xoptional_sequence<xoptional_vector<T, A, BA>> {
 public:
  using self_type       = xoptional_vector;
  using base_type       = xoptional_sequence<self_type>;
  using base_value_type = typename base_type::base_value_type;

  using value_type      = typename base_type::value_type;
  using size_type       = typename base_type::size_type;
  using difference_type = typename base_type::difference_type;
  using reference       = typename base_type::reference;
  using const_reference = typename base_type::const_reference;
  using pointer         = typename base_type::pointer;
  using const_pointer   = typename base_type::const_pointer;

  using iterator               = typename base_type::iterator;
  using const_iterator         = typename base_type::const_iterator;
  using reverse_iterator       = typename base_type::reverse_iterator;
  using const_reverse_iterator = typename base_type::const_reverse_iterator;

  xoptional_vector() = default;
  xoptional_vector(size_type, const base_value_type&);

  template <class CTO, class CBO>
  xoptional_vector(size_type, const xoptional<CTO, CBO>&);

  void resize(size_type);
  void resize(size_type, const base_value_type&);
  template <class CTO, class CBO>
  void resize(size_type, const xoptional<CTO, CBO>&);
};

/**********************************
 * xoptional_iterator declaration *
 **********************************/

template <class ITV, class ITB>
class xoptional_iterator {
 public:
  using self_type = xoptional_iterator<ITV, ITB>;

  // Internal typedefs
  using base_value_type = typename ITV::value_type;
  using base_reference  = typename ITV::reference;

  using flag_type      = typename ITB::value_type;
  using flag_reference = typename ITB::reference;

  // Container typedefs
  using value_type = xoptional<base_value_type, flag_type>;
  using reference  = xoptional<base_reference, flag_reference>;

  using pointer           = std::nullptr_t;
  using difference_type   = typename ITV::difference_type;
  using iterator_category = std::random_access_iterator_tag;

  xoptional_iterator() = default;
  xoptional_iterator(ITV itv, ITB itb);

  self_type& operator++();
  self_type operator++(int);

  reference operator*() const;
  pointer operator->() const;

  self_type& operator--();
  self_type operator--(int);

  self_type& operator+=(difference_type n);
  self_type& operator-=(difference_type n);

  self_type operator+(difference_type n) const;
  self_type operator-(difference_type n) const;
  difference_type operator-(const self_type& rhs) const;

  bool equal(const xoptional_iterator& rhs) const;

 private:
  ITV m_itv;
  ITB m_itb;
};

template <class ITV, class ITB>
bool operator==(const xoptional_iterator<ITV, ITB>&,
                const xoptional_iterator<ITV, ITB>&);

template <class ITV, class ITB>
bool operator!=(const xoptional_iterator<ITV, ITB>&,
                const xoptional_iterator<ITV, ITB>&);

/*************************************
 * xoptional_sequence implementation *
 *************************************/

template <class D>
xoptional_sequence<D>::xoptional_sequence(size_type s, const base_value_type& v)
  : m_values(make_sequence<base_container_type>(s, v)),
    m_flags(make_sequence<flag_container_type>(s, true)) {}

template <class D>
template <class CTO, class CBO>
xoptional_sequence<D>::xoptional_sequence(size_type s,
                                          const xoptional<CTO, CBO>& v)
  : m_values(make_sequence<base_container_type>(s, v.value())),
    m_flags(make_sequence<flag_container_type>(s, v.has_value())) {}

template <class D>
auto xoptional_sequence<D>::empty() const noexcept -> bool {
  return m_values.empty();
}

template <class D>
auto xoptional_sequence<D>::size() const noexcept -> size_type {
  return m_values.size();
}

template <class D>
auto xoptional_sequence<D>::operator[](size_type i) -> reference {
  return reference(m_values[i], m_flags[i]);
}

template <class D>
auto xoptional_sequence<D>::operator[](size_type i) const -> const_reference {
  return const_reference(m_values[i], m_flags[i]);
}

template <class D>
auto xoptional_sequence<D>::front() -> reference {
  return reference(m_values.front(), m_flags.front());
}

template <class D>
auto xoptional_sequence<D>::front() const -> const_reference {
  return const_reference(m_values.front(), m_flags.front());
}

template <class D>
auto xoptional_sequence<D>::back() -> reference {
  return reference(m_values.back(), m_flags.back());
}

template <class D>
auto xoptional_sequence<D>::back() const -> const_reference {
  return const_reference(m_values.back(), m_flags.back());
}

template <class D>
auto xoptional_sequence<D>::begin() noexcept -> iterator {
  return iterator(m_values.begin(), m_flags.begin());
}

template <class D>
auto xoptional_sequence<D>::end() noexcept -> iterator {
  return iterator(m_values.end(), m_flags.end());
}

template <class D>
auto xoptional_sequence<D>::begin() const noexcept -> const_iterator {
  return cbegin();
}

template <class D>
auto xoptional_sequence<D>::end() const noexcept -> const_iterator {
  return cend();
}

template <class D>
auto xoptional_sequence<D>::cbegin() const noexcept -> const_iterator {
  return const_iterator(m_values.cbegin(), m_flags.cbegin());
}

template <class D>
auto xoptional_sequence<D>::cend() const noexcept -> const_iterator {
  return const_iterator(m_values.cend(), m_flags.cend());
}

template <class D>
auto xoptional_sequence<D>::rbegin() noexcept -> reverse_iterator {
  return reverse_iterator(m_values.rbegin(), m_flags.rbegin());
}

template <class D>
auto xoptional_sequence<D>::rend() noexcept -> reverse_iterator {
  return reverse_iterator(m_values.rend(), m_flags.rend());
}

template <class D>
auto xoptional_sequence<D>::rbegin() const noexcept -> const_reverse_iterator {
  return crbegin();
}

template <class D>
auto xoptional_sequence<D>::rend() const noexcept -> const_reverse_iterator {
  return crend();
}

template <class D>
auto xoptional_sequence<D>::crbegin() const noexcept -> const_reverse_iterator {
  return const_reverse_iterator(m_values.crbegin(), m_flags.crbegin());
}

template <class D>
auto xoptional_sequence<D>::crend() const noexcept -> const_reverse_iterator {
  return const_reverse_iterator(m_values.crend(), m_flags.crend());
}

/*******************************************************
 * xoptional_array and xoptional_vector implementation *
 *******************************************************/

template <class T, std::size_t I>
xoptional_array<T, I>::xoptional_array(size_type s, const base_value_type& v)
  : base_type(s, v) {}

template <class T, std::size_t I>
template <class CTO, class CBO>
xoptional_array<T, I>::xoptional_array(size_type s,
                                       const xoptional<CTO, CBO>& v)
  : base_type(s, v) {}

template <class T, class A, class BA>
xoptional_vector<T, A, BA>::xoptional_vector(size_type s,
                                             const base_value_type& v)
  : base_type(s, v) {}

template <class T, class A, class BA>
template <class CTO, class CBO>
xoptional_vector<T, A, BA>::xoptional_vector(size_type s,
                                             const xoptional<CTO, CBO>& v)
  : base_type(s, v) {}

template <class T, class A, class BA>
void xoptional_vector<T, A, BA>::resize(size_type s) {
  // Default to missing
  this->m_values.resize(s);
  this->m_flags.resize(s, false);
}

template <class T, class A, class BA>
void xoptional_vector<T, A, BA>::resize(size_type s, const base_value_type& v) {
  this->m_values.resize(s, v);
  this->m_flags.resize(s, true);
}

template <class T, class A, class BA>
template <class CTO, class CBO>
void xoptional_vector<T, A, BA>::resize(size_type s,
                                        const xoptional<CTO, CBO>& v) {
  this->m_values.resize(s, v.value());
  this->m_flags.resize(s, v.has_value());
}

/*************************************
 * xoptional_iterator implementation *
 *************************************/

template <class ITV, class ITB>
xoptional_iterator<ITV, ITB>::xoptional_iterator(ITV itv, ITB itb)
  : m_itv(itv), m_itb(itb) {}

template <class ITV, class ITB>
auto xoptional_iterator<ITV, ITB>::operator++() -> self_type& {
  ++m_itv;
  ++m_itb;
  return *this;
}

template <class ITV, class ITB>
auto xoptional_iterator<ITV, ITB>::operator++(int) -> self_type {
  self_type tmp(*this);
  ++(*this);
  return tmp;
}

template <class ITV, class ITB>
auto xoptional_iterator<ITV, ITB>::operator--() -> self_type& {
  --m_itv;
  --m_itb;
  return *this;
}

template <class ITV, class ITB>
auto xoptional_iterator<ITV, ITB>::operator--(int) -> self_type {
  self_type tmp(*this);
  --(*this);
  return tmp;
}

template <class ITV, class ITB>
auto xoptional_iterator<ITV, ITB>::operator+=(difference_type n) -> self_type& {
  m_itv += n;
  m_itb += n;
  return *this;
}

template <class ITV, class ITB>
auto xoptional_iterator<ITV, ITB>::operator-=(difference_type n) -> self_type& {
  m_itv -= n;
  m_itb -= n;
  return *this;
}

template <class ITV, class ITB>
auto xoptional_iterator<ITV, ITB>::operator+(difference_type n) const
  -> self_type {
  return self_type(m_itv + n, m_itb + n);
}

template <class ITV, class ITB>
auto xoptional_iterator<ITV, ITB>::operator-(difference_type n) const
  -> self_type {
  return self_type(m_itv - n, m_itb - n);
}

template <class ITV, class ITB>
auto xoptional_iterator<ITV, ITB>::operator-(const self_type& rhs) const
  -> difference_type {
  return m_itv - rhs.m_itv;
}

template <class ITV, class ITB>
auto xoptional_iterator<ITV, ITB>::operator*() const -> reference {
  return reference(*m_itv, *m_itb);
}

template <class ITV, class ITB>
auto xoptional_iterator<ITV, ITB>::operator-> () const -> std::nullptr_t {
  return nullptr;
}

template <class ITV, class ITB>
auto xoptional_iterator<ITV, ITB>::equal(const xoptional_iterator& rhs) const
  -> bool {
  return m_itv == rhs.m_itv && m_itb == rhs.m_itb;
}

template <class ITV, class ITB>
bool operator==(const xoptional_iterator<ITV, ITB>& lhs,
                const xoptional_iterator<ITV, ITB>& rhs) {
  return lhs.equal(rhs);
}

template <class ITV, class ITB>
bool operator!=(const xoptional_iterator<ITV, ITB>& lhs,
                const xoptional_iterator<ITV, ITB>& rhs) {
  return !lhs.equal(rhs);
}

/*******************************************************
 * value() and has_value() xfunctorview implementation *
 *******************************************************/

namespace detail {
template <class E>
struct value_forwarder {
  // internal types
  using xexpression_type         = std::decay_t<E>;
  using optional_type            = typename xexpression_type::value_type;
  using optional_reference       = typename xexpression_type::reference;
  using optional_const_reference = typename xexpression_type::const_reference;

  // types
  using value_type      = typename optional_type::value_type;
  using reference       = typename optional_reference::value_closure;
  using const_reference = typename optional_const_reference::value_closure;
  using pointer         = value_type*;
  using const_pointer   = const value_type*;

  template <class T>
  decltype(auto) operator()(T&& t) const {
    return std::forward<T>(t).value();
  }
};

template <class E>
struct flag_forwarder {
  // internal types
  using xexpression_type         = std::decay_t<E>;
  using optional_type            = typename xexpression_type::value_type;
  using optional_reference       = typename xexpression_type::reference;
  using optional_const_reference = typename xexpression_type::const_reference;

  // types
  using value_type      = typename optional_type::flag_type;
  using reference       = typename optional_reference::flag_closure;
  using const_reference = typename optional_const_reference::flag_closure;
  using pointer         = value_type*;
  using const_pointer   = const value_type*;

  template <class T>
  decltype(auto) operator()(T&& t) const {
    return std::forward<T>(t).has_value();
  }
};
}

template <class E>
auto value(E&& e)
  -> disable_xoptional<typename std::decay_t<E>::value_type, E> {
  return std::forward<E>(e);
}

template <class E>
auto has_value(E&& e)
  -> disable_xoptional<typename std::decay_t<E>::value_type,
                       decltype(ones<bool>(std::forward<E>(e).shape()))> {
  return ones<bool>(std::forward<E>(e).shape());
}

template <class E>
auto value(E&& e)
  -> enable_xoptional<typename std::decay_t<E>::value_type,
                      xfunctorview<detail::value_forwarder<E>, xclosure_t<E>>> {
  using type = xfunctorview<detail::value_forwarder<E>, xclosure_t<E>>;
  return type(std::forward<E>(e));
}

template <class E>
auto has_value(E&& e)
  -> enable_xoptional<typename std::decay_t<E>::value_type,
                      xfunctorview<detail::flag_forwarder<E>, xclosure_t<E>>> {
  using type = xfunctorview<detail::flag_forwarder<E>, xclosure_t<E>>;
  return type(std::forward<E>(e));
}

/***************************
 * value_or implementation *
 ***************************/
}

#endif
