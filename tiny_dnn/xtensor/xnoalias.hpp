/***************************************************************************
* Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XNOALIAS_HPP
#define XNOALIAS_HPP

#include "xsemantic.hpp"

namespace xt {

template <class A>
class noalias_proxy {
 public:
  noalias_proxy(A& a) noexcept;

  template <class E>
  A& operator=(const xexpression<E>& e);

  template <class E>
  A& operator+=(const xexpression<E>& e);

  template <class E>
  A& operator-=(const xexpression<E>& e);

  template <class E>
  A& operator*=(const xexpression<E>& e);

  template <class E>
  A& operator/=(const xexpression<E>& e);

 private:
  A& m_array;
};

template <class A>
noalias_proxy<A> noalias(A& a) noexcept;

/********************************
 * noalias_proxy implementation *
 ********************************/

template <class A>
inline noalias_proxy<A>::noalias_proxy(A& a) noexcept : m_array(a) {}

template <class A>
template <class E>
inline A& noalias_proxy<A>::operator=(const xexpression<E>& e) {
  return m_array.assign(e);
}

template <class A>
template <class E>
inline A& noalias_proxy<A>::operator+=(const xexpression<E>& e) {
  return m_array.plus_assign(e);
}

template <class A>
template <class E>
inline A& noalias_proxy<A>::operator-=(const xexpression<E>& e) {
  return m_array.minus_assign(e);
}

template <class A>
template <class E>
inline A& noalias_proxy<A>::operator*=(const xexpression<E>& e) {
  return m_array.multiplies_assign(e);
}

template <class A>
template <class E>
inline A& noalias_proxy<A>::operator/=(const xexpression<E>& e) {
  return m_array.divides_assign(e);
}

template <class A>
inline noalias_proxy<A> noalias(A& a) noexcept {
  return noalias_proxy<A>(a);
}
}

#endif
