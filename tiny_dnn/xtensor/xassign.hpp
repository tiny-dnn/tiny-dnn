/***************************************************************************
* Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XASSIGN_HPP
#define XASSIGN_HPP

#include <algorithm>
#include "xiterator.hpp"
#include "xtensor_forward.hpp"

namespace xt {
template <class E>
class xexpression;

/********************
 * Assign functions *
 ********************/

template <class E1, class E2>
void assign_data(xexpression<E1>& e1, const xexpression<E2>& e2, bool trivial);

template <class E1, class E2>
bool reshape(xexpression<E1>& e1, const xexpression<E2>& e2);

template <class E1, class E2>
void assign_xexpression(xexpression<E1>& e1, const xexpression<E2>& e2);

template <class E1, class E2>
void computed_assign(xexpression<E1>& e1, const xexpression<E2>& e2);

template <class E1, class E2, class F>
void scalar_computed_assign(xexpression<E1>& e1, const E2& e2, F&& f);

template <class E1, class E2>
void assert_compatible_shape(const xexpression<E1>& e1,
                             const xexpression<E2>& e2);

/*****************
 * data_assigner *
 *****************/

template <class E1, class E2, layout_type L>
class data_assigner {
 public:
  using lhs_iterator = typename E1::stepper;
  using rhs_iterator = typename E2::const_stepper;
  using shape_type   = typename E1::shape_type;
  using index_type   = xindex_type_t<shape_type>;
  using size_type    = typename lhs_iterator::size_type;

  data_assigner(E1& e1, const E2& e2);

  void run();

  void step(size_type i);
  void reset(size_type i);

  void to_end(layout_type);

 private:
  E1& m_e1;

  lhs_iterator m_lhs;
  rhs_iterator m_rhs;
  rhs_iterator m_rhs_end;

  index_type m_index;
};

/********************
 * trivial_assigner *
 ********************/

template <bool index_assign>
struct trivial_assigner {
  template <class E1, class E2>
  static void run(E1& e1, const E2& e2);
};

/***********************************
 * Assign functions implementation *
 ***********************************/

namespace detail {
template <class E1, class E2>
inline bool is_trivial_broadcast(const E1& e1, const E2& e2) {
  return e2.is_trivial_broadcast(e1.strides());
}

template <class D, class E2, class... SL>
inline bool is_trivial_broadcast(const xview<D, SL...>&, const E2&) {
  return false;
}
}

template <class E1, class E2>
inline void assign_data(xexpression<E1>& e1,
                        const xexpression<E2>& e2,
                        bool trivial) {
  E1& de1       = e1.derived_cast();
  const E2& de2 = e2.derived_cast();

  bool trivial_broadcast = trivial && detail::is_trivial_broadcast(de1, de2);
  if (trivial_broadcast) {
    constexpr bool contiguous_layout =
      E1::contiguous_layout && E2::contiguous_layout;
    trivial_assigner<contiguous_layout>::run(de1, de2);
  } else {
    data_assigner<E1, E2, default_assignable_layout(E1::static_layout)>
      assigner(de1, de2);
    assigner.run();
  }
}

template <class E1, class E2>
inline bool reshape(xexpression<E1>& e1, const xexpression<E2>& e2) {
  using shape_type       = typename E1::shape_type;
  using size_type        = typename E1::size_type;
  const E2& de2          = e2.derived_cast();
  size_type size         = de2.dimension();
  shape_type shape       = make_sequence<shape_type>(size, size_type(1));
  bool trivial_broadcast = de2.broadcast_shape(shape);
  e1.derived_cast().reshape(shape);
  return trivial_broadcast;
}

template <class E1, class E2>
inline void assign_xexpression(xexpression<E1>& e1, const xexpression<E2>& e2) {
  bool trivial_broadcast = reshape(e1, e2);
  assign_data(e1, e2, trivial_broadcast);
}

template <class E1, class E2>
inline void computed_assign(xexpression<E1>& e1, const xexpression<E2>& e2) {
  using shape_type = typename E1::shape_type;
  using size_type  = typename E1::size_type;

  E1& de1       = e1.derived_cast();
  const E2& de2 = e2.derived_cast();

  size_type dim          = de2.dimension();
  shape_type shape       = make_sequence<shape_type>(dim, size_type(1));
  bool trivial_broadcast = de2.broadcast_shape(shape);

  if (dim > de1.dimension() || shape > de1.shape()) {
    typename E1::temporary_type tmp(shape);
    assign_data(tmp, e2, trivial_broadcast);
    de1.assign_temporary(std::move(tmp));
  } else {
    assign_data(e1, e2, trivial_broadcast);
  }
}

template <class E1, class E2, class F>
inline void scalar_computed_assign(xexpression<E1>& e1, const E2& e2, F&& f) {
  E1& d = e1.derived_cast();
  std::transform(d.cbegin(), d.cend(), d.begin(),
                 [e2, &f](const auto& v) { return f(v, e2); });
}

template <class E1, class E2>
inline void assert_compatible_shape(const xexpression<E1>& e1,
                                    const xexpression<E2>& e2) {
  using shape_type = typename E1::shape_type;
  using size_type  = typename E1::size_type;
  const E1& de1    = e1.derived_cast();
  const E2& de2    = e2.derived_cast();
  size_type size   = de2.dimension();
  shape_type shape = make_sequence<shape_type>(size, size_type(1));
  de2.broadcast_shape(shape);
  if (shape.size() > de1.shape().size() || shape > de1.shape()) {
    throw broadcast_error(shape, de1.shape());
  }
}

/********************************
 * data_assigner implementation *
 ********************************/

template <class E1, class E2, layout_type L>
inline data_assigner<E1, E2, L>::data_assigner(E1& e1, const E2& e2)
  : m_e1(e1),
    m_lhs(e1.stepper_begin(e1.shape())),
    m_rhs(e2.stepper_begin(e1.shape())),
    m_rhs_end(e2.stepper_end(e1.shape(), L)),
    m_index(make_sequence<index_type>(e1.shape().size(), size_type(0))) {}

template <class E1, class E2, layout_type L>
inline void data_assigner<E1, E2, L>::run() {
  while (m_rhs != m_rhs_end) {
    *m_lhs = *m_rhs;
    stepper_tools<L>::increment_stepper(*this, m_index, m_e1.shape());
  }
}

template <class E1, class E2, layout_type L>
inline void data_assigner<E1, E2, L>::step(size_type i) {
  m_lhs.step(i);
  m_rhs.step(i);
}

template <class E1, class E2, layout_type L>
inline void data_assigner<E1, E2, L>::reset(size_type i) {
  m_lhs.reset(i);
  m_rhs.reset(i);
}

template <class E1, class E2, layout_type L>
inline void data_assigner<E1, E2, L>::to_end(layout_type l) {
  m_lhs.to_end(l);
  m_rhs.to_end(l);
}

/***********************************
 * trivial_assigner implementation *
 ***********************************/

template <bool index_assign>
template <class E1, class E2>
inline void trivial_assigner<index_assign>::run(E1& e1, const E2& e2) {
  using size_type = typename E1::size_type;
  size_type size  = e1.size();
  for (size_type i = 0; i < size; ++i) {
    e1.data_element(i) = e2.data_element(i);
  }
}

template <>
template <class E1, class E2>
inline void trivial_assigner<false>::run(E1& e1, const E2& e2) {
  std::copy(e2.storage_cbegin(), e2.storage_cend(), e1.storage_begin());
}
}

#endif
