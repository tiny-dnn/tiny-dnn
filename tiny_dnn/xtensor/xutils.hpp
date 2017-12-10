/***************************************************************************
* Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XUTILS_HPP
#define XUTILS_HPP

#include <algorithm>
#include <array>
#include <complex>
#include <cstddef>
#include <initializer_list>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "xtensor_config.hpp"

namespace xt {
/****************
 * declarations *
 ****************/

template <class T>
struct remove_class;

template <class F, class... T>
void for_each(F&& f, std::tuple<T...>& t) noexcept(noexcept(std::declval<F>()));

template <class F, class R, class... T>
R accumulate(F&& f,
             R init,
             const std::tuple<T...>& t) noexcept(noexcept(std::declval<F>()));

template <class... T>
struct or_;

template <class... T>
struct and_;

template <bool... B>
struct or_c;

template <bool... B>
struct and_c;

template <std::size_t I, class... Args>
constexpr decltype(auto) argument(Args&&... args) noexcept;

template <class R, class F, class... S>
R apply(std::size_t index,
        F&& func,
        const std::tuple<S...>& s) noexcept(noexcept(std::declval<F>()));

template <class T, class S>
void nested_copy(T&& iter, const S& s);

template <class T, class S>
void nested_copy(T&& iter, std::initializer_list<S> s);

template <class U>
struct initializer_dimension;

template <class R, class T>
constexpr R shape(T t);

template <class T, class S>
constexpr bool check_shape(T t, S first, S last);

template <class C>
bool resize_container(C& c, typename C::size_type size);

template <class T, std::size_t N>
bool resize_container(std::array<T, N>& a,
                      typename std::array<T, N>::size_type size);

template <class S>
S make_sequence(typename S::size_type size, typename S::value_type v);

template <class R, class A>
decltype(auto) forward_sequence(A&& s);

// equivalent to std::size(c) in c++17
template <class C>
constexpr auto sequence_size(const C& c) -> decltype(c.size());

// equivalent to std::size(a) in c++17
template <class T, std::size_t N>
constexpr std::size_t sequence_size(const T (&a)[N]);

/*******************************
 * remove_class implementation *
 *******************************/

template <class T>
struct remove_class {};

template <class C, class R, class... Args>
struct remove_class<R (C::*)(Args...)> {
  typedef R type(Args...);
};

template <class C, class R, class... Args>
struct remove_class<R (C::*)(Args...) const> {
  typedef R type(Args...);
};

template <class T>
using remove_class_t = typename remove_class<T>::type;

/***************************
 * for_each implementation *
 ***************************/

namespace detail {
template <std::size_t I, class F, class... T>
inline typename std::enable_if<I == sizeof...(T), void>::type for_each_impl(
  F&& /*f*/, std::tuple<T...>& /*t*/) noexcept(noexcept(std::declval<F>())) {}

template <std::size_t I, class F, class... T>
  inline typename std::enable_if <
  I<sizeof...(T), void>::type for_each_impl(
    F&& f, std::tuple<T...>& t) noexcept(noexcept(std::declval<F>())) {
  f(std::get<I>(t));
  for_each_impl<I + 1, F, T...>(std::forward<F>(f), t);
}
}

template <class F, class... T>
inline void for_each(F&& f, std::tuple<T...>& t) noexcept(
  noexcept(std::declval<F>())) {
  detail::for_each_impl<0, F, T...>(std::forward<F>(f), t);
}

/*****************************
 * accumulate implementation *
 *****************************/

namespace detail {
template <std::size_t I, class F, class R, class... T>
inline std::enable_if_t<I == sizeof...(T), R> accumulate_impl(
  F&& /*f*/,
  R init,
  const std::tuple<T...>& /*t*/) noexcept(noexcept(std::declval<F>())) {
  return init;
}

template <std::size_t I, class F, class R, class... T>
  inline std::enable_if_t <
  I<sizeof...(T), R> accumulate_impl(
    F&& f,
    R init,
    const std::tuple<T...>& t) noexcept(noexcept(std::declval<F>())) {
  R res = f(init, std::get<I>(t));
  return accumulate_impl<I + 1, F, R, T...>(std::forward<F>(f), res, t);
}
}

template <class F, class R, class... T>
inline R accumulate(F&& f, R init, const std::tuple<T...>& t) noexcept(
  noexcept(std::declval<F>())) {
  return detail::accumulate_impl<0, F, R, T...>(f, init, t);
}

/**********************
 * or_ implementation *
 **********************/

template <>
struct or_<> : std::integral_constant<bool, false> {};

template <class T, class... Ts>
struct or_<T, Ts...>
  : std::integral_constant<bool, T::value || or_<Ts...>::value> {};

/***********************
 * and_ implementation *
 ***********************/

template <>
struct and_<> : std::integral_constant<bool, true> {};

template <class T, class... Ts>
struct and_<T, Ts...>
  : std::integral_constant<bool, T::value && and_<Ts...>::value> {};

/**********************************
 * or_c and and_c implementations *
 **********************************/

template <bool... B>
struct or_c : or_<std::integral_constant<bool, B>...> {};

template <bool... B>
struct and_c : and_<std::integral_constant<bool, B>...> {};

/***************************
 * argument implementation *
 ***************************/

namespace detail {
template <std::size_t I>
struct getter {
  template <class Arg, class... Args>
  static constexpr decltype(auto) get(Arg&& /*arg*/, Args&&... args) noexcept {
    return getter<I - 1>::get(std::forward<Args>(args)...);
  }
};

template <>
struct getter<0> {
  template <class Arg, class... Args>
  static constexpr Arg&& get(Arg&& arg, Args&&... /*args*/) noexcept {
    return std::forward<Arg>(arg);
  }
};
}

template <std::size_t I, class... Args>
constexpr decltype(auto) argument(Args&&... args) noexcept {
  static_assert(I < sizeof...(Args), "I should be lesser than sizeof...(Args)");
  return detail::getter<I>::get(std::forward<Args>(args)...);
}

/************************
 * apply implementation *
 ************************/

namespace detail {
template <class R, class F, std::size_t I, class... S>
R apply_one(F&& func,
            const std::tuple<S...>& s) noexcept(noexcept(std::declval<F>())) {
  return func(std::get<I>(s));
}

template <class R, class F, std::size_t... I, class... S>
R apply(std::size_t index,
        F&& func,
        std::index_sequence<I...> /*seq*/,
        const std::tuple<S...>& s) noexcept(noexcept(std::declval<F>())) {
  using FT = std::add_pointer_t<R(F&&, const std::tuple<S...>&)>;
  static const std::array<FT, sizeof...(I)> ar = {
    {&apply_one<R, F, I, S...>...}};
  return ar[index](std::forward<F>(func), s);
}
}

template <class R, class F, class... S>
inline R apply(std::size_t index, F&& func, const std::tuple<S...>& s) noexcept(
  noexcept(std::declval<F>())) {
  return detail::apply<R>(index, std::forward<F>(func),
                          std::make_index_sequence<sizeof...(S)>(), s);
}

/***************************
 * nested_initializer_list *
 ***************************/

template <class T, std::size_t I>
struct nested_initializer_list {
  using type =
    std::initializer_list<typename nested_initializer_list<T, I - 1>::type>;
};

template <class T>
struct nested_initializer_list<T, 0> {
  using type = T;
};

template <class T, std::size_t I>
using nested_initializer_list_t = typename nested_initializer_list<T, I>::type;

/******************************
 * nested_copy implementation *
 ******************************/

template <class T, class S>
inline void nested_copy(T&& iter, const S& s) {
  *iter++ = s;
}

template <class T, class S>
inline void nested_copy(T&& iter, std::initializer_list<S> s) {
  for (auto it = s.begin(); it != s.end(); ++it) {
    nested_copy(std::forward<T>(iter), *it);
  }
}

/****************************************
 * initializer_dimension implementation *
 ****************************************/

namespace detail {
template <class U>
struct initializer_depth_impl {
  static constexpr std::size_t value = 0;
};

template <class T>
struct initializer_depth_impl<std::initializer_list<T>> {
  static constexpr std::size_t value = 1 + initializer_depth_impl<T>::value;
};
}

template <class U>
struct initializer_dimension {
  static constexpr std::size_t value = detail::initializer_depth_impl<U>::value;
};

/************************************
 * initializer_shape implementation *
 ************************************/

namespace detail {
template <std::size_t I>
struct initializer_shape_impl {
  template <class T>
  static constexpr std::size_t value(T t) {
    return t.size() == 0 ? 0 : initializer_shape_impl<I - 1>::value(*t.begin());
  }
};

template <>
struct initializer_shape_impl<0> {
  template <class T>
  static constexpr std::size_t value(T t) {
    return t.size();
  }
};

template <class R, class U, std::size_t... I>
constexpr R initializer_shape(U t, std::index_sequence<I...>) {
  using size_type = typename R::value_type;
  return {size_type(initializer_shape_impl<I>::value(t))...};
}
}

template <class R, class T>
constexpr R shape(T t) {
  return detail::initializer_shape<R, decltype(t)>(
    t, std::make_index_sequence<initializer_dimension<decltype(t)>::value>());
}

/******************************
 * check_shape implementation *
 ******************************/

namespace detail {
template <class T, class S>
struct predshape {
  constexpr predshape(S first, S last) : m_first(first), m_last(last) {}

  constexpr bool operator()(const T&) const { return m_first == m_last; }

  S m_first;
  S m_last;
};

template <class T, class S>
struct predshape<std::initializer_list<T>, S> {
  constexpr predshape(S first, S last) : m_first(first), m_last(last) {}

  constexpr bool operator()(std::initializer_list<T> t) const {
    return *m_first == t.size() &&
           std::all_of(t.begin(), t.end(),
                       predshape<T, S>(m_first + 1, m_last));
  }

  S m_first;
  S m_last;
};
}

template <class T, class S>
constexpr bool check_shape(T t, S first, S last) {
  return detail::predshape<decltype(t), S>(first, last)(t);
}

/***********************************
 * resize_container implementation *
 ***********************************/

template <class C>
inline bool resize_container(C& c, typename C::size_type size) {
  c.resize(size);
  return true;
}

template <class T, std::size_t N>
inline bool resize_container(std::array<T, N>& /*a*/,
                             typename std::array<T, N>::size_type size) {
  return size == N;
}

/********************************
 * make_sequence implementation *
 ********************************/

namespace detail {
template <class S>
struct sequence_builder {
  using value_type = typename S::value_type;
  using size_type  = typename S::size_type;

  inline static S make(size_type size, value_type v) { return S(size, v); }
};

template <class T, std::size_t N>
struct sequence_builder<std::array<T, N>> {
  using sequence_type = std::array<T, N>;
  using value_type    = typename sequence_type::value_type;
  using size_type     = typename sequence_type::size_type;

  inline static sequence_type make(size_type /*size*/, value_type v) {
    sequence_type s;
    s.fill(v);
    return s;
  }
};
}

template <class S>
inline S make_sequence(typename S::size_type size, typename S::value_type v) {
  return detail::sequence_builder<S>::make(size, v);
}

/***********************************
 * forward_sequence implementation *
 ***********************************/

namespace detail {
template <class R, class A, class E = void>
struct sequence_forwarder {
  template <class T>
  static inline R forward(const T& r) {
    return R(std::begin(r), std::end(r));
  }
};

template <class I, std::size_t L, class A>
struct sequence_forwarder<
  std::array<I, L>,
  A,
  std::enable_if_t<!std::is_same<std::array<I, L>, A>::value>> {
  using R = std::array<I, L>;

  template <class T>
  static inline R forward(const T& r) {
    R ret;
    std::copy(std::begin(r), std::end(r), std::begin(ret));
    return ret;
  }
};

template <class R>
struct sequence_forwarder<R, R> {
  template <class T>
  static inline T&& forward(
    typename std::remove_reference<T>::type& t) noexcept {
    return static_cast<T&&>(t);
  }

  template <class T>
  static inline T&& forward(
    typename std::remove_reference<T>::type&& t) noexcept {
    return static_cast<T&&>(t);
  }
};
}

template <class R, class A>
inline decltype(auto) forward_sequence(A&& s) {
  using forwarder =
    detail::sequence_forwarder<std::decay_t<R>,
                               std::remove_cv_t<std::remove_reference_t<A>>>;
  return forwarder::template forward<A>(s);
}

/*************************************
 * promote_shape and promote_strides *
 *************************************/

namespace detail {
template <class T1, class T2>
constexpr std::common_type_t<T1, T2> imax(const T1& a, const T2& b) {
  return a > b ? a : b;
}

// Variadic meta-function returning the maximal size of std::arrays.
template <class... T>
struct max_array_size;

template <>
struct max_array_size<> {
  static constexpr std::size_t value = 0;
};

template <class T, class... Ts>
struct max_array_size<T, Ts...>
  : std::integral_constant<std::size_t,
                           imax(std::tuple_size<T>::value,
                                max_array_size<Ts...>::value)> {};

// Simple is_array and only_array meta-functions
template <class S>
struct is_array {
  static constexpr bool value = false;
};

template <class T, std::size_t N>
struct is_array<std::array<T, N>> {
  static constexpr bool value = true;
};

template <class... S>
using only_array = and_<is_array<S>...>;

// The promote_index meta-function returns std::vector<promoted_value_type> in
// the
// general case and an array of the promoted value type and maximal size if all
// arguments are of type std::array

template <bool A, class... S>
struct promote_index_impl;

template <class... S>
struct promote_index_impl<false, S...> {
  using type =
    std::vector<typename std::common_type<typename S::value_type...>::type>;
};

template <class... S>
struct promote_index_impl<true, S...> {
  using type =
    std::array<typename std::common_type<typename S::value_type...>::type,
               max_array_size<S...>::value>;
};

template <>
struct promote_index_impl<true> {
  using type = std::array<std::size_t, 0>;
};

template <class... S>
struct promote_index {
  using type = typename promote_index_impl<only_array<S...>::value, S...>::type;
};
}

template <class... S>
using promote_shape_t = typename detail::promote_index<S...>::type;

template <class... S>
using promote_strides_t = typename detail::promote_index<S...>::type;

/**************************
 * closure implementation *
 **************************/

template <class S>
struct closure {
  using underlying_type =
    std::conditional_t<std::is_const<std::remove_reference_t<S>>::value,
                       const std::decay_t<S>,
                       std::decay_t<S>>;
  using type = typename std::conditional<std::is_lvalue_reference<S>::value,
                                         underlying_type&,
                                         underlying_type>::type;
};

template <class S>
using closure_t = typename closure<S>::type;

template <class S>
struct const_closure {
  using underlying_type = const std::decay_t<S>;
  using type = typename std::conditional<std::is_lvalue_reference<S>::value,
                                         underlying_type&,
                                         underlying_type>::type;
};

template <class S>
using const_closure_t = typename const_closure<S>::type;

/******************************
 * ptr_closure implementation *
 ******************************/

template <class S>
struct ptr_closure {
  using underlying_type =
    std::conditional_t<std::is_const<std::remove_reference_t<S>>::value,
                       const std::decay_t<S>,
                       std::decay_t<S>>;
  using type = std::conditional_t<std::is_lvalue_reference<S>::value,
                                  underlying_type*,
                                  underlying_type>;
};

template <class S>
using ptr_closure_t = typename ptr_closure<S>::type;

template <class S>
struct const_ptr_closure {
  using underlying_type = const std::decay_t<S>;
  using type            = std::conditional_t<std::is_lvalue_reference<S>::value,
                                  underlying_type*,
                                  underlying_type>;
};

template <class S>
using const_ptr_closure_t = typename const_ptr_closure<S>::type;

/***************************
 * apply_cv implementation *
 ***************************/

namespace detail {
template <class T,
          class U,
          bool = std::is_const<std::remove_reference_t<T>>::value,
          bool = std::is_volatile<std::remove_reference_t<T>>::value>
struct apply_cv_impl {
  using type = U;
};

template <class T, class U>
struct apply_cv_impl<T, U, true, false> {
  using type = const U;
};

template <class T, class U>
struct apply_cv_impl<T, U, false, true> {
  using type = volatile U;
};

template <class T, class U>
struct apply_cv_impl<T, U, true, true> {
  using type = const volatile U;
};

template <class T, class U>
struct apply_cv_impl<T&, U, false, false> {
  using type = U&;
};

template <class T, class U>
struct apply_cv_impl<T&, U, true, false> {
  using type = const U&;
};

template <class T, class U>
struct apply_cv_impl<T&, U, false, true> {
  using type = volatile U&;
};

template <class T, class U>
struct apply_cv_impl<T&, U, true, true> {
  using type = const volatile U&;
};
}

template <class T, class U>
struct apply_cv {
  using type = typename detail::apply_cv_impl<T, U>::type;
};

template <class T, class U>
using apply_cv_t = typename apply_cv<T, U>::type;

/*****************************
 * is_complex implementation *
 *****************************/

namespace detail {
template <class T>
struct is_complex : public std::false_type {};

template <class T>
struct is_complex<std::complex<T>> : public std::true_type {};
}

template <class T>
struct is_complex {
  static constexpr bool value = detail::is_complex<std::decay_t<T>>::value;
};

/*************************************
 * complex_value_type implementation *
 *************************************/

template <class T>
struct complex_value_type {
  using type = T;
};

template <class T>
struct complex_value_type<std::complex<T>> {
  using type = T;
};

template <class T>
using complex_value_type_t = typename complex_value_type<T>::type;

/*********************************
 * forward_offset implementation *
 *********************************/

namespace detail {

template <class T, class M>
struct forward_type {
  using type = apply_cv_t<T, M>&&;
};

template <class T, class M>
struct forward_type<T&, M> {
  using type = apply_cv_t<T, M>&;
};

template <class T, class M>
using forward_type_t = typename forward_type<T, M>::type;
}

template <class M, std::size_t I, class T>
constexpr detail::forward_type_t<T, M> forward_offset(T&& v) noexcept {
  using forward_type  = detail::forward_type_t<T, M>;
  using cv_value_type = std::remove_reference_t<forward_type>;
  using byte_type     = apply_cv_t<std::remove_reference_t<T>, char>;

  return static_cast<forward_type>(
    *reinterpret_cast<cv_value_type*>(reinterpret_cast<byte_type*>(&v) + I));
}

/**********************************************
 * forward_real & forward_imag implementation *
 **********************************************/

// forward_real

template <class T>
auto forward_real(T&& v)
  -> std::enable_if_t<!is_complex<T>::value,
                      detail::forward_type_t<T, T>>  // real case -> forward
{
  return static_cast<detail::forward_type_t<T, T>>(v);
}

template <class T>
auto forward_real(T&& v) -> std::enable_if_t<
  is_complex<T>::value,
  detail::forward_type_t<T, typename std::decay_t<T>::value_type>>  // complex
                                                                    // case ->
                                                                    // forward
                                                                    // the real
                                                                    // part
{
  return forward_offset<typename std::decay_t<T>::value_type, 0>(v);
}

// forward_imag

template <class T>
auto forward_imag(T &&)
  -> std::enable_if_t<!is_complex<T>::value,
                      std::decay_t<T>>  // real case -> always return 0 by value
{
  return 0;
}

template <class T>
auto forward_imag(T&& v) -> std::enable_if_t<
  is_complex<T>::value,
  detail::forward_type_t<T, typename std::decay_t<T>::value_type>>  // complex
                                                                    // case ->
                                                                    // forwards
                                                                    // the
                                                                    // imaginary
                                                                    // part
{
  using real_type = typename std::decay_t<T>::value_type;
  return forward_offset<real_type, sizeof(real_type)>(v);
}

/**************************
* to_array implementation *
***************************/

namespace detail {
template <class T, std::size_t N, std::size_t... I>
constexpr std::array<std::remove_cv_t<T>, N> to_array_impl(
  T (&a)[N], std::index_sequence<I...>) {
  return {{a[I]...}};
}
}

template <class T, std::size_t N>
constexpr std::array<std::remove_cv_t<T>, N> to_array(T (&a)[N]) {
  return detail::to_array_impl(a, std::make_index_sequence<N>{});
}

/********************************
 * sequence_size implementation *
 ********************************/

// equivalent to std::size(c) in c++17
template <class C>
constexpr auto sequence_size(const C& c) -> decltype(c.size()) {
  return c.size();
}

// equivalent to std::size(a) in c++17
template <class T, std::size_t N>
constexpr std::size_t sequence_size(const T (&)[N]) {
  return N;
}

/*****************************************
 * has_raw_data_interface implementation *
 *****************************************/

template <class T>
class has_raw_data_interface {
  template <class C>
  static std::true_type test(decltype(std::declval<C>().raw_data_offset()));

  template <class C>
  static std::false_type test(...);

 public:
  constexpr static bool value =
    decltype(test<T>(std::size_t(0)))::value == true;
};

/******************
 * enable_if_type *
 ******************/

template <class T>
struct enable_if_type {
  using type = void;
};

/*****************************
 * is_complete implemenation *
 *****************************/

namespace detail {
template <class T>
class is_complete_impl {
  template <class U>
  static auto test(U*) -> std::integral_constant<bool, sizeof(U) == sizeof(U)>;

  static auto test(...) -> std::false_type;

 public:
  using type = decltype(test((T*)0));
};
}

template <class T>
struct is_complete : detail::is_complete_impl<T>::type {};

/*************
 * static_if *
 *************/

namespace static_if_detail {
struct identity {
  template <class T>
  T&& operator()(T&& x) const {
    return std::forward<T>(x);
  }
};
}

template <class TF, class FF>
auto static_if(std::true_type, const TF& tf, const FF&) {
  return tf(static_if_detail::identity());
}

template <class TF, class FF>
auto static_if(std::false_type, const TF&, const FF& ff) {
  return ff(static_if_detail::identity());
}

template <bool cond, class TF, class FF>
auto static_if(const TF& tf, const FF& ff) {
  return static_if(std::integral_constant<bool, cond>(), tf, ff);
}

/********************************************
 * xtrivial_default_construct implemenation *
 ********************************************/

#if !defined(__GNUG__) || defined(_LIBCPP_VERSION) || \
  defined(_GLIBCXX_USE_CXX11_ABI)

template <class T>
using xtrivially_default_constructible =
  std::is_trivially_default_constructible<T>;

#else

template <class T>
using xtrivially_default_constructible =
  std::has_trivial_default_constructor<T>;

#endif
}

#endif
