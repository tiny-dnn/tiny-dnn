/***************************************************************************
* Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

/**
 * @brief functions to obtain xgenerators generating random numbers with given
 * shape
 */

#ifndef XRANDOM_HPP
#define XRANDOM_HPP

#include <functional>
#include <random>
#include <utility>

#include "xgenerator.hpp"

namespace xt {
/*********************
 * Random generators *
 *********************/

namespace random {
using default_engine_type = std::mt19937;
using seed_type           = default_engine_type::result_type;

default_engine_type& get_default_random_engine();
void seed(seed_type seed);

template <class T, class S, class E = random::default_engine_type>
auto rand(const S& shape,
          T lower   = 0,
          T upper   = 1,
          E& engine = random::get_default_random_engine());

template <class T, class S, class E = random::default_engine_type>
auto randint(const S& shape,
             T lower   = 0,
             T upper   = std::numeric_limits<T>::max(),
             E& engine = random::get_default_random_engine());

template <class T, class S, class E = random::default_engine_type>
auto randn(const S& shape,
           T mean    = 0,
           T std_dev = 1,
           E& engine = random::get_default_random_engine());

#ifdef X_OLD_CLANG
template <class T, class I, class E = random::default_engine_type>
auto rand(std::initializer_list<I> shape,
          T lower   = 0,
          T upper   = 1,
          E& engine = random::get_default_random_engine());

template <class T, class I, class E = random::default_engine_type>
auto randint(std::initializer_list<I> shape,
             T lower   = 0,
             T upper   = std::numeric_limits<T>::max(),
             E& engine = random::get_default_random_engine());

template <class T, class I, class E = random::default_engine_type>
auto randn(std::initializer_list<I>,
           T mean    = 0,
           T std_dev = 1,
           E& engine = random::get_default_random_engine());
#else
template <class T,
          class I,
          std::size_t L,
          class E = random::default_engine_type>
auto rand(const I (&shape)[L],
          T lower   = 0,
          T upper   = 1,
          E& engine = random::get_default_random_engine());

template <class T,
          class I,
          std::size_t L,
          class E = random::default_engine_type>
auto randint(const I (&shape)[L],
             T lower   = 0,
             T upper   = std::numeric_limits<T>::max(),
             E& engine = random::get_default_random_engine());

template <class T,
          class I,
          std::size_t L,
          class E = random::default_engine_type>
auto randn(const I (&shape)[L],
           T mean    = 0,
           T std_dev = 1,
           E& engine = random::get_default_random_engine());
#endif
}

namespace detail {
template <class T>
struct random_impl {
  using value_type = T;

  random_impl(std::function<value_type()>&& generator)
    : m_generator(std::move(generator)) {}

  template <class... Args>
  inline value_type operator()(Args...) const {
    return m_generator();
  }

  template <class It>
  inline value_type element(It, It) const {
    return m_generator();
  }

 private:
  std::function<value_type()> m_generator;
};
}

namespace random {
/**
 * Returns a reference to the default random number engine
 */
inline default_engine_type& get_default_random_engine() {
  static default_engine_type mt;
  return mt;
}

/**
 * Seeds the default random number generator with @p seed
 * @param seed The seed
 */
inline void seed(seed_type seed) { get_default_random_engine().seed(seed); }

/**
 * xexpression with specified @p shape containing uniformly distributed random
 * numbers
 * in the interval from @p lower to @p upper, excluding upper.
 *
 * Numbers are drawn from @c std::uniform_real_distribution.
 *
 * @param shape shape of resulting xexpression
 * @param lower lower bound
 * @param upper upper bound
 * @param engine random number engine
 * @tparam T number type to use
 */
template <class T, class S, class E>
inline auto rand(const S& shape, T lower, T upper, E& engine) {
  std::uniform_real_distribution<T> dist(lower, upper);
  return detail::make_xgenerator(
    detail::random_impl<T>(std::bind(dist, std::ref(engine))), shape);
}

/**
 * xexpression with specified @p shape containing uniformly distributed
 * random integers in the interval from @p lower to @p upper, excluding upper.
 *
 * Numbers are drawn from @c std::uniform_int_distribution.
 *
 * @param shape shape of resulting xexpression
 * @param lower lower bound
 * @param upper upper bound
 * @param engine random number engine
 * @tparam T number type to use
 */
template <class T, class S, class E>
inline auto randint(const S& shape, T lower, T upper, E& engine) {
  std::uniform_int_distribution<T> dist(lower, upper - 1);
  return detail::make_xgenerator(
    detail::random_impl<T>(std::bind(dist, std::ref(engine))), shape);
}

/**
 * xexpression with specified @p shape containing numbers sampled from
 * the Normal (Gaussian) random number distribution with mean @p mean and
 * standard deviation @p std_dev.
 *
 * Numbers are drawn from @c std::normal_distribution.
 *
 * @param shape shape of resulting xexpression
 * @param mean mean of normal distribution
 * @param std_dev standard deviation of normal distribution
 * @param engine random number engine
 * @tparam T number type to use
 */
template <class T, class S, class E>
inline auto randn(const S& shape, T mean, T std_dev, E& engine) {
  std::normal_distribution<T> dist(mean, std_dev);
  return detail::make_xgenerator(
    detail::random_impl<T>(std::bind(dist, std::ref(engine))), shape);
}

#ifdef X_OLD_CLANG
template <class T, class I, class E>
inline auto rand(std::initializer_list<I> shape, T lower, T upper, E& engine) {
  std::uniform_real_distribution<T> dist(lower, upper);
  return detail::make_xgenerator(
    detail::random_impl<T>(std::bind(dist, std::ref(engine))), shape);
}

template <class T, class I, class E>
inline auto randint(std::initializer_list<I> shape,
                    T lower,
                    T upper,
                    E& engine) {
  std::uniform_int_distribution<T> dist(lower, upper - 1);
  return detail::make_xgenerator(
    detail::random_impl<T>(std::bind(dist, std::ref(engine))), shape);
}

template <class T, class I, class E>
inline auto randn(std::initializer_list<I> shape,
                  T mean,
                  T std_dev,
                  E& engine) {
  std::normal_distribution<T> dist(mean, std_dev);
  return detail::make_xgenerator(
    detail::random_impl<T>(std::bind(dist, std::ref(engine))), shape);
}
#else
template <class T, class I, std::size_t L, class E>
inline auto rand(const I (&shape)[L], T lower, T upper, E& engine) {
  std::uniform_real_distribution<T> dist(lower, upper);
  return detail::make_xgenerator(
    detail::random_impl<T>(std::bind(dist, std::ref(engine))), shape);
}

template <class T, class I, std::size_t L, class E>
inline auto randint(const I (&shape)[L], T lower, T upper, E& engine) {
  std::uniform_int_distribution<T> dist(lower, upper - 1);
  return detail::make_xgenerator(
    detail::random_impl<T>(std::bind(dist, std::ref(engine))), shape);
}

template <class T, class I, std::size_t L, class E>
inline auto randn(const I (&shape)[L], T mean, T std_dev, E& engine) {
  std::normal_distribution<T> dist(mean, std_dev);
  return detail::make_xgenerator(
    detail::random_impl<T>(std::bind(dist, std::ref(engine))), shape);
}
#endif
}
}

#endif
