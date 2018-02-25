/***************************************************************************
* Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XTENSOR_ACCUMULATOR_HPP
#define XTENSOR_ACCUMULATOR_HPP

#include <algorithm>
#include <numeric>
#include <type_traits>
#include <cstddef>

#include "xtensor_forward.hpp"
#include "xstrides.hpp"
#include "xexpression.hpp"
#include <iostream>

namespace xt
{

#define DEFAULT_STRATEGY_ACCUMULATORS evaluation_strategy::immediate

    /**************
     * accumulate *
     **************/

    namespace detail
    {
        template <class F, class E, class ES>
        xarray<typename std::decay_t<E>::value_type> accumulator_impl(F&&, E&&, std::size_t, ES)
        {
            static_assert(!std::is_same<evaluation_strategy::lazy, ES>::value, "Lazy accumulators not yet implemented.");
        }

        template <class F, class E, class ES>
        xarray<typename std::decay_t<E>::value_type> accumulator_impl(F&&, E&&, ES)
        {
            static_assert(!std::is_same<evaluation_strategy::lazy, ES>::value, "Lazy accumulators not yet implemented.");
        }

        template <class T, class R>
        struct xaccumulator_return_type
        {
            using type = xarray<R>;
        };

        template <class T, std::size_t N, class R>
        struct xaccumulator_return_type<xtensor<T, N>, R>
        {
            using type = xtensor<R, N>;
        };

        template <class T, class R>
        using xaccumulator_return_type_t = typename xaccumulator_return_type<T, R>::type;

        template <class F, class E>
        inline auto accumulator_impl(F&& f, E&& e, std::size_t axis, evaluation_strategy::immediate)
        {
            using function_return_type = typename F::result_type;
            using result_type = xaccumulator_return_type_t<std::decay_t<E>, function_return_type>;

            if (axis >= e.dimension())
            {
                throw std::runtime_error("Axis larger than expression dimension in accumulator.");
            }

            result_type result = e;  // assign + make a copy, we need it anyways

            std::size_t inner_stride = result.strides()[axis];
            std::size_t outer_stride = 1; // this is either going row- or column-wise (strides.back / strides.front)
            std::size_t outer_loop_size = 0;
            std::size_t inner_loop_size = 0;

            auto set_loop_sizes = [&outer_loop_size, &inner_loop_size](auto first, auto last, ptrdiff_t ax)
            {
                outer_loop_size = std::accumulate(first,
                                                  first + ax,
                                                  std::size_t(1), std::multiplies<std::size_t>());

                inner_loop_size = std::accumulate(first + ax,
                                                  last,
                                                  std::size_t(1), std::multiplies<std::size_t>());
            };

            if (result_type::static_layout == layout_type::row_major)
            {
                set_loop_sizes(result.shape().cbegin(), result.shape().cend(), static_cast<ptrdiff_t>(axis));
            }
            else
            {
                set_loop_sizes(result.shape().cbegin(), result.shape().cend(), static_cast<ptrdiff_t>(axis + 1));
                std::swap(inner_loop_size, outer_loop_size);
            }
            inner_loop_size = inner_loop_size - inner_stride;

            std::size_t pos = 0;
            for (std::size_t i = 0; i < outer_loop_size; ++i)
            {
                for (std::size_t j = 0; j < inner_loop_size; ++j)
                {
                    result.data()[pos + inner_stride] = f(result.data()[pos],
                                                          result.data()[pos + inner_stride]);
                    pos += outer_stride;
                }
                pos += inner_stride;
            }
            return result;
        }

        template <class F, class E>
        inline auto accumulator_impl(F&& f, E&& e, evaluation_strategy::immediate)
        {
            using T = typename F::result_type;
            using result_type = xtensor<T, 1>;
            std::size_t sz = e.size();
            auto result = result_type::from_shape({sz});

            auto it = e.template begin<DEFAULT_LAYOUT>();

            result.data()[0] = *it;
            ++it;

            for (std::size_t idx = 0; it != e.template end<DEFAULT_LAYOUT>(); ++it)
            {
                result.data()[idx + 1] = f(result.data()[idx], *it);
                ++idx;
            }
            return result;
        }
    }

    /**
     * Accumulate and flatten array
     * **NOTE** This function is not lazy!
     *
     * @param f functor to use for accumulation
     * @param e xexpression to be accumulated
     * @param evaluation_strategy evaluation strategy of the accumulation
     *
     * @return returns xarray<T> filled with accumulated values
     */
    template <class F, class E, class ES = DEFAULT_STRATEGY_ACCUMULATORS,
              typename std::enable_if_t<!std::is_integral<ES>::value, int> = 0>
    inline auto accumulate(F&& f, E&& e, ES evaluation_strategy = ES())
    {
        // Note we need to check is_integral above in order to prohibit ES = int, and not taking the std::size_t
        // overload below!
        return detail::accumulator_impl(std::forward<F>(f), std::forward<E>(e), evaluation_strategy);
    }

    /**
     * Accumulate over axis
     * **NOTE** This function is not lazy!
     *
     * @param f Functor to use for accumulation
     * @param e xexpression to accumulate
     * @param axis Axis to perform accumulation over
     * @param evaluation_strategy evaluation strategy of the accumulation
     *
     * @return returns xarray<T> filled with accumulated values
     */
    template <class F, class E, class ES = DEFAULT_STRATEGY_ACCUMULATORS>
    inline auto accumulate(F&& f, E&& e, std::size_t axis, ES evaluation_strategy = ES())
    {
        return detail::accumulator_impl(std::forward<F>(f), std::forward<E>(e), axis, evaluation_strategy);
    }

}

#endif