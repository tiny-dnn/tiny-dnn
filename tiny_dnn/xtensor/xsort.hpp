/***************************************************************************
* Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XTENSOR_SORT_HPP
#define XTENSOR_SORT_HPP

#include <algorithm>

#include "xarray.hpp"
#include "xeval.hpp"
#include "xstrided_view.hpp"
#include "xslice.hpp"  // for xnone
#include "xtensor.hpp"

namespace xt
{
    template <class E>
    auto sort(const xexpression<E>& e, placeholders::xtuph /*t*/)
    {
        using value_type = typename E::value_type;
        const auto de = e.derived_cast();
        E ev;
        ev.resize({ de.size() });

        std::copy(de.begin(), de.end(), ev.begin());
        std::sort(ev.begin(), ev.end());

        return ev;
    }

    namespace detail
    {
        template <class E, class F>
        void call_over_leading_axis(E& ev, F&& fct)
        {
            using value_type = typename E::value_type;
            std::size_t n_iters = 1;
            ptrdiff_t secondary_stride;
            if (ev.layout() == layout_type::row_major)
            {
                n_iters = std::accumulate(ev.shape().begin(), ev.shape().end() - 1,
                                          std::size_t(1), std::multiplies<>());
                secondary_stride = static_cast<ptrdiff_t>(ev.strides()[ev.dimension() - 2]);
            }
            else
            {
                n_iters = std::accumulate(ev.shape().begin() + 1, ev.shape().end(),
                                          std::size_t(1), std::multiplies<>());
                secondary_stride = static_cast<ptrdiff_t>(ev.strides()[1]);
            }

            ptrdiff_t offset = 0;

            for (std::size_t i = 0; i < n_iters; ++i, offset += secondary_stride)
            {
                fct(ev.raw_data() + offset, ev.raw_data() + offset + secondary_stride);
            }
        }

        template <class E>
        inline std::size_t leading_axis(const E& e)
        {
            if (e.layout() == layout_type::row_major)
            {
                return e.dimension() - 1;
            }
            else if (e.layout() == layout_type::column_major)
            {
                return 0;
            }
            throw std::runtime_error("Layout not supported.");
        }
    }

    /**
     * Sort xexpression (optionally along axis)
     * The sort is performed using the ``std::sort`` functions.
     * A copy of the xexpression is created and returned.
     *
     * @param e xexpression to sort
     * @param axis axis along which sort is performed
     *
     * @return sorted array (copy)
     */
    template <class E>
    auto sort(const xexpression<E>& e, std::size_t axis)
    {
        using eval_type = typename E::temporary_type;
        using value_type = typename E::value_type;

        const auto& de = e.derived_cast();

        if (de.dimension() == 1)
        {
            return sort(de, xnone());
        }

        eval_type ev;

        if (axis != detail::leading_axis(ev))
        {
            auto axis_numbers = arange<std::size_t>(de.shape().size());
            std::vector<std::size_t> permutation(axis_numbers.begin(), axis_numbers.end());
            permutation.erase(permutation.begin() + ptrdiff_t(axis));
            if (de.layout() == layout_type::row_major)
            {
                permutation.push_back(axis);
            }
            else
            {
                permutation.insert(permutation.begin(), axis);
            }

            // TODO find a more clever way to get reverse permutation?
            std::vector<std::size_t> reverse_permutation;
            for (auto el : axis_numbers)
            {
                auto it = std::find(permutation.begin(), permutation.end(), el);
                reverse_permutation.push_back(std::size_t(std::distance(permutation.begin(), it)));
            }

            ev = transpose(de, permutation);
            detail::call_over_leading_axis(ev, [](auto begin, auto end) { std::sort(begin, end); });
            ev = transpose(ev, reverse_permutation);
            return ev;
        }
        else
        {
            ev = de;
            detail::call_over_leading_axis(ev, [](auto begin, auto end) { std::sort(begin, end); });
            return ev;
        }
    }

    template <class E>
    auto sort(const xexpression<E>& e)
    {
        const auto& de = e.derived_cast();
        return sort(de, de.dimension() - 1);
    }

    namespace detail
    {
        template <class T>
        struct argfunc_result_type
        {
            using type = xarray<std::size_t>;
        };

        template <class T, std::size_t N>
        struct argfunc_result_type<xtensor<T, N>>
        {
            using type = xtensor<std::size_t, N - 1>;
        };

        template <class IT, class F>
        inline std::size_t cmp_idx(IT iter, IT end, ptrdiff_t inc, F&& cmp)
        {
            std::size_t idx = 0;
            double min = *iter;
            iter += inc;
            for (std::size_t i = 1; iter < end; iter += inc, ++i)
            {
                if (cmp(*iter, min))
                {
                    min = *iter;
                    idx = i;
                }
            }
            return idx;
        }

        template <class E, class F>
        xtensor<std::size_t, 0> arg_func_impl(const E& e, F&& f)
        {
            return cmp_idx(e.template begin<DEFAULT_LAYOUT>(),
                           e.template end<DEFAULT_LAYOUT>(), 1,
                           std::forward<F>(f));
        }

        template <class E, class F>
        typename argfunc_result_type<E>::type
        arg_func_impl(const E& e, std::size_t axis, F&& cmp)
        {
            using value_type = typename E::value_type;
            using result_type = typename argfunc_result_type<E>::type;

            if (e.dimension() == 1)
            {
                return arg_func_impl(e, std::forward<F>(cmp));
            }

            xt::dynamic_shape<std::size_t> new_shape = e.shape();
            new_shape.erase(new_shape.begin() + ptrdiff_t(axis));

            result_type result(new_shape);
            auto result_iter = result.begin();

            auto arg_func_lambda = [&result_iter, &cmp](auto begin, auto end) {
                std::size_t idx = 0;
                value_type val = *begin;
                ++begin;
                for (std::size_t i = 1; begin != end; ++begin, ++i)
                {
                    if (cmp(*begin, val))
                    {
                        val = *begin;
                        idx = i;
                    }
                }
                *result_iter = idx;
                ++result_iter;
            };

            if (axis != detail::leading_axis(e))
            {
                E input;
                auto axis_numbers = arange<std::size_t>(e.shape().size());
                std::vector<std::size_t> permutation(axis_numbers.begin(), axis_numbers.end());
                permutation.erase(permutation.begin() + ptrdiff_t(axis));
                if (input.layout() == layout_type::row_major)
                {
                    permutation.push_back(axis);
                }
                else
                {
                    permutation.insert(permutation.begin(), axis);
                }
                // Note we create a copy
                input = transpose(e, permutation);

                detail::call_over_leading_axis(input, arg_func_lambda);
                return result;
            }
            else
            {
                auto&& input = eval(e);
                detail::call_over_leading_axis(input, arg_func_lambda);
                return result;
            }
        }
    }

    template <class E>
    auto argmin(const xexpression<E>& e)
    {
        using value_type = typename E::value_type;
        auto&& ed = eval(e.derived_cast());
        return detail::arg_func_impl(ed, std::less<value_type>());
    }

    /**
     * Find position of minimal value in xexpression
     *
     * @param a xexpression to compute argmin on
     * @param axis select axis (or none)
     *
     * @return returns xarray with positions of minimal value
     */
    template <class E>
    auto argmin(const xexpression<E>& e, std::size_t axis)
    {
        using value_type = typename E::value_type;
        auto&& ed = eval(e.derived_cast());
        return detail::arg_func_impl(ed, axis, std::less<value_type>());
    }

    template <class E>
    auto argmax(const xexpression<E>& e)
    {
        using value_type = typename E::value_type;
        auto&& ed = eval(e.derived_cast());
        return detail::arg_func_impl(ed, std::greater<value_type>());
    }

    /**
     * Find position of maximal value in xexpression
     *
     * @param a xexpression to compute argmin on
     * @param axis select axis (or none)
     *
     * @return returns xarray with positions of minimal value
     */
    template <class E>
    auto argmax(const xexpression<E>& e, std::size_t axis)
    {
        using value_type = typename E::value_type;
        auto&& ed = eval(e.derived_cast());
        return detail::arg_func_impl(ed, axis, std::greater<value_type>());
    }
}

#endif
