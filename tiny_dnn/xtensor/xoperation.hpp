/***************************************************************************
* Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XOPERATION_HPP
#define XOPERATION_HPP

#include <algorithm>
#include <functional>
#include <type_traits>

#include "xfunction.hpp"
#include "xscalar.hpp"
#include "xstrides.hpp"

namespace xt
{

    namespace detail
    {

        /***********
         * helpers *
         ***********/

        template <class T>
        struct identity
        {
            using result_type = T;

            constexpr T operator()(const T& t) const noexcept
            {
                return +t;
            }
        };

        template <class T>
        struct conditional_ternary
        {
            using result_type = T;

            constexpr result_type operator()(const T& t1, const T& t2, const T& t3) const noexcept
            {
                return t1 ? t2 : t3;
            }
        };

        template <template <class...> class F, class... E>
        inline auto make_xfunction(E&&... e) noexcept
        {
            using functor_type = F<common_value_type_t<std::decay_t<E>...>>;
            using result_type = typename functor_type::result_type;
            using type = xfunction<functor_type, result_type, const_xclosure_t<E>...>;
            return type(functor_type(), std::forward<E>(e)...);
        }

        template <template <class...> class F, class... E>
        struct xfunction_type
        {
            using type = xfunction<F<common_value_type_t<std::decay_t<E>...>>,
                                   typename F<common_value_type_t<std::decay_t<E>...>>::result_type,
                                   const_xclosure_t<E>...>;
        };

        // On MSVC, the second argument of enable_if_t is always evaluated, even if the condition is false.
        // Wrapping the xfunction type in the xfunction_type metafunction avoids this evaluation when
        // the condition is false, since it leads to a tricky bug preventing from using operator+ and operator-
        // on vector and arrays iterators.
        template <template <class...> class F, class... E>
        using xfunction_type_t = typename std::enable_if_t<has_xexpression<std::decay_t<E>...>::value,
                                                           xfunction_type<F, E...>>::type;
    }

    /*************
     * operators *
     *************/

    /**
     * @defgroup arithmetic_operators Arithmetic operators
     */

    /**
     * @ingroup arithmetic_operators
     * @brief Identity
     *
     * Returns an \ref xfunction for the element-wise identity
     * of \a e.
     * @param e an \ref xexpression
     * @return an \ref xfunction
     */
    template <class E>
    inline auto operator+(E&& e) noexcept
        -> detail::xfunction_type_t<detail::identity, E>
    {
        return detail::make_xfunction<detail::identity>(std::forward<E>(e));
    }

    /**
    * @ingroup arithmetic_operators
    * @brief Opposite
    *
    * Returns an \ref xfunction for the element-wise opposite
    * of \a e.
    * @param e an \ref xexpression
    * @return an \ref xfunction
    */
    template <class E>
    inline auto operator-(E&& e) noexcept
        -> detail::xfunction_type_t<std::negate, E>
    {
        return detail::make_xfunction<std::negate>(std::forward<E>(e));
    }

    /**
    * @ingroup arithmetic_operators
    * @brief Addition
    *
    * Returns an \ref xfunction for the element-wise addition
    * of \a e1 and \a e2.
    * @param e1 an \ref xexpression or a scalar
    * @param e2 an \ref xexpression or a scalar
    * @return an \ref xfunction
    */
    template <class E1, class E2>
    inline auto operator+(E1&& e1, E2&& e2) noexcept
        -> detail::xfunction_type_t<std::plus, E1, E2>
    {
        return detail::make_xfunction<std::plus>(std::forward<E1>(e1), std::forward<E2>(e2));
    }

    /**
    * @ingroup arithmetic_operators
    * @brief Substraction
    *
    * Returns an \ref xfunction for the element-wise substraction
    * of \a e2 to \a e1.
    * @param e1 an \ref xexpression or a scalar
    * @param e2 an \ref xexpression or a scalar
    * @return an \ref xfunction
    */
    template <class E1, class E2>
    inline auto operator-(E1&& e1, E2&& e2) noexcept
        -> detail::xfunction_type_t<std::minus, E1, E2>
    {
        return detail::make_xfunction<std::minus>(std::forward<E1>(e1), std::forward<E2>(e2));
    }

    /**
    * @ingroup arithmetic_operators
    * @brief Multiplication
    *
    * Returns an \ref xfunction for the element-wise multiplication
    * of \a e1 by \a e2.
    * @param e1 an \ref xexpression or a scalar
    * @param e2 an \ref xexpression or a scalar
    * @return an \ref xfunction
    */
    template <class E1, class E2>
    inline auto operator*(E1&& e1, E2&& e2) noexcept
        -> detail::xfunction_type_t<std::multiplies, E1, E2>
    {
        return detail::make_xfunction<std::multiplies>(std::forward<E1>(e1), std::forward<E2>(e2));
    }

    /**
    * @ingroup arithmetic_operators
    * @brief Division
    *
    * Returns an \ref xfunction for the element-wise division
    * of \a e1 by \a e2.
    * @param e1 an \ref xexpression or a scalar
    * @param e2 an \ref xexpression or a scalar
    * @return an \ref xfunction
    */
    template <class E1, class E2>
    inline auto operator/(E1&& e1, E2&& e2) noexcept
        -> detail::xfunction_type_t<std::divides, E1, E2>
    {
        return detail::make_xfunction<std::divides>(std::forward<E1>(e1), std::forward<E2>(e2));
    }

    /**
     * @defgroup logical_operators Logical operators
     */

    /**
     * @ingroup logical_operators
     * @brief Or
     *
     * Returns an \ref xfunction for the element-wise or
     * of \a e1 and \a e2.
     * @param e1 an \ref xexpression or a scalar
     * @param e2 an \ref xexpression or a scalar
     * @return an \ref xfunction
     */
    template <class E1, class E2>
    inline auto operator||(E1&& e1, E2&& e2) noexcept
        -> detail::xfunction_type_t<std::logical_or, E1, E2>
    {
        return detail::make_xfunction<std::logical_or>(std::forward<E1>(e1), std::forward<E2>(e2));
    }

    /**
    * @ingroup logical_operators
    * @brief And
    *
    * Returns an \ref xfunction for the element-wise and
    * of \a e1 and \a e2.
    * @param e1 an \ref xexpression or a scalar
    * @param e2 an \ref xexpression or a scalar
    * @return an \ref xfunction
    */
    template <class E1, class E2>
    inline auto operator&&(E1&& e1, E2&& e2) noexcept
        -> detail::xfunction_type_t<std::logical_and, E1, E2>
    {
        return detail::make_xfunction<std::logical_and>(std::forward<E1>(e1), std::forward<E2>(e2));
    }

    /**
    * @ingroup logical_operators
    * @brief Not
    *
    * Returns an \ref xfunction for the element-wise not
    * of \a e.
    * @param e an \ref xexpression
    * @return an \ref xfunction
    */
    template <class E>
    inline auto operator!(E&& e) noexcept
        -> detail::xfunction_type_t<std::logical_not, E>
    {
        return detail::make_xfunction<std::logical_not>(std::forward<E>(e));
    }

    /**
     * @defgroup comparison_operators Comparison operators
     */

    /**
     * @ingroup comparison_operators
     * @brief Lesser than
     *
     * Returns an \ref xfunction for the element-wise
     * lesser than comparison of \a e1 and \a e2.
     * @param e1 an \ref xexpression or a scalar
     * @param e2 an \ref xexpression or a scalar
     * @return an \ref xfunction
     */
    template <class E1, class E2>
    inline auto operator<(E1&& e1, E2&& e2) noexcept
        -> detail::xfunction_type_t<std::less, E1, E2>
    {
        return detail::make_xfunction<std::less>(std::forward<E1>(e1), std::forward<E2>(e2));
    }

    /**
    * @ingroup comparison_operators
    * @brief Lesser or equal
    *
    * Returns an \ref xfunction for the element-wise
    * lesser or equal comparison of \a e1 and \a e2.
    * @param e1 an \ref xexpression or a scalar
    * @param e2 an \ref xexpression or a scalar
    * @return an \ref xfunction
    */
    template <class E1, class E2>
    inline auto operator<=(E1&& e1, E2&& e2) noexcept
        -> detail::xfunction_type_t<std::less_equal, E1, E2>
    {
        return detail::make_xfunction<std::less_equal>(std::forward<E1>(e1), std::forward<E2>(e2));
    }

    /**
    * @ingroup comparison_operators
    * @brief Greater than
    *
    * Returns an \ref xfunction for the element-wise
    * greater than comparison of \a e1 and \a e2.
    * @param e1 an \ref xexpression or a scalar
    * @param e2 an \ref xexpression or a scalar
    * @return an \ref xfunction
    */
    template <class E1, class E2>
    inline auto operator>(E1&& e1, E2&& e2) noexcept
        -> detail::xfunction_type_t<std::greater, E1, E2>
    {
        return detail::make_xfunction<std::greater>(std::forward<E1>(e1), std::forward<E2>(e2));
    }

    /**
    * @ingroup comparison_operators
    * @brief Greater or equal
    *
    * Returns an \ref xfunction for the element-wise
    * greater or equal comparison of \a e1 and \a e2.
    * @param e1 an \ref xexpression or a scalar
    * @param e2 an \ref xexpression or a scalar
    * @return an \ref xfunction
    */
    template <class E1, class E2>
    inline auto operator>=(E1&& e1, E2&& e2) noexcept
        -> detail::xfunction_type_t<std::greater_equal, E1, E2>
    {
        return detail::make_xfunction<std::greater_equal>(std::forward<E1>(e1), std::forward<E2>(e2));
    }

    /**
     * @ingroup comparison_operators
     * @brief Equality
     *
     * Returns true if \a e1 and \a e2 have the same shape
     * and hold the same values. Unlike other comparison
     * operators, this does not return an \ref xfunction.
     * @param e1 an \ref xexpression or a scalar
     * @param e2 an \ref xexpression or a scalar
     * @return a boolean
     */
    template <class E1, class E2>
    inline bool operator==(const xexpression<E1>& e1, const xexpression<E2>& e2)
    {
        const E1& de1 = e1.derived_cast();
        const E2& de2 = e2.derived_cast();
        bool res = de1.dimension() == de2.dimension() && std::equal(de1.shape().begin(), de1.shape().end(), de2.shape().begin());
        auto iter1 = de1.xbegin();
        auto iter2 = de2.xbegin();
        auto iter_end = de1.xend();
        while (res && iter1 != iter_end)
        {
            res = (*iter1++ == *iter2++);
        }
        return res;
    }

    /**
    * @ingroup comparison_operators
    * @brief Inequality
    *
    * Returns true if \a e1 and \a e2 have different shapes
    * or hold the different values. Unlike other comparison
    * operators, this does not return an \ref xfunction.
    * @param e1 an \ref xexpression or a scalar
    * @param e2 an \ref xexpression or a scalar
    * @return a boolean
    */
    template <class E1, class E2>
    inline bool operator!=(const xexpression<E1>& e1, const xexpression<E2>& e2)
    {
        return !(e1 == e2);
    }

    /**
    * @ingroup comparison_operators
    * @brief Element-wise equality
    *
    * Returns an \ref xfunction for the element-wise
    * equality of \a e1 and \a e2.
    * @param e1 an \ref xexpression or a scalar
    * @param e2 an \ref xexpression or a scalar
    * @return an \ref xfunction
    */
    template <class E1, class E2>
    inline auto equal(E1&& e1, E2&& e2) noexcept
        -> detail::xfunction_type_t<std::equal_to, E1, E2>
    {
        return detail::make_xfunction<std::equal_to>(std::forward<E1>(e1), std::forward<E2>(e2));
    }

    /**
    * @ingroup comparison_operators
    * @brief Element-wise inequality
    *
    * Returns an \ref xfunction for the element-wise
    * inequality of \a e1 and \a e2.
    * @param e1 an \ref xexpression or a scalar
    * @param e2 an \ref xexpression or a scalar
    * @return an \ref xfunction
    */
    template <class E1, class E2>
    inline auto not_equal(E1&& e1, E2&& e2) noexcept
        -> detail::xfunction_type_t<std::not_equal_to, E1, E2>
    {
        return detail::make_xfunction<std::not_equal_to>(std::forward<E1>(e1), std::forward<E2>(e2));
    }

    /**
    * @ingroup logical_operators
    * @brief Ternary selection
    *
    * Returns an \ref xfunction for the element-wise
    * ternary selection (i.e. operator ? :) of \a e1,
    * \a e2 and \a e3.
    * @param e1 a boolean \ref xexpression
    * @param e2 an \ref xexpression or a scalar
    * @param e3 an \ref xexpression or a scalar
    * @return an \ref xfunction
    */
    template <class E1, class E2, class E3>
    inline auto where(E1&& e1, E2&& e2, E3&& e3) noexcept
        -> detail::xfunction_type_t<detail::conditional_ternary, E1, E2, E3>
    {
        return detail::make_xfunction<detail::conditional_ternary>(std::forward<E1>(e1), std::forward<E2>(e2), std::forward<E3>(e3));
    }

    /**
     * @ingroup logical_operators
     * @brief return vector of indices where T is not zero
     * 
     * @param arr input array
     * @return vector of \a index_types where arr is not equal to zero
     */
    template <class T>
    inline auto nonzero(const T& arr)
        -> std::vector<xindex_type_t<typename T::shape_type>>
    {
        auto shape = arr.shape();
        using index_type = xindex_type_t<typename T::shape_type>;
        using size_type = typename T::size_type;

        auto idx = make_sequence<index_type>(arr.dimension(), 0);
        std::vector<index_type> indices;

        auto next_idx = [&shape](index_type& idx)
        {
            for (int i = int(shape.size() - 1); i >= 0; --i)
            {
                if (idx[i] >= shape[i] - 1)
                {
                    idx[i] = 0;
                }
                else
                {
                    idx[i]++;
                    return idx;
                }
            }
            // return empty index, happens at last iteration step, but remains unused
            return index_type();
        };

        size_type total_size = compute_size(shape);
        for (size_type i = 0; i < total_size; i++, next_idx(idx))
        {
            if (arr.element(std::begin(idx), std::end(idx)))
            {
                indices.push_back(idx);
            }
        }
        return indices;
    }

    /**
     * @ingroup logical_operators
     * @brief return vector of indices where condition is true
     *        (equivalent to \a nonzero(condition))
     * 
     * @param condition input array
     * @return vector of \a index_types where condition is not equal to zero
     */
    template <class T>
    inline auto where(const T& condition)
        -> std::vector<xindex_type_t<typename T::shape_type>>
    {
        return nonzero(condition);
    }

    /**
    * @ingroup logical_operators
    * @brief Any
    *
    * Returns true if any of the values of \a e is truthy,
    * false otherwise.
    * @param e an \ref xexpression
    * @return a boolean
    */
    template <class E>
    inline bool any(E&& e)
    {
        using xtype = std::decay_t<E>;
        if (xtype::static_layout == layout_type::row_major || xtype::static_layout == layout_type::column_major)
        {
            return std::any_of(e.cbegin(), e.cend(),
                               [](const typename std::decay_t<E>::value_type& el) { return el; });
        }
        else
        {
            return std::any_of(e.xbegin(), e.xend(),
                               [](const typename std::decay_t<E>::value_type& el) { return el; });
        }
    }

    /**
    * @ingroup logical_operators
    * @brief Any
    *
    * Returns true if all of the values of \a e are truthy,
    * false otherwise.
    * @param e an \ref xexpression
    * @return a boolean
    */
    template <class E>
    inline bool all(E&& e)
    {
        using xtype = std::decay_t<E>;
        if (xtype::static_layout == layout_type::row_major || xtype::static_layout == layout_type::column_major)
        {
            return std::all_of(e.cbegin(), e.cend(),
                               [](const typename std::decay_t<E>::value_type& el) { return el; });
        }
        else
        {
            return std::all_of(e.xbegin(), e.xend(),
                               [](const typename std::decay_t<E>::value_type& el) { return el; });
        }
    }
}

#endif
