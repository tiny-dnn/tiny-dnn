/***************************************************************************
* Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XSTRIDES_HPP
#define XSTRIDES_HPP

#include <cstddef>
#include <functional>
#include <numeric>

#include "xexception.hpp"
#include "xtensor_forward.hpp"

namespace xt
{

    template <class shape_type>
    auto compute_size(const shape_type& shape) noexcept;

    /***************
     * data offset *
     ***************/

    template <class size_type, class S, size_t dim = 0>
    size_type data_offset(const S& strides) noexcept;

    template <class size_type, class S, size_t dim = 0, class Arg, class... Args>
    size_type data_offset(const S& strides, Arg arg, Args... args) noexcept;

    template <class size_type, class S, class It>
    size_type element_offset(const S& strides, It first, It last) noexcept;

    /*******************
     * strides builder *
     *******************/

    template <class shape_type, class strides_type>
    std::size_t compute_strides(const shape_type& shape, layout_type l, strides_type& strides);

    template <class shape_type, class strides_type, class backstrides_type>
    std::size_t compute_strides(const shape_type& shape, layout_type l,
                                strides_type& strides, backstrides_type& backstrides);

    template <class shape_type, class strides_type>
    void adapt_strides(const shape_type& shape, strides_type& strides) noexcept;

    template <class shape_type, class strides_type, class backstrides_type>
    void adapt_strides(const shape_type& shape, strides_type& strides,
                       backstrides_type& backstrides) noexcept;

    /***********************
     * broadcast functions *
     ***********************/

    template <class S1, class S2>
    bool broadcast_shape(const S1& input, S2& output);

    template <class S1, class S2>
    bool broadcastable(const S1& s1, S2& s2);

    /******************
     * Implementation *
     ******************/

    template <class shape_type>
    inline auto compute_size(const shape_type& shape) noexcept
    {
        using size_type = std::decay_t<typename shape_type::value_type>;
        return std::accumulate(shape.cbegin(), shape.cend(), size_type(1), std::multiplies<size_type>());
    }

    namespace detail
    {
        template <class size_type, class S, std::size_t dim>
        inline size_type raw_data_offset(const S&) noexcept
        {
            return 0;
        }

        template <class size_type, class S, std::size_t dim, class Arg, class... Args>
        inline size_type raw_data_offset(const S& strides, Arg arg, Args... args) noexcept
        {
            return arg * strides[dim] + raw_data_offset<size_type, S, dim + 1>(strides, args...);
        }
    }

    template <class size_type, class S, std::size_t dim>
    inline size_type data_offset(const S&) noexcept
    {
        return 0;
    }

    template <class size_type, class S, std::size_t dim, class Arg, class... Args>
    inline size_type data_offset(const S& strides, Arg arg, Args... args) noexcept
    {
        constexpr std::size_t nargs = sizeof...(Args) + dim + 1;
        if (nargs == strides.size())
        {
            // Correct number of arguments: iterate
            return detail::raw_data_offset<size_type, S, dim, Arg, Args...>(strides, arg, args...);
        }
        else if (nargs > strides.size())
        {
            // Too many arguments: drop the first
            return data_offset<size_type, S, dim>(strides, args...);
        }
        else
        {
            // Too few arguments: right to left scalar product
            auto view = strides.cend() - nargs;
            return detail::raw_data_offset<size_type, const typename S::const_iterator, dim, Arg, Args...>(view, arg, args...);
        }
    }

    template <class size_type, class S, class It>
    inline size_type element_offset(const S& strides, It first, It last) noexcept
    {
        auto size = std::min(static_cast<typename S::size_type>(std::distance(first, last)), strides.size());
        return std::inner_product(last - size, last, strides.cend() - size, size_type(0));
    }

    namespace detail
    {
        template <class shape_type, class strides_type, class bs_ptr>
        inline void adapt_strides(const shape_type& shape, strides_type& strides,
                                  bs_ptr backstrides, typename strides_type::size_type i) noexcept
        {
            if (shape[i] == 1)
                strides[i] = 0;
            (*backstrides)[i] = strides[i] * (shape[i] - 1);
        }

        template <class shape_type, class strides_type>
        inline void adapt_strides(const shape_type& shape, strides_type& strides,
                                  std::nullptr_t, typename strides_type::size_type i) noexcept
        {
            if (shape[i] == 1)
                strides[i] = 0;
        }

        template <class shape_type, class strides_type, class bs_ptr>
        inline std::size_t compute_strides(const shape_type& shape, layout_type l,
                                           strides_type& strides, bs_ptr bs)
        {
            std::size_t data_size = 1;
            if (l == layout_type::row_major)
            {
                for (std::size_t i = strides.size(); i != 0; --i)
                {
                    strides[i - 1] = data_size;
                    data_size = strides[i - 1] * shape[i - 1];
                    adapt_strides(shape, strides, bs, i - 1);
                }
            }
            else
            {
                for (std::size_t i = 0; i < strides.size(); ++i)
                {
                    strides[i] = data_size;
                    data_size = strides[i] * shape[i];
                    adapt_strides(shape, strides, bs, i);
                }
            }
            return data_size;
        }
    }

    template <class shape_type, class strides_type>
    inline std::size_t compute_strides(const shape_type& shape, layout_type l, strides_type& strides)
    {
        return detail::compute_strides(shape, l, strides, nullptr);
    }

    template <class shape_type, class strides_type, class backstrides_type>
    inline std::size_t compute_strides(const shape_type& shape, layout_type l,
                                       strides_type& strides,
                                       backstrides_type& backstrides)
    {
        return detail::compute_strides(shape, l, strides, &backstrides);
    }

    template <class shape_type, class strides_type>
    inline void adapt_strides(const shape_type& shape, strides_type& strides) noexcept
    {
        for (typename shape_type::size_type i = 0; i < shape.size(); ++i)
        {
            detail::adapt_strides(shape, strides, nullptr, i);
        }
    }

    template <class shape_type, class strides_type, class backstrides_type>
    inline void adapt_strides(const shape_type& shape, strides_type& strides,
                              backstrides_type& backstrides) noexcept
    {
        for (typename shape_type::size_type i = 0; i < shape.size(); ++i)
        {
            detail::adapt_strides(shape, strides, &backstrides, i);
        }
    }

    template <class S1, class S2>
    inline bool broadcast_shape(const S1& input, S2& output)
    {
        bool trivial_broadcast = (input.size() == output.size());
        auto input_iter = input.crbegin();
        auto output_iter = output.rbegin();
        for (; input_iter != input.crend(); ++input_iter, ++output_iter)
        {
            if (*output_iter == 1)
            {
                *output_iter = *input_iter;
            }
            else if ((*input_iter != 1) && (*output_iter != *input_iter))
            {
                throw broadcast_error(output, input);
            }
            trivial_broadcast = trivial_broadcast && (*output_iter == *input_iter);
        }
        return trivial_broadcast;
    }

    template <class S1, class S2>
    inline bool broadcastable(const S1& s1, const S2& s2)
    {
        auto iter1 = s1.crbegin();
        auto iter2 = s2.crbegin();
        for (; iter1 != s1.crend() && iter2 != s2.crend(); ++iter1, ++iter2)
        {
            if ((*iter2 != 1) && (*iter1 != 1) && (*iter2 != *iter1))
            {
                return false;
            }
        }
        return true;
    }
}

#endif
