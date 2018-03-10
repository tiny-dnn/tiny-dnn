/***************************************************************************
* Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XTENSOR_REDUCER_HPP
#define XTENSOR_REDUCER_HPP

#include <algorithm>
#include <cstddef>
#include <initializer_list>
#include <iterator>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>
#ifdef X_OLD_CLANG
#include <vector>
#endif

#include "xtl/xfunctional.hpp"
#include "xtl/xsequence.hpp"

#include "xbuilder.hpp"
#include "xexpression.hpp"
#include "xgenerator.hpp"
#include "xiterable.hpp"
#include "xreducer.hpp"
#include "xutils.hpp"

namespace xt
{
    /**********
     * reduce *
     **********/

#define DEFAULT_STRATEGY_REDUCERS evaluation_strategy::lazy

    template <class F, class E, class X, class ES = DEFAULT_STRATEGY_REDUCERS,
              class = std::enable_if_t<!std::is_base_of<evaluation_strategy::base, std::decay_t<X>>::value, int>>
    auto reduce(F&& f, E&& e, X&& axes, ES es = ES()) noexcept;

    template <class F, class E, class ES = DEFAULT_STRATEGY_REDUCERS,
              class = std::enable_if_t<std::is_base_of<evaluation_strategy::base, ES>::value, int>>
    auto reduce(F&& f, E&& e, ES es = ES()) noexcept;

#ifdef X_OLD_CLANG
    template <class F, class E, class I>
    auto reduce(F&& f, E&& e, std::initializer_list<I> axes) noexcept;
#else
    template <class F, class E, class I, std::size_t N, class ES = DEFAULT_STRATEGY_REDUCERS>
    auto reduce(F&& f, E&& e, const I (&axes)[N], ES es = ES()) noexcept;
#endif

    template <class F, class E, class X>
    auto reduce_immediate(F&& f, E&& e, X&& axes)
    {
        using shape_type = dynamic_shape<std::size_t>;
        using accumulate_functor = std::decay_t<decltype(std::get<0>(f))>;
        using result_type = typename accumulate_functor::result_type;

        // retrieve functors from triple struct
        auto acc_fct = std::get<0>(f);
        auto init_fct = std::get<1>(f);
        auto merge_fct = std::get<2>(f);

        shape_type result_shape(e.dimension() - axes.size());
        shape_type iter_shape = e.shape();
        shape_type iter_strides(e.dimension());

        xt::xarray<result_type, std::decay_t<E>::static_layout> result;

        if (!std::is_sorted(axes.cbegin(), axes.cend()))
        {
            throw std::runtime_error("Reducing axes should be sorted");
        }

        // Fast track for complete reduction
        if (e.dimension() == axes.size())
        {
            auto begin = e.data().begin();
            result_type tmp = init_fct(*begin);
            ++begin;
            result(0) = std::accumulate(begin, e.data().end(), tmp, acc_fct);
            return result;
        }

        // axis wise reductions:
        for (std::size_t i = 0, idx = 0; i < e.dimension(); ++i)
        {
            if (std::find(axes.begin(), axes.end(), i) == axes.end())
            {
                // i not in axes!
                result_shape[idx] = e.shape()[i];
                ++idx;
            }
        }

        result.resize(result_shape, e.layout());

        std::size_t ax_idx = (e.layout() == layout_type::row_major) ? axes.size() - 1 : 0;
        std::size_t inner_loop_size = e.strides()[axes[ax_idx]];
        std::size_t inner_stride    = e.strides()[axes[ax_idx]];
        std::size_t outer_loop_size = e.shape()[axes[ax_idx]];

        // The following code merges reduction axes "at the end" (or the beginning for col_major)
        // together by increasing the size of the outer loop where appropriate
        auto merge_loops = [&outer_loop_size, &e](auto it, auto end)
        {
            auto last_ax = *it;
            ++it;
            for (; it != end; ++it)
            {
                // note that we check is_sorted, so this condition is valid
                if (std::abs(ptrdiff_t(*it) - ptrdiff_t(last_ax)) ==  1)
                {
                    last_ax = *it;
                    outer_loop_size *= e.shape()[last_ax];
                }
            }
            return last_ax;
        };

        for (std::size_t i = 0, idx = 0; i < e.dimension(); ++i)
        {
            if (std::find(axes.begin(), axes.end(), i) == axes.end())
            {
                // i not in axes!
                iter_strides[i] = result.strides()[idx];
                ++idx;
            }
        }

        if (e.layout() == layout_type::row_major)
        {
            std::size_t last_ax = merge_loops(axes.rbegin(), axes.rend());

            iter_shape.erase(iter_shape.begin() + ptrdiff_t(last_ax), iter_shape.end());
            iter_strides.erase(iter_strides.begin() + ptrdiff_t(last_ax), iter_strides.end());
        }
        else if (e.layout() == layout_type::column_major)
        {
            // we got column_major here
            std::size_t last_ax = merge_loops(axes.begin(), axes.end());

            // erasing the front vs the back
            iter_shape.erase(iter_shape.begin(), iter_shape.begin() + ptrdiff_t(last_ax + 1));
            iter_strides.erase(iter_strides.begin(), iter_strides.begin() + ptrdiff_t(last_ax + 1));

            // and reversing, to make it work with the same next_idx function
            std::reverse(iter_shape.begin(), iter_shape.end());
            std::reverse(iter_strides.begin(), iter_strides.end());
        }
        else
        {
            throw std::runtime_error("Layout not supported in immediate reduction.");
        }

        xindex temp_idx(iter_shape.size());
        auto next_idx = [&iter_shape, &iter_strides, &temp_idx]()
        {
            std::size_t i = iter_shape.size();
            for (; i > 0; --i)
            {
                if (ptrdiff_t(temp_idx[i - 1]) >= ptrdiff_t(iter_shape[i - 1]) - 1)
                {
                    temp_idx[i - 1] = 0;
                }
                else
                {
                    temp_idx[i - 1]++;
                    break;
                }
            }
            return std::make_pair(i == 0,
                                  std::inner_product(temp_idx.begin(), temp_idx.end(),
                                                     iter_strides.begin(), ptrdiff_t(0)));
        };

        auto begin = e.raw_data();
        auto out = result.raw_data();
        auto out_begin = result.raw_data();

        ptrdiff_t next_stride = 0;

        std::pair<bool, ptrdiff_t> idx_res(false, 0);

        // Remark: eventually some modifications here to make conditions faster where merge + accumulate is the
        // same function (e.g. check std::is_same<decltype(merge_fct), decltype(acc_fct)>::value) ...

        auto merge_border = out;
        bool merge = false;

        // TODO there could be some performance gain by removing merge checking
        //      when axes.size() == 1 and even next_idx could be removed for something simpler (next_stride always the same)
        //      best way to do this would be to create a function that takes (begin, out, outer_loop_size, inner_loop_size, next_idx_lambda)
        // Decide if going about it row-wise or col-wise
        if (inner_stride == 1)
        {
            while(idx_res.first != true)
            {
                // for unknown reasons it's much faster to use a temporary variable and
                // std::accumulate here -- probably some cache behavior
                result_type tmp;
                tmp = init_fct(*begin);
                tmp = std::accumulate(begin + 1, begin + outer_loop_size, tmp, acc_fct);

                // use merge function if necessary
                *out = merge ? merge_fct(*out, tmp) : tmp;

                begin += outer_loop_size;

                idx_res = next_idx();
                next_stride = idx_res.second;
                out = out_begin + next_stride;

                if (out > merge_border)
                {
                    // looped over once
                    merge = false;
                    merge_border = out;
                }
                else
                {
                    merge = true;
                }
            };
        }
        else
        {
            while(idx_res.first != true)
            {
                std::transform(out, out + inner_loop_size, begin, out,
                               [merge, &init_fct, &merge_fct](auto&& v1, auto&& v2)
                               {
                                    return merge ? merge_fct(v1, v2) : init_fct(v2);
                               }
                );

                begin += inner_stride;
                for (std::size_t i = 1; i < outer_loop_size; ++i)
                {
                    std::transform(out, out + inner_loop_size, begin, out, acc_fct);
                    begin += inner_stride;
                }

                idx_res = next_idx();
                next_stride = idx_res.second;
                out = out_begin + next_stride;

                if (out > merge_border)
                {
                    // looped over once
                    merge = false;
                    merge_border = out;
                }
                else
                {
                    merge = true;
                }

            };
        }
        return result;
    }

    /*************
     * xreducer  *
     *************/

    template <class ST, class X>
    struct xreducer_shape_type;

    template <class REDUCE_FUNC, class INIT_FUNC = xtl::identity, class MERGE_FUNC = REDUCE_FUNC>
    struct xreducer_functors
        : public std::tuple<REDUCE_FUNC, INIT_FUNC, MERGE_FUNC>
    {
        using self_type = xreducer_functors<REDUCE_FUNC, INIT_FUNC, MERGE_FUNC>;
        using base_type = std::tuple<REDUCE_FUNC, INIT_FUNC, MERGE_FUNC>;
        using reduce_functor_type = REDUCE_FUNC;
        using init_functor_type = INIT_FUNC;
        using merge_functor_type = MERGE_FUNC;

        xreducer_functors()
            : base_type()
        {
        }

        template <class RF>
        xreducer_functors(RF&& reduce_func)
            : base_type(std::forward<RF>(reduce_func), INIT_FUNC(), reduce_func)
        {
        }

        template <class RF, class IF>
        xreducer_functors(RF&& reduce_func, IF&& init_func)
            : base_type(std::forward<RF>(reduce_func), std::forward<IF>(init_func), reduce_func)
        {
        }

        template <class RF, class IF, class MF>
        xreducer_functors(RF&& reduce_func, IF&& init_func, MF&& merge_func)
            : base_type(std::forward<RF>(reduce_func), std::forward<IF>(init_func), std::forward<MF>(merge_func))
        {
        }
    };

    template <class RF>
    auto make_xreducer_functor(RF&& reduce_func)
    {
        using reducer_type = xreducer_functors<std::remove_reference_t<RF>>;
        return reducer_type(std::forward<RF>(reduce_func));
    }

    template <class RF, class IF>
    auto make_xreducer_functor(RF&& reduce_func, IF&& init_func)
    {
        using reducer_type = xreducer_functors<std::remove_reference_t<RF>, std::remove_reference_t<IF>>;
        return reducer_type(std::forward<RF>(reduce_func), std::forward<IF>(init_func));
    }

    template <class RF, class IF, class MF>
    auto make_xreducer_functor(RF&& reduce_func, IF&& init_func, MF&& merge_func)
    {
        using reducer_type = xreducer_functors<std::remove_reference_t<RF>,
                                               std::remove_reference_t<IF>,
                                               std::remove_reference_t<MF>>;
        return reducer_type(std::forward<RF>(reduce_func), std::forward<IF>(init_func), std::forward<MF>(merge_func));
    }

    template <class F, class CT, class X>
    class xreducer;

    template <class F, class CT, class X>
    class xreducer_stepper;

    template <class F, class CT, class X>
    struct xiterable_inner_types<xreducer<F, CT, X>>
    {
        using xexpression_type = std::decay_t<CT>;
        using inner_shape_type = typename xreducer_shape_type<typename xexpression_type::shape_type, std::decay_t<X>>::type;
        using const_stepper = xreducer_stepper<F, CT, X>;
        using stepper = const_stepper;
    };

    /**
     * @class xreducer
     * @brief Reducing function operating over specified axes.
     *
     * The xreducer class implements an \ref xexpression applying
     * a reducing function to an \ref xexpression over the specified
     * axes.
     *
     * @tparam F a tuple of functors (class \ref xreducer_functors or compatible)
     * @tparam CT the closure type of the \ref xexpression to reduce
     * @tparam X the list of axes
     *
     * The reducer's result_type is deduced from the result type of function
     * <tt>F::reduce_functor_type</tt> when called with elements of the expression @tparam CT.
     *
     * @sa reduce
     */
    template <class F, class CT, class X>
    class xreducer : public xexpression<xreducer<F, CT, X>>,
                     public xconst_iterable<xreducer<F, CT, X>>
    {
    public:

        using self_type = xreducer<F, CT, X>;
        using reduce_functor_type = typename std::decay_t<F>::reduce_functor_type;
        using init_functor_type = typename std::decay_t<F>::init_functor_type;
        using merge_functor_type = typename std::decay_t<F>::merge_functor_type;
        using xexpression_type = std::decay_t<CT>;
        using axes_type = X;

        using substepper_type = typename xexpression_type::const_stepper;
        using value_type = std::decay_t<decltype(std::declval<reduce_functor_type>()(
            std::declval<init_functor_type>()(*std::declval<substepper_type>()), *std::declval<substepper_type>()))>;
        using reference = value_type;
        using const_reference = value_type;
        using pointer = value_type*;
        using const_pointer = const value_type*;

        using size_type = typename xexpression_type::size_type;
        using difference_type = typename xexpression_type::difference_type;

        using iterable_base = xconst_iterable<self_type>;
        using inner_shape_type = typename iterable_base::inner_shape_type;
        using shape_type = inner_shape_type;

        using stepper = typename iterable_base::stepper;
        using const_stepper = typename iterable_base::const_stepper;

        static constexpr layout_type static_layout = layout_type::dynamic;
        static constexpr bool contiguous_layout = false;

        template <class Func, class CTA, class AX>
        xreducer(Func&& func, CTA&& e, AX&& axes);

        size_type size() const noexcept;
        size_type dimension() const noexcept;
        const inner_shape_type& shape() const noexcept;
        layout_type layout() const noexcept;

        template <class... Args>
        const_reference operator()(Args... args) const;
        template <class... Args>
        const_reference at(Args... args) const;
        template <class S>
        disable_integral_t<S, const_reference> operator[](const S& index) const;
        template <class I>
        const_reference operator[](std::initializer_list<I> index) const;
        const_reference operator[](size_type i) const;

        template <class It>
        const_reference element(It first, It last) const;

        template <class S>
        bool broadcast_shape(S& shape, bool reuse_cache = false) const;

        template <class S>
        bool is_trivial_broadcast(const S& strides) const noexcept;

        template <class S>
        const_stepper stepper_begin(const S& shape) const noexcept;
        template <class S>
        const_stepper stepper_end(const S& shape, layout_type) const noexcept;

    private:

        CT m_e;
        reduce_functor_type m_reduce;
        init_functor_type m_init;
        merge_functor_type m_merge;
        axes_type m_axes;
        inner_shape_type m_shape;
        shape_type m_dim_mapping;

        friend class xreducer_stepper<F, CT, X>;
    };

    /*************************
     * reduce implementation *
     *************************/

    namespace detail
    {
        template <class F, class E, class X>
        inline auto reduce_impl(F&& f, E&& e, X&& axes, evaluation_strategy::lazy) noexcept
        {
            using reducer_type = xreducer<F, const_xclosure_t<E>, xtl::const_closure_type_t<X>>;
            return reducer_type(std::forward<F>(f), std::forward<E>(e), std::forward<X>(axes));
        }

        template <class F, class E, class X>
        inline auto reduce_impl(F&& f, E&& e, X&& axes, evaluation_strategy::immediate) noexcept
        {
            return reduce_immediate(std::forward<F>(f), std::forward<E>(e), std::forward<X>(axes));
        }
    }

    /**
     * @brief Returns an \ref xexpression applying the speficied reducing
     * function to an expresssion over the given axes.
     *
     * @param f the reducing function to apply.
     * @param e the \ref xexpression to reduce.
     * @param axes the list of axes.
     * @param evaluation_strategy evaluation strategy to use (lazy (default), or immediate)
     *
     * The returned expression either hold a const reference to \p e or a copy
     * depending on whether \p e is an lvalue or an rvalue.
     */

    template <class F, class E, class X, class ES, class>
    inline auto reduce(F&& f, E&& e, X&& axes, ES evaluation_strategy) noexcept
    {
        return detail::reduce_impl(std::forward<F>(f), std::forward<E>(e), std::forward<X>(axes), evaluation_strategy);
    }

    template <class F, class E, class ES, class>
    inline auto reduce(F&& f, E&& e, ES evaluation_strategy) noexcept
    {
        auto ar = arange(e.dimension());
        return detail::reduce_impl(std::forward<F>(f), std::forward<E>(e), std::move(ar), evaluation_strategy);
    }

#ifdef X_OLD_CLANG
    template <class F, class E, class I>
    inline auto reduce(F&& f, E&& e, std::initializer_list<I> axes) noexcept
    {
        using axes_type = std::vector<typename std::decay_t<E>::size_type>;
        using reducer_type = xreducer<F, const_xclosure_t<E>, axes_type>;
        return reducer_type(std::forward<F>(f), std::forward<E>(e), xtl::forward_sequence<axes_type>(axes));
    }
#else
    template <class F, class E, class I, std::size_t N, class ES>
    inline auto reduce(F&& f, E&& e, const I (&axes)[N], ES evaluation_strategy) noexcept
    {
        using axes_type = std::array<typename std::decay_t<E>::size_type, N>;
        return detail::reduce_impl(std::forward<F>(f), std::forward<E>(e), xtl::forward_sequence<axes_type>(axes), evaluation_strategy);

    }
#endif

    /********************
     * xreducer_stepper *
     ********************/

    template <class F, class CT, class X>
    class xreducer_stepper
    {
    public:

        using self_type = xreducer_stepper<F, CT, X>;
        using xreducer_type = xreducer<F, CT, X>;

        using value_type = typename xreducer_type::value_type;
        using reference = typename xreducer_type::value_type;
        using pointer = typename xreducer_type::const_pointer;
        using size_type = typename xreducer_type::size_type;
        using difference_type = typename xreducer_type::difference_type;

        using xexpression_type = typename xreducer_type::xexpression_type;
        using substepper_type = typename xexpression_type::const_stepper;
        using shape_type = typename xreducer_type::shape_type;

        xreducer_stepper(const xreducer_type& red, size_type offset, bool end = false,
                         layout_type l = default_assignable_layout(xexpression_type::static_layout));

        reference operator*() const;

        void step(size_type dim, size_type n = 1);
        void step_back(size_type dim, size_type n = 1);
        void reset(size_type dim);
        void reset_back(size_type dim);

        void to_begin();
        void to_end(layout_type l);

        bool equal(const self_type& rhs) const;

    private:

        reference aggregate(size_type dim) const;

        substepper_type get_substepper_begin() const;
        size_type get_dim(size_type dim) const noexcept;
        size_type shape(size_type i) const noexcept;
        size_type axis(size_type i) const noexcept;

        const xreducer_type& m_reducer;
        size_type m_offset;
        mutable substepper_type m_stepper;
    };

    template <class F, class CT, class X>
    bool operator==(const xreducer_stepper<F, CT, X>& lhs,
                    const xreducer_stepper<F, CT, X>& rhs);

    template <class F, class CT, class X>
    bool operator!=(const xreducer_stepper<F, CT, X>& lhs,
                    const xreducer_stepper<F, CT, X>& rhs);

    /******************
     * xreducer utils *
     ******************/

    // meta-function returning the shape type for an xreducer
    template <class ST, class X>
    struct xreducer_shape_type
    {
        using type = promote_shape_t<ST, std::decay_t<X>>;
    };

    template <class I1, std::size_t N1, class I2, std::size_t N2>
    struct xreducer_shape_type<std::array<I1, N1>, std::array<I2, N2>>
    {
        using type = std::array<I2, N1 - N2>;
    };

    namespace detail
    {
        template <class InputIt, class ExcludeIt, class OutputIt>
        inline void excluding_copy(InputIt first, InputIt last,
                                   ExcludeIt e_first, ExcludeIt e_last,
                                   OutputIt d_first, OutputIt map_first)
        {
            using difference_type = typename std::iterator_traits<InputIt>::difference_type;
            using value_type = typename std::iterator_traits<OutputIt>::value_type;
            InputIt iter = first;
            while (iter != last && e_first != e_last)
            {
                auto diff = std::distance(first, iter);
                if (diff != difference_type(*e_first))
                {
                    *d_first++ = *iter++;
                    *map_first++ = value_type(diff);
                }
                else
                {
                    ++iter;
                    ++e_first;
                }
            }
            auto diff = std::distance(first, iter);
            auto end = std::distance(iter, last);
            std::iota(map_first, map_first + end, diff);
            std::copy(iter, last, d_first);
        }
    }

    /***************************
     * xreducer implementation *
     ***************************/

    /**
     * @name Constructor
     */
    //@{
    /**
     * Constructs an xreducer expression applying the specified
     * function to the given expression over the given axes.
     *
     * @param func the function to apply
     * @param e the expression to reduce
     * @param axes the axes along which the reduction is performed
     */
    template <class F, class CT, class X>
    template <class Func, class CTA, class AX>
    inline xreducer<F, CT, X>::xreducer(Func&& func, CTA&& e, AX&& axes)
        : m_e(std::forward<CTA>(e))
        , m_reduce(std::get<0>(func))
        , m_init(std::get<1>(func))
        , m_merge(std::get<2>(func))
        , m_axes(std::forward<AX>(axes))
        , m_shape(xtl::make_sequence<inner_shape_type>(m_e.dimension() - m_axes.size(), 0))
        , m_dim_mapping(xtl::make_sequence<shape_type>(m_e.dimension() - m_axes.size(), 0))
    {
        if (!std::is_sorted(m_axes.cbegin(), m_axes.cend()))
        {
            throw std::runtime_error("Reducing axes should be sorted");
        }
        detail::excluding_copy(m_e.shape().cbegin(), m_e.shape().cend(),
                               m_axes.cbegin(), m_axes.cend(),
                               m_shape.begin(), m_dim_mapping.begin());
    }
    //@}

    /**
     * @name Size and shape
     */
    /**
     * Returns the size of the expression.
     */
    template <class F, class CT, class X>
    inline auto xreducer<F, CT, X>::size() const noexcept -> size_type
    {
        return compute_size(shape());
    }

    /**
     * Returns the number of dimensions of the expression.
     */
    template <class F, class CT, class X>
    inline auto xreducer<F, CT, X>::dimension() const noexcept -> size_type
    {
        return m_shape.size();
    }

    /**
     * Returns the shape of the expression.
     */
    template <class F, class CT, class X>
    inline auto xreducer<F, CT, X>::shape() const noexcept -> const inner_shape_type&
    {
        return m_shape;
    }

    /**
     * Returns the shape of the expression.
     */
    template <class F, class CT, class X>
    inline layout_type xreducer<F, CT, X>::layout() const noexcept
    {
        return static_layout;
    }
    //@}

    /**
     * @name Data
     */
    /**
     * Returns a constant reference to the element at the specified position in the reducer.
     * @param args a list of indices specifying the position in the reducer. Indices
     * must be unsigned integers, the number of indices should be equal or greater than
     * the number of dimensions of the reducer.
     */
    template <class F, class CT, class X>
    template <class... Args>
    inline auto xreducer<F, CT, X>::operator()(Args... args) const -> const_reference
    {
        std::array<std::size_t, sizeof...(Args)> arg_array = {{static_cast<std::size_t>(args)...}};
        return element(arg_array.cbegin(), arg_array.cend());
    }

    /**
     * Returns a constant reference to the element at the specified position in the expression,
     * after dimension and bounds checking.
     * @param args a list of indices specifying the position in the function. Indices
     * must be unsigned integers, the number of indices should be equal to the number of dimensions
     * of the expression.
     * @exception std::out_of_range if the number of argument is greater than the number of dimensions
     * or if indices are out of bounds.
     */
    template <class F, class CT, class X>
    template <class... Args>
    inline auto xreducer<F, CT, X>::at(Args... args) const -> const_reference
    {
        check_access(shape(), static_cast<size_type>(args)...);
        return this->operator()(args...);
    }
    /**
     * Returns a constant reference to the element at the specified position in the reducer.
     * @param index a sequence of indices specifying the position in the reducer. Indices
     * must be unsigned integers, the number of indices in the sequence should be equal or greater
     * than the number of dimensions of the reducer.
     */
    template <class F, class CT, class X>
    template <class S>
    inline auto xreducer<F, CT, X>::operator[](const S& index) const
        -> disable_integral_t<S, const_reference>
    {
        return element(index.cbegin(), index.cend());
    }

    template <class F, class CT, class X>
    template <class I>
    inline auto xreducer<F, CT, X>::operator[](std::initializer_list<I> index) const
        -> const_reference
    {
        return element(index.begin(), index.end());
    }

    template <class F, class CT, class X>
    inline auto xreducer<F, CT, X>::operator[](size_type i) const -> const_reference
    {
        return operator()(i);
    }

    /**
     * Returns a constant reference to the element at the specified position in the reducer.
     * @param first iterator starting the sequence of indices
     * @param last iterator ending the sequence of indices
     * The number of indices in the sequence should be equal to or greater
     * than the number of dimensions of the reducer.
     */
    template <class F, class CT, class X>
    template <class It>
    inline auto xreducer<F, CT, X>::element(It first, It last) const -> const_reference
    {
        auto stepper = const_stepper(*this, 0);
        size_type dim = 0;
        while (first != last)
        {
            stepper.step(dim++, std::size_t(*first++));
        }
        return *stepper;
    }
    //@}

    /**
     * @name Broadcasting
     */
    //@{
    /**
     * Broadcast the shape of the reducer to the specified parameter.
     * @param shape the result shape
     * @return a boolean indicating whether the broadcasting is trivial
     */
    template <class F, class CT, class X>
    template <class S>
    inline bool xreducer<F, CT, X>::broadcast_shape(S& shape, bool) const
    {
        return xt::broadcast_shape(m_shape, shape);
    }

    /**
     * Compares the specified strides with those of the container to see whether
     * the broadcasting is trivial.
     * @return a boolean indicating whether the broadcasting is trivial
     */
    template <class F, class CT, class X>
    template <class S>
    inline bool xreducer<F, CT, X>::is_trivial_broadcast(const S& /*strides*/) const noexcept
    {
        return false;
    }
    //@}

    template <class F, class CT, class X>
    template <class S>
    inline auto xreducer<F, CT, X>::stepper_begin(const S& shape) const noexcept -> const_stepper
    {
        size_type offset = shape.size() - dimension();
        return const_stepper(*this, offset);
    }

    template <class F, class CT, class X>
    template <class S>
    inline auto xreducer<F, CT, X>::stepper_end(const S& shape, layout_type l) const noexcept -> const_stepper
    {
        size_type offset = shape.size() - dimension();
        return const_stepper(*this, offset, true, l);
    }

    /***********************************
     * xreducer_stepper implementation *
     ***********************************/

    template <class F, class CT, class X>
    inline xreducer_stepper<F, CT, X>::xreducer_stepper(const xreducer_type& red, size_type offset, bool end, layout_type l)
        : m_reducer(red), m_offset(offset),
          m_stepper(get_substepper_begin())
    {
        if (end)
        {
            to_end(l);
        }
    }

    template <class F, class CT, class X>
    inline auto xreducer_stepper<F, CT, X>::operator*() const -> reference
    {
        reference r = aggregate(0);
        return r;
    }

    template <class F, class CT, class X>
    inline void xreducer_stepper<F, CT, X>::step(size_type dim, size_type n)
    {
        if (dim >= m_offset)
        {
            m_stepper.step(get_dim(dim - m_offset), n);
        }
    }

    template <class F, class CT, class X>
    inline void xreducer_stepper<F, CT, X>::step_back(size_type dim, size_type n)
    {
        if (dim >= m_offset)
        {
            m_stepper.step_back(get_dim(dim - m_offset), n);
        }
    }

    template <class F, class CT, class X>
    inline void xreducer_stepper<F, CT, X>::reset(size_type dim)
    {
        if (dim >= m_offset)
        {
            m_stepper.reset(get_dim(dim - m_offset));
        }
    }

    template <class F, class CT, class X>
    inline void xreducer_stepper<F, CT, X>::reset_back(size_type dim)
    {
        if (dim >= m_offset)
        {
            m_stepper.reset_back(get_dim(dim - m_offset));
        }
    }

    template <class F, class CT, class X>
    inline void xreducer_stepper<F, CT, X>::to_begin()
    {
        m_stepper.to_begin();
    }

    template <class F, class CT, class X>
    inline void xreducer_stepper<F, CT, X>::to_end(layout_type l)
    {
        m_stepper.to_end(l);
    }

    template <class F, class CT, class X>
    inline bool xreducer_stepper<F, CT, X>::equal(const self_type& rhs) const
    {
        return &m_reducer == &(rhs.m_reducer) && m_stepper.equal(rhs.m_stepper);
    }

    template <class F, class CT, class X>
    inline auto xreducer_stepper<F, CT, X>::aggregate(size_type dim) const -> reference
    {
        size_type index = axis(dim);
        size_type size = shape(index);
        reference res;
        if (dim != m_reducer.m_axes.size() - 1)
        {
            res = aggregate(dim + 1);
            for (size_type i = 1; i != size; ++i)
            {
                m_stepper.step(index);
                res = m_reducer.m_merge(res, aggregate(dim + 1));
            }
        }
        else
        {
            res = m_reducer.m_init(*m_stepper);
            for (size_type i = 1; i != size; ++i)
            {
                m_stepper.step(index);
                res = m_reducer.m_reduce(res, *m_stepper);
            }
        }
        m_stepper.reset(index);
        return res;
    }

    template <class F, class CT, class X>
    inline auto xreducer_stepper<F, CT, X>::get_substepper_begin() const -> substepper_type
    {
        return m_reducer.m_e.stepper_begin(m_reducer.m_e.shape());
    }

    template <class F, class CT, class X>
    inline auto xreducer_stepper<F, CT, X>::get_dim(size_type dim) const noexcept -> size_type
    {
        return m_reducer.m_dim_mapping[dim];
    }

    template <class F, class CT, class X>
    inline auto xreducer_stepper<F, CT, X>::shape(size_type i) const noexcept -> size_type
    {
        return m_reducer.m_e.shape()[i];
    }

    template <class F, class CT, class X>
    inline auto xreducer_stepper<F, CT, X>::axis(size_type i) const noexcept -> size_type
    {
        return m_reducer.m_axes[i];
    }

    template <class F, class CT, class X>
    inline bool operator==(const xreducer_stepper<F, CT, X>& lhs,
                           const xreducer_stepper<F, CT, X>& rhs)
    {
        return lhs.equal(rhs);
    }

    template <class F, class CT, class X>
    inline bool operator!=(const xreducer_stepper<F, CT, X>& lhs,
                           const xreducer_stepper<F, CT, X>& rhs)
    {
        return !lhs.equal(rhs);
    }
}

#endif
