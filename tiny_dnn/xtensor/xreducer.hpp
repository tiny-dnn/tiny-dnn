/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XREDUCER_HPP
#define XREDUCER_HPP

#include <algorithm>
#include <cstddef>
#include <initializer_list>
#include <iterator>
#include <stdexcept>
#include <type_traits>
#include <utility>
#ifdef X_OLD_CLANG
#include <vector>
#endif

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

    template <class F, class E, class X>
    auto reduce(F&& f, E&& e, X&& axes) noexcept;

    template <class F, class E>
    auto reduce(F&& f, E&& e) noexcept;

#ifdef X_OLD_CLANG
    template <class F, class E, class I>
    auto reduce(F&& f, E&& e, std::initializer_list<I> axes) noexcept;
#else
    template <class F, class E, class I, std::size_t N>
    auto reduce(F&& f, E&& e, const I (&axes)[N]) noexcept;
#endif

    /*************
     * xreducer  *
     *************/

    template <class ST, class X>
    struct xreducer_shape_type;

    namespace detail
    {
        template <class F, class CT, class X>
        class reducing_iterator;
    }

    template <class F, class CT, class X>
    class xreducer;

    template <class F, class CT, class X>
    struct xiterable_inner_types<xreducer<F, CT, X>>
    {
        using xexpression_type = std::decay_t<CT>;
        using inner_shape_type = typename xreducer_shape_type<typename xexpression_type::shape_type, X>::type;
        using const_stepper = xindexed_stepper<xreducer<F, CT, X>>;
        using stepper = const_stepper;
        using const_broadcast_iterator = xiterator<const_stepper, inner_shape_type*>;
        using broadcast_iterator = const_broadcast_iterator;
        using const_iterator = const_broadcast_iterator;
        using iterator = const_iterator;
    };

    /**
     * @class xreducer
     * @brief Reducing function operating over specified axes.
     *
     * The xreducer class implements an \ref xexpression applying
     * a reducing function to an \ref xexpression over the specified
     * axes.
     *
     * @tparam F the function type
     * @tparam CT the closure type of the \ref xexpression to reduce
     * @tparam X the list of axes
     *
     * @sa reduce
     */
    template <class F, class CT, class X>
    class xreducer : public xexpression<xreducer<F, CT, X>>,
                     public xexpression_const_iterable<xreducer<F, CT, X>>
    {

    public:

        using self_type = xreducer<F, CT, X>;
        using functor_type = typename std::remove_reference<F>::type;
        using xexpression_type = std::decay_t<CT>;
        using axes_type = X;

        using value_type = typename xexpression_type::value_type;
        using reference = value_type;
        using const_reference = value_type;
        using pointer = value_type*;
        using const_pointer = const value_type*;

        using size_type = typename xexpression_type::size_type;
        using difference_type = typename xexpression_type::difference_type;

        using iterable_base = xexpression_const_iterable<self_type>;
        using inner_shape_type = typename iterable_base::inner_shape_type;
        using shape_type = inner_shape_type;

        using stepper = typename iterable_base::stepper;
        using const_stepper = typename iterable_base::const_stepper;

        using broadcast_iterator = typename iterable_base::broadcast_iterator;
        using const_broadcast_iterator = typename iterable_base::const_broadcast_iterator;

        using iterator = typename iterable_base::iterator;
        using const_iterator = typename iterable_base::const_iterator;

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
        const_reference operator[](const xindex& index) const;
        const_reference operator[](size_type i) const;

        template <class It>
        const_reference element(It first, It last) const;

        template <class S>
        bool broadcast_shape(S& shape) const;

        template <class S>
        bool is_trivial_broadcast(const S& strides) const noexcept;

        template <class S>
        const_stepper stepper_begin(const S& shape) const noexcept;
        template <class S>
        const_stepper stepper_end(const S& shape) const noexcept;

    private:

        CT m_e;
        functor_type m_f;
        axes_type m_axes;
        inner_shape_type m_shape;

        using index_type = xindex_type_t<typename xexpression_type::shape_type>;
        mutable index_type m_index;

        friend class detail::reducing_iterator<F, CT, X>;
    };

    /*************************
     * reduce implementation *
     *************************/

    /**
     * @brief Returns an \ref xexpression applying the speficied reducing
     * function to an expresssion over the given axes.
     *
     * @param f the reducing function to apply.
     * @param e the \ref xexpression to reduce.
     * @param axes the list of axes.
     *
     * The returned expression either hold a const reference to \p e or a copy
     * depending on whether \p e is an lvalue or an rvalue.
     */

    template <class F, class E, class X>
    inline auto reduce(F&& f, E&& e, X&& axes) noexcept
    {
        using reducer_type = xreducer<F, const_xclosure_t<E>, const_closure_t<X>>;
        return reducer_type(std::forward<F>(f), std::forward<E>(e), std::forward<X>(axes));
    }

    template <class F, class E>
    inline auto reduce(F&& f, E&& e) noexcept
    {
        auto ar = arange(e.dimension());
        using AR = decltype(ar);
        using reducer_type = xreducer<F, const_xclosure_t<E>, AR>;
        return reducer_type(std::forward<F>(f), std::forward<E>(e), std::move(ar));
    }

#ifdef X_OLD_CLANG
    template <class F, class E, class I>
    inline auto reduce(F&& f, E&& e, std::initializer_list<I> axes) noexcept
    {
        using axes_type = std::vector<typename std::decay_t<E>::size_type>;
        using reducer_type = xreducer<F, const_xclosure_t<E>, axes_type>;
        return reducer_type(std::forward<F>(f), std::forward<E>(e), forward_sequence<axes_type>(axes));
    }
#else
    template <class F, class E, class I, std::size_t N>
    inline auto reduce(F&& f, E&& e, const I (&axes)[N]) noexcept
    {
        using axes_type = std::array<typename std::decay_t<E>::size_type, N>;
        using reducer_type = xreducer<F, const_xclosure_t<E>, axes_type>;
        return reducer_type(std::forward<F>(f), std::forward<E>(e), forward_sequence<axes_type>(axes));
    }
#endif

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
                                   OutputIt d_first)
        {
            using difference_type = typename std::iterator_traits<InputIt>::difference_type;
            InputIt iter = first;
            while (iter != last && e_first != e_last)
            {
                if (std::distance(first, iter) != difference_type(*e_first))
                {
                    *d_first++ = *iter++;
                }
                else
                {
                    ++iter;
                    ++e_first;
                }
            }
            std::copy(iter, last, d_first);
        }

        template <class InputIt, class ExcludeIt, class OutputIt, class T>
        inline void inject(InputIt first, InputIt last,
                           ExcludeIt e_first, ExcludeIt e_last,
                           OutputIt d_first, T default_value)
        {
            using difference_type = typename std::iterator_traits<InputIt>::difference_type;
            OutputIt d_first_bu = d_first;
            while (first != last && e_first != e_last)
            {
                if (std::distance(d_first_bu, d_first) != difference_type(*e_first))
                {
                    *d_first++ = *first++;
                }
                else
                {
                    *d_first++ = default_value;
                    ++e_first;
                }
            }
            std::copy(first, last, d_first);
        }

        // This is not a true iterator since two instances
        // of reducing_iterator on the same xreducer share
        // the same state. However this allows optimization
        // and is not problematic since not in the public
        // interface.
        template <class F, class CT, class X>
        class reducing_iterator
        {

        public:

            using self_type = reducing_iterator<F, CT, X>;
            using reducer_type = xreducer<F, CT, X>;
            using value_type = typename reducer_type::value_type;
            using reference = typename reducer_type::reference;
            using pointer = typename reducer_type::pointer;
            using difference_type = typename reducer_type::difference_type;
            using size_type = typename reducer_type::size_type;
            using iterator_category = std::forward_iterator_tag;

            reducing_iterator() = default;
            reducing_iterator(const reducer_type& reducer, bool end = false);

            self_type& operator++();
            self_type operator++(int);

            reference operator*() const;

            bool equal(const self_type& rhs) const;

        private:

            void increment();

            size_type axes(size_type index) const;
            size_type shape(size_type index) const;

            const reducer_type& m_reducer;
            bool m_end;
        };

        template <class F, class CT, class X>
        inline bool operator==(const reducing_iterator<F, CT, X>& lhs,
                               const reducing_iterator<F, CT, X>& rhs)
        {
            return lhs.equal(rhs);
        }

        template <class F, class CT, class X>
        inline bool operator!=(const reducing_iterator<F, CT, X>& lhs,
                               const reducing_iterator<F, CT, X>& rhs)
        {
            return !(lhs.equal(rhs));
        }

        /*************************************
         * xreducing_iterator implementation *
         *************************************/

        template <class F, class CT, class X>
        inline reducing_iterator<F, CT, X>::reducing_iterator(const reducer_type& reducer, bool end)
            : m_reducer(reducer), m_end(end)
        {
        }

        template <class F, class CT, class X>
        inline auto reducing_iterator<F, CT, X>::operator++() -> self_type&
        {
            increment();
            return *this;
        }

        template <class F, class CT, class X>
        inline auto reducing_iterator<F, CT, X>::operator++(int) -> self_type
        {
            self_type tmp(*this);
            ++(*this);
            return tmp;
        }

        template <class F, class CT, class X>
        inline auto reducing_iterator<F, CT, X>::operator*() const -> reference
        {
            return m_reducer.m_e.element(m_reducer.m_index.cbegin(), m_reducer.m_index.cend());
        }

        template <class F, class CT, class X>
        inline bool reducing_iterator<F, CT, X>::equal(const self_type& rhs) const
        {
            return &m_reducer == &(rhs.m_reducer) && m_end == rhs.m_end;
        }

        template <class F, class CT, class X>
        inline void reducing_iterator<F, CT, X>::increment()
        {
            size_type i = m_reducer.m_axes.size();
            while (i != 0)
            {
                --i;
                if (++(m_reducer.m_index[axes(i)]) != shape(axes(i)))
                {
                    return;
                }
                else
                {
                    m_reducer.m_index[axes(i)] = 0;
                }
            }
            if (i == 0)
            {
                m_end = true;
            }
        }

        template <class F, class CT, class X>
        inline auto reducing_iterator<F, CT, X>::axes(size_type index) const -> size_type
        {
            return m_reducer.m_axes[index];
        }

        template <class F, class CT, class X>
        inline auto reducing_iterator<F, CT, X>::shape(size_type index) const -> size_type
        {
            return m_reducer.m_e.shape()[index];
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
        : m_e(std::forward<CTA>(e)), m_f(std::forward<Func>(func)), m_axes(std::forward<AX>(axes)),
          m_shape(make_sequence<shape_type>(m_e.dimension() - m_axes.size(), 0)),
          m_index(make_sequence<index_type>(m_e.dimension(), 0))
    {
        if (!std::is_sorted(m_axes.cbegin(), m_axes.cend()))
        {
            throw std::runtime_error("Reducing axes should be sorted");
        }
        detail::excluding_copy(m_e.shape().begin(), m_e.shape().end(),
                               m_axes.begin(), m_axes.end(),
                               m_shape.begin());
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
        std::array<std::size_t, sizeof...(Args)> arg_array = {static_cast<std::size_t>(args)...};
        return element(arg_array.cbegin(), arg_array.cend());
    }

    /**
     * Returns a constant reference to the element at the specified position in the reducer.
     * @param index a sequence of indices specifying the position in the reducer. Indices
     * must be unsigned integers, the number of indices in the sequence should be equal or greater
     * than the number of dimensions of the reducer.
     */
    template <class F, class CT, class X>
    inline auto xreducer<F, CT, X>::operator[](const xindex& index) const -> const_reference
    {
        return element(index.cbegin(), index.cend());
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
     * The number of indices in the squence should be equal to or greater
     * than the number of dimensions of the reducer.
     */
    template <class F, class CT, class X>
    template <class It>
    inline auto xreducer<F, CT, X>::element(It first, It last) const -> const_reference
    {
        detail::inject(first, last, m_axes.cbegin(), m_axes.cend(),
                       m_index.begin(), size_type(0));
        using iter_type = detail::reducing_iterator<F, CT, X>;
        iter_type iter = iter_type(*this);
        iter_type iter_end = iter_type(*this, true);
        value_type init_value = *iter;
        value_type res = std::accumulate(++iter, iter_end, init_value, m_f);
        return res;
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
    inline bool xreducer<F, CT, X>::broadcast_shape(S& shape) const
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
        return const_stepper(this, offset);
    }

    template <class F, class CT, class X>
    template <class S>
    inline auto xreducer<F, CT, X>::stepper_end(const S& shape) const noexcept -> const_stepper
    {
        size_type offset = shape.size() - dimension();
        return const_stepper(this, offset, true);
    }
}

#endif
