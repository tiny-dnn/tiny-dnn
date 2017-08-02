/***************************************************************************
* Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XGENERATOR_HPP
#define XGENERATOR_HPP

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <tuple>
#include <type_traits>
#include <utility>

#include "xexpression.hpp"
#include "xiterable.hpp"
#include "xstrides.hpp"
#include "xutils.hpp"

namespace xt
{

    /**************
     * xgenerator *
     **************/

    template <class F, class R, class S>
    class xgenerator;

    template <class C, class R, class S>
    struct xiterable_inner_types<xgenerator<C, R, S>>
    {
        using inner_shape_type = S;
        using const_stepper = xindexed_stepper<xgenerator<C, R, S>>;
        using stepper = const_stepper;
        using const_iterator = xiterator<const_stepper, inner_shape_type*, DEFAULT_LAYOUT>;
        using iterator = const_iterator;
        using const_reverse_iterator = std::reverse_iterator<const_iterator>;
        using reverse_iterator = std::reverse_iterator<iterator>;
    };

    /**
     * @class xgenerator
     * @brief Multidimensional function operating on indices.
     *
     * The xgenerator class implements a multidimensional function,
     * generating a value from the supplied indices.
     *
     * @tparam F the function type
     * @tparam R the return type of the function
     * @tparam S the shape type of the generator
     */
    template <class F, class R, class S>
    class xgenerator : public xexpression<xgenerator<F, R, S>>,
                       public xexpression_const_iterable<xgenerator<F, R, S>>
    {
    public:

        using self_type = xgenerator<F, R, S>;
        using functor_type = typename std::remove_reference<F>::type;

        using value_type = R;
        using reference = value_type;
        using const_reference = value_type;
        using pointer = value_type*;
        using const_pointer = const value_type*;
        using size_type = std::size_t;
        using difference_type = std::ptrdiff_t;

        using iterable_base = xexpression_const_iterable<self_type>;
        using inner_shape_type = typename iterable_base::inner_shape_type;
        using shape_type = inner_shape_type;
        using strides_type = S;

        using stepper = typename iterable_base::stepper;
        using const_stepper = typename iterable_base::const_stepper;

        static constexpr layout_type static_layout = layout_type::any;
        static constexpr bool contiguous_layout = false;

        template <class Func>
        xgenerator(Func&& f, const S& shape) noexcept;

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

        template <class O>
        bool broadcast_shape(O& shape) const;

        template <class O>
        bool is_trivial_broadcast(const O& /*strides*/) const noexcept;

        template <class O>
        const_stepper stepper_begin(const O& shape) const noexcept;
        template <class O>
        const_stepper stepper_end(const O& shape, layout_type) const noexcept;

    private:

        functor_type m_f;
        inner_shape_type m_shape;
    };

    /*****************************
     * xgenerator implementation *
     *****************************/

    /**
     * @name Constructor
     */
    //@{
    /**
     * Constructs an xgenerator applying the specified function over the 
     * given shape.
     * @param f the function to apply
     * @param shape the shape of the xgenerator
     */
    template <class F, class R, class S>
    template <class Func>
    inline xgenerator<F, R, S>::xgenerator(Func&& f, const S& shape) noexcept
        : m_f(std::forward<Func>(f)), m_shape(shape)
    {
    }
    //@}

    /**
     * @name Size and shape
     */
    //@{
    /**
     * Returns the size of the expression.
     */
    template <class F, class R, class S>
    inline auto xgenerator<F, R, S>::size() const noexcept -> size_type
    {
        return compute_size(shape());
    }

    /**
     * Returns the number of dimensions of the function.
     */
    template <class F, class R, class S>
    inline auto xgenerator<F, R, S>::dimension() const noexcept -> size_type
    {
        return m_shape.size();
    }

    /**
     * Returns the shape of the xgenerator.
     */
    template <class F, class R, class S>
    inline auto xgenerator<F, R, S>::shape() const noexcept -> const inner_shape_type&
    {
        return m_shape;
    }

    template <class F, class R, class S>
    inline layout_type xgenerator<F, R, S>::layout() const noexcept
    {
        return static_layout;
    }

    //@}

    /**
     * @name Data
     */
    /**
     * Returns the evaluated element at the specified position in the function.
     * @param args a list of indices specifying the position in the function. Indices
     * must be unsigned integers, the number of indices should be equal or greater than
     * the number of dimensions of the function.
     */
    template <class F, class R, class S>
    template <class... Args>
    inline auto xgenerator<F, R, S>::operator()(Args... args) const -> const_reference
    {
        XTENSOR_ASSERT(check_index(shape(), args...));
        return m_f(args...);
    }

    template <class F, class R, class S>
    inline auto xgenerator<F, R, S>::operator[](const xindex& index) const -> const_reference
    {
        return element(index.begin(), index.end());
    }

    template <class F, class R, class S>
    inline auto xgenerator<F, R, S>::operator[](size_type i) const -> const_reference
    {
        return operator()(i);
    }

    /**
     * Returns a constant reference to the element at the specified position in the function.
     * @param first iterator starting the sequence of indices
     * @param last iterator ending the sequence of indices
     * The number of indices in the sequence should be equal to or greater
     * than the number of dimensions of the container.
     */
    template <class F, class R, class S>
    template <class It>
    inline auto xgenerator<F, R, S>::element(It first, It last) const -> const_reference
    {
        XTENSOR_ASSERT(check_element_index(shape(), first, last));
        return m_f.element(first, last);
    }
    //@}

    /**
     * @name Broadcasting
     */
    //@{
    /**
     * Broadcast the shape of the function to the specified parameter.
     * @param shape the result shape
     * @return a boolean indicating whether the broadcasting is trivial
     */
    template <class F, class R, class S>
    template <class O>
    inline bool xgenerator<F, R, S>::broadcast_shape(O& shape) const
    {
        return xt::broadcast_shape(m_shape, shape);
    }

    /**
     * Compares the specified strides with those of the container to see whether
     * the broadcasting is trivial.
     * @return a boolean indicating whether the broadcasting is trivial
     */
    template <class F, class R, class S>
    template <class O>
    inline bool xgenerator<F, R, S>::is_trivial_broadcast(const O& /*strides*/) const noexcept
    {
        return false;
    }
    //@}

    template <class F, class R, class S>
    template <class O>
    inline auto xgenerator<F, R, S>::stepper_begin(const O& shape) const noexcept -> const_stepper
    {
        size_type offset = shape.size() - dimension();
        return const_stepper(this, offset);
    }

    template <class F, class R, class S>
    template <class O>
    inline auto xgenerator<F, R, S>::stepper_end(const O& shape, layout_type) const noexcept -> const_stepper
    {
        size_type offset = shape.size() - dimension();
        return const_stepper(this, offset, true);
    }

    namespace detail
    {
#ifdef X_OLD_CLANG
        template <class Functor, class I>
        inline auto make_xgenerator(Functor&& f, std::initializer_list<I> shape) noexcept
        {
            using shape_type = std::vector<std::size_t>;
            using type = xgenerator<Functor, typename Functor::value_type, shape_type>;
            return type(std::forward<Functor>(f), forward_sequence<shape_type>(shape));
        }
#else
        template <class Functor, class I, std::size_t L>
        inline auto make_xgenerator(Functor&& f, const I (&shape)[L]) noexcept
        {
            using shape_type = std::array<std::size_t, L>;
            using type = xgenerator<Functor, typename Functor::value_type, shape_type>;
            return type(std::forward<Functor>(f), forward_sequence<shape_type>(shape));
        }
#endif

        template <class Functor, class S>
        inline auto make_xgenerator(Functor&& f, S&& shape) noexcept
        {
            using type = xgenerator<Functor, typename Functor::value_type, std::decay_t<S>>;
            return type(std::forward<Functor>(f), std::forward<S>(shape));
        }
    }
}

#endif
