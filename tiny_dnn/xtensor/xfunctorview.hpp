/***************************************************************************
* Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XFUNCTORVIEW_HPP
#define XFUNCTORVIEW_HPP

#include <algorithm>
#include <array>
#include <cstddef>
#include <type_traits>
#include <utility>

#include "xtensor/xexpression.hpp"
#include "xtensor/xiterator.hpp"
#include "xtensor/xsemantic.hpp"
#include "xtensor/xutils.hpp"

#include "xtensor/xarray.hpp"
#include "xtensor/xtensor.hpp"

namespace xt
{

    /****************************
     * xfunctorview declaration *
     ****************************/

    template <class F, class IT>
    class xfunctor_iterator;

    template <class F, class ST>
    class xfunctor_stepper;

    template <class F, class CT>
    class xfunctorview;

    /*******************************
     * xfunctorview_temporary_type *
     *******************************/

    namespace detail
    {
        template <class F, class S, layout_type L>
        struct functorview_temporary_type_impl
        {
            using type = xarray<typename F::value_type, L>;
        };

        template <class F, class T, std::size_t N, layout_type L>
        struct functorview_temporary_type_impl<F, std::array<T, N>, L>
        {
            using type = xtensor<typename F::value_type, N, L>;
        };
    }

    template <class F, class E>
    struct xfunctorview_temporary_type
    {
        using type = typename detail::functorview_temporary_type_impl<F, typename E::shape_type, E::static_layout>::type;
    };

    template <class F, class CT>
    struct xcontainer_inner_types<xfunctorview<F, CT>>
    {
        using xexpression_type = std::decay_t<CT>;
        using temporary_type = typename xfunctorview_temporary_type<F, xexpression_type>::type;
    };

#define DL DEFAULT_LAYOUT
    /**
     * @class xfunctorview
     * @brief View of an xexpression .
     *
     * The xfunctorview class is an expression addressing its elements by applying a functor to the
     * corresponding element of an underlying expression. Unlike e.g. xgenerator, an xfunctorview is
     * an lvalue. It is used e.g. to access real and imaginary parts of complex expressions.
     * expressions.
     * xfunctorview is not meant to be used directly, but through helper functions such
     * as \ref real or \ref imag.
     *
     * @tparam F the functor type to be applied to the elements of specified expression.
     * @tparam CT the closure type of the \ref xexpression type underlying this view
     *
     * @sa real, imag
     */
    template <class F, class CT>
    class xfunctorview : public xview_semantic<xfunctorview<F, CT>>
    {
    public:

        using self_type = xfunctorview<F, CT>;
        using xexpression_type = std::decay_t<CT>;
        using semantic_base = xview_semantic<self_type>;
        using functor_type = typename std::decay_t<F>;

        using value_type = typename functor_type::value_type;
        using reference = typename functor_type::reference;
        using const_reference = typename functor_type::const_reference;
        using pointer = typename functor_type::pointer;
        using const_pointer = typename functor_type::const_pointer;
        using size_type = typename xexpression_type::size_type;
        using difference_type = typename xexpression_type::difference_type;

        using shape_type = typename xexpression_type::shape_type;

        using stepper = xfunctor_stepper<functor_type, typename xexpression_type::stepper>;
        using const_stepper = xfunctor_stepper<functor_type, typename xexpression_type::const_stepper>;

        using iterator = xfunctor_iterator<functor_type, typename xexpression_type::iterator>;
        using const_iterator = xfunctor_iterator<functor_type, typename xexpression_type::const_iterator>;

        template <layout_type L>
        using broadcast_iterator = xfunctor_iterator<functor_type, typename xexpression_type::template broadcast_iterator<L>>;
        template <layout_type L>
        using const_broadcast_iterator = xfunctor_iterator<functor_type, typename xexpression_type::template const_broadcast_iterator<L>>;

        template <class S, layout_type L>
        using shaped_xiterator = xfunctor_iterator<functor_type, xiterator<typename xexpression_type::stepper, S, L>>;
        template <class S, layout_type L>
        using const_shaped_xiterator = xfunctor_iterator<functor_type, xiterator<typename xexpression_type::const_stepper, S, L>>;

        using reverse_iterator = xfunctor_iterator<functor_type, typename xexpression_type::reverse_iterator>;
        using const_reverse_iterator = xfunctor_iterator<functor_type, typename xexpression_type::const_reverse_iterator>;

        template <layout_type L>
        using reverse_broadcast_iterator = xfunctor_iterator<functor_type, typename xexpression_type::template reverse_broadcast_iterator<L>>;
        template <layout_type L>
        using const_reverse_broadcast_iterator = xfunctor_iterator<functor_type, typename xexpression_type::template const_reverse_broadcast_iterator<L>>;

        template <class S, layout_type L>
        using reverse_shaped_xiterator = xfunctor_iterator<functor_type, typename xexpression_type::template reverse_shaped_xiterator<S, L>>;
        template <class S, layout_type L>
        using const_reverse_shaped_xiterator = xfunctor_iterator<functor_type, typename xexpression_type::template const_reverse_shaped_xiterator<S, L>>;

        static constexpr layout_type static_layout = xexpression_type::static_layout;
        static constexpr bool contiguous_layout = false;

        xfunctorview(CT) noexcept;

        template <class Func, class E>
        xfunctorview(Func&&, E&&) noexcept;

        template <class E>
        self_type& operator=(const xexpression<E>& e);

        template <class E>
        disable_xexpression<E, self_type>& operator=(const E& e);

        size_type size() const noexcept;
        size_type dimension() const noexcept;
        const shape_type& shape() const noexcept;
        layout_type layout() const noexcept;

        template <class... Args>
        reference operator()(Args... args);
        reference operator[](const xindex& index);
        reference operator[](size_type i);

        template <class IT>
        reference element(IT first, IT last);

        template <class... Args>
        const_reference operator()(Args... args) const;
        const_reference operator[](const xindex& index) const;
        const_reference operator[](size_type i) const;

        template <class IT>
        const_reference element(IT first, IT last) const;

        template <class S>
        bool broadcast_shape(S& shape) const;

        template <class S>
        bool is_trivial_broadcast(const S& strides) const;

        iterator begin() noexcept;
        iterator end() noexcept;

        const_iterator begin() const noexcept;
        const_iterator end() const noexcept;
        const_iterator cbegin() const noexcept;
        const_iterator cend() const noexcept;

        template <layout_type L = DL>
        broadcast_iterator<L> xbegin() noexcept;
        template <layout_type L = DL>
        broadcast_iterator<L> xend() noexcept;

        template <layout_type L = DL>
        const_broadcast_iterator<L> xbegin() const noexcept;
        template <layout_type L = DL>
        const_broadcast_iterator<L> xend() const noexcept;
        template <layout_type L = DL>
        const_broadcast_iterator<L> cxbegin() const noexcept;
        template <layout_type L = DL>
        const_broadcast_iterator<L> cxend() const noexcept;

        template <class S, layout_type L = DL>
        shaped_xiterator<S, L> xbegin(const S& shape) noexcept;
        template <class S, layout_type L = DL>
        shaped_xiterator<S, L> xend(const S& shape) noexcept;

        template <class S, layout_type L = DL>
        const_shaped_xiterator<S, L> xbegin(const S& shape) const noexcept;
        template <class S, layout_type L = DL>
        const_shaped_xiterator<S, L> xend(const S& shape) const noexcept;
        template <class S, layout_type L = DL>
        const_shaped_xiterator<S, L> cxbegin(const S& shape) const noexcept;
        template <class S, layout_type L = DL>
        const_shaped_xiterator<S, L> cxend(const S& shape) const noexcept;

        reverse_iterator rbegin() noexcept;
        reverse_iterator rend() noexcept;

        const_reverse_iterator rbegin() const noexcept;
        const_reverse_iterator rend() const noexcept;
        const_reverse_iterator crbegin() const noexcept;
        const_reverse_iterator crend() const noexcept;

        template <layout_type L = DL>
        reverse_broadcast_iterator<L> xrbegin() noexcept;
        template <layout_type L = DL>
        reverse_broadcast_iterator<L> xrend() noexcept;

        template <layout_type L = DL>
        const_reverse_broadcast_iterator<L> xrbegin() const noexcept;
        template <layout_type L = DL>
        const_reverse_broadcast_iterator<L> xrend() const noexcept;
        template <layout_type L = DL>
        const_reverse_broadcast_iterator<L> cxrbegin() const noexcept;
        template <layout_type L = DL>
        const_reverse_broadcast_iterator<L> cxrend() const noexcept;

        template <class S, layout_type L = DL>
        reverse_shaped_xiterator<S, L> xrbegin(const S& shape) noexcept;
        template <class S, layout_type L = DL>
        reverse_shaped_xiterator<S, L> xrend(const S& shape) noexcept;

        template <class S, layout_type L = DL>
        const_reverse_shaped_xiterator<S, L> xrbegin(const S& shape) const noexcept;
        template <class S, layout_type L = DL>
        const_reverse_shaped_xiterator<S, L> xrend(const S& shape) const noexcept;
        template <class S, layout_type L = DL>
        const_reverse_shaped_xiterator<S, L> cxrbegin(const S& shape) const noexcept;
        template <class S, layout_type L = DL>
        const_reverse_shaped_xiterator<S, L> cxrend(const S& shape) const noexcept;

        template <class S>
        stepper stepper_begin(const S& shape) noexcept;
        template <class S>
        stepper stepper_end(const S& shape, layout_type l) noexcept;
        template <class S>
        const_stepper stepper_begin(const S& shape) const noexcept;
        template <class S>
        const_stepper stepper_end(const S& shape, layout_type l) const noexcept;

    private:

        CT m_e;
        functor_type m_functor;

        using temporary_type = typename xcontainer_inner_types<self_type>::temporary_type;
        void assign_temporary_impl(temporary_type&& tmp);
        friend class xview_semantic<xfunctorview<F, CT>>;
    };

#undef DL

    /*********************************
     * xfunctor_iterator declaration *
     *********************************/

    template <class F, class IT>
    class xfunctor_iterator
    {
    public:

        using functor_type = std::decay_t<F>;
        using value_type = typename functor_type::value_type;

        using subiterator_traits = std::iterator_traits<IT>;

        using reference = apply_cv_t<typename subiterator_traits::reference, value_type>;
        using pointer = std::remove_reference_t<reference>*;
        using difference_type = typename subiterator_traits::difference_type;
        using iterator_category = typename subiterator_traits::iterator_category;

        using self_type = xfunctor_iterator<F, IT>;

        xfunctor_iterator(const IT&, const functor_type*);

        self_type& operator++();
        self_type operator++(int);

        reference operator*() const;
        pointer operator->() const;

        bool equal(const xfunctor_iterator& rhs) const;

    private:

        IT m_it;
        const functor_type* p_functor;

        template <class F_, class IT_>
        friend xfunctor_iterator<F_, IT_> operator+(xfunctor_iterator<F_, IT_>, xfunctor_iterator<F_, IT_>);

        template <class F_, class IT_>
        friend typename xfunctor_iterator<F_, IT_>::difference_type operator-(xfunctor_iterator<F_, IT_>, xfunctor_iterator<F_, IT_>);
    };

    template <class F, class IT>
    bool operator==(const xfunctor_iterator<F, IT>& lhs,
                    const xfunctor_iterator<F, IT>& rhs);

    template <class F, class IT>
    bool operator!=(const xfunctor_iterator<F, IT>& lhs,
                    const xfunctor_iterator<F, IT>& rhs);

    template <class F, class IT>
    xfunctor_iterator<F, IT> operator+(xfunctor_iterator<F, IT> it1, xfunctor_iterator<F, IT> it2)
    {
        return xfunctor_iterator<F, IT>(it1.m_it + it2.m_it);
    }

    template <class F, class IT>
    typename xfunctor_iterator<F, IT>::difference_type operator-(xfunctor_iterator<F, IT> it1, xfunctor_iterator<F, IT> it2)
    {
        return it1.m_it - it2.m_it;
    }

    /********************************
     * xfunctor_stepper declaration *
     ********************************/

    template <class F, class ST>
    class xfunctor_stepper
    {
    public:

        using functor_type = std::decay_t<F>;

        using value_type = typename functor_type::value_type;
        using reference = apply_cv_t<typename ST::reference, value_type>;
        using pointer = std::remove_reference_t<reference>*;
        using size_type = typename ST::size_type;
        using difference_type = typename ST::difference_type;

        xfunctor_stepper() = default;
        xfunctor_stepper(const ST&, const functor_type*);

        reference operator*() const;

        void step(size_type dim, size_type n = 1);
        void step_back(size_type dim, size_type n = 1);
        void reset(size_type dim);
        void reset_back(size_type dim);

        void to_begin();
        void to_end(layout_type);

        bool equal(const xfunctor_stepper& rhs) const;

    private:

        ST m_stepper;
        const functor_type* p_functor;
    };

    template <class F, class ST>
    bool operator==(const xfunctor_stepper<F, ST>& lhs,
                    const xfunctor_stepper<F, ST>& rhs);

    template <class F, class ST>
    bool operator!=(const xfunctor_stepper<F, ST>& lhs,
                    const xfunctor_stepper<F, ST>& rhs);

    /*******************************
     * xfunctorview implementation *
     *******************************/

    /**
     * @name Constructors
     */
    //@{

    /**
     * Constructs an xfunctorview expression wrappering the specified \ref xexpression.
     *
     * @param e the underlying expression
     */
    template <class F, class CT>
    inline xfunctorview<F, CT>::xfunctorview(CT e) noexcept
        : m_e(e), m_functor(functor_type())
    {
    }

    /**
    * Constructs an xfunctorview expression wrappering the specified \ref xexpression.
    *
    * @param func the functor to be applied to the elements of the underlying expression.
    * @param e the underlying expression
    */
    template <class F, class CT>
    template <class Func, class E>
    inline xfunctorview<F, CT>::xfunctorview(Func&& func, E&& e) noexcept
        : m_e(std::forward<E>(e)), m_functor(std::forward<Func>(func))
    {
    }
    //@}

    /**
     * @name Extended copy semantic
     */
    //@{
    /**
     * The extended assignment operator.
     */
    template <class F, class CT>
    template <class E>
    inline auto xfunctorview<F, CT>::operator=(const xexpression<E>& e) -> self_type&
    {
        bool cond = (e.derived_cast().shape().size() == dimension()) && std::equal(shape().begin(), shape().end(), e.derived_cast().shape().begin());
        if (!cond)
        {
            semantic_base::operator=(broadcast(e.derived_cast(), shape()));
        }
        else
        {
            semantic_base::operator=(e);
        }
        return *this;
    }
    //@}

    template <class F, class CT>
    template <class E>
    inline auto xfunctorview<F, CT>::operator=(const E& e) -> disable_xexpression<E, self_type>&
    {
        std::fill(begin(), end(), e);
        return *this;
    }

    template <class F, class CT>
    inline void xfunctorview<F, CT>::assign_temporary_impl(temporary_type&& tmp)
    {
        std::copy(tmp.cbegin(), tmp.cend(), xbegin());
    }

    /**
     * @name Size and shape
     */
    /**
     * Returns the size of the expression.
     */
    template <class F, class CT>
    inline auto xfunctorview<F, CT>::size() const noexcept -> size_type
    {
        return m_e.size();
    }

    /**
     * Returns the number of dimensions of the expression.
     */
    template <class F, class CT>
    inline auto xfunctorview<F, CT>::dimension() const noexcept -> size_type
    {
        return m_e.dimension();
    }

    /**
     * Returns the shape of the expression.
     */
    template <class F, class CT>
    inline auto xfunctorview<F, CT>::shape() const noexcept -> const shape_type&
    {
        return m_e.shape();
    }

    /**
     * Returns the layout_type of the expression.
     */
    template <class F, class CT>
    inline layout_type xfunctorview<F, CT>::layout() const noexcept
    {
        return m_e.layout();
    }
    //@}

    /**
     * @name Data
     */
    /**
     * Returns a reference to the element at the specified position in the expression.
     * @param args a list of indices specifying the position in the function. Indices
     * must be unsigned integers, the number of indices should be equal or greater than
     * the number of dimensions of the expression.
     */
    template <class F, class CT>
    template <class... Args>
    inline auto xfunctorview<F, CT>::operator()(Args... args) -> reference
    {
        return m_functor(m_e(args...));
    }

    /**
     * Returns a reference to the element at the specified position in the expression.
     * @param index a sequence of indices specifying the position in the function. Indices
     * must be unsigned integers, the number of indices in the sequence should be equal or greater
     * than the number of dimensions of the container.
     */
    template <class F, class CT>
    inline auto xfunctorview<F, CT>::operator[](const xindex& index) -> reference
    {
        return m_functor(m_e[index]);
    }

    template <class F, class CT>
    inline auto xfunctorview<F, CT>::operator[](size_type i) -> reference
    {
        return operator()(i);
    }

    /**
     * Returns a reference to the element at the specified position in the expression.
     * @param first iterator starting the sequence of indices
     * @param last iterator ending the sequence of indices
     * The number of indices in the sequence should be equal to or greater
     * than the number of dimensions of the function.
     */
    template <class F, class CT>
    template <class IT>
    inline auto xfunctorview<F, CT>::element(IT first, IT last) -> reference
    {
        return m_functor(m_e.element(first, last));
    }

    /**
     * Returns a constant reference to the element at the specified position in the expression.
     * @param args a list of indices specifying the position in the function. Indices
     * must be unsigned integers, the number of indices should be equal or greater than
     * the number of dimensions of the expression.
     */
    template <class F, class CT>
    template <class... Args>
    inline auto xfunctorview<F, CT>::operator()(Args... args) const -> const_reference
    {
        return m_functor(m_e(args...));
    }

    /**
     * Returns a constant reference to the element at the specified position in the expression.
     * @param index a sequence of indices specifying the position in the function. Indices
     * must be unsigned integers, the number of indices in the sequence should be equal or greater
     * than the number of dimensions of the container.
     */
    template <class F, class CT>
    inline auto xfunctorview<F, CT>::operator[](const xindex& index) const -> const_reference
    {
        return m_functor(m_e[index]);
    }

    template <class F, class CT>
    inline auto xfunctorview<F, CT>::operator[](size_type i) const -> const_reference
    {
        return operator()(i);
    }

    /**
     * Returns a constant reference to the element at the specified position in the expression.
     * @param first iterator starting the sequence of indices
     * @param last iterator ending the sequence of indices
     * The number of indices in the sequence should be equal to or greater
     * than the number of dimensions of the function.
     */
    template <class F, class CT>
    template <class IT>
    inline auto xfunctorview<F, CT>::element(IT first, IT last) const -> const_reference
    {
        return m_functor(m_e.element(first, last));
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
    template <class F, class CT>
    template <class S>
    inline bool xfunctorview<F, CT>::broadcast_shape(S& shape) const
    {
        return m_e.broadcast_shape(shape);
    }

    /**
     * Compares the specified strides with those of the container to see whether
     * the broadcasting is trivial.
     * @return a boolean indicating whether the broadcasting is trivial
     */
    template <class F, class CT>
    template <class S>
    inline bool xfunctorview<F, CT>::is_trivial_broadcast(const S& strides) const
    {
        return m_e.is_trivial_broadcast(strides);
    }
    //@}

    /**
     * @name Iterators
     */
    //@{
    /**
     * Returns an iterator to the first element of the expression.
     */
    template <class F, class CT>
    inline auto xfunctorview<F, CT>::begin() noexcept -> iterator
    {
        return iterator(m_e.begin(), &m_functor);
    }

    /**
     * Returns an iterator to the element following the last element
     * of the expression.
     */
    template <class F, class CT>
    inline auto xfunctorview<F, CT>::end() noexcept -> iterator
    {
        return iterator(m_e.end(), &m_functor);
    }

    /**
     * Returns a constant iterator to the first element of the expression.
     */
    template <class F, class CT>
    inline auto xfunctorview<F, CT>::begin() const noexcept -> const_iterator
    {
        return const_iterator(m_e.cbegin(), &m_functor);
    }

    /**
     * Returns a constant iterator to the element following the last element
     * of the expression.
     */
    template <class F, class CT>
    inline auto xfunctorview<F, CT>::end() const noexcept -> const_iterator
    {
        return const_iterator(m_e.cend(), &m_functor);
    }

    /**
     * Returns a constant iterator to the first element of the expression.
     */
    template <class F, class CT>
    inline auto xfunctorview<F, CT>::cbegin() const noexcept -> const_iterator
    {
        return const_iterator(m_e.cbegin(), &m_functor);
    }

    /**
     * Returns a constant iterator to the element following the last element
     * of the expression.
     */
    template <class F, class CT>
    inline auto xfunctorview<F, CT>::cend() const noexcept -> const_iterator
    {
        return const_iterator(m_e.cend(), &m_functor);
    }
    //@}

    /**
     * @name Broadcast iterators
     */
    //@{
    /**
     * Returns an iterator to the first element of the expression.
     * @tparam L layout used for the traversal. Default value is \c DEFAULT_LAYOUT.
     */
    template <class F, class CT>
    template <layout_type L>
    inline auto xfunctorview<F, CT>::xbegin() noexcept -> broadcast_iterator<L>
    {
        return broadcast_iterator<L>(m_e.template xbegin<L>(), &m_functor);
    }

    /**
     * Returns an iterator to the element following the last element
     * of the expression.
     * @tparam L layout used for the traversal. Default value is \c DEFAULT_LAYOUT.
     */
    template <class F, class CT>
    template <layout_type L>
    inline auto xfunctorview<F, CT>::xend() noexcept -> broadcast_iterator<L>
    {
        return broadcast_iterator<L>(m_e.template xend<L>(), &m_functor);
    }

    /**
     * Returns a constant iterator to the first element of the expression.
     * @tparam L layout used for the traversal. Default value is \c DEFAULT_LAYOUT.
     */
    template <class F, class CT>
    template <layout_type L>
    inline auto xfunctorview<F, CT>::xbegin() const noexcept -> const_broadcast_iterator<L>
    {
        return cxbegin<L>();
    }

    /**
     * Returns a constant iterator to the element following the last element
     * of the expression.
     * @tparam L layout used for the traversal. Default value is \c DEFAULT_LAYOUT.
     */
    template <class F, class CT>
    template <layout_type L>
    inline auto xfunctorview<F, CT>::xend() const noexcept -> const_broadcast_iterator<L>
    {
        return cxend<L>();
    }

    /**
     * Returns a constant iterator to the first element of the expression.
     * @tparam L layout used for the traversal. Default value is \c DEFAULT_LAYOUT.
     */
    template <class F, class CT>
    template <layout_type L>
    inline auto xfunctorview<F, CT>::cxbegin() const noexcept -> const_broadcast_iterator<L>
    {
        return const_broadcast_iterator<L>(m_e.template cxbegin<L>(), &m_functor);
    }

    /**
     * Returns a constant iterator to the element following the last element
     * of the expression.
     * @tparam L layout used for the traversal. Default value is \c DEFAULT_LAYOUT.
     */
    template <class F, class CT>
    template <layout_type L>
    inline auto xfunctorview<F, CT>::cxend() const noexcept -> const_broadcast_iterator<L>
    {
        return const_broadcast_iterator<L>(m_e.template cxend<L>(), &m_functor);
    }

    /**
     * Returns a constant iterator to the first element of the expression. The
     * iteration is broadcasted to the specified shape.
     * @param shape the shape used for broadcasting
     * @tparam S type of the \c shape parameter.
     * @tparam L layout used for the traversal. Default value is \c DEFAULT_LAYOUT.
     */
    template <class F, class CT>
    template <class S, layout_type L>
    inline auto xfunctorview<F, CT>::xbegin(const S& shape) noexcept -> shaped_xiterator<S, L>
    {
        return shaped_xiterator<S, L>(m_e.template xbegin<S, L>(shape), &m_functor);
    }

    /**
     * Returns a constant iterator to the element following the last element of the
     * expression. The iteration is broadcasted to the specified shape.
     * @param shape the shape used for broadcasting
     * @tparam S type of the \c shape parameter.
     * @tparam L layout used for the traversal. Default value is \c DEFAULT_LAYOUT.
     */
    template <class F, class CT>
    template <class S, layout_type L>
    inline auto xfunctorview<F, CT>::xend(const S& shape) noexcept -> shaped_xiterator<S, L>
    {
        return shaped_xiterator<S, L>(m_e.template xend<S, L>(shape), &m_functor);
    }

    /**
     * Returns a constant iterator to the first element of the expression. The
     * iteration is broadcasted to the specified shape.
     * @param shape the shape used for broadcasting
     * @tparam S type of the \c shape parameter.
     * @tparam L layout used for the traversal. Default value is \c DEFAULT_LAYOUT.
     */
    template <class F, class CT>
    template <class S, layout_type L>
    inline auto xfunctorview<F, CT>::xbegin(const S& shape) const noexcept -> const_shaped_xiterator<S, L>
    {
        return cxbegin<S, L>(shape);
    }

    /**
     * Returns a constant iterator to the element following the last element of the
     * expression. The iteration is broadcasted to the specified shape.
     * @param shape the shape used for broadcasting
     * @tparam S type of the \c shape parameter.
     * @tparam L layout used for the traversal. Default value is \c DEFAULT_LAYOUT.
     */
    template <class F, class CT>
    template <class S, layout_type L>
    inline auto xfunctorview<F, CT>::xend(const S& shape) const noexcept -> const_shaped_xiterator<S, L>
    {
        return cxend<S, L>(shape);
    }

    /**
     * Returns a constant iterator to the first element of the expression. The
     * iteration is broadcasted to the specified shape.
     * @param shape the shape used for broadcasting
     * @tparam S type of the \c shape parameter.
     * @tparam L layout used for the traversal. Default value is \c DEFAULT_LAYOUT.
     */
    template <class F, class CT>
    template <class S, layout_type L>
    inline auto xfunctorview<F, CT>::cxbegin(const S& shape) const noexcept -> const_shaped_xiterator<S, L>
    {
        return const_shaped_xiterator<S, L>(m_e.template cxbegin<S, L>(shape), &m_functor);
    }

    /**
     * Returns a constant iterator to the element following the last element of the
     * expression. The iteration is broadcasted to the specified shape.
     * @param shape the shape used for broadcasting
     * @tparam S type of the \c shape parameter.
     * @tparam L layout used for the traversal. Default value is \c DEFAULT_LAYOUT.
     */
    template <class F, class CT>
    template <class S, layout_type L>
    inline auto xfunctorview<F, CT>::cxend(const S& shape) const noexcept -> const_shaped_xiterator<S, L>
    {
        return const_shaped_xiterator<S, L>(m_e.template cxend<S, L>(shape), &m_functor);
    }
    //@}

    /**
     * @name Reverse iterators
     */
    //@{
    /**
     * Returns an iterator to the first element of the reversed expression.
     */
    template <class F, class CT>
    inline auto xfunctorview<F, CT>::rbegin() noexcept -> reverse_iterator
    {
        return reverse_iterator(m_e.rbegin(), &m_functor);
    }

    /**
     * Returns an iterator to the element following the last element
     * of the reversed expression.
     */
    template <class F, class CT>
    inline auto xfunctorview<F, CT>::rend() noexcept -> reverse_iterator
    {
        return reverse_iterator(m_e.rend(), &m_functor);
    }

    /**
     * Returns a constant iterator to the first element of the reversed expression.
     */
    template <class F, class CT>
    inline auto xfunctorview<F, CT>::rbegin() const noexcept -> const_reverse_iterator
    {
        return crbegin();
    }

    /**
     * Returns a constant iterator to the element following the last element
     * of the reversed expression.
     */
    template <class F, class CT>
    inline auto xfunctorview<F, CT>::rend() const noexcept -> const_reverse_iterator
    {
        return crend();
    }

    /**
     * Returns a constant iterator to the first element of the reversed expression.
     */
    template <class F, class CT>
    inline auto xfunctorview<F, CT>::crbegin() const noexcept -> const_reverse_iterator
    {
        return const_reverse_iterator(m_e.crbegin(), &m_functor);
    }

    /**
     * Returns a constant iterator to the element following the last element
     * of the reversed expression.
     */
    template <class F, class CT>
    inline auto xfunctorview<F, CT>::crend() const noexcept -> const_reverse_iterator
    {
        return const_reverse_iterator(m_e.crend(), &m_functor);
    }
    //@}

    /**
     * @name Reverse broadcast iterators
     */
    //@{
    /**
     * Returns an iterator to the first element of the reversed expression.
     * @tparam L layout used for the traversal. Default value is \c DEFAULT_LAYOUT.
     */
    template <class F, class CT>
    template <layout_type L>
    inline auto xfunctorview<F, CT>::xrbegin() noexcept -> reverse_broadcast_iterator<L>
    {
        return reverse_broadcast_iterator<L>(m_e.template xrbegin<L>(), &m_functor);
    }

    /**
     * Returns an iterator to the element following the last element
     * of the reversed expression.
     * @tparam L layout used for the traversal. Default value is \c DEFAULT_LAYOUT.
     */
    template <class F, class CT>
    template <layout_type L>
    inline auto xfunctorview<F, CT>::xrend() noexcept -> reverse_broadcast_iterator<L>
    {
        return reverse_broadcast_iterator<L>(m_e.template xrend<L>(), &m_functor);
    }

    /**
     * Returns a constant iterator to the first element of the reversed expression.
     * @tparam L layout used for the traversal. Default value is \c DEFAULT_LAYOUT.
     */
    template <class F, class CT>
    template <layout_type L>
    inline auto xfunctorview<F, CT>::xrbegin() const noexcept -> const_reverse_broadcast_iterator<L>
    {
        return cxrbegin<L>();
    }

    /**
     * Returns a constant iterator to the element following the last element
     * of the reversed expression.
     * @tparam L layout used for the traversal. Default value is \c DEFAULT_LAYOUT.
     */
    template <class F, class CT>
    template <layout_type L>
    inline auto xfunctorview<F, CT>::xrend() const noexcept -> const_reverse_broadcast_iterator<L>
    {
        return cxrend<L>();
    }

    /**
     * Returns a constant iterator to the first element of the reversed expression.
     * @tparam L layout used for the traversal. Default value is \c DEFAULT_LAYOUT.
     */
    template <class F, class CT>
    template <layout_type L>
    inline auto xfunctorview<F, CT>::cxrbegin() const noexcept -> const_reverse_broadcast_iterator<L>
    {
        return const_reverse_broadcast_iterator<L>(m_e.template cxrbegin<L>(), &m_functor);
    }

    /**
     * Returns a constant iterator to the element following the last element
     * of the reversed expression.
     * @tparam L layout used for the traversal. Default value is \c DEFAULT_LAYOUT.
     */
    template <class F, class CT>
    template <layout_type L>
    inline auto xfunctorview<F, CT>::cxrend() const noexcept -> const_reverse_broadcast_iterator<L>
    {
        return const_reverse_broadcast_iterator<L>(m_e.template cxrend<L>(), &m_functor);
    }

    /**
     * Returns an iterator to the first element of the expression. The
     * iteration is broadcasted to the specified shape.
     * @param shape the shape used for broadcasting
     * @tparam S type of the \c shape parameter.
     * @tparam L layout used for the traversal. Default value is \c DEFAULT_LAYOUT.
     */
    template <class F, class CT>
    template <class S, layout_type L>
    inline auto xfunctorview<F, CT>::xrbegin(const S& shape) noexcept -> reverse_shaped_xiterator<S, L>
    {
        return reverse_shaped_xiterator<S, L>(m_e.template xrbegin<S, L>(shape), &m_functor);
    }

    /**
     * Returns an iterator to the element following the last element of the
     * reversed expression. The iteration is broadcasted to the specified shape.
     * @param shape the shape used for broadcasting
     * @tparam S type of the \c shape parameter.
     * @tparam L layout used for the traversal. Default value is \c DEFAULT_LAYOUT.
     */
    template <class F, class CT>
    template <class S, layout_type L>
    inline auto xfunctorview<F, CT>::xrend(const S& shape) noexcept -> reverse_shaped_xiterator<S, L>
    {
        return reverse_shaped_xiterator<S, L>(m_e.template xrend<S, L>(shape), &m_functor);
    }

    /**
     * Returns a constant iterator to the first element of the reversed expression.
     * The iteration is broadcasted to the specified shape.
     * @param shape the shape used for broadcasting
     * @tparam S type of the \c shape parameter.
     * @tparam L layout used for the traversal. Default value is \c DEFAULT_LAYOUT.
     */
    template <class F, class CT>
    template <class S, layout_type L>
    inline auto xfunctorview<F, CT>::xrbegin(const S& shape) const noexcept -> const_reverse_shaped_xiterator<S, L>
    {
        return cxrbegin<S, L>(shape);
    }

    /**
     * Returns a constant iterator to the element following the last element
     * of the reversed expression.
     * @param shape the shape used for broadcasting
     * @tparam S type of the \c shape parameter.
     * @tparam L layout used for the traversal. Default value is \c DEFAULT_LAYOUT.
     */
    template <class F, class CT>
    template <class S, layout_type L>
    inline auto xfunctorview<F, CT>::xrend(const S& shape) const noexcept -> const_reverse_shaped_xiterator<S, L>
    {
        return cxrend<S, L>();
    }

    /**
     * Returns a constant iterator to the first element of the reversed expression.
     * The iteration is broadcasted to the specified shape.
     * @param shape the shape used for broadcasting
     * @tparam S type of the \c shape parameter.
     * @tparam L layout used for the traversal. Default value is \c DEFAULT_LAYOUT.
     */
    template <class F, class CT>
    template <class S, layout_type L>
    inline auto xfunctorview<F, CT>::cxrbegin(const S& shape) const noexcept -> const_reverse_shaped_xiterator<S, L>
    {
        return const_reverse_shaped_xiterator<S, L>(m_e.template cxrbegin<S, L>(), &m_functor);
    }

    /**
     * Returns a constant iterator to the element following the last element
     * of the reversed expression.
     * @param shape the shape used for broadcasting
     * @tparam S type of the \c shape parameter.
     * @tparam L layout used for the traversal. Default value is \c DEFAULT_LAYOUT.
     */
    template <class F, class CT>
    template <class S, layout_type L>
    inline auto xfunctorview<F, CT>::cxrend(const S& shape) const noexcept -> const_reverse_shaped_xiterator<S, L>
    {
        return const_reverse_shaped_xiterator<S, L>(m_e.template cxrend<S, L>(shape), &m_functor);
    }
    //@}

    /***************
     * stepper api *
     ***************/

    template <class F, class CT>
    template <class S>
    inline auto xfunctorview<F, CT>::stepper_begin(const S& shape) noexcept -> stepper
    {
        return stepper(m_e.stepper_begin(shape), &m_functor);
    }

    template <class F, class CT>
    template <class S>
    inline auto xfunctorview<F, CT>::stepper_end(const S& shape, layout_type l) noexcept -> stepper
    {
        return stepper(m_e.stepper_end(shape, l), &m_functor);
    }

    template <class F, class CT>
    template <class S>
    inline auto xfunctorview<F, CT>::stepper_begin(const S& shape) const noexcept -> const_stepper
    {
        const xexpression_type& const_m_e = m_e;
        return const_stepper(const_m_e.stepper_begin(shape), &m_functor);
    }

    template <class F, class CT>
    template <class S>
    inline auto xfunctorview<F, CT>::stepper_end(const S& shape, layout_type l) const noexcept -> const_stepper
    {
        const xexpression_type& const_m_e = m_e;
        return const_stepper(const_m_e.stepper_end(shape, l), &m_functor);
    }

    /************************************
     * xfunctor_iterator implementation *
     ************************************/

    template <class F, class IT>
    xfunctor_iterator<F, IT>::xfunctor_iterator(const IT& it, const functor_type* pf)
        : m_it(it), p_functor(pf)
    {
    }

    template <class F, class IT>
    auto xfunctor_iterator<F, IT>::operator++() -> self_type&
    {
        ++m_it;
        return *this;
    }

    template <class F, class IT>
    auto xfunctor_iterator<F, IT>::operator++(int) -> self_type
    {
        self_type tmp(*this);
        ++m_it;
        return tmp;
    }

    template <class F, class IT>
    auto xfunctor_iterator<F, IT>::operator*() const -> reference
    {
        return (*p_functor)(*m_it);
    }

    template <class F, class IT>
    auto xfunctor_iterator<F, IT>::operator->() const -> pointer
    {
        return &((*p_functor)(*m_it));
    }

    template <class F, class IT>
    auto xfunctor_iterator<F, IT>::equal(const xfunctor_iterator& rhs) const -> bool
    {
        return m_it == rhs.m_it;
    }

    template <class F, class IT>
    bool operator==(const xfunctor_iterator<F, IT>& lhs,
                    const xfunctor_iterator<F, IT>& rhs)
    {
        return lhs.equal(rhs);
    }

    template <class F, class IT>
    bool operator!=(const xfunctor_iterator<F, IT>& lhs,
                    const xfunctor_iterator<F, IT>& rhs)
    {
        return !lhs.equal(rhs);
    }

    /***********************************
     * xfunctor_stepper implementation *
     ***********************************/

    template <class F, class ST>
    xfunctor_stepper<F, ST>::xfunctor_stepper(const ST& stepper, const functor_type* pf)
        : m_stepper(stepper), p_functor(pf)
    {
    }

    template <class F, class ST>
    auto xfunctor_stepper<F, ST>::operator*() const -> reference
    {
        return (*p_functor)(*m_stepper);
    }

    template <class F, class ST>
    void xfunctor_stepper<F, ST>::step(size_type dim, size_type n)
    {
        m_stepper.step(dim, n);
    }

    template <class F, class ST>
    void xfunctor_stepper<F, ST>::step_back(size_type dim, size_type n)
    {
        m_stepper.step_back(dim, n);
    }

    template <class F, class ST>
    void xfunctor_stepper<F, ST>::reset(size_type dim)
    {
        m_stepper.reset(dim);
    }

    template <class F, class ST>
    void xfunctor_stepper<F, ST>::reset_back(size_type dim)
    {
        m_stepper.reset_back(dim);
    }

    template <class F, class ST>
    void xfunctor_stepper<F, ST>::to_begin()
    {
        m_stepper.to_begin();
    }

    template <class F, class ST>
    void xfunctor_stepper<F, ST>::to_end(layout_type l)
    {
        m_stepper.to_end(l);
    }

    template <class F, class ST>
    auto xfunctor_stepper<F, ST>::equal(const xfunctor_stepper& rhs) const -> bool
    {
        return m_stepper == rhs.m_stepper;
    }

    template <class F, class ST>
    bool operator==(const xfunctor_stepper<F, ST>& lhs,
                    const xfunctor_stepper<F, ST>& rhs)
    {
        return lhs.equal(rhs);
    }

    template <class F, class ST>
    bool operator!=(const xfunctor_stepper<F, ST>& lhs,
                    const xfunctor_stepper<F, ST>& rhs)
    {
        return !lhs.equal(rhs);
    }
}
#endif
