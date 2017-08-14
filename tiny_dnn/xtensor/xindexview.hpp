/***************************************************************************
* Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XINDEXVIEW_HPP
#define XINDEXVIEW_HPP

#include <algorithm>
#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

#include "xexpression.hpp"
#include "xiterable.hpp"
#include "xstrides.hpp"
#include "xutils.hpp"

namespace xt
{

    template <class CT, class I>
    class xindexview;

    template <class CT, class I>
    struct xcontainer_inner_types<xindexview<CT, I>>
    {
        using xexpression_type = std::decay_t<CT>;
        using temporary_type = xarray<typename xexpression_type::value_type, xexpression_type::static_layout>;
    };

    template <class CT, class I>
    struct xiterable_inner_types<xindexview<CT, I>>
    {
        using inner_shape_type = std::array<std::size_t, 1>;
        using const_stepper = xindexed_stepper<xindexview<CT, I>>;
        using stepper = xindexed_stepper<xindexview<CT, I>, false>;
        using const_iterator = xiterator<const_stepper, inner_shape_type*, DEFAULT_LAYOUT>;
        using iterator = xiterator<stepper, inner_shape_type*, DEFAULT_LAYOUT>;
        using const_reverse_iterator = std::reverse_iterator<const_iterator>;
        using reverse_iterator = std::reverse_iterator<iterator>;
    };

    /**************
     * xindexview *
     **************/

    /**
     * @class xindexview
     * @brief View of an xexpression from vector of indices.
     *
     * The xindexview class implements a flat (1D) view into a multidimensional
     * xexpression yielding the values at the indices of the index array.
     * xindexview is not meant to be used directly, but only with the \ref index_view
     * and \ref filter helper functions.
     *
     * @tparam CT the closure type of the \ref xexpression type underlying this view
     * @tparam I the index array type of the view
     *
     * @sa index_view, filter
     */
    template <class CT, class I>
    class xindexview : public xview_semantic<xindexview<CT, I>>,
                       public xexpression_iterable<xindexview<CT, I>>
    {
    public:

        using self_type = xindexview<CT, I>;
        using xexpression_type = std::decay_t<CT>;
        using semantic_base = xview_semantic<self_type>;

        using value_type = typename xexpression_type::value_type;
        using reference = typename xexpression_type::reference;
        using const_reference = typename xexpression_type::const_reference;
        using pointer = typename xexpression_type::pointer;
        using const_pointer = typename xexpression_type::const_pointer;
        using size_type = typename xexpression_type::size_type;
        using difference_type = typename xexpression_type::difference_type;

        using iterable_base = xexpression_iterable<self_type>;
        using inner_shape_type = typename iterable_base::inner_shape_type;
        using shape_type = inner_shape_type;
        using strides_type = shape_type;

        using indices_type = I;

        using stepper = typename iterable_base::stepper;
        using const_stepper = typename iterable_base::const_stepper;

        using temporary_type = typename xcontainer_inner_types<self_type>::temporary_type;
        using base_index_type = xindex_type_t<shape_type>;

        static constexpr layout_type static_layout = layout_type::dynamic;
        static constexpr bool contiguous_layout = false;

        template <class I2>
        xindexview(CT e, I2&& indices) noexcept;

        template <class E>
        self_type& operator=(const xexpression<E>& e);

        template <class E>
        disable_xexpression<E, self_type>& operator=(const E& e);

        size_type size() const noexcept;
        size_type dimension() const noexcept;
        const inner_shape_type& shape() const noexcept;
        layout_type layout() const noexcept;

        reference operator()();
        template <class... Args>
        reference operator()(std::size_t idx, Args... /*args*/);
        reference operator[](const xindex& index);
        reference operator[](size_type i);

        template <class It>
        reference element(It first, It last);

        const_reference operator()() const;
        template <class... Args>
        const_reference operator()(std::size_t idx, Args... /*args*/) const;
        const_reference operator[](const xindex& index) const;
        const_reference operator[](size_type i) const;

        template <class It>
        const_reference element(It first, It last) const;

        template <class O>
        bool broadcast_shape(O& shape) const;

        template <class O>
        bool is_trivial_broadcast(const O& /*strides*/) const noexcept;

        template <class ST>
        stepper stepper_begin(const ST& shape);
        template <class ST>
        stepper stepper_end(const ST& shape, layout_type);

        template <class ST>
        const_stepper stepper_begin(const ST& shape) const;
        template <class ST>
        const_stepper stepper_end(const ST& shape, layout_type) const;

    private:

        CT m_e;
        const indices_type m_indices;
        const inner_shape_type m_shape;

        void assign_temporary_impl(temporary_type&& tmp);

        friend class xview_semantic<xindexview<CT, I>>;
    };

    /***************
     * xfiltration *
     ***************/

    /**
     * @class xfiltration
     * @brief Filter of a xexpression for fast scalar assign.
     *
     * The xfiltration class implements a lazy filtration of a multidimentional
     * \ref xexpression, optimized for scalar and computed scalar assignments.
     * Actually, the \ref xfiltration class IS NOT an \ref xexpression and the
     * scalar and computed scalar assignments are the only method it provides.
     * The filtering condition is not evaluated until the filtration is assigned.
     *
     * xfiltration is not meant to be used directly, but only with the \ref filtration
     * helper function.
     *
     * @tparam ECT the closure type of the \ref xexpression type underlying this filtration
     * @tparam CCR the closure type of the filtering \ref xexpression type
     *
     * @sa filtration
     */
    template <class ECT, class CCT>
    class xfiltration
    {
    public:

        using self_type = xfiltration<ECT, CCT>;
        using xexpression_type = std::decay_t<ECT>;
        using const_reference = typename xexpression_type::const_reference;

        xfiltration(ECT e, CCT condition);

        template <class E>
        disable_xexpression<E, self_type&> operator=(const E&);

        template <class E>
        disable_xexpression<E, self_type&> operator+=(const E&);

        template <class E>
        disable_xexpression<E, self_type&> operator-=(const E&);

        template <class E>
        disable_xexpression<E, self_type&> operator*=(const E&);

        template <class E>
        disable_xexpression<E, self_type&> operator/=(const E&);

    private:

        template <class F>
        self_type& apply(F&& func);

        ECT m_e;
        CCT m_condition;
    };

    /*****************************
     * xindexview implementation *
     *****************************/

    /**
     * @name Constructor
     */
    //@{
    /**
     * Constructs an xindexview, selecting the indices specified by \a indices.
     * The resulting xexpression has a 1D shape with a length of n for n indices.
     * 
     * @param e the underlying xexpression for this view
     * @param indices the indices to select
     */
    template <class CT, class I>
    template <class I2>
    inline xindexview<CT, I>::xindexview(CT e, I2&& indices) noexcept
        : m_e(e), m_indices(std::forward<I2>(indices)), m_shape({m_indices.size()})
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
    template <class CT, class I>
    template <class E>
    inline auto xindexview<CT, I>::operator=(const xexpression<E>& e) -> self_type&
    {
        return semantic_base::operator=(e);
    }
    //@}

    template <class CT, class I>
    template <class E>
    inline auto xindexview<CT, I>::operator=(const E& e) -> disable_xexpression<E, self_type>&
    {
        std::fill(this->begin(), this->end(), e);
        return *this;
    }

    template <class CT, class I>
    inline void xindexview<CT, I>::assign_temporary_impl(temporary_type&& tmp)
    {
        std::copy(tmp.cbegin(), tmp.cend(), this->xbegin());
    }

    /**
     * @name Size and shape
     */
    //@{
    /**
     * Returns the size of the xindexview.
     */
    template <class CT, class I>
    inline auto xindexview<CT, I>::size() const noexcept -> size_type
    {
        return compute_size(shape());
    }

    /**
     * Returns the number of dimensions of the xindexview.
     */
    template <class CT, class I>
    inline auto xindexview<CT, I>::dimension() const noexcept -> size_type
    {
        return 1;
    }

    /**
     * Returns the shape of the xindexview.
     */
    template <class CT, class I>
    inline auto xindexview<CT, I>::shape() const noexcept -> const inner_shape_type&
    {
        return m_shape;
    }

    template <class CT, class I>
    inline layout_type xindexview<CT, I>::layout() const noexcept
    {
        return static_layout;
    }

    //@}

    /**
     * @name Data
     */
    template <class CT, class I>
    inline auto xindexview<CT, I>::operator()() -> reference
    {
        return m_e();
    }

    template <class CT, class I>
    inline auto xindexview<CT, I>::operator()() const -> const_reference
    {
        return m_e();
    }

    template <class CT, class I>
    template <class... Args>
    inline auto xindexview<CT, I>::operator()(std::size_t idx, Args... /*args*/) -> reference
    {
        return m_e[m_indices[idx]];
    }

    /**
     * Returns the element at the specified position in the xindexview. 
     * 
     * @param idx the position in the view
     */
    template <class CT, class I>
    template <class... Args>
    inline auto xindexview<CT, I>::operator()(std::size_t idx, Args... /*args*/) const -> const_reference
    {
        return m_e[m_indices[idx]];
    }

    template <class CT, class I>
    inline auto xindexview<CT, I>::operator[](const xindex& index) -> reference
    {
        return m_e[m_indices[index[0]]];
    }

    template <class CT, class I>
    inline auto xindexview<CT, I>::operator[](size_type i) -> reference
    {
        return operator()(i);
    }

    template <class CT, class I>
    inline auto xindexview<CT, I>::operator[](const xindex& index) const -> const_reference
    {
        return m_e[m_indices[index[0]]];
    }

    template <class CT, class I>
    inline auto xindexview<CT, I>::operator[](size_type i) const -> const_reference
    {
        return operator()(i);
    }

    /**
     * Returns a reference to the element at the specified position in the xindexview.
     * @param first iterator starting the sequence of indices
     * The number of indices in the sequence should be equal to or greater 1.
     */
    template <class CT, class I>
    template <class It>
    inline auto xindexview<CT, I>::element(It first, It /*last*/) -> reference
    {
        return m_e[m_indices[(*first)]];
    }

    template <class CT, class I>
    template <class It>
    inline auto xindexview<CT, I>::element(It first, It /*last*/) const -> const_reference
    {
        return m_e[m_indices[(*first)]];
    }
    //@}

    /**
     * @name Broadcasting
     */
    //@{
    /**
     * Broadcast the shape of the xindexview to the specified parameter.
     * @param shape the result shape
     * @return a boolean indicating whether the broadcasting is trivial
     */
    template <class CT, class I>
    template <class O>
    inline bool xindexview<CT, I>::broadcast_shape(O& shape) const
    {
        return xt::broadcast_shape(m_shape, shape);
    }

    /**
     * Compares the specified strides with those of the container to see whether
     * the broadcasting is trivial.
     * @return a boolean indicating whether the broadcasting is trivial
     */
    template <class CT, class I>
    template <class O>
    inline bool xindexview<CT, I>::is_trivial_broadcast(const O& /*strides*/) const noexcept
    {
        return false;
    }
    //@}

    /***************
     * stepper api *
     ***************/

    template <class CT, class I>
    template <class ST>
    inline auto xindexview<CT, I>::stepper_begin(const ST& shape) -> stepper
    {
        size_type offset = shape.size() - dimension();
        return stepper(this, offset);
    }

    template <class CT, class I>
    template <class ST>
    inline auto xindexview<CT, I>::stepper_end(const ST& shape, layout_type) -> stepper
    {
        size_type offset = shape.size() - dimension();
        return stepper(this, offset, true);
    }

    template <class CT, class I>
    template <class ST>
    inline auto xindexview<CT, I>::stepper_begin(const ST& shape) const -> const_stepper
    {
        size_type offset = shape.size() - dimension();
        return const_stepper(this, offset);
    }

    template <class CT, class I>
    template <class ST>
    inline auto xindexview<CT, I>::stepper_end(const ST& shape, layout_type) const -> const_stepper
    {
        size_type offset = shape.size() - dimension();
        return const_stepper(this, offset, true);
    }

    /******************************
     * xfiltration implementation *
     ******************************/

    /**
     * @name Constructor
     */
    //@{
    /**
     * Constructs a xfiltration on the given expression \c e, selecting
     * the elements matching the specified \c condition. 
     *
     * @param e the \ref xexpression to filter.
     * @param condition the filtering \ref xexpression to apply.
     */
    template <class ECT, class CCT>
    inline xfiltration<ECT, CCT>::xfiltration(ECT e, CCT condition)
        : m_e(e), m_condition(condition)
    {
    }
    //@}

    /**
     * @name Extended copy semantic
     */
    //@{
    /**
     * Assigns the scalar \c e to \c *this.
     * @param e the scalar to assign.
     * @return a reference to \ *this.
     */
    template <class ECT, class CCT>
    template <class E>
    inline auto xfiltration<ECT, CCT>::operator=(const E& e) -> disable_xexpression<E, self_type&>
    {
        return apply([this, &e](const_reference v, bool cond) { return cond ? e : v; });
    }
    //@}

    /**
     * @name Computed assignement
     */
    //@{
    /**
     * Adds the scalar \c e to \c *this.
     * @param e the scalar to add.
     * @return a reference to \c *this.
     */
    template <class ECT, class CCT>
    template <class E>
    inline auto xfiltration<ECT, CCT>::operator+=(const E& e) -> disable_xexpression<E, self_type&>
    {
        return apply([this, &e](const_reference v, bool cond) { return cond ? v + e : v; });
    }

    /**
     * Subtracts the scalar \c e from \c *this.
     * @param e the scalar to subtract.
     * @return a reference to \c *this.
     */
    template <class ECT, class CCT>
    template <class E>
    inline auto xfiltration<ECT, CCT>::operator-=(const E& e) -> disable_xexpression<E, self_type&>
    {
        return apply([this, &e](const_reference v, bool cond) { return cond ? v - e : v; });
    }

    /**
     * Multiplies \c *this with the scalar \c e.
     * @param e the scalar involved in the operation.
     * @return a reference to \c *this.
     */
    template <class ECT, class CCT>
    template <class E>
    inline auto xfiltration<ECT, CCT>::operator*=(const E& e) -> disable_xexpression<E, self_type&>
    {
        return apply([this, &e](const_reference v, bool cond) { return cond ? v * e : v; });
    }

    /**
     * Divides \c *this by the scalar \c e.
     * @param e the scalar involved in the operation.
     * @return a reference to \c *this.
     */
    template <class ECT, class CCT>
    template <class E>
    inline auto xfiltration<ECT, CCT>::operator/=(const E& e) -> disable_xexpression<E, self_type&>
    {
        return apply([this, &e](const_reference v, bool cond) { return cond ? v / e : v; });
    }

    template <class ECT, class CCT>
    template <class F>
    inline auto xfiltration<ECT, CCT>::apply(F&& func) -> self_type&
    {
        std::transform(m_e.cbegin(), m_e.cend(), m_condition.cbegin(), m_e.begin(), func);
        return *this;
    }

    /**
     * @brief creates an indexview from a container of indices.
     *        
     * Returns a 1D view with the elements at \a indices selected.
     *
     * @param e the underlying xexpression
     * @param indices the indices to select
     * 
     * \code{.cpp}
     * xarray<double> a = {{1,5,3}, {4,5,6}};
     * b = index_view(a, {{0, 0}, {1, 0}, {1, 1}});
     * std::cout << b << std::endl; // {1, 4, 5}
     * b += 100;
     * std::cout << a << std::endl; // {{101, 5, 3}, {104, 105, 6}}
     * \endcode
     */
    template <class E, class I>
    inline auto index_view(E&& e, I&& indices) noexcept
    {
        using view_type = xindexview<xclosure_t<E>, std::decay_t<I>>;
        return view_type(std::forward<E>(e), std::forward<I>(indices));
    }
#ifdef X_OLD_CLANG
    template <class E, class I>
    inline auto index_view(E&& e, std::initializer_list<std::initializer_list<I>> indices) noexcept
    {
        std::vector<xindex> idx;
        for (auto it = indices.begin(); it != indices.end(); ++it)
        {
            idx.emplace_back(xindex(it->begin(), it->end()));
        }
        using view_type = xindexview<xclosure_t<E>, std::vector<xindex>>;
        return view_type(std::forward<E>(e), std::move(idx));
    }
#else
    template <class E, std::size_t L>
    inline auto index_view(E&& e, const xindex (&indices)[L]) noexcept
    {
        using view_type = xindexview<xclosure_t<E>, std::array<xindex, L>>;
        return view_type(std::forward<E>(e), to_array(indices));
    }
#endif

    /**
     * @brief creates a view into \a e filtered by \a condition.
     *        
     * Returns a 1D view with the elements selected where \a condition evaluates to \em true.
     * This is equivalent to \verbatim{index_view(e, where(condition));}\endverbatim
     * The returned view is not optimal if you just want to assign a scalar to the filtered
     * elements. In that case, you should consider using the \ref filtration function
     * instead.
     *
     * @param e the underlying xexpression
     * @param condition xexpression with shape of \a e which selects indices
     *
     * \code{.cpp}
     * xarray<double> a = {{1,5,3}, {4,5,6}};
     * b = filter(a, a >= 5);
     * std::cout << b << std::endl; // {5, 5, 6}
     * \endcode
     *
     * \sa filtration
     */
    template <class E, class O>
    inline auto filter(E&& e, O&& condition) noexcept
    {
        auto indices = where(std::forward<O>(condition));
        using view_type = xindexview<xclosure_t<E>, decltype(indices)>;
        return view_type(std::forward<E>(e), std::move(indices));
    }

    /**
     * @brief creates a filtration of \c e filtered by \a condition.
     *
     * Returns a lazy filtration optimized for scalar assignment.
     * Actually, scalar assignment and computed scalar assignments
     * are the only available methods of the filtration, the filtration
     * IS NOT an \ref xexpression.
     *
     * @param e the \ref xexpression to filter
     * @param condition the filtering \ref xexpression
     *
     * \code{.cpp}
     * xarray<double> a = {{1,5,3}, {4,5,6}};
     * filtration(a, a >= 5) += 2;
     * std::cout << a << std::endl; // {{1, 7, 3}, {4, 7, 8}}
     * \endcode
     */
    template <class E, class C>
    inline auto filtration(E&& e, C&& condition) noexcept
    {
        using filtration_type = xfiltration<xclosure_t<E>, xclosure_t<C>>;
        return filtration_type(std::forward<E>(e), std::forward<C>(condition));
    }
}

#endif
