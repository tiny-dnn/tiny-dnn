/***************************************************************************
* Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XITERABLE_HPP
#define XITERABLE_HPP

#include "xiterator.hpp"

namespace xt
{

    /*******************
     * xconst_iterable *
     *******************/

    template <class D>
    struct xiterable_inner_types;

#define DL DEFAULT_LAYOUT

    /**
     * @class xconst_iterable
     * @brief Base class for multidimensional iterable constant expressions
     *
     * The xconst_iterable class defines the interface for multidimensional
     * constant expressions that can be iterated.
     *
     * @tparam D The derived type, i.e. the inheriting class for which xconst_iterable
     *           provides the interface.
     */
    template <class D>
    class xconst_iterable
    {
    public:

        using derived_type = D;

        using iterable_types = xiterable_inner_types<D>;
        using inner_shape_type = typename iterable_types::inner_shape_type;

        using stepper = typename iterable_types::stepper;
        using const_stepper = typename iterable_types::const_stepper;

        using iterator = typename iterable_types::iterator;
        using const_iterator = typename iterable_types::const_iterator;

        template <layout_type L>
        using broadcast_iterator = xiterator<stepper, inner_shape_type*, L>;
        template <layout_type L>
        using const_broadcast_iterator = xiterator<const_stepper, inner_shape_type*, L>;

        template <class S, layout_type L>
        using shaped_xiterator = xiterator<stepper, S, L>;
        template <class S, layout_type L>
        using const_shaped_xiterator = xiterator<const_stepper, S, L>;

        using reverse_iterator = typename iterable_types::reverse_iterator;
        using const_reverse_iterator = typename iterable_types::const_reverse_iterator;

        template <layout_type L>
        using reverse_broadcast_iterator = std::reverse_iterator<broadcast_iterator<L>>;
        template <layout_type L>
        using const_reverse_broadcast_iterator = std::reverse_iterator<const_broadcast_iterator<L>>;

        template <class S, layout_type L>
        using reverse_shaped_xiterator = std::reverse_iterator<shaped_xiterator<S, L>>;
        template <class S, layout_type L>
        using const_reverse_shaped_xiterator = std::reverse_iterator<const_shaped_xiterator<S, L>>;

        template <layout_type L = DL>
        const_broadcast_iterator<L> xbegin() const noexcept;
        template <layout_type L = DL>
        const_broadcast_iterator<L> xend() const noexcept;
        template <layout_type L = DL>
        const_broadcast_iterator<L> cxbegin() const noexcept;
        template <layout_type L = DL>
        const_broadcast_iterator<L> cxend() const noexcept;

        template <class S, layout_type L = DL>
        const_shaped_xiterator<S, L> xbegin(const S& shape) const noexcept;
        template <class S, layout_type L = DL>
        const_shaped_xiterator<S, L> xend(const S& shape) const noexcept;
        template <class S, layout_type L = DL>
        const_shaped_xiterator<S, L> cxbegin(const S& shape) const noexcept;
        template <class S, layout_type L = DL>
        const_shaped_xiterator<S, L> cxend(const S& shape) const noexcept;

        template <layout_type L = DL>
        const_reverse_broadcast_iterator<L> xrbegin() const noexcept;
        template <layout_type L = DL>
        const_reverse_broadcast_iterator<L> xrend() const noexcept;
        template <layout_type L = DL>
        const_reverse_broadcast_iterator<L> cxrbegin() const noexcept;
        template <layout_type L = DL>
        const_reverse_broadcast_iterator<L> cxrend() const noexcept;

        template <class S, layout_type L = DL>
        const_reverse_shaped_xiterator<S, L> xrbegin(const S& shape) const noexcept;
        template <class S, layout_type L = DL>
        const_reverse_shaped_xiterator<S, L> xrend(const S& shape) const noexcept;
        template <class S, layout_type L = DL>
        const_reverse_shaped_xiterator<S, L> cxrbegin(const S& shape) const noexcept;
        template <class S, layout_type L = DL>
        const_reverse_shaped_xiterator<S, L> cxrend(const S& shape) const noexcept;

    protected:

        const inner_shape_type& get_shape() const;

    private:

        template <layout_type L>
        const_broadcast_iterator<L> get_cxbegin(bool reverse) const noexcept;
        template <layout_type L>
        const_broadcast_iterator<L> get_cxend(bool reverse) const noexcept;

        template <class S, layout_type L>
        const_shaped_xiterator<S, L> get_cxbegin(const S& shape, bool reverse) const noexcept;
        template <class S, layout_type L>
        const_shaped_xiterator<S, L> get_cxend(const S& shape, bool reverse) const noexcept;

        template <class S>
        const_stepper get_stepper_begin(const S& shape) const noexcept;
        template <class S>
        const_stepper get_stepper_end(const S& shape, layout_type l) const noexcept;

        const derived_type& derived_cast() const;
    };

    /*************
     * xiterable *
     *************/

    /**
     * @class xiterable
     * @brief Base class for multidimensional iterable expressions
     *
     * The xiterable class defines the interface for multidimensional
     * expressions that can be iterated.
     *
     * @tparam D The derived type, i.e. the inheriting class for which xiterable
     *           provides the interface.
     */
    template <class D>
    class xiterable : public xconst_iterable<D>
    {
    public:

        using derived_type = D;

        using base_type = xconst_iterable<D>;
        using inner_shape_type = typename base_type::inner_shape_type;

        using stepper = typename base_type::stepper;
        using const_stepper = typename base_type::const_stepper;

        using iterator = typename base_type::iterator;
        using const_iterator = typename base_type::const_iterator;

        template <layout_type L>
        using broadcast_iterator = typename base_type::template broadcast_iterator<L>;
        template <layout_type L>
        using const_broadcast_iterator = typename base_type::template const_broadcast_iterator<L>;

        template <class S, layout_type L>
        using shaped_xiterator = typename base_type::template shaped_xiterator<S, L>;
        template <class S, layout_type L>
        using const_shaped_xiterator = typename base_type::template const_shaped_xiterator<S, L>;

        using reverse_iterator = typename base_type::reverse_iterator;
        using const_reverse_iterator = typename base_type::const_reverse_iterator;

        template <layout_type L>
        using reverse_broadcast_iterator = typename base_type::template reverse_broadcast_iterator<L>;
        template <layout_type L>
        using const_reverse_broadcast_iterator = typename base_type::template const_reverse_broadcast_iterator<L>;

        template <class S, layout_type L>
        using reverse_shaped_xiterator = typename base_type::template reverse_shaped_xiterator<S, L>;
        template <class S, layout_type L>
        using const_reverse_shaped_xiterator = typename base_type::template const_reverse_shaped_xiterator<S, L>;

        template <layout_type L = DL>
        broadcast_iterator<L> xbegin() noexcept;
        template <layout_type L = DL>
        broadcast_iterator<L> xend() noexcept;

        using base_type::xbegin;
        using base_type::xend;

        template <class S, layout_type L = DL>
        shaped_xiterator<S, L> xbegin(const S& shape) noexcept;
        template <class S, layout_type L = DL>
        shaped_xiterator<S, L> xend(const S& shape) noexcept;

        template <layout_type L = DL>
        reverse_broadcast_iterator<L> xrbegin() noexcept;
        template <layout_type L = DL>
        reverse_broadcast_iterator<L> xrend() noexcept;

        using base_type::xrbegin;
        using base_type::xrend;

        template <class S, layout_type L = DL>
        reverse_shaped_xiterator<S, L> xrbegin(const S& shape) noexcept;
        template <class S, layout_type L = DL>
        reverse_shaped_xiterator<S, L> xrend(const S& shape) noexcept;

    private:

        template <layout_type L>
        broadcast_iterator<L> get_xbegin(bool reverse) noexcept;
        template <layout_type L>
        broadcast_iterator<L> get_xend(bool reverse) noexcept;

        template <class S, layout_type L>
        shaped_xiterator<S, L> get_xbegin(const S& shape, bool reverse) noexcept;
        template <class S, layout_type L>
        shaped_xiterator<S, L> get_xend(const S& shape, bool reverse) noexcept;

        template <class S>
        stepper get_stepper_begin(const S& shape) noexcept;
        template <class S>
        stepper get_stepper_end(const S& shape, layout_type l) noexcept;

        derived_type& derived_cast();
    };

#undef DL

    /******************************
     * xexpression_const_iterable *
     ******************************/

    /**
     * @class xexpression_const_iterable
     * @brief Base class for multidimensional iterable constant expressions
     *        that don't store any data
     *
     * The xexpression_const_iterable class defines the interface for multidimensional
     * constant expressions that don't store any data and that can be iterated.
     *
     * @tparam D The derived type, i.e. the inheriting class for which xexpression_const_iterable
     *           provides the interface.
     */
    template <class D>
    class xexpression_const_iterable : public xconst_iterable<D>
    {
    public:

        using base_type = xconst_iterable<D>;
        using inner_shape_type = typename base_type::inner_shape_type;

        using stepper = typename base_type::stepper;
        using const_stepper = typename base_type::const_stepper;

        using iterator = typename base_type::iterator;
        using const_iterator = typename base_type::const_iterator;

        template <layout_type L>
        using broadcast_iterator = typename base_type::template broadcast_iterator<L>;
        template <layout_type L>
        using const_broadcast_iterator = typename base_type::template const_broadcast_iterator<L>;

        template <class S, layout_type L>
        using shaped_xiterator = typename base_type::template shaped_xiterator<S, L>;
        template <class S, layout_type L>
        using const_shaped_xiterator = typename base_type::template const_shaped_xiterator<S, L>;

        using reverse_iterator = typename base_type::reverse_iterator;
        using const_reverse_iterator = typename base_type::const_reverse_iterator;

        template <layout_type L>
        using reverse_broadcast_iterator = typename base_type::template reverse_broadcast_iterator<L>;
        template <layout_type L>
        using const_reverse_broadcast_iterator = typename base_type::template const_reverse_broadcast_iterator<L>;

        template <class S, layout_type L>
        using reverse_shaped_xiterator = typename base_type::template reverse_shaped_xiterator<S, L>;
        template <class S, layout_type L>
        using const_reverse_shaped_xiterator = typename base_type::template const_reverse_shaped_xiterator<S, L>;

        const_iterator begin() const noexcept;
        const_iterator end() const noexcept;
        const_iterator cbegin() const noexcept;
        const_iterator cend() const noexcept;

        const_reverse_iterator rbegin() const noexcept;
        const_reverse_iterator rend() const noexcept;
        const_reverse_iterator crbegin() const noexcept;
        const_reverse_iterator crend() const noexcept;
    };

    /************************
     * xexpression_iterable *
     ************************/

    /**
     * @class xexpression_iterable
     * @brief Base class for multidimensional iterable expressions
     *        that don't store any data
     *
     * The xexpression_iterable class defines the interface for multidimensional
     * expressions that don't store any data and that can be iterated.
     *
     * @tparam D The derived type, i.e.the inheriting class for which xexpression_iterable
     *           provides the interface.
     */
    template <class D>
    class xexpression_iterable : public xiterable<D>
    {
    public:

        using base_type = xiterable<D>;
        using inner_shape_type = typename base_type::inner_shape_type;

        using stepper = typename base_type::stepper;
        using const_stepper = typename base_type::const_stepper;

        using iterator = typename base_type::iterator;
        using const_iterator = typename base_type::const_iterator;

        template <layout_type L>
        using broadcast_iterator = typename base_type::template broadcast_iterator<L>;
        template <layout_type L>
        using const_broadcast_iterator = typename base_type::template const_broadcast_iterator<L>;

        template <class S, layout_type L>
        using shaped_xiterator = typename base_type::template shaped_xiterator<S, L>;
        template <class S, layout_type L>
        using const_shaped_xiterator = typename base_type::template const_shaped_xiterator<S, L>;

        using reverse_iterator = typename base_type::reverse_iterator;
        using const_reverse_iterator = typename base_type::const_reverse_iterator;

        template <layout_type L>
        using reverse_broadcast_iterator = typename base_type::template reverse_broadcast_iterator<L>;
        template <layout_type L>
        using const_reverse_broadcast_iterator = typename base_type::template const_reverse_broadcast_iterator<L>;

        template <class S, layout_type L>
        using reverse_shaped_xiterator = typename base_type::template reverse_shaped_xiterator<S, L>;
        template <class S, layout_type L>
        using const_reverse_shaped_xiterator = typename base_type::template const_reverse_shaped_xiterator<S, L>;

        iterator begin() noexcept;
        iterator end() noexcept;

        const_iterator begin() const noexcept;
        const_iterator end() const noexcept;
        const_iterator cbegin() const noexcept;
        const_iterator cend() const noexcept;

        reverse_iterator rbegin() noexcept;
        reverse_iterator rend() noexcept;

        const_reverse_iterator rbegin() const noexcept;
        const_reverse_iterator rend() const noexcept;
        const_reverse_iterator crbegin() const noexcept;
        const_reverse_iterator crend() const noexcept;
    };

    /**********************************
     * xconst_iterable implementation *
     **********************************/

    /**
     * @name Constant broadcast iterators
     */
    //@{
    /**
     * Returns a constant iterator to the first element of the expression.
     * @tparam L layout used for the traversal. Default value is \c DEFAULT_LAYOUT.
     */
    template <class D>
    template <layout_type L>
    inline auto xconst_iterable<D>::xbegin() const noexcept -> const_broadcast_iterator<L>
    {
        return cxbegin<L>();
    }

    /**
     * Returns a constant iterator to the element following the last element
     * of the expression.
     * @tparam L layout used for the traversal. Default value is \c DEFAULT_LAYOUT.
     */
    template <class D>
    template <layout_type L>
    inline auto xconst_iterable<D>::xend() const noexcept -> const_broadcast_iterator<L>
    {
        return cxend<L>();
    }

    /**
     * Returns a constant iterator to the first element of the expression.
     * @tparam L layout used for the traversal. Default value is \c DEFAULT_LAYOUT.
     */
    template <class D>
    template <layout_type L>
    inline auto xconst_iterable<D>::cxbegin() const noexcept -> const_broadcast_iterator<L>
    {
        return get_cxbegin<L>(false);
    }

    /**
     * Returns a constant iterator to the element following the last element
     * of the expression.
     * @tparam L layout used for the traversal. Default value is \c DEFAULT_LAYOUT.
     */
    template <class D>
    template <layout_type L>
    inline auto xconst_iterable<D>::cxend() const noexcept -> const_broadcast_iterator<L>
    {
        return get_cxend<L>(false);
    }

    /**
     * Returns a constant iterator to the first element of the expression. The
     * iteration is broadcasted to the specified shape.
     * @param shape the shape used for broadcasting
     * @tparam S type of the \c shape parameter.
     * @tparam L layout used for the traversal. Default value is \c DEFAULT_LAYOUT.
     */
    template <class D>
    template <class S, layout_type L>
    inline auto xconst_iterable<D>::xbegin(const S& shape) const noexcept -> const_shaped_xiterator<S, L>
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
    template <class D>
    template <class S, layout_type L>
    inline auto xconst_iterable<D>::xend(const S& shape) const noexcept -> const_shaped_xiterator<S, L>
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
    template <class D>
    template <class S, layout_type L>
    inline auto xconst_iterable<D>::cxbegin(const S& shape) const noexcept -> const_shaped_xiterator<S, L>
    {
        return get_cxbegin<S, L>(shape, false);
    }

    /**
     * Returns a constant iterator to the element following the last element of the
     * expression. The iteration is broadcasted to the specified shape.
     * @param shape the shape used for broadcasting
     * @tparam S type of the \c shape parameter.
     * @tparam L layout used for the traversal. Default value is \c DEFAULT_LAYOUT.
     */
    template <class D>
    template <class S, layout_type L>
    inline auto xconst_iterable<D>::cxend(const S& shape) const noexcept -> const_shaped_xiterator<S, L>
    {
        return get_cxend<S, L>(shape, false);
    }
    //@}

    /**
     * @name Constant reverse broadcast iterators
     */
    //@{
    /**
     * Returns a constant iterator to the first element of the reversed expression.
     * @tparam L layout used for the traversal. Default value is \c DEFAULT_LAYOUT.
     */
    template <class D>
    template <layout_type L>
    inline auto xconst_iterable<D>::xrbegin() const noexcept -> const_reverse_broadcast_iterator<L>
    {
        return cxrbegin<L>();
    }

    /**
     * Returns a constant iterator to the element following the last element
     * of the reversed expression.
     * @tparam L layout used for the traversal. Default value is \c DEFAULT_LAYOUT.
     */
    template <class D>
    template <layout_type L>
    inline auto xconst_iterable<D>::xrend() const noexcept -> const_reverse_broadcast_iterator<L>
    {
        return cxrend<L>();
    }

    /**
     * Returns a constant iterator to the first element of the reversed expression.
     * @tparam L layout used for the traversal. Default value is \c DEFAULT_LAYOUT.
     */
    template <class D>
    template <layout_type L>
    inline auto xconst_iterable<D>::cxrbegin() const noexcept -> const_reverse_broadcast_iterator<L>
    {
        return const_reverse_broadcast_iterator<L>(get_cxend<L>(true));
    }

    /**
     * Returns a constant iterator to the element following the last element
     * of the reversed expression.
     * @tparam L layout used for the traversal. Default value is \c DEFAULT_LAYOUT.
     */
    template <class D>
    template <layout_type L>
    inline auto xconst_iterable<D>::cxrend() const noexcept -> const_reverse_broadcast_iterator<L>
    {
        return const_reverse_broadcast_iterator<L>(get_cxbegin<L>(true));
    }

    /**
     * Returns a constant iterator to the first element of the reversed expression.
     * The iteration is broadcasted to the specified shape.
     * @param shape the shape used for broadcasting
     * @tparam S type of the \c shape parameter.
     * @tparam L layout used for the traversal. Default value is \c DEFAULT_LAYOUT.
     */
    template <class D>
    template <class S, layout_type L>
    inline auto xconst_iterable<D>::xrbegin(const S& shape) const noexcept -> const_reverse_shaped_xiterator<S, L>
    {
        return cxrbegin<S, L>(shape);
    }

    /**
     * Returns a constant iterator to the element following the last element of the
     * reversed expression. The iteration is broadcasted to the specified shape.
     * @param shape the shape used for broadcasting
     * @tparam S type of the \c shape parameter.
     * @tparam L layout used for the traversal. Default value is \c DEFAULT_LAYOUT.
     */
    template <class D>
    template <class S, layout_type L>
    inline auto xconst_iterable<D>::xrend(const S& shape) const noexcept -> const_reverse_shaped_xiterator<S, L>
    {
        return cxrend<S, L>(shape);
    }

    /**
     * Returns a constant iterator to the first element of the reversed expression.
     * The iteration is broadcasted to the specified shape.
     * @param shape the shape used for broadcasting
     * @tparam S type of the \c shape parameter.
     * @tparam L layout used for the traversal. Default value is \c DEFAULT_LAYOUT.
     */
    template <class D>
    template <class S, layout_type L>
    inline auto xconst_iterable<D>::cxrbegin(const S& shape) const noexcept -> const_reverse_shaped_xiterator<S, L>
    {
        return const_reverse_shaped_xiterator<S, L>(get_cxend<S, L>(shape, true));
    }

    /**
     * Returns a constant iterator to the element following the last element of the
     * reversed expression. The iteration is broadcasted to the specified shape.
     * @param shape the shape used for broadcasting
     * @tparam S type of the \c shape parameter.
     * @tparam L layout used for the traversal. Default value is \c DEFAULT_LAYOUT.
     */
    template <class D>
    template <class S, layout_type L>
    inline auto xconst_iterable<D>::cxrend(const S& shape) const noexcept -> const_reverse_shaped_xiterator<S, L>
    {
        return const_reverse_shaped_xiterator<S, L>(get_cxbegin<S, L>(shape, true));
    }
    //@}

    template <class D>
    template <layout_type L>
    inline auto xconst_iterable<D>::get_cxbegin(bool reverse) const noexcept -> const_broadcast_iterator<L>
    {
        return const_broadcast_iterator<L>(get_stepper_begin(get_shape()), &get_shape(), reverse);
    }

    template <class D>
    template <layout_type L>
    inline auto xconst_iterable<D>::get_cxend(bool reverse) const noexcept -> const_broadcast_iterator<L>
    {
        return const_broadcast_iterator<L>(get_stepper_end(get_shape(), L), &get_shape(), reverse);
    }

    template <class D>
    template <class S, layout_type L>
    inline auto xconst_iterable<D>::get_cxbegin(const S& shape, bool reverse) const noexcept -> const_shaped_xiterator<S, L>
    {
        return const_shaped_xiterator<S, L>(get_stepper_begin(shape), shape, reverse);
    }

    template <class D>
    template <class S, layout_type L>
    inline auto xconst_iterable<D>::get_cxend(const S& shape, bool reverse) const noexcept -> const_shaped_xiterator<S, L>
    {
        return const_shaped_xiterator<S, L>(get_stepper_end(shape, L), shape, reverse);
    }

    template <class D>
    template <class S>
    inline auto xconst_iterable<D>::get_stepper_begin(const S& shape) const noexcept -> const_stepper
    {
        return derived_cast().stepper_begin(shape);
    }

    template <class D>
    template <class S>
    inline auto xconst_iterable<D>::get_stepper_end(const S& shape, layout_type l) const noexcept -> const_stepper
    {
        return derived_cast().stepper_end(shape, l);
    }

    template <class D>
    inline auto xconst_iterable<D>::get_shape() const -> const inner_shape_type&
    {
        return derived_cast().shape();
    }

    template <class D>
    inline auto xconst_iterable<D>::derived_cast() const -> const derived_type&
    {
        return *static_cast<const derived_type*>(this);
    }

    /****************************
     * xiterable implementation *
     ****************************/

    /**
     * @name Broadcast iterators
     */
    //@{
    /**
     * Returns an iterator to the first element of the expression.
     * @tparam L layout used for the traversal. Default value is \c DEFAULT_LAYOUT.
     */
    template <class D>
    template <layout_type L>
    inline auto xiterable<D>::xbegin() noexcept -> broadcast_iterator<L>
    {
        return get_xbegin<L>(false);
    }

    /**
     * Returns an iterator to the element following the last element
     * of the expression.
     * @tparam L layout used for the traversal. Default value is \c DEFAULT_LAYOUT.
     */
    template <class D>
    template <layout_type L>
    inline auto xiterable<D>::xend() noexcept -> broadcast_iterator<L>
    {
        return get_xend<L>(false);
    }

    /**
     * Returns an iterator to the first element of the expression. The
     * iteration is broadcasted to the specified shape.
     * @param shape the shape used for broadcasting
     * @tparam S type of the \c shape parameter.
     * @tparam L layout used for the traversal. Default value is \c DEFAULT_LAYOUT.
     */
    template <class D>
    template <class S, layout_type L>
    inline auto xiterable<D>::xbegin(const S& shape) noexcept -> shaped_xiterator<S, L>
    {
        return get_xbegin<S, L>(shape, false);
    }

    /**
     * Returns an iterator to the element following the last element of the
     * expression. The iteration is broadcasted to the specified shape.
     * @param shape the shape used for broadcasting
     * @tparam S type of the \c shape parameter.
     * @tparam L layout used for the traversal. Default value is \c DEFAULT_LAYOUT.
     */
    template <class D>
    template <class S, layout_type L>
    inline auto xiterable<D>::xend(const S& shape) noexcept -> shaped_xiterator<S, L>
    {
        return get_xend<S, L>(shape, false);
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
    template <class D>
    template <layout_type L>
    inline auto xiterable<D>::xrbegin() noexcept -> reverse_broadcast_iterator<L>
    {
        return reverse_broadcast_iterator<L>(get_xend<L>(true));
    }

    /**
     * Returns an iterator to the element following the last element
     * of the reversed expression.
     * @tparam L layout used for the traversal. Default value is \c DEFAULT_LAYOUT.
     */
    template <class D>
    template <layout_type L>
    inline auto xiterable<D>::xrend() noexcept -> reverse_broadcast_iterator<L>
    {
        return reverse_broadcast_iterator<L>(get_xbegin<L>(true));
    }

    /**
     * Returns an iterator to the first element of the reversed expression. The
     * iteration is broadcasted to the specified shape.
     * @param shape the shape used for broadcasting
     * @tparam S type of the \c shape parameter.
     * @tparam L layout used for the traversal. Default value is \c DEFAULT_LAYOUT.
     */
    template <class D>
    template <class S, layout_type L>
    inline auto xiterable<D>::xrbegin(const S& shape) noexcept -> reverse_shaped_xiterator<S, L>
    {
        return reverse_shaped_xiterator<S, L>(get_xend<S, L>(shape, true));
    }

    /**
     * Returns an iterator to the element following the last element of the
     * reversed expression. The iteration is broadcasted to the specified shape.
     * @param shape the shape used for broadcasting
     * @tparam S type of the \c shape parameter.
     * @tparam L layout used for the traversal. Default value is \c DEFAULT_LAYOUT.
     */
    template <class D>
    template <class S, layout_type L>
    inline auto xiterable<D>::xrend(const S& shape) noexcept -> reverse_shaped_xiterator<S, L>
    {
        return reverse_shaped_xiterator<S, L>(get_xbegin<S, L>(shape, true));
    }
    //@}

    template <class D>
    template <layout_type L>
    inline auto xiterable<D>::get_xbegin(bool reverse) noexcept -> broadcast_iterator<L>
    {
        return broadcast_iterator<L>(get_stepper_begin(this->get_shape()), &(this->get_shape()), reverse);
    }

    template <class D>
    template <layout_type L>
    inline auto xiterable<D>::get_xend(bool reverse) noexcept -> broadcast_iterator<L>
    {
        return broadcast_iterator<L>(get_stepper_end(this->get_shape(), L), &(this->get_shape()), reverse);
    }

    template <class D>
    template <class S, layout_type L>
    inline auto xiterable<D>::get_xbegin(const S& shape, bool reverse) noexcept -> shaped_xiterator<S, L>
    {
        return shaped_xiterator<S, L>(get_stepper_begin(shape), shape, reverse);
    }

    template <class D>
    template <class S, layout_type L>
    inline auto xiterable<D>::get_xend(const S& shape, bool reverse) noexcept -> shaped_xiterator<S, L>
    {
        return shaped_xiterator<S, L>(get_stepper_end(shape, L), shape, reverse);
    }

    template <class D>
    template <class S>
    inline auto xiterable<D>::get_stepper_begin(const S& shape) noexcept -> stepper
    {
        return derived_cast().stepper_begin(shape);
    }

    template <class D>
    template <class S>
    inline auto xiterable<D>::get_stepper_end(const S& shape, layout_type l) noexcept -> stepper
    {
        return derived_cast().stepper_end(shape, l);
    }

    template <class D>
    inline auto xiterable<D>::derived_cast() -> derived_type&
    {
        return *static_cast<derived_type*>(this);
    }

    /*********************************************
     * xexpression_const_iterable implementation *
     *********************************************/

    /**
     * @name Constant Iterators
     */
    //@{
    /**
     * Returns a constant iterator to the first element of the expression.
     */
    template <class D>
    inline auto xexpression_const_iterable<D>::begin() const noexcept -> const_iterator
    {
        return this->cxbegin();
    }

    /**
     * Returns a constant iterator to the element following the last element
     * of the expression.
     */
    template <class D>
    inline auto xexpression_const_iterable<D>::end() const noexcept -> const_iterator
    {
        return this->cxend();
    }

    /**
     * Returns a constant iterator to the first element of the expression.
     */
    template <class D>
    inline auto xexpression_const_iterable<D>::cbegin() const noexcept -> const_iterator
    {
        return this->cxbegin();
    }

    /**
     * Returns a constant iterator to the element following the last element
     * of the expression.
     */
    template <class D>
    inline auto xexpression_const_iterable<D>::cend() const noexcept -> const_iterator
    {
        return this->cxend();
    }
    //@}

    /**
     * @name Constant Reverse Iterators
     */
    //@{
    /**
     * Returns a constant iterator to the first element of the reversed expression.
     */
    template <class D>
    inline auto xexpression_const_iterable<D>::rbegin() const noexcept -> const_reverse_iterator
    {
        return this->cxrbegin();
    }

    /**
     * Returns a constant iterator to the element following the last element
     * of the reversed expression.
     */
    template <class D>
    inline auto xexpression_const_iterable<D>::rend() const noexcept -> const_reverse_iterator
    {
        return this->cxrend();
    }

    /**
     * Returns a constant iterator to the first element of the reversed expression.
     */
    template <class D>
    inline auto xexpression_const_iterable<D>::crbegin() const noexcept -> const_reverse_iterator
    {
        return this->cxrbegin();
    }

    /**
     * Returns a constant iterator to the element following the last element
     * of the reversed expression.
     */
    template <class D>
    inline auto xexpression_const_iterable<D>::crend() const noexcept -> const_reverse_iterator
    {
        return this->cxrend();
    }
    //@}

    /***************************************
     * xexpression_iterable implementation *
     ***************************************/

    /**
     * @name Iterators
     */
    //@{
    /**
     * Returns an iterator to the first element of the expression.
     */
    template <class D>
    inline auto xexpression_iterable<D>::begin() noexcept -> iterator
    {
        return this->xbegin();
    }

    /**
     * Returns an iterator to the element following the last element of
     * the expression.
     */
    template <class D>
    inline auto xexpression_iterable<D>::end() noexcept -> iterator
    {
        return this->xend();
    }
    //@}

    /**
     * @name Constant Iterators
     */
    /**
     * Returns a constant iterator to the first element of the expression.
     */
    template <class D>
    inline auto xexpression_iterable<D>::begin() const noexcept -> const_iterator
    {
        return this->cxbegin();
    }

    /**
     * Returns a constant iterator to the element following the last element
     * of the expression.
     */
    template <class D>
    inline auto xexpression_iterable<D>::end() const noexcept -> const_iterator
    {
        return this->cxend();
    }

    /**
     * Returns a constant iterator to the first element of the expression.
     */
    template <class D>
    inline auto xexpression_iterable<D>::cbegin() const noexcept -> const_iterator
    {
        return this->cxbegin();
    }

    /**
     * Returns a constant iterator to the element following the last element
     * of the expression.
     */
    template <class D>
    inline auto xexpression_iterable<D>::cend() const noexcept -> const_iterator
    {
        return this->cxend();
    }
    //@}

    /**
     * @name Reverse Iterators
     */
    //@{
    /**
     * Returns an iterator to the first element of the reversed expression.
     */
    template <class D>
    inline auto xexpression_iterable<D>::rbegin() noexcept -> reverse_iterator
    {
        return this->xrbegin();
    }

    /**
     * Returns an iterator to the element following the last element
     * of the reversed expression.
     */
    template <class D>
    inline auto xexpression_iterable<D>::rend() noexcept -> reverse_iterator
    {
        return this->xrend();
    }
    //@}

    /**
     * @name Constant Reverse Iterators
     */
    //@{
    /**
     * Returns a constant iterator to the first element of the reversed expression.
     */
    template <class D>
    inline auto xexpression_iterable<D>::rbegin() const noexcept -> const_reverse_iterator
    {
        return this->cxrbegin();
    }

    /**
     * Returns a constant iterator to the element following the last element
     * of the reversed expression.
     */
    template <class D>
    inline auto xexpression_iterable<D>::rend() const noexcept -> const_reverse_iterator
    {
        return this->cxrend();
    }

    /**
     * Returns a constant iterator to the first element of the reversed expression.
     */
    template <class D>
    inline auto xexpression_iterable<D>::crbegin() const noexcept -> const_reverse_iterator
    {
        return this->cxrbegin();
    }

    /**
     * Returns a constant iterator to the element following the last element
     * of the reversed expression.
     */
    template <class D>
    inline auto xexpression_iterable<D>::crend() const noexcept -> const_reverse_iterator
    {
        return this->cxrend();
    }
    //@}
}

#endif
