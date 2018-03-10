/***************************************************************************
* Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XOPTIONAL_ASSEMBLY_BASE_HPP
#define XOPTIONAL_ASSEMBLY_BASE_HPP

#include "xiterable.hpp"
#include "xtensor_forward.hpp"

namespace xt
{
    template <class D, bool is_const>
    class xoptional_assembly_stepper;

    template <class D, bool is_const>
    class xoptional_assembly_iterator;

#define DL DEFAULT_LAYOUT

    /***************************
     * xoptional_assembly_base *
     ***************************/

    /**
     * @class xcontainer
     * @brief Base class for dense multidimensional optional assemblies.
     *
     * The xoptional_assembly_base class defines the interface for dense multidimensional
     * optional assembly classes. Optional assembly classes hold optional values and are
     * optimized for tensor operations. xoptional_assembly_base does not embed any data
     * container, this responsibility is delegated to the inheriting classes.
     *
     * @tparam D The derived type, i.e. the inheriting class for which xoptional_assembly_base
     *           provides the interface.
     */
    template <class D>
    class xoptional_assembly_base : private xiterable<D>
    {
    public:

        using self_type = xoptional_assembly_base<D>;
        using derived_type = D;
        using inner_types = xcontainer_inner_types<D>;

        using value_expression = typename inner_types::value_expression;
        using base_value_type = typename value_expression::value_type;
        using base_reference = typename value_expression::reference;
        using base_const_reference = typename value_expression::const_reference;

        using flag_expression = typename inner_types::flag_expression;
        using flag_type = typename flag_expression::value_type;
        using flag_reference = typename flag_expression::reference;
        using flag_const_reference = typename flag_expression::const_reference;

        using value_type = xtl::xoptional<base_value_type, flag_type>;
        using reference = xtl::xoptional<base_reference, flag_reference>;
        using const_reference = xtl::xoptional<base_const_reference, flag_const_reference>;
        using pointer = xtl::xclosure_pointer<reference>;
        using const_pointer = xtl::xclosure_pointer<const_reference>;
        using size_type = typename value_expression::size_type;
        using difference_type = typename value_expression::difference_type;
        using simd_value_type = xsimd::simd_type<value_type>;

        using shape_type = typename value_expression::shape_type;
        using strides_type = typename value_expression::strides_type;
        using backstrides_type = typename value_expression::backstrides_type;

        using inner_shape_type = typename value_expression::inner_shape_type;
        using inner_strides_type = typename value_expression::inner_strides_type;
        using inner_backstrides_type = typename value_expression::inner_backstrides_type;

        using iterable_base = xiterable<D>;
        using stepper = typename iterable_base::stepper;
        using const_stepper = typename iterable_base::const_stepper;

        static constexpr layout_type static_layout = value_expression::static_layout;
        static constexpr bool contiguous_layout = value_expression::contiguous_layout;

        using expression_tag = xoptional_expression_tag;

        template <layout_type L>
        using layout_iterator = typename iterable_base::template layout_iterator<L>;
        template <layout_type L>
        using const_layout_iterator = typename iterable_base::template const_layout_iterator<L>;
        template <layout_type L>
        using reverse_layout_iterator = typename iterable_base::template reverse_layout_iterator<L>;
        template <layout_type L>
        using const_reverse_layout_iterator = typename iterable_base::template const_reverse_layout_iterator<L>;

        template <class S, layout_type L>
        using broadcast_iterator = typename iterable_base::template broadcast_iterator<S, L>;
        template <class S, layout_type L>
        using const_broadcast_iterator = typename iterable_base::template const_broadcast_iterator<S, L>;
        template <class S, layout_type L>
        using reverse_broadcast_iterator = typename iterable_base::template reverse_broadcast_iterator<S, L>;
        template <class S, layout_type L>
        using const_reverse_broadcast_iterator = typename iterable_base::template const_reverse_broadcast_iterator<S, L>;

        using storage_iterator = xoptional_assembly_iterator<D, false>;
        using const_storage_iterator = xoptional_assembly_iterator<D, true>;
        using reverse_storage_iterator = std::reverse_iterator<storage_iterator>;
        using const_reverse_storage_iterator = std::reverse_iterator<const_storage_iterator>;

        using iterator = typename iterable_base::iterator;
        using const_iterator = typename iterable_base::const_iterator;
        using reverse_iterator = typename iterable_base::reverse_iterator;
        using const_reverse_iterator = typename iterable_base::const_reverse_iterator;

        size_type size() const noexcept;
        constexpr size_type dimension() const noexcept;

        const inner_shape_type& shape() const noexcept;
        const inner_strides_type& strides() const noexcept;
        const inner_backstrides_type& backstrides() const noexcept;

        template <class S = shape_type>
        void resize(const S& shape, bool force = false);
        template <class S = shape_type>
        void resize(const S& shape, layout_type l);
        template <class S = shape_type>
        void resize(const S& shape, const strides_type& strides);

        template <class S = shape_type>
        void reshape(const S& shape, layout_type layout = static_layout);

        layout_type layout() const noexcept;

        template <class... Args>
        reference operator()(Args... args);

        template <class... Args>
        const_reference operator()(Args... args) const;

        template <class... Args>
        reference at(Args... args);

        template <class... Args>
        const_reference at(Args... args) const;

        template <class S>
        disable_integral_t<S, reference> operator[](const S& index);
        template <class I>
        reference operator[](std::initializer_list<I> index);
        reference operator[](size_type i);

        template <class S>
        disable_integral_t<S, const_reference> operator[](const S& index) const;
        template <class I>
        const_reference operator[](std::initializer_list<I> index) const;
        const_reference operator[](size_type i) const;

        template <class It>
        reference element(It first, It last);
        template <class It>
        const_reference element(It first, It last) const;

        template <class S>
        bool broadcast_shape(S& shape, bool reuse_cache = false) const;

        template <class S>
        bool is_trivial_broadcast(const S& strides) const noexcept;

        using iterable_base::begin;
        using iterable_base::end;
        using iterable_base::cbegin;
        using iterable_base::cend;
        using iterable_base::rbegin;
        using iterable_base::rend;
        using iterable_base::crbegin;
        using iterable_base::crend;

        template <layout_type L = DL>
        storage_iterator storage_begin() noexcept;
        template <layout_type L = DL>
        storage_iterator storage_end() noexcept;

        template <layout_type L = DL>
        const_storage_iterator storage_begin() const noexcept;
        template <layout_type L = DL>
        const_storage_iterator storage_end() const noexcept;
        template <layout_type L = DL>
        const_storage_iterator storage_cbegin() const noexcept;
        template <layout_type L = DL>
        const_storage_iterator storage_cend() const noexcept;

        template <layout_type L = DL>
        reverse_storage_iterator storage_rbegin() noexcept;
        template <layout_type L = DL>
        reverse_storage_iterator storage_rend() noexcept;

        template <layout_type L = DL>
        const_reverse_storage_iterator storage_rbegin() const noexcept;
        template <layout_type L = DL>
        const_reverse_storage_iterator storage_rend() const noexcept;
        template <layout_type L = DL>
        const_reverse_storage_iterator storage_crbegin() const noexcept;
        template <layout_type L = DL>
        const_reverse_storage_iterator storage_crend() const noexcept;

        template <class S>
        stepper stepper_begin(const S& shape) noexcept;
        template <class S>
        stepper stepper_end(const S& shape, layout_type l) noexcept;

        template <class S>
        const_stepper stepper_begin(const S& shape) const noexcept;
        template <class S>
        const_stepper stepper_end(const S& shape, layout_type l) const noexcept;

        reference data_element(size_type i);
        const_reference data_element(size_type i) const;

        value_expression& value() noexcept;
        const value_expression& value() const noexcept;

        flag_expression& has_value() noexcept;
        const flag_expression& has_value() const noexcept;

    protected:

        xoptional_assembly_base() = default;
        ~xoptional_assembly_base() = default;

        xoptional_assembly_base(const xoptional_assembly_base&) = default;
        xoptional_assembly_base& operator=(const xoptional_assembly_base&) = default;

        xoptional_assembly_base(xoptional_assembly_base&&) = default;
        xoptional_assembly_base& operator=(xoptional_assembly_base&&) = default;

    private:

        derived_type& derived_cast() noexcept;
        const derived_type& derived_cast() const noexcept;

        friend class xiterable<D>;
        friend class xconst_iterable<D>;
    };

#undef DL

    /*******************************
     * xoptional_assembly_iterator *
     *******************************/

    namespace detail
    {
        template <class D, bool is_const>
        using get_optional_reference_t = std::conditional_t<is_const,
                                                            typename xoptional_assembly_base<D>::const_reference,
                                                            typename xoptional_assembly_base<D>::reference>;

        template <class D, bool is_const>
        using get_optional_pointer_t = std::conditional_t<is_const,
                                                          typename xoptional_assembly_base<D>::const_pointer,
                                                          typename xoptional_assembly_base<D>::pointer>;
    }

    template <class D, bool is_const>
    class xoptional_assembly_iterator
        : public xtl::xrandom_access_iterator_base<xoptional_assembly_iterator<D, is_const>,
                                                   typename xoptional_assembly_base<D>::value_type,
                                                   typename xoptional_assembly_base<D>::difference_type,
                                                   detail::get_optional_pointer_t<D, is_const>,
                                                   detail::get_optional_reference_t<D, is_const>>

    {
    public:

        using self_type = xoptional_assembly_iterator<D, is_const>;
        using assembly_type = xoptional_assembly_base<D>;
        using value_type = typename xoptional_assembly_base<D>::value_type;
        using reference = detail::get_optional_reference_t<D, is_const>;
        using pointer = detail::get_optional_pointer_t<D, is_const>;
        using difference_type = typename assembly_type::difference_type;
        using iterator_category = std::random_access_iterator_tag;

        using value_expression = typename assembly_type::value_expression;
        using flag_expression = typename assembly_type::flag_expression;
        using value_iterator = std::conditional_t<is_const,
                                                  typename value_expression::const_storage_iterator,
                                                  typename value_expression::storage_iterator>;
        using flag_iterator = std::conditional_t<is_const,
                                                 typename flag_expression::const_storage_iterator,
                                                 typename flag_expression::storage_iterator>;

        xoptional_assembly_iterator(value_iterator vit, flag_iterator fit) noexcept;

        self_type& operator++();
        self_type& operator--();

        self_type& operator+=(difference_type n);
        self_type& operator-=(difference_type n);

        difference_type operator-(const self_type& rhs) const;

        reference operator*() const;
        pointer operator->() const;

        bool equal(const self_type& rhs) const;
        bool less_than(const self_type& rhs) const;

    private:

        value_iterator m_vit;
        flag_iterator m_fit;
    };

    template <class D, bool is_const>
    bool operator==(const xoptional_assembly_iterator<D, is_const>& lhs,
                    const xoptional_assembly_iterator<D, is_const>& rhs);

    template <class D, bool is_const>
    bool operator<(const xoptional_assembly_iterator<D, is_const>& lhs,
                   const xoptional_assembly_iterator<D, is_const>& rhs);

    /******************************
     * xoptional_assembly_stepper *
     ******************************/

    template <class D, bool is_const>
    class xoptional_assembly_stepper
    {
    public:

        using self_type = xoptional_assembly_stepper<D, is_const>;
        using assembly_type = xoptional_assembly_base<D>;
        using value_type = typename assembly_type::value_type;
        using reference = std::conditional_t<is_const,
                                             typename assembly_type::const_reference,
                                             typename assembly_type::reference>;
        using pointer = std::conditional_t<is_const,
                                           typename assembly_type::const_pointer,
                                           typename assembly_type::pointer>;
        using size_type = typename assembly_type::size_type;
        using difference_type = typename assembly_type::difference_type;
        using value_expression = typename assembly_type::value_expression;
        using flag_expression = typename assembly_type::flag_expression;
        using value_stepper = std::conditional_t<is_const,
                                                 typename value_expression::const_stepper,
                                                 typename value_expression::stepper>;
        using flag_stepper = std::conditional_t<is_const,
                                                typename flag_expression::const_stepper,
                                                typename flag_expression::stepper>;

        xoptional_assembly_stepper(value_stepper vs, flag_stepper fs) noexcept;

        void step(size_type dim, size_type n = 1);
        void step_back(size_type dim, size_type n = 1);
        void reset(size_type dim);
        void reset_back(size_type dim);

        void to_begin();
        void to_end(layout_type l);

        reference operator*() const;

        bool equal(const self_type& rhs) const;

    private:

        value_stepper m_vs;
        flag_stepper m_fs;
    };

    template <class D, bool is_const>
    bool operator==(const xoptional_assembly_stepper<D, is_const>& lhs,
                    const xoptional_assembly_stepper<D, is_const>& rhs);

    template <class D, bool is_const>
    bool operator!=(const xoptional_assembly_stepper<D, is_const>& lhs,
                    const xoptional_assembly_stepper<D, is_const>& rhs);

    /******************************************
     * xoptional_assembly_base implementation *
     ******************************************/

    /**
     * @name Size and shape
     */
    //@{
    /**
     * Returns the number of element in the optional assembly.
     */
    template <class D>
    inline auto xoptional_assembly_base<D>::size() const noexcept -> size_type
    {
        return value().size();
    }

    /**
     * Returns the number of dimensions of the optional assembly.
     */
    template <class D>
    inline auto constexpr xoptional_assembly_base<D>::dimension() const noexcept -> size_type
    {
        return value().dimension();
    }

    /**
     * Returns the shape of the optional assembly.
     */
    template <class D>
    inline auto xoptional_assembly_base<D>::shape() const noexcept -> const inner_shape_type&
    {
        return value().shape();
    }

    /**
     * Returns the strides of the optional assembly.
     */
    template <class D>
    inline auto xoptional_assembly_base<D>::strides() const noexcept -> const inner_strides_type&
    {
        return value().strides();
    }

    /**
     * Returns the backstrides of the optional assembly.
     */
    template <class D>
    inline auto xoptional_assembly_base<D>::backstrides() const noexcept -> const inner_backstrides_type&
    {
        return value().backstrides();
    }
    //@}

    /**
     * Resizes the optional assembly.
     * @param shape the new shape
     * @param force force reshaping, even if the shape stays the same (default: false)
     */
    template <class D>
    template <class S>
    inline void xoptional_assembly_base<D>::resize(const S& shape, bool force)
    {
        value().resize(shape, force);
        has_value().resize(shape, force);
    }

    /**
     * Resizes the optional assembly.
     * @param shape the new shape
     * @param l the new layout_type
     */
    template <class D>
    template <class S>
    inline void xoptional_assembly_base<D>::resize(const S& shape, layout_type l)
    {
        value().resize(shape, l);
        has_value().resize(shape, l);
    }

    /**
     * Resizes the optional assembly.
     * @param shape the new shape
     * @param strides the new strides
     */
    template <class D>
    template <class S>
    inline void xoptional_assembly_base<D>::resize(const S& shape, const strides_type& strides)
    {
        value().resize(shape, strides);
        has_value().resize(shape, strides);
    }

    /**
     * Reshapes the optional assembly.
     * @param shape the new shape
     * @param layout the new layout
     */
    template <class D>
    template <class S>
    inline void xoptional_assembly_base<D>::reshape(const S& shape, layout_type layout)
    {
        value().reshape(shape, layout);
        has_value().reshape(shape, layout);
    }

    /**
     * Return the layout_type of the container
     * @return layout_type of the container
     */
    template <class D>
    inline layout_type xoptional_assembly_base<D>::layout() const noexcept
    {
        return value().layout();
    }

    /**
     * @name Data
     */
    //@{
    /**
     * Returns a reference to the element at the specified position in the optional assembly.
     * @param args a list of indices specifying the position in the optional assembly. Indices
     * must be unsigned integers, the number of indices should be equal or greater than
     * the number of dimensions of the optional assembly.
     */
    template <class D>
    template <class... Args>
    inline auto xoptional_assembly_base<D>::operator()(Args... args) -> reference
    {
        return reference(value()(args...), has_value()(args...));
    }

    /**
     * Returns a constant reference to the element at the specified position in the optional assembly.
     * @param args a list of indices specifying the position in the optional assembly. Indices
     * must be unsigned integers, the number of indices should be equal or greater than
     * the number of dimensions of the optional assembly.
     */
    template <class D>
    template <class... Args>
    inline auto xoptional_assembly_base<D>::operator()(Args... args) const -> const_reference
    {
        return const_reference(value()(args...), has_value()(args...));
    }

    /**
     * Returns a reference to the element at the specified position in the optional assembly,
     * after dimension and bounds checking.
     * @param args a list of indices specifying the position in the optional assembly. Indices
     * must be unsigned integers, the number of indices should be equal to the number of dimensions
     * of the optional assembly.
     * @exception std::out_of_range if the number of argument is greater than the number of dimensions
     * or if indices are out of bounds.
     */
    template <class D>
    template <class... Args>
    inline auto xoptional_assembly_base<D>::at(Args... args) -> reference
    {
        return reference(value().at(args...), has_value().at(args...));
    }

    /**
     * Returns a constant reference to the element at the specified position in the optional assembly,
     * after dimension and bounds checking.
     * @param args a list of indices specifying the position in the optional assembly. Indices
     * must be unsigned integers, the number of indices should be equal to the number of dimensions
     * of the optional assembly.
     * @exception std::out_of_range if the number of argument is greater than the number of dimensions
     * or if indices are out of bounds.
     */
    template <class D>
    template <class... Args>
    inline auto xoptional_assembly_base<D>::at(Args... args) const -> const_reference
    {
        return const_reference(value().at(args...), has_value().at(args...));
    }

    /**
     * Returns a reference to the element at the specified position in the optional assembly.
     * @param index a sequence of indices specifying the position in the optional assembly. Indices
     * must be unsigned integers, the number of indices in the list should be equal or greater
     * than the number of dimensions of the optional assembly.
     */
    template <class D>
    template <class S>
    inline auto xoptional_assembly_base<D>::operator[](const S& index)
        -> disable_integral_t<S, reference>
    {
        return reference(value()[index], has_value()[index]);
    }

    template <class D>
    template <class I>
    inline auto xoptional_assembly_base<D>::operator[](std::initializer_list<I> index)
        -> reference
    {
        return reference(value()[index], has_value()[index]);
    }

    template <class D>
    inline auto xoptional_assembly_base<D>::operator[](size_type i) -> reference
    {
        return reference(value()[i], has_value()[i]);
    }

    /**
     * Returns a constant reference to the element at the specified position in the optional assembly.
     * @param index a sequence of indices specifying the position in the optional assembly. Indices
     * must be unsigned integers, the number of indices in the list should be equal or greater
     * than the number of dimensions of the optional assembly.
     */
    template <class D>
    template <class S>
    inline auto xoptional_assembly_base<D>::operator[](const S& index) const
        -> disable_integral_t<S, const_reference>
    {
        return const_reference(value()[index], has_value()[index]);
    }

    template <class D>
    template <class I>
    inline auto xoptional_assembly_base<D>::operator[](std::initializer_list<I> index) const
        -> const_reference
    {
        return const_reference(value()[index], has_value()[index]);
    }

    template <class D>
    inline auto xoptional_assembly_base<D>::operator[](size_type i) const -> const_reference
    {
        return const_reference(value()[i], has_value()[i]);
    }

    /**
     * Returns a reference to the element at the specified position in the optional assembly.
     * @param first iterator starting the sequence of indices
     * @param last iterator ending the sequence of indices
     * The number of indices in the sequence should be equal to or greater
     * than the number of dimensions of the optional assembly.
     */
    template <class D>
    template <class It>
    inline auto xoptional_assembly_base<D>::element(It first, It last) -> reference
    {
        return reference(value().element(first, last), has_value().element(first, last));
    }

    /**
     * Returns a constant reference to the element at the specified position in the optional assembly.
     * @param first iterator starting the sequence of indices
     * @param last iterator ending the sequence of indices
     * The number of indices in the sequence should be equal to or greater
     * than the number of dimensions of the optional assembly.
     */
    template <class D>
    template <class It>
    inline auto xoptional_assembly_base<D>::element(It first, It last) const -> const_reference
    {
        return const_reference(value().element(first, last), has_value().element(first, last));
    }
    //@}

    /**
     * @name Broadcasting
     */
    //@{
    /**
     * Broadcast the shape of the optional assembly to the specified parameter.
     * @param shape the result shape
     * @return a boolean indicating whether the broadcasting is trivial
     */
    template <class D>
    template <class S>
    inline bool xoptional_assembly_base<D>::broadcast_shape(S& shape, bool reuse_cache) const
    {
        bool res = value().broadcast_shape(shape, reuse_cache);
        return res && has_value().broadcast_shape(shape, reuse_cache);
    }

    /**
     * Compares the specified strides with those of the optional assembly to see whether
     * the broadcasting is trivial.
     * @return a boolean indicating whether the broadcasting is trivial
     */
    template <class D>
    template <class S>
    inline bool xoptional_assembly_base<D>::is_trivial_broadcast(const S& strides) const noexcept
    {
        return value().is_trivial_broadcast(strides) && has_value().is_trivial_broadcast(strides);
    }
    //@}

    template <class D>
    template <layout_type L>
    inline auto xoptional_assembly_base<D>::storage_begin() noexcept -> storage_iterator
    {
        return storage_iterator(value().template storage_begin<L>(),
                                has_value().template storage_begin<L>());
    }

    template <class D>
    template <layout_type L>
    inline auto xoptional_assembly_base<D>::storage_end() noexcept -> storage_iterator
    {
        return storage_iterator(value().template storage_end<L>(),
                                has_value().template storage_end<L>());
    }

    template <class D>
    template <layout_type L>
    inline auto xoptional_assembly_base<D>::storage_begin() const noexcept -> const_storage_iterator
    {
        return storage_cbegin<L>();
    }

    template <class D>
    template <layout_type L>
    inline auto xoptional_assembly_base<D>::storage_end() const noexcept -> const_storage_iterator
    {
        return storage_cend<L>();
    }

    template <class D>
    template <layout_type L>
    inline auto xoptional_assembly_base<D>::storage_cbegin() const noexcept -> const_storage_iterator
    {
        return const_storage_iterator(value().template storage_cbegin<L>(),
                                      has_value().template storage_begin<L>());
    }

    template <class D>
    template <layout_type L>
    inline auto xoptional_assembly_base<D>::storage_cend() const noexcept -> const_storage_iterator
    {
        return const_storage_iterator(value().template storage_cend<L>(),
                                      has_value().template storage_end<L>());
    }

    template <class D>
    template <layout_type L>
    inline auto xoptional_assembly_base<D>::storage_rbegin() noexcept -> reverse_storage_iterator
    {
        return reverse_storage_iterator(storage_end<L>());
    }

    template <class D>
    template <layout_type L>
    inline auto xoptional_assembly_base<D>::storage_rend() noexcept -> reverse_storage_iterator
    {
        return reverse_storage_iterator(storage_begin<L>());
    }

    template <class D>
    template <layout_type L>
    inline auto xoptional_assembly_base<D>::storage_rbegin() const noexcept -> const_reverse_storage_iterator
    {
        return storage_crbegin<L>();
    }

    template <class D>
    template <layout_type L>
    inline auto xoptional_assembly_base<D>::storage_rend() const noexcept -> const_reverse_storage_iterator
    {
        return storage_crend<L>();
    }

    template <class D>
    template <layout_type L>
    inline auto xoptional_assembly_base<D>::storage_crbegin() const noexcept -> const_reverse_storage_iterator
    {
        return const_reverse_storage_iterator(storage_cend<L>());
    }

    template <class D>
    template <layout_type L>
    inline auto xoptional_assembly_base<D>::storage_crend() const noexcept -> const_reverse_storage_iterator
    {
        return const_reverse_storage_iterator(storage_begin<L>());
    }

    template <class D>
    template <class S>
    inline auto xoptional_assembly_base<D>::stepper_begin(const S& shape) noexcept -> stepper
    {
        return stepper(value().stepper_begin(shape), has_value().stepper_begin(shape));
    }

    template <class D>
    template <class S>
    inline auto xoptional_assembly_base<D>::stepper_end(const S& shape, layout_type l) noexcept -> stepper
    {
        return stepper(value().stepper_end(shape, l), has_value().stepper_end(shape, l));
    }

    template <class D>
    template <class S>
    inline auto xoptional_assembly_base<D>::stepper_begin(const S& shape) const noexcept -> const_stepper
    {
        return const_stepper(value().stepper_begin(shape), has_value().stepper_begin(shape));
    }

    template <class D>
    template <class S>
    inline auto xoptional_assembly_base<D>::stepper_end(const S& shape, layout_type l) const noexcept -> const_stepper
    {
        return const_stepper(value().stepper_end(shape, l), has_value().stepper_end(shape, l));
    }

    template <class D>
    inline auto xoptional_assembly_base<D>::data_element(size_type i) -> reference
    {
        return reference(value().data_element(i), has_value().data_element(i));
    }

    template <class D>
    inline auto xoptional_assembly_base<D>::data_element(size_type i) const -> const_reference
    {
        return const_reference(value().data_element(i), has_value().data_element(i));
    }

    /**
     * Return an expression for the values of the optional assembly.
     */
    template <class D>
    inline auto xoptional_assembly_base<D>::value() noexcept -> value_expression&
    {
        return derived_cast().value_impl();
    }

    /**
     * Return a constant expression for the values of the optional assembly.
     */
    template <class D>
    inline auto xoptional_assembly_base<D>::value() const noexcept -> const value_expression&
    {
        return derived_cast().value_impl();
    }

    /**
     * Return an expression for the missing mask of the optional assembly.
     */
    template <class D>
    inline auto xoptional_assembly_base<D>::has_value() noexcept -> flag_expression&
    {
        return derived_cast().has_value_impl();
    }

    /**
     * Return a constant expression for the missing mask of the optional assembly.
     */
    template <class D>
    inline auto xoptional_assembly_base<D>::has_value() const noexcept -> const flag_expression&
    {
        return derived_cast().has_value_impl();
    }

    template <class D>
    inline auto xoptional_assembly_base<D>::derived_cast() noexcept -> derived_type&
    {
        return *static_cast<derived_type*>(this);
    }

    template <class D>
    inline auto xoptional_assembly_base<D>::derived_cast() const noexcept -> const derived_type&
    {
        return *static_cast<const derived_type*>(this);
    }

    /**********************************************
     * xoptional_assembly_iterator implementation *
     **********************************************/

    template <class D, bool C>
    inline xoptional_assembly_iterator<D, C>::xoptional_assembly_iterator(value_iterator vit, flag_iterator fit) noexcept
        : m_vit(vit), m_fit(fit)
    {
    }

    template <class D, bool C>
    inline auto xoptional_assembly_iterator<D, C>::operator++() -> self_type&
    {
        ++m_vit;
        ++m_fit;
        return *this;
    }

    template <class D, bool C>
    inline auto xoptional_assembly_iterator<D, C>::operator--() -> self_type&
    {
        --m_vit;
        --m_fit;
        return *this;
    }

    template <class D, bool C>
    inline auto xoptional_assembly_iterator<D, C>::operator+=(difference_type n) -> self_type&
    {
        m_vit += n;
        m_fit += n;
        return *this;
    }

    template <class D, bool C>
    inline auto xoptional_assembly_iterator<D, C>::operator-=(difference_type n) -> self_type&
    {
        m_vit -= n;
        m_fit -= n;
        return *this;
    }

    template <class D, bool C>
    inline auto xoptional_assembly_iterator<D, C>::operator-(const self_type& rhs) const -> difference_type
    {
        return m_vit - rhs.m_vit;
    }

    template <class D, bool C>
    inline auto xoptional_assembly_iterator<D, C>::operator*() const -> reference
    {
        return reference(*m_vit, *m_fit);
    }

    template <class D, bool C>
    inline auto xoptional_assembly_iterator<D, C>::operator-> () const -> pointer
    {
        return &(this->operator*());
    }

    template <class D, bool C>
    inline bool xoptional_assembly_iterator<D, C>::equal(const self_type& rhs) const
    {
        return m_vit.equal(rhs.m_vit) && m_fit.equal(rhs.m_fit);
    }

    template <class D, bool C>
    inline bool xoptional_assembly_iterator<D, C>::less_than(const self_type& rhs) const
    {
        return m_vit.less_than(rhs.m_vit) && m_fit.less_than(rhs.m_fit);
    }

    template <class D, bool is_const>
    inline bool operator==(const xoptional_assembly_iterator<D, is_const>& lhs,
                           const xoptional_assembly_iterator<D, is_const>& rhs)
    {
        return lhs.equal(rhs);
    }

    template <class D, bool is_const>
    inline bool operator<(const xoptional_assembly_iterator<D, is_const>& lhs,
                          const xoptional_assembly_iterator<D, is_const>& rhs)
    {
        return lhs.less_than(rhs);
    }

    /*********************************************
     * xoptional_assembly_stepper implementation *
     *********************************************/

    template <class D, bool C>
    inline xoptional_assembly_stepper<D, C>::xoptional_assembly_stepper(value_stepper vs, flag_stepper fs) noexcept
        : m_vs(vs), m_fs(fs)
    {
    }

    template <class D, bool C>
    inline void xoptional_assembly_stepper<D, C>::step(size_type dim, size_type n)
    {
        m_vs.step(dim, n);
        m_fs.step(dim, n);
    }

    template <class D, bool C>
    inline void xoptional_assembly_stepper<D, C>::step_back(size_type dim, size_type n)
    {
        m_vs.step_back(dim, n);
        m_fs.step_back(dim, n);
    }

    template <class D, bool C>
    inline void xoptional_assembly_stepper<D, C>::reset(size_type dim)
    {
        m_vs.reset(dim);
        m_fs.reset(dim);
    }

    template <class D, bool C>
    inline void xoptional_assembly_stepper<D, C>::reset_back(size_type dim)
    {
        m_vs.reset_back(dim);
        m_fs.reset_back(dim);
    }

    template <class D, bool C>
    inline void xoptional_assembly_stepper<D, C>::to_begin()
    {
        m_vs.to_begin();
        m_fs.to_begin();
    }

    template <class D, bool C>
    inline void xoptional_assembly_stepper<D, C>::to_end(layout_type l)
    {
        m_vs.to_end(l);
        m_fs.to_end(l);
    }

    template <class D, bool C>
    inline auto xoptional_assembly_stepper<D, C>::operator*() const -> reference
    {
        return reference(*m_vs, *m_fs);
    }

    template <class D, bool C>
    inline bool xoptional_assembly_stepper<D, C>::equal(const self_type& rhs) const
    {
        return m_vs.equal(rhs.m_vs) && m_fs.equal(rhs.m_fs);
    }

    template <class D, bool is_const>
    inline bool operator==(const xoptional_assembly_stepper<D, is_const>& lhs,
                           const xoptional_assembly_stepper<D, is_const>& rhs)
    {
        return lhs.equal(rhs);
    }

    template <class D, bool is_const>
    inline bool operator!=(const xoptional_assembly_stepper<D, is_const>& lhs,
                           const xoptional_assembly_stepper<D, is_const>& rhs)
    {
        return !(lhs == rhs);
    }
}

#endif
