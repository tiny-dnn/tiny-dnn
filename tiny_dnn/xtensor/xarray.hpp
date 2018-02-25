/***************************************************************************
* Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XTENSOR_ARRAY_HPP
#define XTENSOR_ARRAY_HPP

#include <algorithm>
#include <initializer_list>
#include <utility>

#include "xtl/xsequence.hpp"

#include "xbuffer_adaptor.hpp"
#include "xcontainer.hpp"
#include "xsemantic.hpp"

namespace xt
{

    /********************************
     * xarray_container declaration *
     ********************************/

    template <class EC, layout_type L, class SC, class Tag>
    struct xcontainer_inner_types<xarray_container<EC, L, SC, Tag>>
    {
        using container_type = EC;
        using shape_type = SC;
        using strides_type = shape_type;
        using backstrides_type = shape_type;
        using inner_shape_type = shape_type;
        using inner_strides_type = strides_type;
        using inner_backstrides_type = backstrides_type;
        using temporary_type = xarray_container<EC, L, SC, Tag>;
        static constexpr layout_type layout = L;
    };

    template <class EC, layout_type L, class SC, class Tag>
    struct xiterable_inner_types<xarray_container<EC, L, SC, Tag>>
        : xcontainer_iterable_types<xarray_container<EC, L, SC, Tag>>
    {
    };

    /**
     * @class xarray_container
     * @brief Dense multidimensional container with tensor semantic.
     *
     * The xarray_container class implements a dense multidimensional container
     * with tensor semantic.
     *
     * @tparam EC The type of the container holding the elements.
     * @tparam L The layout_type of the container.
     * @tparam SC The type of the containers holding the shape and the strides.
     * @tparam Tag The expression tag.
     * @sa xarray
     */
    template <class EC, layout_type L, class SC, class Tag>
    class xarray_container : public xstrided_container<xarray_container<EC, L, SC, Tag>>,
                             public xcontainer_semantic<xarray_container<EC, L, SC, Tag>>
    {
    public:

        using self_type = xarray_container<EC, L, SC, Tag>;
        using base_type = xstrided_container<self_type>;
        using semantic_base = xcontainer_semantic<self_type>;
        using container_type = typename base_type::container_type;
        using allocator_type = typename base_type::allocator_type;
        using value_type = typename base_type::value_type;
        using reference = typename base_type::reference;
        using const_reference = typename base_type::const_reference;
        using pointer = typename base_type::pointer;
        using const_pointer = typename base_type::const_pointer;
        using shape_type = typename base_type::shape_type;
        using inner_shape_type = typename base_type::inner_shape_type;
        using strides_type = typename base_type::strides_type;
        using backstrides_type = typename base_type::backstrides_type;
        using inner_strides_type = typename base_type::inner_strides_type;
        using temporary_type = typename semantic_base::temporary_type;
        using expression_tag = Tag;

        xarray_container();
        explicit xarray_container(const shape_type& shape, layout_type l = L);
        explicit xarray_container(const shape_type& shape, const_reference value, layout_type l = L);
        explicit xarray_container(const shape_type& shape, const strides_type& strides);
        explicit xarray_container(const shape_type& shape, const strides_type& strides, const_reference value);
        explicit xarray_container(container_type&& data, inner_shape_type&& shape, inner_strides_type&& strides);

        xarray_container(const value_type& t);
        xarray_container(nested_initializer_list_t<value_type, 1> t);
        xarray_container(nested_initializer_list_t<value_type, 2> t);
        xarray_container(nested_initializer_list_t<value_type, 3> t);
        xarray_container(nested_initializer_list_t<value_type, 4> t);
        xarray_container(nested_initializer_list_t<value_type, 5> t);

        template <class S = shape_type>
        static xarray_container from_shape(S&& s);

        ~xarray_container() = default;

        xarray_container(const xarray_container&) = default;
        xarray_container& operator=(const xarray_container&) = default;

        xarray_container(xarray_container&&) = default;
        xarray_container& operator=(xarray_container&&) = default;

        template <class E>
        xarray_container(const xexpression<E>& e);

        template <class E>
        xarray_container& operator=(const xexpression<E>& e);

    private:

        container_type m_data;

        container_type& data_impl() noexcept;
        const container_type& data_impl() const noexcept;

        friend class xcontainer<xarray_container<EC, L, SC, Tag>>;
    };

    /******************************
     * xarray_adaptor declaration *
     ******************************/

    template <class EC, layout_type L, class SC, class Tag>
    struct xcontainer_inner_types<xarray_adaptor<EC, L, SC, Tag>>
    {
        using container_type = std::remove_reference_t<EC>;
        using shape_type = SC;
        using strides_type = shape_type;
        using backstrides_type = shape_type;
        using inner_shape_type = shape_type;
        using inner_strides_type = strides_type;
        using inner_backstrides_type = backstrides_type;
        using temporary_type = xarray_container<temporary_container_t<container_type>, L, SC, Tag>;
        static constexpr layout_type layout = L;
    };

    template <class EC, layout_type L, class SC, class Tag>
    struct xiterable_inner_types<xarray_adaptor<EC, L, SC, Tag>>
        : xcontainer_iterable_types<xarray_adaptor<EC, L, SC, Tag>>
    {
    };

    /**
     * @class xarray_adaptor
     * @brief Dense multidimensional container adaptor with
     * tensor semantic.
     *
     * The xarray_adaptor class implements a dense multidimensional
     * container adaptor with tensor semantic. It is used to provide
     * a multidimensional container semantic and a tensor semantic to
     * stl-like containers.
     *
     * @tparam EC The closure for the container type to adapt.
     * @tparam L The layout_type of the adaptor.
     * @tparam SC The type of the containers holding the shape and the strides.
     * @tparam Tag The expression tag.
     */
    template <class EC, layout_type L, class SC, class Tag>
    class xarray_adaptor : public xstrided_container<xarray_adaptor<EC, L, SC, Tag>>,
                           public xcontainer_semantic<xarray_adaptor<EC, L, SC, Tag>>
    {
    public:

        using container_closure_type = EC;

        using self_type = xarray_adaptor<EC, L, SC, Tag>;
        using base_type = xstrided_container<self_type>;
        using semantic_base = xcontainer_semantic<self_type>;
        using container_type = typename base_type::container_type;
        using allocator_type = typename base_type::allocator_type;
        using shape_type = typename base_type::shape_type;
        using strides_type = typename base_type::strides_type;
        using backstrides_type = typename base_type::backstrides_type;
        using temporary_type = typename semantic_base::temporary_type;
        using expression_tag = Tag;

        xarray_adaptor(container_type&& data);
        xarray_adaptor(const container_type& data);

        template <class D>
        xarray_adaptor(D&& data, const shape_type& shape, layout_type l = L);

        template <class D>
        xarray_adaptor(D&& data, const shape_type& shape, const strides_type& strides);

        ~xarray_adaptor() = default;

        xarray_adaptor(const xarray_adaptor&) = default;
        xarray_adaptor& operator=(const xarray_adaptor&);

        xarray_adaptor(xarray_adaptor&&) = default;
        xarray_adaptor& operator=(xarray_adaptor&&);
        xarray_adaptor& operator=(temporary_type&&);

        template <class E>
        xarray_adaptor& operator=(const xexpression<E>& e);

    private:

        container_closure_type m_data;

        container_type& data_impl() noexcept;
        const container_type& data_impl() const noexcept;


        friend class xcontainer<xarray_adaptor<EC, L, SC, Tag>>;
    };

    /***********************************
     * xarray_container implementation *
     ***********************************/

    /**
     * @name Constructors
     */
    //@{
    /**
     * Allocates an uninitialized xarray_container that holds 0 element.
     */
    template <class EC, layout_type L, class SC, class Tag>
    inline xarray_container<EC, L, SC, Tag>::xarray_container()
        : base_type(), m_data(1, value_type())
    {
    }

    /**
     * Allocates an uninitialized xarray_container with the specified shape and
     * layout_type.
     * @param shape the shape of the xarray_container
     * @param l the layout_type of the xarray_container
     */
    template <class EC, layout_type L, class SC, class Tag>
    inline xarray_container<EC, L, SC, Tag>::xarray_container(const shape_type& shape, layout_type l)
        : base_type()
    {
        base_type::resize(shape, l);
    }

    /**
     * Allocates an xarray_container with the specified shape and layout_type. Elements
     * are initialized to the specified value.
     * @param shape the shape of the xarray_container
     * @param value the value of the elements
     * @param l the layout_type of the xarray_container
     */
    template <class EC, layout_type L, class SC, class Tag>
    inline xarray_container<EC, L, SC, Tag>::xarray_container(const shape_type& shape, const_reference value, layout_type l)
        : base_type()
    {
        base_type::resize(shape, l);
        std::fill(m_data.begin(), m_data.end(), value);
    }

    /**
     * Allocates an uninitialized xarray_container with the specified shape and strides.
     * @param shape the shape of the xarray_container
     * @param strides the strides of the xarray_container
     */
    template <class EC, layout_type L, class SC, class Tag>
    inline xarray_container<EC, L, SC, Tag>::xarray_container(const shape_type& shape, const strides_type& strides)
        : base_type()
    {
        base_type::resize(shape, strides);
    }

    /**
     * Allocates an uninitialized xarray_container with the specified shape and strides.
     * Elements are initialized to the specified value.
     * @param shape the shape of the xarray_container
     * @param strides the strides of the xarray_container
     * @param value the value of the elements
     */
    template <class EC, layout_type L, class SC, class Tag>
    inline xarray_container<EC, L, SC, Tag>::xarray_container(const shape_type& shape, const strides_type& strides, const_reference value)
        : base_type()
    {
        base_type::resize(shape, strides);
        std::fill(m_data.begin(), m_data.end(), value);
    }

    /**
     * Allocates an xarray_container that holds a single element initialized to the
     * specified value.
     * @param t the value of the element
     */
    template <class EC, layout_type L, class SC, class Tag>
    inline xarray_container<EC, L, SC, Tag>::xarray_container(const value_type& t)
        : base_type()
    {
        base_type::resize(xt::shape<shape_type>(t), true);
        nested_copy(m_data.begin(), t);
    }

    /**
     * Allocates an xarray_container by moving specified data, shape and strides
     *
     * @param data the data for the xarray_container
     * @param shape the shape of the xarray_container
     * @param strides the strides of the xarray_container
     */
    template <class EC, layout_type L, class SC, class Tag>
    inline xarray_container<EC, L, SC, Tag>::xarray_container(container_type&& data, inner_shape_type&& shape, inner_strides_type&& strides)
        : base_type(std::move(shape), std::move(strides)), m_data(std::move(data))
    {
    }
    //@}

    /**
     * @name Constructors from initializer list
     */
    //@{
    /**
     * Allocates a one-dimensional xarray_container.
     * @param t the elements of the xarray_container
     */
    template <class EC, layout_type L, class SC, class Tag>
    inline xarray_container<EC, L, SC, Tag>::xarray_container(nested_initializer_list_t<value_type, 1> t)
        : base_type()
    {
        base_type::resize(xt::shape<shape_type>(t));
        L == layout_type::row_major ? nested_copy(m_data.begin(), t) : nested_copy(this->template begin<layout_type::row_major>(), t);
    }

    /**
     * Allocates a two-dimensional xarray_container.
     * @param t the elements of the xarray_container
     */
    template <class EC, layout_type L, class SC, class Tag>
    inline xarray_container<EC, L, SC, Tag>::xarray_container(nested_initializer_list_t<value_type, 2> t)
        : base_type()
    {
        base_type::resize(xt::shape<shape_type>(t));
        L == layout_type::row_major ? nested_copy(m_data.begin(), t) : nested_copy(this->template begin<layout_type::row_major>(), t);
    }

    /**
     * Allocates a three-dimensional xarray_container.
     * @param t the elements of the xarray_container
     */
    template <class EC, layout_type L, class SC, class Tag>
    inline xarray_container<EC, L, SC, Tag>::xarray_container(nested_initializer_list_t<value_type, 3> t)
        : base_type()
    {
        base_type::resize(xt::shape<shape_type>(t));
        L == layout_type::row_major ? nested_copy(m_data.begin(), t) : nested_copy(this->template begin<layout_type::row_major>(), t);
    }

    /**
     * Allocates a four-dimensional xarray_container.
     * @param t the elements of the xarray_container
     */
    template <class EC, layout_type L, class SC, class Tag>
    inline xarray_container<EC, L, SC, Tag>::xarray_container(nested_initializer_list_t<value_type, 4> t)
        : base_type()
    {
        base_type::resize(xt::shape<shape_type>(t));
        L == layout_type::row_major ? nested_copy(m_data.begin(), t) : nested_copy(this->template begin<layout_type::row_major>(), t);
    }

    /**
     * Allocates a five-dimensional xarray_container.
     * @param t the elements of the xarray_container
     */
    template <class EC, layout_type L, class SC, class Tag>
    inline xarray_container<EC, L, SC, Tag>::xarray_container(nested_initializer_list_t<value_type, 5> t)
        : base_type()
    {
        base_type::resize(xt::shape<shape_type>(t));
        L == layout_type::row_major ? nested_copy(m_data.begin(), t) : nested_copy(this->template begin<layout_type::row_major>(), t);
    }
    //@}

    /**
     * Allocates and returns an xarray_container with the specified shape.
     * @param s the shape of the xarray_container
     */
    template <class EC, layout_type L, class SC, class Tag>
    template <class S>
    inline xarray_container<EC, L, SC, Tag> xarray_container<EC, L, SC, Tag>::from_shape(S&& s)
    {
        shape_type shape = xtl::forward_sequence<shape_type>(s);
        return self_type(shape);
    }

    /**
     * @name Extended copy semantic
     */
    //@{
    /**
     * The extended copy constructor.
     */
    template <class EC, layout_type L, class SC, class Tag>
    template <class E>
    inline xarray_container<EC, L, SC, Tag>::xarray_container(const xexpression<E>& e)
        : base_type()
    {
        // Avoids unintialized data because of (m_shape == shape) condition
        // in resize (called by assign), which is always true when dimension == 0.
        if (e.derived_cast().dimension() == 0)
        {
            detail::resize_data_container(m_data, std::size_t(1));
        }
        semantic_base::assign(e);
    }

    /**
     * The extended assignment operator.
     */
    template <class EC, layout_type L, class SC, class Tag>
    template <class E>
    inline auto xarray_container<EC, L, SC, Tag>::operator=(const xexpression<E>& e) -> self_type&
    {
        return semantic_base::operator=(e);
    }
    //@}

    template <class EC, layout_type L, class SC, class Tag>
    inline auto xarray_container<EC, L, SC, Tag>::data_impl() noexcept -> container_type&
    {
        return m_data;
    }

    template <class EC, layout_type L, class SC, class Tag>
    inline auto xarray_container<EC, L, SC, Tag>::data_impl() const noexcept -> const container_type&
    {
        return m_data;
    }

    /******************
     * xarray_adaptor *
     ******************/

    /**
     * @name Constructors
     */
    //@{
    /**
     * Constructs an xarray_adaptor of the given stl-like container.
     * @param data the container to adapt
     */
    template <class EC, layout_type L, class SC, class Tag>
    inline xarray_adaptor<EC, L, SC, Tag>::xarray_adaptor(container_type&& data)
        : base_type(), m_data(std::move(data))
    {
    }

    /**
     * Constructs an xarray_adaptor of the given stl-like container.
     * @param data the container to adapt
     */
    template <class EC, layout_type L, class SC, class Tag>
    inline xarray_adaptor<EC, L, SC, Tag>::xarray_adaptor(const container_type& data)
        : base_type(), m_data(data)
    {
    }

    /**
     * Constructs an xarray_adaptor of the given stl-like container,
     * with the specified shape and layout_type.
     * @param data the container to adapt
     * @param shape the shape of the xarray_adaptor
     * @param l the layout_type of the xarray_adaptor
     */
    template <class EC, layout_type L, class SC, class Tag>
    template <class D>
    inline xarray_adaptor<EC, L, SC, Tag>::xarray_adaptor(D&& data, const shape_type& shape, layout_type l)
        : base_type(), m_data(std::forward<D>(data))
    {
        base_type::resize(shape, l);
    }

    /**
     * Constructs an xarray_adaptor of the given stl-like container,
     * with the specified shape and strides.
     * @param data the container to adapt
     * @param shape the shape of the xarray_adaptor
     * @param strides the strides of the xarray_adaptor
     */
    template <class EC, layout_type L, class SC, class Tag>
    template <class D>
    inline xarray_adaptor<EC, L, SC, Tag>::xarray_adaptor(D&& data, const shape_type& shape, const strides_type& strides)
        : base_type(), m_data(std::forward<D>(data))
    {
        base_type::resize(shape, strides);
    }
    //@}

    template <class EC, layout_type L, class SC, class Tag>
    inline auto xarray_adaptor<EC, L, SC, Tag>::operator=(const xarray_adaptor& rhs) -> self_type&
    {
        base_type::operator=(rhs);
        m_data = rhs.m_data;
        return *this;
    }

    template <class EC, layout_type L, class SC, class Tag>
    inline auto xarray_adaptor<EC, L, SC, Tag>::operator=(xarray_adaptor&& rhs) -> self_type&
    {
        base_type::operator=(std::move(rhs));
        m_data = rhs.m_data;
        return *this;
    }

    template <class EC, layout_type L, class SC, class Tag>
    inline auto xarray_adaptor<EC, L, SC, Tag>::operator=(temporary_type&& rhs) -> self_type&
    {
        base_type::shape_impl() = std::move(const_cast<shape_type&>(rhs.shape()));
        base_type::strides_impl() = std::move(const_cast<strides_type&>(rhs.strides()));
        base_type::backstrides_impl() = std::move(const_cast<backstrides_type&>(rhs.backstrides()));
        m_data = std::move(rhs.data());
        return *this;
    }

    /**
     * @name Extended copy semantic
     */
    //@{
    /**
     * The extended assignment operator.
     */
    template <class EC, layout_type L, class SC, class Tag>
    template <class E>
    inline auto xarray_adaptor<EC, L, SC, Tag>::operator=(const xexpression<E>& e) -> self_type&
    {
        return semantic_base::operator=(e);
    }
    //@}

    template <class EC, layout_type L, class SC, class Tag>
    inline auto xarray_adaptor<EC, L, SC, Tag>::data_impl() noexcept -> container_type&
    {
        return m_data;
    }

    template <class EC, layout_type L, class SC, class Tag>
    inline auto xarray_adaptor<EC, L, SC, Tag>::data_impl() const noexcept -> const container_type&
    {
        return m_data;
    }
}

#endif
