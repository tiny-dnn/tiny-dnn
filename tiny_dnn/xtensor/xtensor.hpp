/***************************************************************************
* Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XTENSOR_TENSOR_HPP
#define XTENSOR_TENSOR_HPP

#include <algorithm>
#include <array>
#include <cstddef>
#include <utility>
#include <vector>

#include "xbuffer_adaptor.hpp"
#include "xcontainer.hpp"
#include "xsemantic.hpp"

namespace xt
{

    /***********************
     * xtensor declaration *
     ***********************/

    template <class EC, std::size_t N, layout_type L, class Tag>
    struct xcontainer_inner_types<xtensor_container<EC, N, L, Tag>>
    {
        using container_type = EC;
        using shape_type = std::array<typename container_type::size_type, N>;
        using strides_type = shape_type;
        using backstrides_type = shape_type;
        using inner_shape_type = shape_type;
        using inner_strides_type = strides_type;
        using inner_backstrides_type = backstrides_type;
        using temporary_type = xtensor_container<EC, N, L, Tag>;
        static constexpr layout_type layout = L;
    };

    template <class EC, std::size_t N, layout_type L, class Tag>
    struct xiterable_inner_types<xtensor_container<EC, N, L, Tag>>
        : xcontainer_iterable_types<xtensor_container<EC, N, L, Tag>>
    {
    };

    /**
     * @class xtensor_container
     * @brief Dense multidimensional container with tensor semantic and fixed
     * dimension.
     *
     * The xtensor_container class implements a dense multidimensional container
     * with tensor semantic and fixed dimension
     *
     * @tparam EC The type of the container holding the elements.
     * @tparam N The dimension of the container.
     * @tparam L The layout_type of the tensor.
     * @tparam Tag The expression tag.
     * @sa xtensor
     */
    template <class EC, size_t N, layout_type L, class Tag>
    class xtensor_container : public xstrided_container<xtensor_container<EC, N, L, Tag>>,
                              public xcontainer_semantic<xtensor_container<EC, N, L, Tag>>
    {
    public:

        using self_type = xtensor_container<EC, N, L, Tag>;
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

        xtensor_container();
        xtensor_container(nested_initializer_list_t<value_type, N> t);
        explicit xtensor_container(const shape_type& shape, layout_type l = L);
        explicit xtensor_container(const shape_type& shape, const_reference value, layout_type l = L);
        explicit xtensor_container(const shape_type& shape, const strides_type& strides);
        explicit xtensor_container(const shape_type& shape, const strides_type& strides, const_reference value);
        explicit xtensor_container(container_type&& data, inner_shape_type&& shape, inner_strides_type&& strides);

        template <class S = shape_type>
        static xtensor_container from_shape(S&& s);

        ~xtensor_container() = default;

        xtensor_container(const xtensor_container&) = default;
        xtensor_container& operator=(const xtensor_container&) = default;

        xtensor_container(xtensor_container&&) = default;
        xtensor_container& operator=(xtensor_container&&) = default;

        template <class E>
        xtensor_container(const xexpression<E>& e);

        template <class E>
        xtensor_container& operator=(const xexpression<E>& e);

    private:

        container_type m_data;

        container_type& data_impl() noexcept;
        const container_type& data_impl() const noexcept;

        friend class xcontainer<xtensor_container<EC, N, L, Tag>>;
    };

    /*****************************************
     * xtensor_container_adaptor declaration *
     *****************************************/

    template <class EC, std::size_t N, layout_type L, class Tag>
    struct xcontainer_inner_types<xtensor_adaptor<EC, N, L, Tag>>
    {
        using container_type = std::remove_reference_t<EC>;
        using shape_type = std::array<typename container_type::size_type, N>;
        using strides_type = shape_type;
        using backstrides_type = shape_type;
        using inner_shape_type = shape_type;
        using inner_strides_type = strides_type;
        using inner_backstrides_type = backstrides_type;
        using temporary_type = xtensor_container<temporary_container_t<container_type>, N, L, Tag>;
        static constexpr layout_type layout = L;
    };

    template <class EC, std::size_t N, layout_type L>
    struct xiterable_inner_types<xtensor_adaptor<EC, N, L>>
        : xcontainer_iterable_types<xtensor_adaptor<EC, N, L>>
    {
    };

    /**
     * @class xtensor_adaptor
     * @brief Dense multidimensional container adaptor with tensor semantic
     * and fixed dimension.
     *
     * The xtensor_adaptor class implements a dense multidimensional
     * container adaptor with tensor semantic and fixed dimension. It
     * is used to provide a multidimensional container semantic and a
     * tensor semantic to stl-like containers.
     *
     * @tparam EC The closure for the container type to adapt.
     * @tparam N The dimension of the adaptor.
     * @tparam L The layout_type of the adaptor.
     * @tparam Tag The expression tag.
     */
    template <class EC, std::size_t N, layout_type L, class Tag>
    class xtensor_adaptor : public xstrided_container<xtensor_adaptor<EC, N, L, Tag>>,
                            public xcontainer_semantic<xtensor_adaptor<EC, N, L, Tag>>
    {
    public:

        using container_closure_type = EC;

        using self_type = xtensor_adaptor<EC, N, L, Tag>;
        using base_type = xstrided_container<self_type>;
        using semantic_base = xcontainer_semantic<self_type>;
        using container_type = typename base_type::container_type;
        using allocator_type = typename base_type::allocator_type;
        using shape_type = typename base_type::shape_type;
        using strides_type = typename base_type::strides_type;
        using backstrides_type = typename base_type::backstrides_type;
        using temporary_type = typename semantic_base::temporary_type;
        using expression_tag = Tag;

        xtensor_adaptor(container_type&& data);
        xtensor_adaptor(const container_type& data);

        template <class D>
        xtensor_adaptor(D&& data, const shape_type& shape, layout_type l = L);

        template <class D>
        xtensor_adaptor(D&& data, const shape_type& shape, const strides_type& strides);

        ~xtensor_adaptor() = default;

        xtensor_adaptor(const xtensor_adaptor&) = default;
        xtensor_adaptor& operator=(const xtensor_adaptor&);

        xtensor_adaptor(xtensor_adaptor&&) = default;
        xtensor_adaptor& operator=(xtensor_adaptor&&);
        xtensor_adaptor& operator=(temporary_type&&);

        template <class E>
        xtensor_adaptor& operator=(const xexpression<E>& e);

    private:

        container_closure_type m_data;

        container_type& data_impl() noexcept;
        const container_type& data_impl() const noexcept;

        friend class xcontainer<xtensor_adaptor<EC, N, L, Tag>>;
    };

    /************************************
     * xtensor_container implementation *
     ************************************/

    /**
     * @name Constructors
     */
    //@{
    /**
     * Allocates an uninitialized xtensor_container that holds 0 element.
     */
    template <class EC, std::size_t N, layout_type L, class Tag>
    inline xtensor_container<EC, N, L, Tag>::xtensor_container()
        : base_type(), m_data(1, value_type())
    {
    }

    /**
     * Allocates an xtensor_container with nested initializer lists.
     */
    template <class EC, std::size_t N, layout_type L, class Tag>
    inline xtensor_container<EC, N, L, Tag>::xtensor_container(nested_initializer_list_t<value_type, N> t)
        : base_type()
    {
        base_type::resize(xt::shape<shape_type>(t), true);
        L == layout_type::row_major ? nested_copy(m_data.begin(), t) : nested_copy(this->template begin<layout_type::row_major>(), t);
    }

    /**
     * Allocates an uninitialized xtensor_container with the specified shape and
     * layout_type.
     * @param shape the shape of the xtensor_container
     * @param l the layout_type of the xtensor_container
     */
    template <class EC, std::size_t N, layout_type L, class Tag>
    inline xtensor_container<EC, N, L, Tag>::xtensor_container(const shape_type& shape, layout_type l)
        : base_type()
    {
        base_type::resize(shape, l);
    }

    /**
     * Allocates an xtensor_container with the specified shape and layout_type. Elements
     * are initialized to the specified value.
     * @param shape the shape of the xtensor_container
     * @param value the value of the elements
     * @param l the layout_type of the xtensor_container
     */
    template <class EC, std::size_t N, layout_type L, class Tag>
    inline xtensor_container<EC, N, L, Tag>::xtensor_container(const shape_type& shape, const_reference value, layout_type l)
        : base_type()
    {
        base_type::resize(shape, l);
        std::fill(m_data.begin(), m_data.end(), value);
    }

    /**
     * Allocates an uninitialized xtensor_container with the specified shape and strides.
     * @param shape the shape of the xtensor_container
     * @param strides the strides of the xtensor_container
     */
    template <class EC, std::size_t N, layout_type L, class Tag>
    inline xtensor_container<EC, N, L, Tag>::xtensor_container(const shape_type& shape, const strides_type& strides)
        : base_type()
    {
        base_type::resize(shape, strides);
    }

    /**
     * Allocates an uninitialized xtensor_container with the specified shape and strides.
     * Elements are initialized to the specified value.
     * @param shape the shape of the xtensor_container
     * @param strides the strides of the xtensor_container
     * @param value the value of the elements
     */
    template <class EC, std::size_t N, layout_type L, class Tag>
    inline xtensor_container<EC, N, L, Tag>::xtensor_container(const shape_type& shape, const strides_type& strides, const_reference value)
        : base_type()
    {
        base_type::resize(shape, strides);
        std::fill(m_data.begin(), m_data.end(), value);
    }

    /**
     * Allocates an xtensor_container by moving specified data, shape and strides
     *
     * @param data the data for the xtensor_container
     * @param shape the shape of the xtensor_container
     * @param strides the strides of the xtensor_container
     */
    template <class EC, std::size_t N, layout_type L, class Tag>
    inline xtensor_container<EC, N, L, Tag>::xtensor_container(container_type&& data, inner_shape_type&& shape, inner_strides_type&& strides)
        : base_type(std::move(shape), std::move(strides)), m_data(std::move(data))
    {
    }

    template <class EC, std::size_t N, layout_type L, class Tag>
    template <class S>
    inline xtensor_container<EC, N, L, Tag> xtensor_container<EC, N, L, Tag>::from_shape(S&& s)
    {
        if (s.size() != N)
        {
            throw std::runtime_error("Cannot change dimension of xtensor.");
        }
        shape_type shape = xtl::forward_sequence<shape_type>(s);
        return self_type(shape);
    }
    //@}

    /**
     * @name Extended copy semantic
     */
    //@{
    /**
     * The extended copy constructor.
     */
    template <class EC, std::size_t N, layout_type L, class Tag>
    template <class E>
    inline xtensor_container<EC, N, L, Tag>::xtensor_container(const xexpression<E>& e)
        : base_type()
    {
        // Avoids unintialized data because of (m_shape == shape) condition
        // in resize (called by assign), which is always true when size() == 1.
        // The condition dimension() == 0 as in xarray is not sufficient because
        // the shape is always initialized since it has a static number of dimensions.
        if (e.derived_cast().size() == 1)
        {
            detail::resize_data_container(m_data, std::size_t(1));
        }
        semantic_base::assign(e);
    }

    /**
     * The extended assignment operator.
     */
    template <class EC, std::size_t N, layout_type L, class Tag>
    template <class E>
    inline auto xtensor_container<EC, N, L, Tag>::operator=(const xexpression<E>& e) -> self_type&
    {
        return semantic_base::operator=(e);
    }
    //@}

    template <class EC, std::size_t N, layout_type L, class Tag>
    inline auto xtensor_container<EC, N, L, Tag>::data_impl() noexcept -> container_type&
    {
        return m_data;
    }

    template <class EC, std::size_t N, layout_type L, class Tag>
    inline auto xtensor_container<EC, N, L, Tag>::data_impl() const noexcept -> const container_type&
    {
        return m_data;
    }

    /*******************
     * xtensor_adaptor *
     *******************/

    /**
     * @name Constructors
     */
    //@{
    /**
     * Constructs an xtensor_adaptor of the given stl-like container.
     * @param data the container to adapt
     */
    template <class EC, std::size_t N, layout_type L, class Tag>
    inline xtensor_adaptor<EC, N, L, Tag>::xtensor_adaptor(container_type&& data)
        : base_type(), m_data(std::move(data))
    {
    }

    /**
     * Constructs an xtensor_adaptor of the given stl-like container.
     * @param data the container to adapt
     */
    template <class EC, std::size_t N, layout_type L, class Tag>
    inline xtensor_adaptor<EC, N, L, Tag>::xtensor_adaptor(const container_type& data)
        : base_type(), m_data(data)
    {
    }

    /**
     * Constructs an xtensor_adaptor of the given stl-like container,
     * with the specified shape and layout_type.
     * @param data the container to adapt
     * @param shape the shape of the xtensor_adaptor
     * @param l the layout_type of the xtensor_adaptor
     */
    template <class EC, std::size_t N, layout_type L, class Tag>
    template <class D>
    inline xtensor_adaptor<EC, N, L, Tag>::xtensor_adaptor(D&& data, const shape_type& shape, layout_type l)
        : base_type(), m_data(std::forward<D>(data))
    {
        base_type::resize(shape, l);
    }

    /**
     * Constructs an xtensor_adaptor of the given stl-like container,
     * with the specified shape and strides.
     * @param data the container to adapt
     * @param shape the shape of the xtensor_adaptor
     * @param strides the strides of the xtensor_adaptor
     */
    template <class EC, std::size_t N, layout_type L, class Tag>
    template <class D>
    inline xtensor_adaptor<EC, N, L, Tag>::xtensor_adaptor(D&& data, const shape_type& shape, const strides_type& strides)
        : base_type(), m_data(std::forward<D>(data))
    {
        base_type::resize(shape, strides);
    }
    //@}

    template <class EC, std::size_t N, layout_type L, class Tag>
    inline auto xtensor_adaptor<EC, N, L, Tag>::operator=(const xtensor_adaptor& rhs) -> self_type&
    {
        base_type::operator=(rhs);
        m_data = rhs.m_data;
        return *this;
    }

    template <class EC, std::size_t N, layout_type L, class Tag>
    inline auto xtensor_adaptor<EC, N, L, Tag>::operator=(xtensor_adaptor&& rhs) -> self_type&
    {
        base_type::operator=(std::move(rhs));
        m_data = rhs.m_data;
        return *this;
    }

    template <class EC, std::size_t N, layout_type L, class Tag>
    inline auto xtensor_adaptor<EC, N, L, Tag>::operator=(temporary_type&& rhs) -> self_type&
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
    template <class EC, std::size_t N, layout_type L, class Tag>
    template <class E>
    inline auto xtensor_adaptor<EC, N, L, Tag>::operator=(const xexpression<E>& e) -> self_type&
    {
        return semantic_base::operator=(e);
    }
    //@}

    template <class EC, std::size_t N, layout_type L, class Tag>
    inline auto xtensor_adaptor<EC, N, L, Tag>::data_impl() noexcept -> container_type&
    {
        return m_data;
    }

    template <class EC, std::size_t N, layout_type L, class Tag>
    inline auto xtensor_adaptor<EC, N, L, Tag>::data_impl() const noexcept -> const container_type&
    {
        return m_data;
    }
}

#endif
