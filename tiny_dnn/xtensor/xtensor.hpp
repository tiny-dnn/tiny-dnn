/***************************************************************************
* Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XTENSOR_HPP
#define XTENSOR_HPP

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

    template <class EC, std::size_t N, layout_type L>
    struct xcontainer_inner_types<xtensor_container<EC, N, L>>
    {
        using container_type = EC;
        using shape_type = std::array<typename container_type::size_type, N>;
        using strides_type = shape_type;
        using backstrides_type = shape_type;
        using inner_shape_type = shape_type;
        using inner_strides_type = strides_type;
        using inner_backstrides_type = backstrides_type;
        using temporary_type = xtensor_container<EC, N, L>;
    };

    template <class EC, std::size_t N, layout_type L>
    struct xiterable_inner_types<xtensor_container<EC, N, L>>
        : xcontainer_iterable_types<xtensor_container<EC, N, L>>
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
     * @sa xtensor
     */
    template <class EC, size_t N, layout_type L>
    class xtensor_container : public xstrided_container<xtensor_container<EC, N, L>, L>,
                              public xcontainer_semantic<xtensor_container<EC, N, L>>
    {
    public:

        using self_type = xtensor_container<EC, N, L>;
        using base_type = xstrided_container<self_type, L>;
        using semantic_base = xcontainer_semantic<self_type>;
        using container_type = typename base_type::container_type;
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

        friend class xcontainer<xtensor_container<EC, N, L>>;
    };

    /*****************************************
     * xtensor_container_adaptor declaration *
     *****************************************/

    template <class EC, std::size_t N, layout_type L = DEFAULT_LAYOUT>
    class xtensor_adaptor;

    template <class EC, std::size_t N, layout_type L>
    struct xcontainer_inner_types<xtensor_adaptor<EC, N, L>>
    {
        using container_type = EC;
        using shape_type = std::array<typename container_type::size_type, N>;
        using strides_type = shape_type;
        using backstrides_type = shape_type;
        using inner_shape_type = shape_type;
        using inner_strides_type = strides_type;
        using inner_backstrides_type = backstrides_type;
        using temporary_type = xtensor_container<EC, N, L>;
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
     * @tparam EC The container type to adapt.
     * @tparam N The dimension of the adaptor.
     * @tparam L The layout_type of the adaptor.
     */
    template <class EC, std::size_t N, layout_type L>
    class xtensor_adaptor : public xstrided_container<xtensor_adaptor<EC, N, L>, L>,
                            public xadaptor_semantic<xtensor_adaptor<EC, N, L>>
    {
    public:

        using self_type = xtensor_adaptor<EC, N, L>;
        using base_type = xstrided_container<self_type, L>;
        using semantic_base = xadaptor_semantic<self_type>;
        using container_type = typename base_type::container_type;
        using shape_type = typename base_type::shape_type;
        using strides_type = typename base_type::strides_type;
        using backstrides_type = typename base_type::backstrides_type;

        using container_closure_type = adaptor_closure_t<container_type>;

        xtensor_adaptor(container_closure_type data);
        xtensor_adaptor(container_closure_type data, const shape_type& shape, layout_type l = layout_type::row_major);
        xtensor_adaptor(container_closure_type data, const shape_type& shape, const strides_type& strides);

        ~xtensor_adaptor() = default;

        xtensor_adaptor(const xtensor_adaptor&) = default;
        xtensor_adaptor& operator=(const xtensor_adaptor&);

        xtensor_adaptor(xtensor_adaptor&&) = default;
        xtensor_adaptor& operator=(xtensor_adaptor&&);

        template <class E>
        xtensor_adaptor& operator=(const xexpression<E>& e);

    private:

        container_closure_type m_data;

        container_type& data_impl() noexcept;
        const container_type& data_impl() const noexcept;

        using temporary_type = typename xcontainer_inner_types<self_type>::temporary_type;
        void assign_temporary_impl(temporary_type&& tmp);

        friend class xcontainer<xtensor_adaptor<EC, N, L>>;
        friend class xadaptor_semantic<xtensor_adaptor<EC, N, L>>;
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
    template <class EC, std::size_t N, layout_type L>
    inline xtensor_container<EC, N, L>::xtensor_container()
        : base_type(), m_data(1, value_type())
    {
    }

    /**
     * Allocates an xtensor_container with nested initializer lists.
     */
    template <class EC, std::size_t N, layout_type L>
    inline xtensor_container<EC, N, L>::xtensor_container(nested_initializer_list_t<value_type, N> t)
        : base_type()
    {
        base_type::reshape(xt::shape<shape_type>(t), true);
        L == layout_type::row_major ? nested_copy(m_data.begin(), t) : nested_copy(this->template xbegin<layout_type::row_major>(), t);
    }

    /**
     * Allocates an uninitialized xtensor_container with the specified shape and
     * layout_type.
     * @param shape the shape of the xtensor_container
     * @param l the layout_type of the xtensor_container
     */
    template <class EC, std::size_t N, layout_type L>
    inline xtensor_container<EC, N, L>::xtensor_container(const shape_type& shape, layout_type l)
        : base_type()
    {
        base_type::reshape(shape, l);
    }

    /**
     * Allocates an xtensor_container with the specified shape and layout_type. Elements
     * are initialized to the specified value.
     * @param shape the shape of the xtensor_container
     * @param value the value of the elements
     * @param l the layout_type of the xtensor_container
     */
    template <class EC, std::size_t N, layout_type L>
    inline xtensor_container<EC, N, L>::xtensor_container(const shape_type& shape, const_reference value, layout_type l)
        : base_type()
    {
        base_type::reshape(shape, l);
        std::fill(m_data.begin(), m_data.end(), value);
    }

    /**
     * Allocates an uninitialized xtensor_container with the specified shape and strides.
     * @param shape the shape of the xtensor_container
     * @param strides the strides of the xtensor_container
     */
    template <class EC, std::size_t N, layout_type L>
    inline xtensor_container<EC, N, L>::xtensor_container(const shape_type& shape, const strides_type& strides)
        : base_type()
    {
        base_type::reshape(shape, strides);
    }

    /**
     * Allocates an uninitialized xtensor_container with the specified shape and strides.
     * Elements are initialized to the specified value.
     * @param shape the shape of the xtensor_container
     * @param strides the strides of the xtensor_container
     * @param value the value of the elements
     */
    template <class EC, std::size_t N, layout_type L>
    inline xtensor_container<EC, N, L>::xtensor_container(const shape_type& shape, const strides_type& strides, const_reference value)
        : base_type()
    {
        base_type::reshape(shape, strides);
        std::fill(m_data.begin(), m_data.end(), value);
    }

    /**
     * Allocates an xtensor_container by moving specified data, shape and strides
     *
     * @param data the data for the xtensor_container
     * @param shape the shape of the xtensor_container
     * @param strides the strides of the xtensor_container
     */
    template <class EC, std::size_t N, layout_type L>
    inline xtensor_container<EC, N, L>::xtensor_container(container_type&& data, inner_shape_type&& shape, inner_strides_type&& strides)
        : base_type(std::move(shape), std::move(strides)), m_data(std::move(data))
    {
    }

    template <class EC, std::size_t N, layout_type L>
    template <class S>
    inline xtensor_container<EC, N, L> xtensor_container<EC, N, L>::from_shape(S&& s)
    {
        if (s.size() != N)
        {
            throw std::runtime_error("Cannot change dimension of xtensor.");
        }
        shape_type shape = forward_sequence<shape_type>(s);
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
    template <class EC, std::size_t N, layout_type L>
    template <class E>
    inline xtensor_container<EC, N, L>::xtensor_container(const xexpression<E>& e)
        : base_type()
    {
        // Avoids unintialized data because of (m_shape == shape) condition
        // in reshape (called by assign), which is always true when dimension == 0.
        if (e.derived_cast().dimension() == 0)
        {
            m_data.resize(1);
        }
        semantic_base::assign(e);
    }

    /**
     * The extended assignment operator.
     */
    template <class EC, std::size_t N, layout_type L>
    template <class E>
    inline auto xtensor_container<EC, N, L>::operator=(const xexpression<E>& e) -> self_type&
    {
        return semantic_base::operator=(e);
    }
    //@}

    template <class EC, std::size_t N, layout_type L>
    inline auto xtensor_container<EC, N, L>::data_impl() noexcept -> container_type&
    {
        return m_data;
    }

    template <class EC, std::size_t N, layout_type L>
    inline auto xtensor_container<EC, N, L>::data_impl() const noexcept -> const container_type&
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
    template <class EC, std::size_t N, layout_type L>
    inline xtensor_adaptor<EC, N, L>::xtensor_adaptor(container_closure_type data)
        : base_type(), m_data(std::forward<container_closure_type>(data))
    {
    }

    /**
     * Constructs an xtensor_adaptor of the given stl-like container,
     * with the specified shape and layout_type.
     * @param data the container to adapt
     * @param shape the shape of the xtensor_adaptor
     * @param l the layout_type of the xtensor_adaptor
     */
    template <class EC, std::size_t N, layout_type L>
    inline xtensor_adaptor<EC, N, L>::xtensor_adaptor(container_closure_type data, const shape_type& shape, layout_type l)
        : base_type(), m_data(std::forward<container_closure_type>(data))
    {
        base_type::reshape(shape, l);
    }

    /**
     * Constructs an xtensor_adaptor of the given stl-like container,
     * with the specified shape and strides.
     * @param data the container to adapt
     * @param shape the shape of the xtensor_adaptor
     * @param strides the strides of the xtensor_adaptor
     */
    template <class EC, std::size_t N, layout_type L>
    inline xtensor_adaptor<EC, N, L>::xtensor_adaptor(container_closure_type data, const shape_type& shape, const strides_type& strides)
        : base_type(), m_data(std::forward<container_closure_type>(data))
    {
        base_type::reshape(shape, strides);
    }
    //@}

    template <class EC, std::size_t N, layout_type L>
    inline auto xtensor_adaptor<EC, N, L>::operator=(const xtensor_adaptor& rhs) -> self_type&
    {
        base_type::operator=(rhs);
        m_data = rhs.m_data;
        return *this;
    }

    template <class EC, std::size_t N, layout_type L>
    inline auto xtensor_adaptor<EC, N, L>::operator=(xtensor_adaptor&& rhs) -> self_type&
    {
        base_type::operator=(std::move(rhs));
        m_data = rhs.m_data;
        return *this;
    }

    /**
     * @name Extended copy semantic
     */
    //@{
    /**
     * The extended assignment operator.
     */
    template <class EC, std::size_t N, layout_type L>
    template <class E>
    inline auto xtensor_adaptor<EC, N, L>::operator=(const xexpression<E>& e) -> self_type&
    {
        return semantic_base::operator=(e);
    }
    //@}

    template <class EC, std::size_t N, layout_type L>
    inline auto xtensor_adaptor<EC, N, L>::data_impl() noexcept -> container_type&
    {
        return m_data;
    }

    template <class EC, std::size_t N, layout_type L>
    inline auto xtensor_adaptor<EC, N, L>::data_impl() const noexcept -> const container_type&
    {
        return m_data;
    }

    template <class EC, std::size_t N, layout_type L>
    inline void xtensor_adaptor<EC, N, L>::assign_temporary_impl(temporary_type&& tmp)
    {
        base_type::shape_impl() = std::move(const_cast<shape_type&>(tmp.shape()));
        base_type::strides_impl() = std::move(const_cast<strides_type&>(tmp.strides()));
        base_type::backstrides_impl() = std::move(const_cast<backstrides_type&>(tmp.backstrides()));
        m_data = std::move(tmp.data());
    }
}

#endif
