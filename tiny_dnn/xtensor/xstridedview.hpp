/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XSTRIDEDVIEW_HPP
#define XSTRIDEDVIEW_HPP

#include <algorithm>
#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

#include "xexpression.hpp"
#include "xiterable.hpp"
#include "xstrides.hpp"
#include "xutils.hpp"
#include "xview.hpp"

namespace xt
{
    template <class CT, class CD>
    class xstrided_view;

    template <class CT, class CD>
    struct xcontainer_inner_types<xstrided_view<CT, CD>>
    {
        using xexpression_type = std::decay_t<CT>;
        using temporary_type = xarray<typename xexpression_type::value_type>;
    };

    template <class CT, class CD>
    struct xiterable_inner_types<xstrided_view<CT, CD>>
    {
        using inner_shape_type = typename std::decay_t<CT>::shape_type;
        using inner_strides_type = inner_shape_type;
        using inner_backstrides_type_type = inner_shape_type;
        using const_stepper = xindexed_stepper<xstrided_view<CT, CD>>;
        using stepper = xindexed_stepper<xstrided_view<CT, CD>, false>;
        using const_broadcast_iterator = xiterator<const_stepper, inner_shape_type*>;
        using broadcast_iterator = xiterator<stepper, inner_shape_type*>;
        using const_iterator = const_broadcast_iterator;
        using iterator = broadcast_iterator;
    };

    /*****************
     * xstrided_view *
     *****************/

    /**
     * @class xstrided_view
     * @brief View of an xexpression using strides
     *
     * The xstrided_view class implements a view utilizing an offset and strides 
     * into a multidimensional xcontainer. The xstridedview is currently used 
     * to implement `transpose`.
     * @tparam CT the closure type of the \ref xexpression type underlying this view
     * @tparam CD the closure type of the underlying data container
     * 
     * @sa stridedview, transpose
     */
    template <class CT, class CD>
    class xstrided_view : public xview_semantic<xstrided_view<CT, CD>>,
                          public xexpression_iterable<xstrided_view<CT, CD>>
    {

    public:

        using self_type = xstrided_view<CT, CD>;
        using xexpression_type = std::decay_t<CT>;
        using semantic_base = xview_semantic<self_type>;

        using value_type = typename xexpression_type::value_type;
        using reference = typename xexpression_type::reference;
        using const_reference = typename xexpression_type::const_reference;
        using pointer = typename xexpression_type::pointer;
        using const_pointer = typename xexpression_type::const_pointer;
        using size_type = typename xexpression_type::size_type;
        using difference_type = typename xexpression_type::difference_type;

        using underlying_container_type = CD;

        using iterable_base = xexpression_iterable<self_type>;
        using inner_shape_type = typename iterable_base::inner_shape_type;
        using shape_type = inner_shape_type;
        using strides_type = shape_type;
        using backstrides_type = shape_type;
        using closure_type = const self_type;

        using stepper = typename iterable_base::stepper;
        using const_stepper = typename iterable_base::const_stepper;

        using broadcast_iterator = typename iterable_base::broadcast_iterator;
        using const_broadcast_iterator = typename iterable_base::const_broadcast_iterator;

        using iterator = typename iterable_base::iterator;
        using const_iterator = typename iterable_base::const_iterator;

        using temporary_type = typename xcontainer_inner_types<self_type>::temporary_type;
        using base_index_type = xindex_type_t<shape_type>;

        template <class I>
        xstrided_view(CT e, I&& shape, I&& strides, std::size_t offset) noexcept;

        template <class E>
        self_type& operator=(const xexpression<E>& e);

        template <class E>
        disable_xexpression<E, self_type>& operator=(const E& e);

        size_type size() const noexcept;
        size_type dimension() const noexcept;
        const shape_type& shape() const noexcept;
        const strides_type& strides() const noexcept;
        const backstrides_type& backstrides() const noexcept;

        reference operator()();
        template <class... Args>
        reference operator()(Args... args);
        reference operator[](const xindex& index);
        reference operator[](size_type i);

        template <class It>
        reference element(It first, It last);

        const_reference operator()() const;
        template <class... Args>
        const_reference operator()(Args... args) const;
        const_reference operator[](const xindex& index) const;
        const_reference operator[](size_type i) const;

        template <class It>
        const_reference element(It first, It last) const;

        template <class O>
        bool broadcast_shape(O& shape) const;

        template <class O>
        bool is_trivial_broadcast(const O& strides) const noexcept;

        template <class ST>
        stepper stepper_begin(const ST& shape);
        template <class ST>
        stepper stepper_end(const ST& shape);

        template <class ST>
        const_stepper stepper_begin(const ST& shape) const;
        template <class ST>
        const_stepper stepper_end(const ST& shape) const;

        underlying_container_type& data() noexcept;
        const underlying_container_type& data() const noexcept;

        value_type* raw_data() noexcept;
        const value_type* raw_data() const noexcept;

        size_type raw_data_offset() const noexcept;

    private:

        CT m_e;
        CD m_data;
        shape_type m_shape;
        strides_type m_strides;
        backstrides_type m_backstrides;
        std::size_t m_offset;

        void assign_temporary_impl(temporary_type& tmp);

        friend class xview_semantic<xstrided_view<CT, CD>>;
    };

    /*****************************
     * xstrided_view implementation *
     *****************************/

    /**
     * @name Constructor
     */
    //@{
    /**
     * Constructs an xstrided_view, selecting the indices specified by \a indices.
     * The resulting xexpression has a 1D shape with a length of n for n indices.
     * 
     * @param e the underlying xexpression for this view
     * @param indices the indices to select
     */
    template <class CT, class CD>
    template <class I>
    inline xstrided_view<CT, CD>::xstrided_view(CT e, I&& shape, I&& strides, std::size_t offset) noexcept
        : m_e(e), m_data(m_e.data()), m_shape(std::forward<I>(shape)), m_strides(std::forward<I>(strides)), m_offset(offset)
    {
        m_backstrides = make_sequence<backstrides_type>(m_shape.size(), 0);
        adapt_strides(m_shape, m_strides, m_backstrides);
    }
    //@}

    /**
     * @name Extended copy semantic
     */
    //@{
    /**
     * The extended assignment operator.
     */
    template <class CT, class CD>
    template <class E>
    inline auto xstrided_view<CT, CD>::operator=(const xexpression<E>& e) -> self_type&
    {
        return semantic_base::operator=(e);
    }
    //@}

    template <class CT, class CD>
    template <class E>
    inline auto xstrided_view<CT, CD>::operator=(const E& e) -> disable_xexpression<E, self_type>&
    {
        std::fill(this->begin(), this->end(), e);
        return *this;
    }

    template <class CT, class CD>
    inline void xstrided_view<CT, CD>::assign_temporary_impl(temporary_type& tmp)
    {
        std::copy(tmp.cbegin(), tmp.cend(), this->xbegin());
    }

    /**
     * @name Size and shape
     */
    //@{
    /**
     * Returns the size of the xstrided_view.
     */
    template <class CT, class CD>
    inline auto xstrided_view<CT, CD>::size() const noexcept -> size_type
    {
        return compute_size(shape());
    }

    /**
     * Returns the number of dimensions of the xstrided_view.
     */
    template <class CT, class CD>
    inline auto xstrided_view<CT, CD>::dimension() const noexcept -> size_type
    {
        return m_shape.size();
    }

    /**
     * Returns the shape of the xstrided_view.
     */
    template <class CT, class CD>
    inline auto xstrided_view<CT, CD>::shape() const noexcept -> const shape_type&
    {
        return m_shape;
    }

    template <class CT, class CD>
    inline auto xstrided_view<CT, CD>::strides() const noexcept -> const strides_type&
    {
        return m_strides;
    }

    template <class CT, class CD>
    inline auto xstrided_view<CT, CD>::backstrides() const noexcept -> const backstrides_type&
    {
        return m_backstrides;
    }

    template <class CT, class CD>
    inline auto xstrided_view<CT, CD>::data() noexcept -> underlying_container_type&
    {
        return m_e.data();
    }

    template <class CT, class CD>
    inline auto xstrided_view<CT, CD>::data() const noexcept -> const underlying_container_type&
    {
        return m_e.data();
    }

    template <class CT, class CD>
    inline auto xstrided_view<CT, CD>::raw_data() noexcept -> value_type*
    {
        return m_e.raw_data();
    }

    template <class CT, class CD>
    inline auto xstrided_view<CT, CD>::raw_data() const noexcept -> const value_type*
    {
        return m_e.raw_data();
    }

    template <class CT, class CD>
    inline auto xstrided_view<CT, CD>::raw_data_offset() const noexcept -> size_type
    {
        return m_offset;
    }
    //@}

    /**
     * @name Data
     */
    template <class CT, class CD>
    inline auto xstrided_view<CT, CD>::operator()() -> reference
    {
        return m_e();
    }

    template <class CT, class CD>
    inline auto xstrided_view<CT, CD>::operator()() const -> const_reference
    {
        return m_e();
    }

    template <class CT, class CD>
    template <class... Args>
    inline auto xstrided_view<CT, CD>::operator()(Args... args) -> reference
    {
        XTENSOR_ASSERT(check_index(shape(), args...));
        size_type index = m_offset + data_offset<size_type>(strides(), static_cast<size_type>(args)...);
        return m_data[index];
    }

    /**
     * Returns the element at the specified position in the xstrided_view. 
     * 
     * @param idx the position in the view
     */
    template <class CT, class CD>
    template <class... Args>
    inline auto xstrided_view<CT, CD>::operator()(Args... args) const -> const_reference
    {
        XTENSOR_ASSERT(check_index(shape(), args...));
        size_type index = m_offset + data_offset<size_type>(strides(), static_cast<size_type>(args)...);
        return m_data[index];
    }

    template <class CT, class CD>
    inline auto xstrided_view<CT, CD>::operator[](const xindex& index) -> reference
    {
        return element(index.cbegin(), index.cend());
    }

    template <class CT, class CD>
    inline auto xstrided_view<CT, CD>::operator[](size_type i) -> reference
    {
        return operator()(i);
    }

    template <class CT, class CD>
    inline auto xstrided_view<CT, CD>::operator[](const xindex& index) const -> const_reference
    {
        return element(index.cbegin(), index.cend());
    }

    template <class CT, class CD>
    inline auto xstrided_view<CT, CD>::operator[](size_type i) const -> const_reference
    {
        return operator()(i);
    }

    /**
     * Returns a reference to the element at the specified position in the xstrided_view.
     * @param first iterator starting the sequence of indices
     * The number of indices in the squence should be equal to or greater 1.
     */
    template <class CT, class CD>
    template <class It>
    inline auto xstrided_view<CT, CD>::element(It first, It last) -> reference
    {
        return m_data[m_offset + element_offset<size_type>(strides(), first, last)];
    }

    template <class CT, class CD>
    template <class It>
    inline auto xstrided_view<CT, CD>::element(It first, It last) const -> const_reference
    {
        return m_data[m_offset + element_offset<size_type>(strides(), first, last)];
    }
    //@}

    /**
     * @name Broadcasting
     */
    //@{
    /**
     * Broadcast the shape of the xstrided_view to the specified parameter.
     * @param shape the result shape
     * @return a boolean indicating whether the broadcasting is trivial
     */
    template <class CT, class CD>
    template <class O>
    inline bool xstrided_view<CT, CD>::broadcast_shape(O& shape) const
    {
        return xt::broadcast_shape(m_shape, shape);
    }

    /**
     * Compares the specified strides with those of the container to see whether
     * the broadcasting is trivial.
     * @return a boolean indicating whether the broadcasting is trivial
     */
    template <class CT, class CD>
    template <class O>
    inline bool xstrided_view<CT, CD>::is_trivial_broadcast(const O& str) const noexcept
    {
        return str.size() == strides().size() &&
            std::equal(str.cbegin(), str.cend(), strides().begin());
    }
    //@}

    /***************
     * stepper api *
     ***************/

    template <class CT, class CD>
    template <class ST>
    inline auto xstrided_view<CT, CD>::stepper_begin(const ST& shape) -> stepper
    {
        size_type offset = shape.size() - dimension();
        return stepper(this, offset);
    }

    template <class CT, class CD>
    template <class ST>
    inline auto xstrided_view<CT, CD>::stepper_end(const ST& shape) -> stepper
    {
        size_type offset = shape.size() - dimension();
        return stepper(this, offset, true);
    }

    template <class CT, class CD>
    template <class ST>
    inline auto xstrided_view<CT, CD>::stepper_begin(const ST& shape) const -> const_stepper
    {
        size_type offset = shape.size() - dimension();
        return const_stepper(this, offset);
    }

    template <class CT, class CD>
    template <class ST>
    inline auto xstrided_view<CT, CD>::stepper_end(const ST& shape) const -> const_stepper
    {
        size_type offset = shape.size() - dimension();
        return const_stepper(this, offset, true);
    }

    /**
     * Construct a strided view from an xexpression, shape, strides and offset.
     *
     * @param e xexpression
     * @param shape the shape of the view
     * @param strides the new strides of the view
     * @param offset the offset of the first element in the underlying container
     *
     * @tparam E type of xexpression
     * @tparam I shape and strides type
     *
     * @return the view
     */
    template <class E, class I>
    inline auto strided_view(E&& e, I&& shape, I&& strides, std::size_t offset = 0) noexcept
    {
        using view_type = xstrided_view<xclosure_t<E>, decltype(e.data())>;
        return view_type(std::forward<E>(e), std::forward<I>(shape), std::forward<I>(strides), offset);
    }

    /****************************
     * transpose implementation *
     ****************************/

    namespace detail
    {
        template <class E, class S>
        inline auto transpose_impl(E&& e, S&& permutation, check_policy::none)
        {
            if (container_size(permutation) != e.dimension())
            {
                throw transpose_error("Permutation does not have the same size as shape");
            }

            // permute stride and shape
            using strides_type = typename std::decay_t<E>::strides_type;
            strides_type temp_strides;
            resize_container(temp_strides, e.strides().size());

            using shape_type = typename std::decay_t<E>::shape_type;
            shape_type temp_shape;
            resize_container(temp_shape, e.shape().size());

            for (std::size_t i = 0; i < e.shape().size(); ++i)
            {
                if (std::size_t(permutation[i]) >= e.dimension())
                {
                    throw transpose_error("Permutation contains wrong axis");
                }
                temp_shape[i] = e.shape()[permutation[i]];
                temp_strides[i] = e.strides()[permutation[i]];
            }
            using view_type = xstrided_view<xclosure_t<E>, decltype(e.data())>;
            return view_type(std::forward<E>(e), std::move(temp_shape), std::move(temp_strides), 0);
        }

        template <class E, class S>
        inline auto transpose_impl(E&& e, S&& permutation, check_policy::full)
        {
            // check if axis appears twice in permutation
            for (std::size_t i = 0; i < container_size(permutation); ++i)
            {
                for (std::size_t j = i + 1; j < container_size(permutation); ++j)
                {
                    if (permutation[i] == permutation[j])
                    {
                        throw transpose_error("Permutation contains axis more than once");
                    }
                }
            }
            return transpose_impl(std::forward<E>(e), std::forward<S>(permutation), check_policy::none());
        }
    }

    template <class E>
    inline auto transpose(E&& e) noexcept
    {
        using shape_type = typename std::decay_t<E>::shape_type;

        shape_type shape;
        resize_container(shape, e.shape().size());
        std::copy(e.shape().rbegin(), e.shape().rend(), shape.begin());

        shape_type strides;
        resize_container(strides, e.strides().size());
        std::copy(e.strides().rbegin(), e.strides().rend(), strides.begin());

        using view_type = xstrided_view<xclosure_t<E>, decltype(e.data())>;
        return view_type(std::forward<E>(e), std::move(shape), std::move(strides), 0);
    }

    /**
     * Returns a transpose view by permuting the xexpression e with @p permutation.
     * @param permutation the sequence containing permutation
     * @param check_policy the check level (check_policy::full() or check_policy::none())
     * @tparam Tag selects the level of error checking on permutation vector defaults to check_policy::none.
     */
    template <class E, class S, class Tag = check_policy::none>
    inline auto transpose(E&& e, S&& permutation, Tag check_policy = Tag())
    {
        return detail::transpose_impl(std::forward<E>(e), std::forward<S>(permutation), check_policy);
    }

#ifdef X_OLD_CLANG
    template <class E, class I, class Tag = check_policy::none>
    inline auto transpose(E&& e, std::initializer_list<I> permutation, Tag check_policy = Tag())
    {
        std::vector<I> perm(permutation);
        return detail::transpose_impl(std::forward<E>(e), std::move(perm), check_policy);
    }
#else
    template <class E, class I, std::size_t N, class Tag = check_policy::none>
    inline auto transpose(E&& e, const I (&permutation)[N], Tag check_policy = Tag())
    {
        return detail::transpose_impl(std::forward<E>(e), permutation, check_policy);
    }
#endif

}

#endif
