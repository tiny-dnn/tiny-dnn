/***************************************************************************
* Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
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
    template <class CT, class S, class CD>
    class xstrided_view;

    template <class CT, class S, class CD>
    struct xcontainer_inner_types<xstrided_view<CT, S, CD>>
    {
        using xexpression_type = std::decay_t<CT>;
        using temporary_type = xarray<typename xexpression_type::value_type>;
    };

    template <class CT, class S, class CD>
    struct xiterable_inner_types<xstrided_view<CT, S, CD>>
    {
        using inner_shape_type = S;
        using inner_strides_type = inner_shape_type;
        using inner_backstrides_type_type = inner_shape_type;
        using const_stepper = xstepper<const xstrided_view<CT, S, CD>>;
        using stepper = xstepper<xstrided_view<CT, S, CD>>;
        using const_iterator = xiterator<const_stepper, inner_shape_type*, DEFAULT_LAYOUT>;
        using iterator = xiterator<stepper, inner_shape_type*, DEFAULT_LAYOUT>;
        using const_reverse_iterator = std::reverse_iterator<const_iterator>;
        using reverse_iterator = std::reverse_iterator<iterator>;
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
    template <class CT, class S, class CD>
    class xstrided_view : public xview_semantic<xstrided_view<CT, S, CD>>,
                          public xexpression_iterable<xstrided_view<CT, S, CD>>
    {
    public:

        using self_type = xstrided_view<CT, S, CD>;
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

        static constexpr layout_type static_layout = layout_type::dynamic;
        static constexpr bool contiguous_layout = false;

        using temporary_type = typename xcontainer_inner_types<self_type>::temporary_type;
        using base_index_type = xindex_type_t<shape_type>;

        xstrided_view(CT e, S&& shape, S&& strides, std::size_t offset) noexcept;
        xstrided_view(CT e, CD data, S&& shape, S&& strides, std::size_t offset) noexcept;

        template <class E>
        self_type& operator=(const xexpression<E>& e);

        template <class E>
        disable_xexpression<E, self_type>& operator=(const E& e);

        size_type size() const noexcept;
        size_type dimension() const noexcept;
        const shape_type& shape() const noexcept;
        const strides_type& strides() const noexcept;
        const backstrides_type& backstrides() const noexcept;
        layout_type layout() const noexcept;

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
        stepper stepper_end(const ST& shape, layout_type l);

        template <class ST>
        const_stepper stepper_begin(const ST& shape) const;
        template <class ST>
        const_stepper stepper_end(const ST& shape, layout_type l) const;

        using container_iterator = typename std::decay_t<CD>::iterator;
        using const_container_iterator = typename std::decay_t<CD>::const_iterator;

        underlying_container_type& data() noexcept;
        const underlying_container_type& data() const noexcept;

        value_type* raw_data() noexcept;
        const value_type* raw_data() const noexcept;

        size_type raw_data_offset() const noexcept;

    protected:

        container_iterator data_xbegin() noexcept;
        const_container_iterator data_xbegin() const noexcept;
        container_iterator data_xend(layout_type l) noexcept;
        const_container_iterator data_xend(layout_type l) const noexcept;

    private:

        template <class C>
        friend class xstepper;

        template <class It>
        It data_xbegin_impl(It begin) const noexcept;

        template <class It>
        It data_xend_impl(It end, layout_type l) const noexcept;

        void assign_temporary_impl(temporary_type&& tmp);

        CT m_e;
        CD m_data;
        shape_type m_shape;
        strides_type m_strides;
        backstrides_type m_backstrides;
        std::size_t m_offset;

        friend class xview_semantic<xstrided_view<CT, S, CD>>;
    };

    /********************************
     * xstrided_view implementation *
     ********************************/

    /**
     * @name Constructor
     */
    //@{
    /**
     * Constructs an xstrided_view 
     * 
     * @param e the underlying xexpression for this view
     * @param shape the shape of the view
     * @param strides the strides of the view
     * @param offset the offset of the first element in the underlying container
     */
    template <class CT, class S, class CD>
    inline xstrided_view<CT, S, CD>::xstrided_view(CT e, S&& shape, S&& strides, std::size_t offset) noexcept
        : m_e(e), m_data(m_e.data()), m_shape(std::forward<S>(shape)), m_strides(std::forward<S>(strides)), m_offset(offset)
    {
        m_backstrides = make_sequence<backstrides_type>(m_shape.size(), 0);
        adapt_strides(m_shape, m_strides, m_backstrides);
    }

    template <class CT, class S, class CD>
    inline xstrided_view<CT, S, CD>::xstrided_view(CT e, CD data, S&& shape, S&& strides, std::size_t offset) noexcept
        : m_e(e), m_data(data), m_shape(std::forward<S>(shape)), m_strides(std::forward<S>(strides)), m_offset(offset)
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
    template <class CT, class S, class CD>
    template <class E>
    inline auto xstrided_view<CT, S, CD>::operator=(const xexpression<E>& e) -> self_type&
    {
        return semantic_base::operator=(e);
    }
    //@}

    template <class CT, class S, class CD>
    template <class E>
    inline auto xstrided_view<CT, S, CD>::operator=(const E& e) -> disable_xexpression<E, self_type>&
    {
        std::fill(this->begin(), this->end(), e);
        return *this;
    }

    template <class CT, class S, class CD>
    inline void xstrided_view<CT, S, CD>::assign_temporary_impl(temporary_type&& tmp)
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
    template <class CT, class S, class CD>
    inline auto xstrided_view<CT, S, CD>::size() const noexcept -> size_type
    {
        return compute_size(shape());
    }

    /**
     * Returns the number of dimensions of the xstrided_view.
     */
    template <class CT, class S, class CD>
    inline auto xstrided_view<CT, S, CD>::dimension() const noexcept -> size_type
    {
        return m_shape.size();
    }

    /**
     * Returns the shape of the xstrided_view.
     */
    template <class CT, class S, class CD>
    inline auto xstrided_view<CT, S, CD>::shape() const noexcept -> const shape_type&
    {
        return m_shape;
    }

    template <class CT, class S, class CD>
    inline auto xstrided_view<CT, S, CD>::strides() const noexcept -> const strides_type&
    {
        return m_strides;
    }

    template <class CT, class S, class CD>
    inline auto xstrided_view<CT, S, CD>::backstrides() const noexcept -> const backstrides_type&
    {
        return m_backstrides;
    }

    template <class CT, class S, class CD>
    inline auto xstrided_view<CT, S, CD>::layout() const noexcept -> layout_type
    {
        return layout_type::row_major;
    }

    template <class CT, class S, class CD>
    inline auto xstrided_view<CT, S, CD>::data() noexcept -> underlying_container_type&
    {
        return m_e.data();
    }

    template <class CT, class S, class CD>
    inline auto xstrided_view<CT, S, CD>::data() const noexcept -> const underlying_container_type&
    {
        return m_e.data();
    }

    template <class CT, class S, class CD>
    inline auto xstrided_view<CT, S, CD>::raw_data() noexcept -> value_type*
    {
        return m_e.raw_data();
    }

    template <class CT, class S, class CD>
    inline auto xstrided_view<CT, S, CD>::raw_data() const noexcept -> const value_type*
    {
        return m_e.raw_data();
    }

    template <class CT, class S, class CD>
    inline auto xstrided_view<CT, S, CD>::raw_data_offset() const noexcept -> size_type
    {
        return m_offset;
    }
    //@}

    /**
     * @name Data
     */
    template <class CT, class S, class CD>
    inline auto xstrided_view<CT, S, CD>::operator()() -> reference
    {
        return m_e();
    }

    template <class CT, class S, class CD>
    inline auto xstrided_view<CT, S, CD>::operator()() const -> const_reference
    {
        return m_e();
    }

    template <class CT, class S, class CD>
    template <class... Args>
    inline auto xstrided_view<CT, S, CD>::operator()(Args... args) -> reference
    {
        XTENSOR_ASSERT(check_index(shape(), args...));
        size_type index = m_offset + data_offset<size_type>(strides(), static_cast<size_type>(args)...);
        return m_data[index];
    }

    /**
     * Returns the element at the specified position in the xstrided_view. 
     * 
     * @param args a list of indices specifying the position in the view. Indices
     * must be unsigned integers, the number of indices should be equal or greater than
     * the number of dimensions of the view.
     */
    template <class CT, class S, class CD>
    template <class... Args>
    inline auto xstrided_view<CT, S, CD>::operator()(Args... args) const -> const_reference
    {
        XTENSOR_ASSERT(check_index(shape(), args...));
        size_type index = m_offset + data_offset<size_type>(strides(), static_cast<size_type>(args)...);
        return m_data[index];
    }

    template <class CT, class S, class CD>
    inline auto xstrided_view<CT, S, CD>::operator[](const xindex& index) -> reference
    {
        return element(index.cbegin(), index.cend());
    }

    template <class CT, class S, class CD>
    inline auto xstrided_view<CT, S, CD>::operator[](size_type i) -> reference
    {
        return operator()(i);
    }

    template <class CT, class S, class CD>
    inline auto xstrided_view<CT, S, CD>::operator[](const xindex& index) const -> const_reference
    {
        return element(index.cbegin(), index.cend());
    }

    template <class CT, class S, class CD>
    inline auto xstrided_view<CT, S, CD>::operator[](size_type i) const -> const_reference
    {
        return operator()(i);
    }

    /**
     * Returns a reference to the element at the specified position in the xstrided_view.
     * @param first iterator starting the sequence of indices
     * @param last iterator ending the sequence of indices
     * The number of indices in the sequence should be equal to or greater than the the number
     * of dimensions of the container..
     */
    template <class CT, class S, class CD>
    template <class It>
    inline auto xstrided_view<CT, S, CD>::element(It first, It last) -> reference
    {
        return m_data[m_offset + element_offset<size_type>(strides(), first, last)];
    }

    template <class CT, class S, class CD>
    template <class It>
    inline auto xstrided_view<CT, S, CD>::element(It first, It last) const -> const_reference
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
    template <class CT, class S, class CD>
    template <class O>
    inline bool xstrided_view<CT, S, CD>::broadcast_shape(O& shape) const
    {
        return xt::broadcast_shape(m_shape, shape);
    }

    /**
     * Compares the specified strides with those of the container to see whether
     * the broadcasting is trivial.
     * @return a boolean indicating whether the broadcasting is trivial
     */
    template <class CT, class S, class CD>
    template <class O>
    inline bool xstrided_view<CT, S, CD>::is_trivial_broadcast(const O& str) const noexcept
    {
        return str.size() == strides().size() &&
            std::equal(str.cbegin(), str.cend(), strides().begin());
    }
    //@}

    /***************
     * stepper api *
     ***************/

    template <class CT, class S, class CD>
    template <class ST>
    inline auto xstrided_view<CT, S, CD>::stepper_begin(const ST& shape) -> stepper
    {
        size_type offset = shape.size() - dimension();
        return stepper(this, data_xbegin(), offset);
    }

    template <class CT, class S, class CD>
    template <class ST>
    inline auto xstrided_view<CT, S, CD>::stepper_end(const ST& shape, layout_type l) -> stepper
    {
        size_type offset = shape.size() - dimension();
        return stepper(this, data_xend(l), offset);
    }

    template <class CT, class S, class CD>
    template <class ST>
    inline auto xstrided_view<CT, S, CD>::stepper_begin(const ST& shape) const -> const_stepper
    {
        size_type offset = shape.size() - dimension();
        return const_stepper(this, data_xbegin(), offset);
    }

    template <class CT, class S, class CD>
    template <class ST>
    inline auto xstrided_view<CT, S, CD>::stepper_end(const ST& shape, layout_type l) const -> const_stepper
    {
        size_type offset = shape.size() - dimension();
        return const_stepper(this, data_xend(l), offset);
    }

    template <class CT, class S, class CD>
    template <class It>
    inline It xstrided_view<CT, S, CD>::data_xbegin_impl(It begin) const noexcept
    {
        return begin + m_offset;
    }

    template <class CT, class S, class CD>
    template <class It>
    inline It xstrided_view<CT, S, CD>::data_xend_impl(It end, layout_type l) const noexcept
    {
        if (dimension() == 0)
        {
            return end;
        }
        else
        {
            auto leading_stride = (l == layout_type::row_major ? strides().back() : strides().front());
            return end - 1 + leading_stride;
        }
    }

    template <class CT, class S, class CD>
    inline auto xstrided_view<CT, S, CD>::data_xbegin() noexcept -> container_iterator
    {
        return data_xbegin_impl(m_data.begin());
    }

    template <class CT, class S, class CD>
    inline auto xstrided_view<CT, S, CD>::data_xbegin() const noexcept -> const_container_iterator
    {
        return data_xbegin_impl(m_data.begin());
    }

    template <class CT, class S, class CD>
    inline auto xstrided_view<CT, S, CD>::data_xend(layout_type l) noexcept -> container_iterator
    {
        return data_xend_impl(m_data.end(), l);
    }

    template <class CT, class S, class CD>
    inline auto xstrided_view<CT, S, CD>::data_xend(layout_type l) const noexcept -> const_container_iterator
    {
        return data_xend_impl(m_data.end(), l);
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
        using view_type = xstrided_view<xclosure_t<E>, I, decltype(e.data())>;
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
            if (sequence_size(permutation) != e.dimension())
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
            using view_type = xstrided_view<xclosure_t<E>, shape_type, decltype(e.data())>;
            return view_type(std::forward<E>(e), std::move(temp_shape), std::move(temp_strides), 0);
        }

        template <class E, class S>
        inline auto transpose_impl(E&& e, S&& permutation, check_policy::full)
        {
            // check if axis appears twice in permutation
            for (std::size_t i = 0; i < sequence_size(permutation); ++i)
            {
                for (std::size_t j = i + 1; j < sequence_size(permutation); ++j)
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

        using view_type = xstrided_view<xclosure_t<E>, shape_type, decltype(e.data())>;
        return view_type(std::forward<E>(e), std::move(shape), std::move(strides), 0);
    }

    /**
     * Returns a transpose view by permuting the xexpression e with @p permutation.
     * @param e the input expression
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

    namespace detail
    {
        template <class CT>
        class expression_adaptor
        {
        public:
            using xexpression_type = std::decay_t<CT>;
            using shape_type = typename xexpression_type::shape_type;
            using index_type = xindex_type_t<shape_type>;
            using size_type = typename xexpression_type::size_type;
            using value_type = typename xexpression_type::value_type;
            using reference = typename xexpression_type::reference;

            expression_adaptor(CT&& e) : m_e(e)
            {
                resize_container(m_index, m_e.dimension());
                m_size = compute_size(m_e.shape());
                compute_strides(m_e.shape(), layout_type::row_major, m_strides);
            }

            const reference operator[](std::size_t idx) const
            {
                std::div_t dv{};
                for (size_type i = 0; i < m_strides.size(); ++i)
                {
                    dv = std::div((int) idx, (int) m_strides[i]);
                    idx = dv.rem;
                    m_index[i] = dv.quot;
                }
                return m_e.element(m_index.begin(), m_index.end());
            }

            size_type size() const
            {
                return m_size;
            }

        private:
            CT m_e;
            shape_type m_strides;
            mutable index_type m_index;
            size_type m_size;
        };
    }

    class slice_vector : private std::vector<std::array<long int, 3>>
    {

    public:
        using index_type = long int;
        using base_type = std::vector<std::array<index_type, 3>>;

        // propagating interface
        using base_type::begin;
        using base_type::cbegin;
        using base_type::end;
        using base_type::cend;
        using base_type::size;
        using base_type::operator[];
        using base_type::push_back;

        inline slice_vector() = default;

        template <class E, class... Args>
        inline slice_vector(const xexpression<E>& e, Args... args)
        {
            const auto& de = e.derived_cast();
            m_shape.resize(de.shape().size());
            std::copy(de.shape().begin(), de.shape().end(), m_shape.begin());
            append(args...);
        }

        template <class... Args>
        inline slice_vector(const std::vector<std::size_t>& shape, Args... args)
        {
            m_shape = shape;
            append(args...);
        }

        template <class T, class... Args>
        inline void append(const T& s, Args... args)
        {
            push_back(s);
            append(args...);
        }

        inline void append()
        {
        }

        template <class T>
        inline void push_back(const xslice<T>& s)
        {
            auto ds = s.derived_cast();
            base_type::push_back({ds(0), (index_type)ds.size(), (index_type)ds.step_size()});
        }

        template <class A, class B, class C>
        inline void push_back(const xrange_adaptor<A, B, C>& s)
        {
            auto idx = size() - newaxis_count;
            if (idx >= m_shape.size())
            {
                throw std::runtime_error("Too many slices in slice vector for shape");
            }
            auto ds = s.get(m_shape[idx]);
            base_type::push_back({(index_type)ds(0), (index_type)ds.size(), (index_type)ds.step_size()});
        }

        inline void push_back(xall_tag /*s*/)
        {
            auto idx = size() - newaxis_count;
            if (idx >= m_shape.size())
            {
                throw std::runtime_error("Too many slices in slice vector for shape");
            }
            base_type::push_back({0, (index_type)m_shape[idx], 1});
        }

        inline void push_back(xnewaxis_tag /*s*/)
        {
            ++newaxis_count;
            base_type::push_back({-1, 0, 0});
        }

        inline void push_back(index_type i)
        {
            base_type::push_back({i, 0, 0});
        }

    private:

        std::vector<std::size_t> m_shape;
        std::size_t newaxis_count = 0;
    };

    namespace detail
    {
        template <class E, std::enable_if_t<has_raw_data_interface<std::decay_t<E>>::value>* = nullptr>
        inline auto&& get_data(E&& e)
        {
            return e.data();
        }

        template <class E, std::enable_if_t<has_raw_data_interface<std::decay_t<E>>::value>* = nullptr>
        inline std::size_t get_offset(E&& e)
        {
            return e.raw_data_offset();
        }

        template <class E, std::enable_if_t<has_raw_data_interface<std::decay_t<E>>::value>* = nullptr>
        inline auto&& get_strides(E&& e)
        {
            return e.strides();
        }

        template <class E, std::enable_if_t<!has_raw_data_interface<std::decay_t<E>>::value>* = nullptr>
        inline auto get_data(E&& e) -> expression_adaptor<xclosure_t<E>>
        {
            return std::move(expression_adaptor<xclosure_t<E>>(e));
        }

        template <class E, std::enable_if_t<!has_raw_data_interface<std::decay_t<E>>::value>* = nullptr>
        inline std::size_t get_offset(E&& /*e*/)
        {
            return 0;
        }

        template <class E, std::enable_if_t<!has_raw_data_interface<std::decay_t<E>>::value>* = nullptr>
        inline auto get_strides(E&& e)
        {
            std::vector<std::size_t> strides;
            strides.resize(e.shape().size());
            compute_strides(e.shape(), layout_type::row_major, strides);
            return strides;
        }
    }

    template <class E, class S>
    inline auto dynamic_view(E&& e, S&& slices)
    {
        // Compute dimension
        std::size_t dimension = e.dimension();

        for (const auto& el : slices)
        {
            if (el[0] >= 0 && el[1] == 0)
            {
                // treat this like a single integral and remove from shape
                --dimension;
            }
            else if (el[0] == -1 && el[1] == 0)
            {
                // treat this like a new axis
                ++dimension;
            }
        }

        // Compute strided view

        std::size_t offset = detail::get_offset(e);
        using shape_type = typename std::vector<std::size_t>;

        shape_type new_shape(dimension);
        shape_type new_strides(dimension);

        auto old_shape = e.shape();
        auto&& old_strides = detail::get_strides(e);

        std::size_t i = 0;
        std::size_t idx = 0;
        std::size_t newaxis_skip = 0;

        for (; i < slices.size(); ++i)
        {
            if (slices[i][0] >= 0)
            {
                offset += slices[i][0] * old_strides[i];
            }

            if (slices[i][1] != 0 && slices[i][2] != 0)
            {
                new_shape[idx] = slices[i][1];
                new_strides[idx] = slices[i][2] * old_strides[i - newaxis_skip];
                ++idx;
            }
            else if (slices[i][0] == -1)  // newaxis
            {
                new_shape[idx] = 1;
                new_strides[idx] = 0;
                ++newaxis_skip;
                ++idx;
            }
        }

        for (; i < old_shape.size(); ++i)
        {
            new_shape[idx] = old_shape[i];
            new_strides[idx] = old_strides[i];
            ++idx;
        }

        auto data = detail::get_data(e);

        using view_type = xstrided_view<xclosure_t<E>, shape_type, decltype(data)>;
        return view_type(std::forward<E>(e), std::forward<decltype(data)>(data), std::move(new_shape), std::move(new_strides), offset);
    }
}

#endif
