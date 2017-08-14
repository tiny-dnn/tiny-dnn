/***************************************************************************
* Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XSCALAR_HPP
#define XSCALAR_HPP

#include <array>
#include <cstddef>
#include <utility>

#include "xexpression.hpp"
#include "xiterable.hpp"
#include "xlayout.hpp"

namespace xt
{

    /***********
     * xscalar *
     ***********/

    // xscalar is a cheap wrapper for a scalar value as an xexpression.

    template <bool is_const, class CT>
    class xscalar_stepper;

    template <bool is_const, class CT>
    class xdummy_iterator;

    template <class CT>
    class xscalar;

    template <class CT>
    struct xiterable_inner_types<xscalar<CT>>
    {
        using value_type = std::decay_t<CT>;
        using inner_shape_type = std::array<std::size_t, 0>;
        using iterator = value_type*;
        using const_iterator = const value_type*;
        using const_stepper = xscalar_stepper<true, CT>;
        using stepper = xscalar_stepper<false, CT>;
        using reverse_iterator = std::reverse_iterator<iterator>;
        using const_reverse_iterator = std::reverse_iterator<const_iterator>;
    };

    template <class CT>
    class xscalar : public xexpression<xscalar<CT>>,
                    public xiterable<xscalar<CT>>
    {
    public:

        using self_type = xscalar<CT>;

        using value_type = std::decay_t<CT>;
        using reference = value_type&;
        using const_reference = const value_type&;
        using pointer = value_type*;
        using const_pointer = const value_type*;
        using size_type = std::size_t;
        using difference_type = std::ptrdiff_t;

        using iterable_base = xiterable<self_type>;
        using inner_shape_type = typename iterable_base::inner_shape_type;
        using shape_type = inner_shape_type;

        using stepper = typename iterable_base::stepper;
        using const_stepper = typename iterable_base::const_stepper;

        using iterator = typename iterable_base::iterator;
        using const_iterator = typename iterable_base::const_iterator;

        using dummy_iterator = xdummy_iterator<false, CT>;
        using const_dummy_iterator = xdummy_iterator<true, CT>;

        static constexpr layout_type static_layout = layout_type::any;
        static constexpr bool contiguous_layout = true;

        xscalar(CT value) noexcept;

        size_type size() const noexcept;
        size_type dimension() const noexcept;
        const shape_type& shape() const noexcept;
        layout_type layout() const noexcept;

        template <class... Args>
        reference operator()(Args...) noexcept;
        reference operator[](const xindex&) noexcept;
        reference operator[](size_type) noexcept;

        template <class... Args>
        const_reference operator()(Args...) const noexcept;
        const_reference operator[](const xindex&) const noexcept;
        const_reference operator[](size_type) const noexcept;

        template <class It>
        reference element(It, It) noexcept;

        template <class It>
        const_reference element(It, It) const noexcept;

        template <class S>
        bool broadcast_shape(S& shape) const noexcept;

        template <class S>
        bool is_trivial_broadcast(const S& strides) const noexcept;

        iterator begin() noexcept;
        iterator end() noexcept;

        const_iterator begin() const noexcept;
        const_iterator end() const noexcept;
        const_iterator cbegin() const noexcept;
        const_iterator cend() const noexcept;

        template <class S>
        stepper stepper_begin(const S& shape) noexcept;
        template <class S>
        stepper stepper_end(const S& shape, layout_type l) noexcept;

        template <class S>
        const_stepper stepper_begin(const S& shape) const noexcept;
        template <class S>
        const_stepper stepper_end(const S& shape, layout_type l) const noexcept;

        dummy_iterator dummy_begin() noexcept;
        dummy_iterator dummy_end() noexcept;

        const_dummy_iterator dummy_begin() const noexcept;
        const_dummy_iterator dummy_end() const noexcept;

        reference data_element(size_type i) noexcept;
        const_reference data_element(size_type i) const noexcept;

    private:

        CT m_value;
    };

    template <class T>
    xscalar<T&> xref(T& t);

    template <class T>
    xscalar<const T&> xcref(T& t);

    /*******************
     * xscalar_stepper *
     *******************/

    template <bool is_const, class CT>
    class xscalar_stepper
    {
    public:

        using self_type = xscalar_stepper<is_const, CT>;
        using container_type = std::conditional_t<is_const,
                                                  const xscalar<CT>,
                                                  xscalar<CT>>;

        using value_type = typename container_type::value_type;
        using reference = std::conditional_t<is_const,
                                             typename container_type::const_reference,
                                             typename container_type::reference>;
        using pointer = std::conditional_t<is_const,
                                           typename container_type::const_pointer,
                                           typename container_type::pointer>;
        using size_type = typename container_type::size_type;
        using difference_type = typename container_type::difference_type;

        xscalar_stepper(container_type* c) noexcept;

        reference operator*() const noexcept;

        void step(size_type dim, size_type n = 1) noexcept;
        void step_back(size_type dim, size_type n = 1) noexcept;
        void reset(size_type dim) noexcept;
        void reset_back(size_type dim) noexcept;

        void to_begin() noexcept;
        void to_end(layout_type l) noexcept;

        bool equal(const self_type& rhs) const noexcept;

    private:

        container_type* p_c;
    };

    template <bool is_const, class CT>
    bool operator==(const xscalar_stepper<is_const, CT>& lhs,
                    const xscalar_stepper<is_const, CT>& rhs) noexcept;

    template <bool is_const, class CT>
    bool operator!=(const xscalar_stepper<is_const, CT>& lhs,
                    const xscalar_stepper<is_const, CT>& rhs) noexcept;

    /*******************
     * xdummy_iterator *
     *******************/

    template <bool is_const, class CT>
    class xdummy_iterator
    {
    public:

        using self_type = xdummy_iterator<is_const, CT>;
        using container_type = std::conditional_t<is_const,
                                                  const xscalar<CT>,
                                                  xscalar<CT>>;

        using value_type = typename container_type::value_type;
        using reference = std::conditional_t<is_const,
                                             typename container_type::const_reference,
                                             typename container_type::reference>;
        using pointer = std::conditional_t<is_const,
                                           typename container_type::const_pointer,
                                           typename container_type::pointer>;
        using difference_type = typename container_type::difference_type;
        using iterator_category = std::forward_iterator_tag;

        explicit xdummy_iterator(container_type* c) noexcept;

        self_type& operator++() noexcept;
        self_type operator++(int) noexcept;

        reference operator*() const noexcept;

        bool equal(const self_type& rhs) const noexcept;

    private:

        container_type* p_c;
    };

    template <bool is_const, class CT>
    bool operator==(const xdummy_iterator<is_const, CT>& lhs,
                    const xdummy_iterator<is_const, CT>& rhs) noexcept;

    template <bool is_const, class CT>
    bool operator!=(const xdummy_iterator<is_const, CT>& lhs,
                    const xdummy_iterator<is_const, CT>& rhs) noexcept;

    /*******************************
     * trivial_begin / trivial_end *
     *******************************/

    namespace detail
    {
        template <class CT>
        constexpr auto trivial_begin(xscalar<CT>& c) -> decltype(c.dummy_begin())
        {
            return c.dummy_begin();
        }

        template <class CT>
        constexpr auto trivial_end(xscalar<CT>& c) -> decltype(c.dummy_end())
        {
            return c.dummy_end();
        }

        template <class CT>
        constexpr auto trivial_begin(const xscalar<CT>& c) -> decltype(c.dummy_begin())
        {
            return c.dummy_begin();
        }

        template <class CT>
        constexpr auto trivial_end(const xscalar<CT>& c) -> decltype(c.dummy_end())
        {
            return c.dummy_end();
        }
    }

    /**************************
     * xscalar implementation *
     **************************/

    template <class CT>
    inline xscalar<CT>::xscalar(CT value) noexcept
        : m_value(value)
    {
    }

    template <class CT>
    inline auto xscalar<CT>::size() const noexcept -> size_type
    {
        return 1;
    }

    template <class CT>
    inline auto xscalar<CT>::dimension() const noexcept -> size_type
    {
        return 0;
    }

    template <class CT>
    inline auto xscalar<CT>::shape() const noexcept -> const shape_type&
    {
        static std::array<size_type, 0> zero_shape;
        return zero_shape;
    }

    template <class CT>
    inline layout_type xscalar<CT>::layout() const noexcept
    {
        return static_layout;
    }

    template <class CT>
    template <class... Args>
    inline auto xscalar<CT>::operator()(Args...) noexcept -> reference
    {
        return m_value;
    }

    template <class CT>
    inline auto xscalar<CT>::operator[](const xindex&) noexcept -> reference
    {
        return m_value;
    }

    template <class CT>
    inline auto xscalar<CT>::operator[](size_type) noexcept -> reference
    {
        return m_value;
    }

    template <class CT>
    template <class... Args>
    inline auto xscalar<CT>::operator()(Args...) const noexcept -> const_reference
    {
        return m_value;
    }

    template <class CT>
    inline auto xscalar<CT>::operator[](const xindex&) const noexcept -> const_reference
    {
        return m_value;
    }

    template <class CT>
    inline auto xscalar<CT>::operator[](size_type) const noexcept -> const_reference
    {
        return m_value;
    }

    template <class CT>
    template <class It>
    inline auto xscalar<CT>::element(It, It) noexcept -> reference
    {
        return m_value;
    }

    template <class CT>
    template <class It>
    inline auto xscalar<CT>::element(It, It) const noexcept -> const_reference
    {
        return m_value;
    }

    template <class CT>
    template <class S>
    inline bool xscalar<CT>::broadcast_shape(S&) const noexcept
    {
        return true;
    }

    template <class CT>
    template <class S>
    inline bool xscalar<CT>::is_trivial_broadcast(const S&) const noexcept
    {
        return true;
    }

    template <class CT>
    inline auto xscalar<CT>::begin() noexcept -> iterator
    {
        return &m_value;
    }

    template <class CT>
    inline auto xscalar<CT>::end() noexcept -> iterator
    {
        return &m_value + 1;
    }

    template <class CT>
    inline auto xscalar<CT>::begin() const noexcept -> const_iterator
    {
        return &m_value;
    }

    template <class CT>
    inline auto xscalar<CT>::end() const noexcept -> const_iterator
    {
        return &m_value + 1;
    }

    template <class CT>
    inline auto xscalar<CT>::cbegin() const noexcept -> const_iterator
    {
        return &m_value;
    }

    template <class CT>
    inline auto xscalar<CT>::cend() const noexcept -> const_iterator
    {
        return &m_value + 1;
    }

    template <class CT>
    template <class S>
    inline auto xscalar<CT>::stepper_begin(const S&) noexcept -> stepper
    {
        return stepper(this);
    }

    template <class CT>
    template <class S>
    inline auto xscalar<CT>::stepper_end(const S&, layout_type) noexcept -> stepper
    {
        return stepper(this + 1);
    }

    template <class CT>
    template <class S>
    inline auto xscalar<CT>::stepper_begin(const S&) const noexcept -> const_stepper
    {
        return const_stepper(this);
    }

    template <class CT>
    template <class S>
    inline auto xscalar<CT>::stepper_end(const S&, layout_type) const noexcept -> const_stepper
    {
        return const_stepper(this + 1);
    }

    template <class CT>
    inline auto xscalar<CT>::dummy_begin() noexcept -> dummy_iterator
    {
        return dummy_iterator(this);
    }

    template <class CT>
    inline auto xscalar<CT>::dummy_end() noexcept -> dummy_iterator
    {
        return dummy_iterator(this);
    }

    template <class CT>
    inline auto xscalar<CT>::dummy_begin() const noexcept -> const_dummy_iterator
    {
        return const_dummy_iterator(this);
    }

    template <class CT>
    inline auto xscalar<CT>::dummy_end() const noexcept -> const_dummy_iterator
    {
        return const_dummy_iterator(this);
    }

    template <class CT>
    inline auto xscalar<CT>::data_element(size_type) noexcept->reference
    {
        return m_value;
    }
    
    template <class CT>
    inline auto xscalar<CT>::data_element(size_type) const noexcept->const_reference
    {
        return m_value;
    }

    template <class T>
    inline xscalar<T&> xref(T& t)
    {
        return xscalar<T&>(t);
    }

    template <class T>
    inline xscalar<const T&> xcref(T& t)
    {
        return xscalar<const T&>(t);
    }

    /**********************************
     * xscalar_stepper implementation *
     **********************************/

    template <bool is_const, class CT>
    inline xscalar_stepper<is_const, CT>::xscalar_stepper(container_type* c) noexcept
        : p_c(c)
    {
    }

    template <bool is_const, class CT>
    inline auto xscalar_stepper<is_const, CT>::operator*() const noexcept -> reference
    {
        return p_c->operator()();
    }

    template <bool is_const, class CT>
    inline void xscalar_stepper<is_const, CT>::step(size_type /*dim*/, size_type /*n*/) noexcept
    {
    }

    template <bool is_const, class CT>
    inline void xscalar_stepper<is_const, CT>::step_back(size_type /*dim*/, size_type /*n*/) noexcept
    {
    }

    template <bool is_const, class CT>
    inline void xscalar_stepper<is_const, CT>::reset(size_type /*dim*/) noexcept
    {
    }

    template <bool is_const, class CT>
    inline void xscalar_stepper<is_const, CT>::reset_back(size_type /*dim*/) noexcept
    {
    }

    template <bool is_const, class CT>
    inline void xscalar_stepper<is_const, CT>::to_begin() noexcept
    {
        p_c = p_c->stepper_begin(p_c->shap()).pc;
    }

    template <bool is_const, class CT>
    inline void xscalar_stepper<is_const, CT>::to_end(layout_type l) noexcept
    {
        p_c = p_c->stepper_end(p_c->shape(), l).p_c;
    }

    template <bool is_const, class CT>
    inline bool xscalar_stepper<is_const, CT>::equal(const self_type& rhs) const noexcept
    {
        return (p_c == rhs.p_c);
    }

    template <bool is_const, class CT>
    inline bool operator==(const xscalar_stepper<is_const, CT>& lhs,
                           const xscalar_stepper<is_const, CT>& rhs) noexcept
    {
        return lhs.equal(rhs);
    }

    template <bool is_const, class CT>
    inline bool operator!=(const xscalar_stepper<is_const, CT>& lhs,
                           const xscalar_stepper<is_const, CT>& rhs) noexcept
    {
        return !(lhs.equal(rhs));
    }

    /**********************************
     * xdummy_iterator implementation *
     **********************************/

    template <bool is_const, class CT>
    inline xdummy_iterator<is_const, CT>::xdummy_iterator(container_type* c) noexcept
        : p_c(c)
    {
    }

    template <bool is_const, class CT>
    inline auto xdummy_iterator<is_const, CT>::operator++() noexcept -> self_type&
    {
        return *this;
    }

    template <bool is_const, class CT>
    inline auto xdummy_iterator<is_const, CT>::operator++(int) noexcept -> self_type
    {
        self_type tmp(*this);
        ++(*this);
        return tmp;
    }

    template <bool is_const, class CT>
    inline auto xdummy_iterator<is_const, CT>::operator*() const noexcept -> reference
    {
        return p_c->operator()();
    }

    template <bool is_const, class CT>
    inline bool xdummy_iterator<is_const, CT>::equal(const self_type& rhs) const noexcept
    {
        return p_c == rhs.p_c;
    }

    template <bool is_const, class CT>
    inline bool operator==(const xdummy_iterator<is_const, CT>& lhs,
                           const xdummy_iterator<is_const, CT>& rhs) noexcept
    {
        return lhs.equal(rhs);
    }

    template <bool is_const, class CT>
    inline bool operator!=(const xdummy_iterator<is_const, CT>& lhs,
                           const xdummy_iterator<is_const, CT>& rhs) noexcept
    {
        return !(lhs.equal(rhs));
    }
}

#endif
