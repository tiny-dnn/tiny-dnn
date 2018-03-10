/***************************************************************************
* Copyright (c) 2016, Sylvain Corlay and Johan Mabille                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XDYNAMIC_BITSET_HPP
#define XDYNAMIC_BITSET_HPP

#include <climits>
#include <type_traits>
#include <vector>

#include "xclosure.hpp"
#include "xiterator_base.hpp"

namespace xtl
{
    template <class B, class A, bool is_const>
    class xbitset_reference;

    template <class B, class A, bool is_const>
    class xbitset_iterator;

    /******************
     * xdyamic_bitset *
     ******************/

    template <class B, class Allocator = std::allocator<B>>
    class xdynamic_bitset
    {
    public:

        using self_type = xdynamic_bitset<B, Allocator>;
        using allocator_type = Allocator;
        using block_type = B;
        using storage_type = std::vector<B, Allocator>;
        using value_type = bool;
        using reference = xbitset_reference<block_type, allocator_type, false>;
        using const_reference = xbitset_reference<block_type, allocator_type, true>;

        using pointer = typename reference::pointer;
        using const_pointer = typename const_reference::pointer;
        using size_type = typename storage_type::size_type;
        using difference_type = typename storage_type::difference_type;
        using iterator = xbitset_iterator<block_type, allocator_type, false>;
        using const_iterator = xbitset_iterator<block_type, allocator_type, true>;
        using reverse_iterator = std::reverse_iterator<iterator>;
        using const_reverse_iterator = std::reverse_iterator<const_iterator>;

        xdynamic_bitset();
        explicit xdynamic_bitset(const allocator_type& allocator);
        xdynamic_bitset(size_type count, bool b, const allocator_type& alloc = allocator_type());
        explicit xdynamic_bitset(size_type count, const allocator_type& alloc = allocator_type());
        template <class BlockInputIt>
        xdynamic_bitset(BlockInputIt first, BlockInputIt last, const allocator_type& alloc = allocator_type());
        xdynamic_bitset(std::initializer_list<bool> init, const allocator_type& alloc = allocator_type());

        void assign(size_type count, bool b);
        template <class BlockInputIt>
        void assign(BlockInputIt first, BlockInputIt last);
        void assign(std::initializer_list<bool> init);

        allocator_type get_allocator() const;

        bool empty() const noexcept;
        size_type size() const noexcept;
        size_type max_size() const noexcept;
        void reserve(size_type new_cap);
        size_type capacity() const noexcept;

        void resize(size_type size, bool b = false);
        void clear() noexcept;
        void push_back(bool b);
        void pop_back();

        void swap(self_type& rhs);

        reference at(size_type i);
        const_reference at(size_type i) const;

        reference operator[](size_type i);
        const_reference operator[](size_type i) const;

        reference front();
        const_reference front() const;

        reference back();
        const_reference back() const;

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

        self_type& operator&=(const self_type& rhs);
        self_type& operator|=(const self_type& rhs);
        self_type& operator^=(const self_type& rhs);

        self_type operator<<(size_type pos);
        self_type& operator<<=(size_type pos);
        self_type operator>>(size_type pos);
        self_type& operator>>=(size_type pos);

        self_type& set();
        self_type& set(size_type pos, value_type value = true);

        self_type& reset();
        self_type& reset(size_type pos);

        self_type& flip();
        self_type& flip(size_type pos);

        bool all() const noexcept;
        bool any() const noexcept;
        bool none() const noexcept;
        size_type count() const noexcept;

        size_type block_count() const noexcept;
        block_type* data() noexcept;
        const block_type* data() const noexcept;

        bool operator==(const self_type& rhs) const noexcept;
        bool operator!=(const self_type& rhs) const noexcept;
        bool operator<(const self_type& rhs) const noexcept;
        bool operator<=(const self_type& rhs) const noexcept;
        bool operator>(const self_type& rhs) const noexcept;
        bool operator>=(const self_type& rhs) const noexcept;

    private:

        static constexpr std::size_t s_bits_per_block = CHAR_BIT * sizeof(block_type);

        size_type compute_block_count(size_type bits_count) const noexcept;
        size_type block_index(size_type pos) const noexcept;
        size_type bit_index(size_type pos) const noexcept;
        block_type bit_mask(size_type pos) const noexcept;
        size_type count_extra_bits() const noexcept;
        void zero_unused_bits();

        storage_type m_buffer;
        size_type m_size;
    };

    template <class B, class A>
    xdynamic_bitset<B, A> operator~(const xdynamic_bitset<B, A>& lhs);

    template <class B, class A>
    xdynamic_bitset<B, A> operator&(const xdynamic_bitset<B, A>& lhs, const xdynamic_bitset<B, A>& rhs);

    template <class B, class A>
    xdynamic_bitset<B, A> operator|(const xdynamic_bitset<B, A>& lhs, const xdynamic_bitset<B, A>& rhs);

    template <class B, class A>
    xdynamic_bitset<B, A> operator^(const xdynamic_bitset<B, A>& lhs, const xdynamic_bitset<B, A>& rhs);

    template <class B, class A>
    void swap(const xdynamic_bitset<B, A>& lhs, const xdynamic_bitset<B, A>& rhs);

    /*********************
     * xbitset_reference *
     *********************/

    template <class B, class A, bool is_const>
    class xbitset_reference
    {
    public:
        
        using bitset_type = xdynamic_bitset<B, A>;
        using self_type = xbitset_reference<B, A, is_const>;
        using pointer = std::conditional_t<is_const,
                                           const xclosure_pointer<const self_type>,
                                           xclosure_pointer<self_type>>;

        operator bool() const noexcept;

        xbitset_reference(const self_type&) = default;
        xbitset_reference(self_type&&) = default;

        self_type& operator=(const self_type&) noexcept;
        self_type& operator=(self_type&&) noexcept;
        self_type& operator=(bool) noexcept;

        bool operator~() const noexcept;

        self_type& operator&=(bool) noexcept;
        self_type& operator|=(bool) noexcept;
        self_type& operator^=(bool) noexcept;
        self_type& flip() noexcept;

        pointer operator&() noexcept;

    private:

        using block_type = typename bitset_type::block_type;
        using closure_type = std::conditional_t<is_const, const block_type&, block_type&>;

        xbitset_reference(closure_type block, block_type pos);

        void assign(bool) noexcept;
        void set() noexcept;
        void reset() noexcept;

        closure_type m_block;
        const block_type m_mask;

        friend class xdynamic_bitset<B, A>;
    };

    /********************
     * xbitset_iterator *
     ********************/

    template <class B, class A, bool is_const>
    class xbitset_iterator : public xrandom_access_iterator_base<xbitset_iterator<B, A, is_const>,
                                                                 typename xdynamic_bitset<B, A>::value_type,
                                                                 typename xdynamic_bitset<B, A>::difference_type,
                                                                 std::conditional_t<is_const,
                                                                                    typename xdynamic_bitset<B, A>::const_pointer,
                                                                                    typename xdynamic_bitset<B, A>::pointer>,
                                                                 std::conditional_t<is_const,
                                                                                    typename xdynamic_bitset<B, A>::const_reference,
                                                                                    typename xdynamic_bitset<B, A>::reference>>
    {
    public:

        using self_type = xbitset_iterator<B, A, is_const>;
        using container_type = xdynamic_bitset<B, A>;
        using value_type = typename container_type::value_type;
        using reference = std::conditional_t<is_const,
                                             typename container_type::const_reference,
                                             typename container_type::reference>;
        using pointer = std::conditional_t<is_const,
                                           typename container_type::const_pointer,
                                           typename container_type::pointer>;
        using size_type = typename container_type::size_type;
        using difference_type = typename container_type::difference_type;
        using base_type = xrandom_access_iterator_base<self_type, value_type, difference_type, pointer, reference>;

        using container_reference = std::conditional_t<is_const, const container_type&, container_type&>;
        using container_pointer = std::conditional_t<is_const, const container_type*, container_type*>;

        xbitset_iterator() noexcept;
        xbitset_iterator(container_reference c, size_type index) noexcept;

        self_type& operator++();
        self_type& operator--();

        self_type& operator+=(difference_type n);
        self_type& operator-=(difference_type n);

        difference_type operator-(const self_type& rhs) const;

        reference operator*() const;
        pointer operator->() const;

        bool operator==(const self_type& rhs) const;
        bool operator<(const self_type& rhs) const;

    private:

        container_pointer p_container;
        size_type m_index;
    };

    /**********************************
     * xdynamic_bitset implementation *
     **********************************/

    template <class B, class A>
    inline xdynamic_bitset<B, A>::xdynamic_bitset()
        : xdynamic_bitset(0, allocator_type())
    {
    }

    template <class B, class A>
    inline xdynamic_bitset<B, A>::xdynamic_bitset(const allocator_type& allocator)
        : xdynamic_bitset(0, allocator)
    {
    }

    template <class B, class A>
    inline xdynamic_bitset<B, A>::xdynamic_bitset(size_type count, bool b, const allocator_type& alloc)
        : m_buffer(compute_block_count(count), b ? ~block_type(0) : block_type(0), alloc), m_size(count)
    {
        zero_unused_bits();
    }

    template <class B, class A>
    inline xdynamic_bitset<B, A>::xdynamic_bitset(size_type count, const allocator_type& alloc)
        : m_buffer(compute_block_count(count), block_type(0), alloc), m_size(count)
    {
    }

    template <class B, class A>
    template <class BlockInputIt>
    inline xdynamic_bitset<B, A>::xdynamic_bitset(BlockInputIt first, BlockInputIt last, const allocator_type& alloc)
        : m_buffer(first, last, alloc)
    {
        m_size = m_buffer.size() * s_bits_per_block;
    }

    template <class B, class A>
    inline xdynamic_bitset<B, A>::xdynamic_bitset(std::initializer_list<bool> init, const allocator_type& alloc)
        : xdynamic_bitset(init.size(), alloc)
    {
        std::copy(init.begin(), init.end(), begin());
    }

    template <class B, class A>
    inline void xdynamic_bitset<B, A>::assign(size_type count, bool b)
    {
        resize(count);
        b ? set() : reset();
    }

    template <class B, class A>
    template <class BlockInputIt>
    inline void xdynamic_bitset<B, A>::assign(BlockInputIt first, BlockInputIt last)
    {
        resize(std::distance(first, last) * s_bits_per_block);
        std::copy(first, last, m_buffer.begin());
    }

    template <class B, class A>
    inline void xdynamic_bitset<B, A>::assign(std::initializer_list<bool> init)
    {
        resize(init.size());
        std::copy(init.begin(), init.end(), begin());
    }

    template <class B, class A>
    inline auto xdynamic_bitset<B, A>::get_allocator() const -> allocator_type
    {
        return m_buffer.get_allocator();
    }

    template <class B, class A>
    inline bool xdynamic_bitset<B, A>::empty() const noexcept
    {
        return m_buffer.empty();
    }

    template <class B, class A>
    inline auto xdynamic_bitset<B, A>::size() const noexcept -> size_type
    {
        return m_size;
    }

    template <class B, class A>
    inline auto xdynamic_bitset<B, A>::max_size() const noexcept -> size_type
    {
        return m_buffer.max_size() * s_bits_per_block;
    }

    template <class B, class A>
    inline void xdynamic_bitset<B, A>::reserve(size_type new_cap)
    {
        m_buffer.reserve(compute_block_count(new_cap));
    }

    template <class B, class A>
    inline auto xdynamic_bitset<B, A>::capacity() const noexcept -> size_type
    {
        return m_buffer.capacity() * s_bits_per_block;
    }

    template <class B, class A>
    inline void xdynamic_bitset<B, A>::resize(size_type size, bool b)
    {
        size_type old_block_count = block_count();
        size_type new_block_count = compute_block_count(size);
        block_type value = b ? ~block_type(0) : block_type(0);

        if (new_block_count != old_block_count)
        {
            m_buffer.resize(new_block_count, value);
        }

        if (b && size > m_size)
        {
            size_type extra_bits = count_extra_bits();
            if (extra_bits > 0)
            {
                m_buffer[old_block_count - 1] |= (value << extra_bits);
            }
        }

        m_size = size;
        zero_unused_bits();
    }

    template <class B, class A>
    inline void xdynamic_bitset<B, A>::clear() noexcept
    {
        m_buffer.clear();
        m_size = size_type(0);
    }

    template <class B, class A>
    inline void xdynamic_bitset<B, A>::push_back(bool b)
    {
        size_type s = size();
        resize(s + 1);
        set(s, b);
    }

    template <class B, class A>
    inline void xdynamic_bitset<B, A>::pop_back()
    {
        size_type old_block_count = m_buffer.size();
        size_type new_block_count = compute_block_count(m_size - 1);

        if (new_block_count != old_block_count)
        {
            m_buffer.pop_back();
        }

        --m_size;
        zero_unused_bits();
    }

    template <class B, class A>
    inline void xdynamic_bitset<B, A>::swap(self_type& rhs)
    {
        using std::swap;
        swap(m_buffer, rhs.m_buffer);
        swap(m_size, rhs.m_size);
    }

    template <class B, class A>
    inline auto xdynamic_bitset<B, A>::at(size_type i) -> reference
    {
        return reference(m_buffer.at(block_index(i)), bit_index(i));
    }

    template <class B, class A>
    inline auto xdynamic_bitset<B, A>::at(size_type i) const -> const_reference
    {
        return const_reference(m_buffer.at(block_index(i)), bit_index(i));
    }

    template <class B, class A>
    inline auto xdynamic_bitset<B, A>::operator[](size_type i) -> reference
    {
        return reference(m_buffer[block_index(i)], bit_index(i));
    }

    template <class B, class A>
    inline auto xdynamic_bitset<B, A>::operator[](size_type i) const -> const_reference
    {
        return const_reference(m_buffer[block_index(i)], bit_index(i));
    }

    template <class B, class A>
    inline auto xdynamic_bitset<B, A>::front() -> reference
    {
        return (*this)[0];
    }

    template <class B, class A>
    inline auto xdynamic_bitset<B, A>::front() const -> const_reference
    {
        return (*this)[0];
    }

    template <class B, class A>
    inline auto xdynamic_bitset<B, A>::back() -> reference
    {
        return (*this)[m_size - 1];
    }

    template <class B, class A>
    inline auto xdynamic_bitset<B, A>::back() const -> const_reference
    {
        return (*this)[m_size - 1];
    }

    template <class B, class A>
    inline auto xdynamic_bitset<B, A>::begin() noexcept -> iterator
    {
        return iterator(*this, size_type(0));
    }

    template <class B, class A>
    inline auto xdynamic_bitset<B, A>::end() noexcept -> iterator
    {
        return iterator(*this, size());
    }

    template <class B, class A>
    inline auto xdynamic_bitset<B, A>::begin() const noexcept -> const_iterator
    {
        return cbegin();
    }

    template <class B, class A>
    inline auto xdynamic_bitset<B, A>::end() const noexcept -> const_iterator
    {
        return cend();
    }

    template <class B, class A>
    inline auto xdynamic_bitset<B, A>::cbegin() const noexcept -> const_iterator
    {
        return const_iterator(*this, size_type(0));
    }

    template <class B, class A>
    inline auto xdynamic_bitset<B, A>::cend() const noexcept -> const_iterator
    {
        return const_iterator(*this, size());
    }

    template <class B, class A>
    inline auto xdynamic_bitset<B, A>::rbegin() noexcept -> reverse_iterator
    {
        return reverse_iterator(end());
    }

    template <class B, class A>
    inline auto xdynamic_bitset<B, A>::rend() noexcept -> reverse_iterator
    {
        return reverse_iterator(begin());
    }

    template <class B, class A>
    inline auto xdynamic_bitset<B, A>::rbegin() const noexcept -> const_reverse_iterator
    {
        return crbegin();
    }

    template <class B, class A>
    inline auto xdynamic_bitset<B, A>::rend() const noexcept -> const_reverse_iterator
    {
        return crend();
    }

    template <class B, class A>
    inline auto xdynamic_bitset<B, A>::crbegin() const noexcept -> const_reverse_iterator
    {
        return const_reverse_iterator(cend());
    }

    template <class B, class A>
    inline auto xdynamic_bitset<B, A>::crend() const noexcept -> const_reverse_iterator
    {
        return const_reverse_iterator(cbegin());
    }

    template <class B, class A>
    inline auto xdynamic_bitset<B, A>::operator&=(const self_type& rhs) -> self_type&
    {
        size_type size = block_count();
        for (size_type i = 0; i < size; ++i)
        {
            m_buffer[i] &= rhs.m_buffer[i];
        }
        return *this;
    }

    template <class B, class A>
    inline auto xdynamic_bitset<B, A>::operator|=(const self_type& rhs) -> self_type&
    {
        size_type size = block_count();
        for (size_type i = 0; i < size; ++i)
        {
            m_buffer[i] |= rhs.m_buffer[i];
        }
        return *this;
    }

    template <class B, class A>
    inline auto xdynamic_bitset<B, A>::operator^=(const self_type& rhs) -> self_type&
    {
        size_type size = block_count();
        for (size_type i = 0; i < size; ++i)
        {
            m_buffer[i] ^= rhs.m_buffer[i];
        }
        return *this;
    }

    template <class B, class A>
    inline auto xdynamic_bitset<B, A>::operator<<(size_type pos) -> self_type
    {
        self_type tmp(*this);
        tmp <<= pos;
        return tmp;
    }

    template <class B, class A>
    inline auto xdynamic_bitset<B, A>::operator<<=(size_type pos) -> self_type&
    {
        if (pos >= m_size)
        {
            return reset();
        }

        if (pos > 0)
        {
            size_type last = block_count() - 1;
            size_type div = pos / s_bits_per_block;
            size_type r = bit_index(pos);
            block_type* b = &m_buffer[0];

            if (r != 0)
            {
                size_type rs = s_bits_per_block - r;
                for (size_type i = last - div; i > 0; --i)
                {
                    b[i + div] = (b[i] << r) | (b[i - 1] >> rs);
                }
                b[div] = b[0] << r;
            }
            else
            {
                for (size_type i = last - div; i > 0; --i)
                {
                    b[i + div] = b[i];
                }
                b[div] = b[0];
            }

            std::fill_n(m_buffer.begin(), div, block_type(0));
            zero_unused_bits();
        }
        return *this;
    }

    template <class B, class A>
    inline auto xdynamic_bitset<B, A>::operator>>(size_type pos) -> self_type
    {
        self_type tmp(*this);
        tmp >>= pos;
        return tmp;
    }

    template <class B, class A>
    inline auto xdynamic_bitset<B, A>::operator>>=(size_type pos) -> self_type&
    {
        if (pos >= m_size)
        {
            return reset();
        }

        if (pos > 0)
        {
            size_type last = block_count() - 1;
            size_type div = pos / s_bits_per_block;
            size_type r = bit_index(pos);
            block_type* b = &m_buffer[0];

            if (r != 0)
            {
                size_type ls = s_bits_per_block - r;
                for (size_type i = div; i < last; ++i)
                {
                    b[i - div] = (b[i] >> r) | (b[i + 1] << ls);
                }
                b[last - div] = b[last] >> r;
            }
            else
            {
                for (size_type i = div; i <= last; ++i)
                {
                    b[i - div] = b[i];
                }
            }

            std::fill_n(m_buffer.begin() + (block_count() - div), div, block_type(0));
        }
        return *this;
    }

    template <class B, class A>
    inline auto xdynamic_bitset<B, A>::set() -> self_type&
    {
        std::fill(m_buffer.begin(), m_buffer.end(), ~block_type(0));
        zero_unused_bits();
        return *this;
    }

    template <class B, class A>
    inline auto xdynamic_bitset<B, A>::set(size_type pos, value_type value) -> self_type&
    {
        if (value)
        {
            m_buffer[block_index(pos)] |= bit_mask(pos);
        }
        else
        {
            reset(pos);
        }
        return *this;
    }

    template <class B, class A>
    inline auto xdynamic_bitset<B, A>::reset() -> self_type&
    {
        std::fill(m_buffer.begin(), m_buffer.end(), block_type(0));
        return *this;
    }

    template <class B, class A>
    inline auto xdynamic_bitset<B, A>::reset(size_type pos) -> self_type&
    {
        m_buffer[block_index(pos)] &= ~bit_mask(pos);
        return *this;
    }

    template <class B, class A>
    inline auto xdynamic_bitset<B, A>::flip() -> self_type&
    {
        size_type size = block_count();
        for (size_type i = 0; i < size; ++i)
        {
            m_buffer[i] = ~m_buffer[i];
        }
        zero_unused_bits();
        return *this;
    }

    template <class B, class A>
    inline auto xdynamic_bitset<B, A>::flip(size_type pos) -> self_type&
    {
        m_buffer[block_index(pos)] ^= bit_mask(pos);
        return *this;
    }

    template <class B, class A>
    inline bool xdynamic_bitset<B, A>::all() const noexcept
    {
        if (empty())
            return true;

        size_type extrabits = count_extra_bits();
        constexpr block_type all_ones = ~block_type(0);

        size_type size = extrabits != 0 ? block_count() - 1 : block_count();
        for (size_type i = 0; i < size; ++i)
        {
            if (m_buffer[i] != all_ones)
                return false;
        }

        if (extrabits != 0)
        {
            block_type mask = ~(~block_type(0) << extrabits);
            if (m_buffer.back() != mask)
                return false;
        }

        return true;
    }

    template <class B, class A>
    inline bool xdynamic_bitset<B, A>::any() const noexcept
    {
        size_type size = block_count();
        for (size_type i = 0; i < size; ++i)
        {
            if (m_buffer[i])
                return true;
        }
        return false;
    }

    template <class B, class A>
    inline bool xdynamic_bitset<B, A>::none() const noexcept
    {
        return !any();
    }

    template <class B, class A>
    inline auto xdynamic_bitset<B, A>::count() const noexcept -> size_type
    {
        static constexpr unsigned char table[] =
        {
            0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
            1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
            1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
            2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
            1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
            2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
            2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
            3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 4, 5, 5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8
        };
        size_type res = 0;
        const unsigned char* p = static_cast<const unsigned char*>(static_cast<const void*>(&m_buffer[0]));
        size_type length = m_buffer.size() * sizeof(block_type);
        for (size_type i = 0; i < length; ++i, ++p)
        {
            res += table[*p];
        }
        return res;
    }

    template <class B, class A>
    inline auto xdynamic_bitset<B, A>::block_count() const noexcept -> size_type
    {
        return m_buffer.size();
    }

    template <class B, class A>
    inline auto xdynamic_bitset<B, A>::data() noexcept -> block_type*
    {
        return m_buffer.data();
    }

    template <class B, class A>
    inline auto xdynamic_bitset<B, A>::data() const noexcept -> const block_type*
    {
        return m_buffer.data();
    }

    template <class B, class A>
    inline bool xdynamic_bitset<B, A>::operator==(const self_type& rhs) const noexcept
    {
        return m_size == rhs.m_size && m_buffer == rhs.m_buffer;
    }

    template <class B, class A>
    inline bool xdynamic_bitset<B, A>::operator!=(const self_type& rhs) const noexcept
    {
        return !(*this == rhs);
    }

    template <class B, class A>
    inline bool xdynamic_bitset<B, A>::operator<(const self_type& rhs) const noexcept
    {
        return m_size < rhs.m_size || (m_size == rhs.m_size && m_buffer < rhs.m_buffer);
    }

    template <class B, class A>
    inline bool xdynamic_bitset<B, A>::operator<=(const self_type& rhs) const noexcept
    {
        return !(*this > rhs);
    }

    template <class B, class A>
    inline bool xdynamic_bitset<B, A>::operator>(const self_type& rhs) const noexcept
    {
        return rhs < *this;
    }

    template <class B, class A>
    inline bool xdynamic_bitset<B, A>::operator>=(const self_type& rhs) const noexcept
    {
        return rhs <= *this;
    }

    template <class B, class A>
    inline auto xdynamic_bitset<B, A>::compute_block_count(size_type bits_count) const noexcept -> size_type
    {
        return bits_count / s_bits_per_block
            + static_cast<size_type>(bits_count % s_bits_per_block != 0);
    }

    template <class B, class A>
    inline auto xdynamic_bitset<B, A>::block_index(size_type pos) const noexcept -> size_type
    {
        return pos / s_bits_per_block;
    }

    template <class B, class A>
    inline auto xdynamic_bitset<B, A>::bit_index(size_type pos) const noexcept -> size_type
    {
        return pos % s_bits_per_block;
    }

    template <class B, class A>
    inline auto xdynamic_bitset<B, A>::bit_mask(size_type pos) const noexcept -> block_type
    {
        return block_type(1) << bit_index(pos);
    }

    template <class B, class A>
    inline auto xdynamic_bitset<B, A>::count_extra_bits() const noexcept -> size_type
    {
        return bit_index(size());
    }

    template <class B, class A>
    inline void xdynamic_bitset<B, A>::zero_unused_bits()
    {
        size_type extra_bits = count_extra_bits();
        //std::cout << "nb extra bits = " << extra_bits << std::endl;
        if (extra_bits != 0)
        {
            m_buffer.back() &= ~(~block_type(0) << extra_bits);
        }
    }

    template <class B, class A>
    inline xdynamic_bitset<B, A> operator~(const xdynamic_bitset<B, A>& lhs)
    {
        xdynamic_bitset<B, A> res(lhs);
        res.flip();
        return res;
    }

    template <class B, class A>
    inline xdynamic_bitset<B, A> operator&(const xdynamic_bitset<B, A>& lhs, const xdynamic_bitset<B, A>& rhs)
    {
        xdynamic_bitset<B, A> res(lhs);
        res &= rhs;
        return res;
    }

    template <class B, class A>
    inline xdynamic_bitset<B, A> operator|(const xdynamic_bitset<B, A>& lhs, const xdynamic_bitset<B, A>& rhs)
    {
        xdynamic_bitset<B, A> res(lhs);
        res |= rhs;
        return res;
    }

    template <class B, class A>
    inline xdynamic_bitset<B, A> operator^(const xdynamic_bitset<B, A>& lhs, const xdynamic_bitset<B, A>& rhs)
    {
        xdynamic_bitset<B, A> res(lhs);
        res ^= rhs;
        return res;
    }

    template <class B, class A>
    inline void swap(const xdynamic_bitset<B, A>& lhs, const xdynamic_bitset<B, A>& rhs)
    {
        return lhs.swap(rhs);
    }

    /************************************
     * xbitset_reference implementation *
     ************************************/

    template <class B, class A, bool C>
    inline xbitset_reference<B, A, C>::xbitset_reference(closure_type block, block_type pos)
        : m_block(block), m_mask(block_type(1) << pos)
    {
    }

    template <class B, class A, bool C>
    inline xbitset_reference<B, A, C>::operator bool() const noexcept
    {
        return (m_block & m_mask) != 0;
    }

    template <class B, class A, bool C>
    inline auto xbitset_reference<B, A, C>::operator=(const self_type& rhs) noexcept -> self_type&
    {
        assign(rhs);
        return *this;
    }

    template <class B, class A, bool C>
    inline auto xbitset_reference<B, A, C>::operator=(self_type&& rhs) noexcept -> self_type&
    {
        assign(rhs);
        return *this;
    }

    template <class B, class A, bool C>
    inline auto xbitset_reference<B, A, C>::operator=(bool rhs) noexcept -> self_type&
    {
        assign(rhs);
        return *this;
    }

    template <class B, class A, bool C>
    inline bool xbitset_reference<B, A, C>::operator~() const noexcept
    {
        return (m_block & m_mask) == 0;
    }

    template <class B, class A, bool C>
    inline auto xbitset_reference<B, A, C>::operator&=(bool rhs) noexcept -> self_type&
    {
        if (!rhs)
        {
            reset();
        }
        return *this;
    }

    template <class B, class A, bool C>
    inline auto xbitset_reference<B, A, C>::operator|=(bool rhs) noexcept -> self_type&
    {
        if (rhs)
        {
            set();
        }
        return *this;
    }

    template <class B, class A, bool C>
    inline auto xbitset_reference<B, A, C>::operator^=(bool rhs) noexcept -> self_type&
    {
        return rhs ? flip() : *this;
    }

    template <class B, class A, bool C>
    inline auto xbitset_reference<B, A, C>::flip() noexcept -> self_type&
    {
        m_block ^= m_mask;
        return *this;
    }

    template <class B, class A, bool C>
    inline auto xbitset_reference<B, A, C>::operator&() noexcept -> pointer
    {
        return pointer(*this);
    }

    template <class B, class A, bool C>
    inline void xbitset_reference<B, A, C>::assign(bool rhs) noexcept
    {
        rhs ? set() : reset();
    }

    template <class B, class A, bool C>
    inline void xbitset_reference<B, A, C>::set() noexcept
    {
        m_block |= m_mask;
    }

    template <class B, class A, bool C>
    inline void xbitset_reference<B, A, C>::reset() noexcept
    {
        m_block &= ~m_mask;
    }

    /***********************************
     * xbitset_iterator implementation *
     ***********************************/

    template <class B, class A, bool C>
    inline xbitset_iterator<B, A, C>::xbitset_iterator() noexcept
        : p_container(nullptr), m_index(0)
    {
    }

    template <class B, class A, bool C>
    inline xbitset_iterator<B, A, C>::xbitset_iterator(container_reference c, size_type index) noexcept
        : p_container(&c), m_index(index)
    {
    }

    template <class B, class A, bool C>
    inline auto xbitset_iterator<B, A, C>::operator++() -> self_type&
    {
        ++m_index;
        return *this;
    }

    template <class B, class A, bool C>
    inline auto xbitset_iterator<B, A, C>::operator--() -> self_type&
    {
        --m_index;
        return *this;
    }

    template <class B, class A, bool C>
    inline auto xbitset_iterator<B, A, C>::operator+=(difference_type n) -> self_type&
    {
        difference_type res = static_cast<difference_type>(m_index) + n;
        m_index = static_cast<size_type>(res);
        return *this;
    }

    template <class B, class A, bool C>
    inline auto xbitset_iterator<B, A, C>::operator-=(difference_type n) -> self_type&
    {
        difference_type res = static_cast<difference_type>(m_index) - n;
        m_index = static_cast<size_type>(res);
        return *this;
    }

    template <class B, class A, bool C>
    inline auto xbitset_iterator<B, A, C>::operator-(const self_type& rhs) const -> difference_type
    {
        return difference_type(m_index - rhs.m_index);
    }

    template <class B, class A, bool C>
    inline auto xbitset_iterator<B, A, C>::operator*() const -> reference
    {
        return (*p_container)[m_index];
    }

    template <class B, class A, bool C>
    inline auto xbitset_iterator<B, A, C>::operator->() const -> pointer
    {
        return &(operator*());
    }

    template <class B, class A, bool C>
    inline bool xbitset_iterator<B, A, C>::operator==(const self_type& rhs) const
    {
        return p_container == rhs.p_container && m_index == rhs.m_index;
    }

    template <class B, class A, bool C>
    inline bool xbitset_iterator<B, A, C>::operator<(const self_type& rhs) const
    {
        return p_container == rhs.p_container && m_index < rhs.m_index;
    }
}

#endif
