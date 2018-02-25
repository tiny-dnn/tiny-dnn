/***************************************************************************
* Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XTENSOR_BUFFER_ADAPTOR_HPP
#define XTENSOR_BUFFER_ADAPTOR_HPP

#include <algorithm>
#include <functional>
#include <iterator>
#include <memory>
#include <stdexcept>

#include "xtl/xclosure.hpp"

#include "xstorage.hpp"

namespace xt
{
    /******************************
     * xbuffer_adator declaration *
     ******************************/

    struct no_ownership
    {
    };

    struct acquire_ownership
    {
    };

    template <class CP, class O = no_ownership, class A = std::allocator<std::remove_pointer_t<std::remove_reference_t<CP>>>>
    class xbuffer_adaptor;

    /*********************************
     * xbuffer_adator implementation *
     *********************************/

    namespace detail
    {
        template <class CP, class A>
        class xbuffer_storage
        {
        public:

            using self_type = xbuffer_storage<CP, A>;
            using allocator_type = A;
            using value_type = typename allocator_type::value_type;
            using reference = typename allocator_type::reference;
            using const_reference = typename allocator_type::const_reference;
            using pointer = typename allocator_type::pointer;
            using const_pointer = typename allocator_type::const_pointer;
            using size_type = typename allocator_type::size_type;
            using difference_type = typename allocator_type::difference_type;

            xbuffer_storage();

            template <class P>
            xbuffer_storage(P&& data, size_type size, const allocator_type& alloc = allocator_type());

            size_type size() const noexcept;
            void resize(size_type size);

            pointer data() noexcept;
            const_pointer data() const noexcept;

            void swap(self_type& rhs) noexcept;

        private:

            pointer p_data;
            size_type m_size;
        };

        template <class CP, class A>
        class xbuffer_owner_storage
        {
        public:

            using self_type = xbuffer_owner_storage<CP, A>;
            using allocator_type = A;
            using value_type = typename allocator_type::value_type;
            using reference = typename allocator_type::reference;
            using const_reference = typename allocator_type::const_reference;
            using pointer = typename allocator_type::pointer;
            using const_pointer = typename allocator_type::const_pointer;
            using size_type = typename allocator_type::size_type;
            using difference_type = typename allocator_type::difference_type;

            xbuffer_owner_storage() = default;

            template <class P>
            xbuffer_owner_storage(P&& data, size_type size, const allocator_type& alloc = allocator_type());

            ~xbuffer_owner_storage();

            xbuffer_owner_storage(const self_type&) = delete;
            self_type& operator=(const self_type&);

            xbuffer_owner_storage(self_type&&);
            self_type& operator=(self_type&&);

            size_type size() const noexcept;
            void resize(size_type size);

            pointer data() noexcept;
            const_pointer data() const noexcept;

            allocator_type get_allocator() const noexcept;

            void swap(self_type& rhs) noexcept;

        private:

            xtl::xclosure_wrapper<CP> m_data;
            size_type m_size;
            bool m_moved_from;
            allocator_type m_allocator;
        };

        template <class CP, class A, class O>
        struct get_buffer_storage
        {
            using type = xbuffer_storage<CP, A>;
        };

        template <class CP, class A>
        struct get_buffer_storage<CP, A, acquire_ownership>
        {
            using type = xbuffer_owner_storage<CP, A>;
        };

        template <class CP, class A, class O>
        using buffer_storage_t = typename get_buffer_storage<CP, A, O>::type;
    }

    template <class CP, class O, class A>
    class xbuffer_adaptor : private detail::buffer_storage_t<CP, A, O>
    {
    public:

        using base_type = detail::buffer_storage_t<CP, A, O>;
        using self_type = xbuffer_adaptor<CP, O, A>;
        using allocator_type = typename base_type::allocator_type;
        using value_type = typename base_type::value_type;
        using reference = typename base_type::reference;
        using const_reference = typename base_type::const_reference;
        using pointer = typename base_type::pointer;
        using const_pointer = typename base_type::const_pointer;
        using temporary_type = uvector<value_type, allocator_type>;

        using size_type = typename base_type::size_type;
        using difference_type = typename base_type::difference_type;

        using iterator = pointer;
        using const_iterator = const_pointer;
        using reverse_iterator = std::reverse_iterator<iterator>;
        using const_reverse_iterator = std::reverse_iterator<const_iterator>;

        xbuffer_adaptor() = default;

        template <class P>
        xbuffer_adaptor(P&& data, size_type size, const allocator_type& alloc = allocator_type());

        ~xbuffer_adaptor() = default;

        xbuffer_adaptor(const self_type&) = default;
        self_type& operator=(const self_type&) = default;

        xbuffer_adaptor(self_type&&) = default;
        xbuffer_adaptor& operator=(self_type&&) = default;

        self_type& operator=(temporary_type&&);

        bool empty() const noexcept;
        using base_type::size;
        using base_type::resize;

        reference operator[](size_type i);
        const_reference operator[](size_type i) const;

        reference front();
        const_reference front() const;

        reference back();
        const_reference back() const;

        iterator begin();
        iterator end();

        const_iterator begin() const;
        const_iterator end() const;
        const_iterator cbegin() const;
        const_iterator cend() const;

        reverse_iterator rbegin();
        reverse_iterator rend();

        const_reverse_iterator rbegin() const;
        const_reverse_iterator rend() const;
        const_reverse_iterator crbegin() const;
        const_reverse_iterator crend() const;

        using base_type::data;
        using base_type::swap;
    };

    template <class CP, class O, class A>
    bool operator==(const xbuffer_adaptor<CP, O, A>& lhs,
                    const xbuffer_adaptor<CP, O, A>& rhs);

    template <class CP, class O, class A>
    bool operator!=(const xbuffer_adaptor<CP, O, A>& lhs,
                    const xbuffer_adaptor<CP, O, A>& rhs);

    template <class CP, class O, class A>
    bool operator<(const xbuffer_adaptor<CP, O, A>& lhs,
                   const xbuffer_adaptor<CP, O, A>& rhs);

    template <class CP, class O, class A>
    bool operator<=(const xbuffer_adaptor<CP, O, A>& lhs,
                    const xbuffer_adaptor<CP, O, A>& rhs);

    template <class CP, class O, class A>
    bool operator>(const xbuffer_adaptor<CP, O, A>& lhs,
                   const xbuffer_adaptor<CP, O, A>& rhs);

    template <class CP, class O, class A>
    bool operator>=(const xbuffer_adaptor<CP, O, A>& lhs,
                    const xbuffer_adaptor<CP, O, A>& rhs);

    template <class CP, class O, class A>
    void swap(xbuffer_adaptor<CP, O, A>& lhs,
              xbuffer_adaptor<CP, O, A>& rhs) noexcept;

    /************************************
     * temporary_container metafunction *
     ************************************/

    template <class C>
    struct temporary_container
    {
        using type = C;
    };

    template <class CP, class O, class A>
    struct temporary_container<xbuffer_adaptor<CP, O, A>>
    {
        using type = typename xbuffer_adaptor<CP, O, A>::temporary_type;
    };

    template <class C>
    using temporary_container_t = typename temporary_container<C>::type;

    /**********************************
     * xbuffer_storage implementation *
     **********************************/

    namespace detail
    {
        template <class CP, class A>
        inline xbuffer_storage<CP, A>::xbuffer_storage()
            : p_data(nullptr), m_size(0)
        {
        }

        template <class CP, class A>
        template <class P>
        inline xbuffer_storage<CP, A>::xbuffer_storage(P&& data, size_type size, const allocator_type&)
            : p_data(std::forward<P>(data)), m_size(size)
        {
        }

        template <class CP, class A>
        inline auto xbuffer_storage<CP, A>::size() const noexcept -> size_type
        {
            return m_size;
        }

        template <class CP, class A>
        inline void xbuffer_storage<CP, A>::resize(size_type size)
        {
            if (size != m_size)
            {
                throw std::runtime_error("xbuffer_storage not resizable");
            }
        }

        template <class CP, class A>
        inline auto xbuffer_storage<CP, A>::data() noexcept -> pointer
        {
            return p_data;
        }

        template <class CP, class A>
        inline auto xbuffer_storage<CP, A>::data() const noexcept -> const_pointer
        {
            return p_data;
        }

        template <class CP, class A>
        inline void xbuffer_storage<CP, A>::swap(self_type& rhs) noexcept
        {
            using std::swap;
            swap(p_data, rhs.p_data);
            swap(m_size, rhs.m_size);
        }
    }

    /****************************************
     * xbuffer_owner_storage implementation *
     ****************************************/

    namespace detail
    {
        template <class CP, class A>
        template <class P>
        inline xbuffer_owner_storage<CP, A>::xbuffer_owner_storage(P&& data, size_type size, const allocator_type& alloc)
            : m_data(std::forward<P>(data)), m_size(size), m_moved_from(false), m_allocator(alloc)
        {
        }

        template <class CP, class A>
        inline xbuffer_owner_storage<CP, A>::~xbuffer_owner_storage()
        {
            if (!m_moved_from)
            {
                safe_destroy_deallocate(m_allocator, m_data.get(), m_size);
                m_size = 0;
            }
        }

        template <class CP, class A>
        inline auto xbuffer_owner_storage<CP, A>::operator=(const self_type& rhs) -> self_type&
        {
            using std::swap;
            if (this != &rhs)
            {
                allocator_type al = std::allocator_traits<allocator_type>::select_on_container_copy_construction(rhs.get_allocator());
                pointer tmp = safe_init_allocate(al, rhs.m_size);
                if (xtrivially_default_constructible<value_type>::value)
                {
                    std::uninitialized_copy(rhs.m_data.get(), rhs.m_data.get() + rhs.m_size, tmp);
                }
                else
                {
                    std::copy(rhs.m_data.get(), rhs.m_data.get() + rhs.m_size, tmp);
                }
                swap(m_data.get(), tmp);
                m_size = rhs.m_size;
                swap(m_allocator, al);
                safe_destroy_deallocate(al, tmp, m_size);
            }
            return *this;
        }

        template <class CP, class A>
        inline xbuffer_owner_storage<CP, A>::xbuffer_owner_storage(self_type&& rhs)
            : m_data(std::move(rhs.m_data)), m_size(std::move(rhs.m_size)), m_moved_from(std::move(rhs.m_moved_from)), m_allocator(std::move(rhs.m_allocator))
        {
            rhs.m_moved_from = true;
            rhs.m_size = 0;
        }

        template <class CP, class A>
        inline auto xbuffer_owner_storage<CP, A>::operator=(self_type&& rhs) -> self_type&
        {
            swap(rhs);
            rhs.m_moved_from = true;
            return *this;
        }

        template <class CP, class A>
        inline auto xbuffer_owner_storage<CP, A>::size() const noexcept -> size_type
        {
            return m_size;
        }

        template <class CP, class A>
        void xbuffer_owner_storage<CP, A>::resize(size_type size)
        {
            using std::swap;
            if (size != m_size)
            {
                pointer tmp = safe_init_allocate(m_allocator, size);
                swap(m_data.get(), tmp);
                swap(m_size, size);
                safe_destroy_deallocate(m_allocator, tmp, size);
            }
        }

        template <class CP, class A>
        inline auto xbuffer_owner_storage<CP, A>::data() noexcept -> pointer
        {
            return m_data.get();
        }

        template <class CP, class A>
        inline auto xbuffer_owner_storage<CP, A>::data() const noexcept -> const_pointer
        {
            return m_data.get();
        }

        template <class CP, class A>
        inline auto xbuffer_owner_storage<CP, A>::get_allocator() const noexcept -> allocator_type
        {
            return allocator_type(m_allocator);
        }

        template <class CP, class A>
        inline void xbuffer_owner_storage<CP, A>::swap(self_type& rhs) noexcept
        {
            using std::swap;
            swap(m_data, rhs.m_data);
            swap(m_size, rhs.m_size);
            swap(m_allocator, rhs.m_allocator);
        }
    }

    /**********************************
     * xbuffer_adaptor implementation *
     **********************************/

    template <class CP, class O, class A>
    template <class P>
    inline xbuffer_adaptor<CP, O, A>::xbuffer_adaptor(P&& data, size_type size, const allocator_type& alloc)
        : base_type(std::forward<P>(data), size, alloc)
    {
    }

    template <class CP, class O, class A>
    inline auto xbuffer_adaptor<CP, O, A>::operator=(temporary_type&& tmp) -> self_type&
    {
        base_type::resize(tmp.size());
        std::copy(tmp.cbegin(), tmp.cend(), begin());
        return *this;
    }

    template <class CP, class O, class A>
    bool xbuffer_adaptor<CP, O, A>::empty() const noexcept
    {
        return size() == 0;
    }

    template <class CP, class O, class A>
    inline auto xbuffer_adaptor<CP, O, A>::operator[](size_type i) -> reference
    {
        return data()[i];
    }

    template <class CP, class O, class A>
    inline auto xbuffer_adaptor<CP, O, A>::operator[](size_type i) const -> const_reference
    {
        return data()[i];
    }

    template <class CP, class O, class A>
    inline auto xbuffer_adaptor<CP, O, A>::front() -> reference
    {
        return data()[0];
    }

    template <class CP, class O, class A>
    inline auto xbuffer_adaptor<CP, O, A>::front() const -> const_reference
    {
        return data()[0];
    }

    template <class CP, class O, class A>
    inline auto xbuffer_adaptor<CP, O, A>::back() -> reference
    {
        return data()[size() - 1];
    }

    template <class CP, class O, class A>
    inline auto xbuffer_adaptor<CP, O, A>::back() const -> const_reference
    {
        return data()[size() - 1];
    }

    template <class CP, class O, class A>
    inline auto xbuffer_adaptor<CP, O, A>::begin() -> iterator
    {
        return data();
    }

    template <class CP, class O, class A>
    inline auto xbuffer_adaptor<CP, O, A>::end() -> iterator
    {
        return data() + size();
    }

    template <class CP, class O, class A>
    inline auto xbuffer_adaptor<CP, O, A>::begin() const -> const_iterator
    {
        return data();
    }

    template <class CP, class O, class A>
    inline auto xbuffer_adaptor<CP, O, A>::end() const -> const_iterator
    {
        return data() + size();
    }

    template <class CP, class O, class A>
    inline auto xbuffer_adaptor<CP, O, A>::cbegin() const -> const_iterator
    {
        return begin();
    }

    template <class CP, class O, class A>
    inline auto xbuffer_adaptor<CP, O, A>::cend() const -> const_iterator
    {
        return end();
    }

    template <class CP, class O, class A>
    inline auto xbuffer_adaptor<CP, O, A>::rbegin() -> reverse_iterator
    {
        return reverse_iterator(end());
    }

    template <class CP, class O, class A>
    inline auto xbuffer_adaptor<CP, O, A>::rend() -> reverse_iterator
    {
        return reverse_iterator(begin());
    }

    template <class CP, class O, class A>
    inline auto xbuffer_adaptor<CP, O, A>::rbegin() const -> const_reverse_iterator
    {
        return const_reverse_iterator(end());
    }

    template <class CP, class O, class A>
    inline auto xbuffer_adaptor<CP, O, A>::rend() const -> const_reverse_iterator
    {
        return const_reverse_iterator(begin());
    }

    template <class CP, class O, class A>
    inline auto xbuffer_adaptor<CP, O, A>::crbegin() const -> const_reverse_iterator
    {
        return rbegin();
    }

    template <class CP, class O, class A>
    inline auto xbuffer_adaptor<CP, O, A>::crend() const -> const_reverse_iterator
    {
        return rend();
    }

    template <class CP, class O, class A>
    inline bool operator==(const xbuffer_adaptor<CP, O, A>& lhs,
                           const xbuffer_adaptor<CP, O, A>& rhs)
    {
        return lhs.size() == rhs.size() && std::equal(lhs.begin(), lhs.end(), rhs.begin());
    }

    template <class CP, class O, class A>
    inline bool operator!=(const xbuffer_adaptor<CP, O, A>& lhs,
                           const xbuffer_adaptor<CP, O, A>& rhs)
    {
        return !(lhs == rhs);
    }

    template <class CP, class O, class A>
    inline bool operator<(const xbuffer_adaptor<CP, O, A>& lhs,
                          const xbuffer_adaptor<CP, O, A>& rhs)
    {
        return std::lexicographical_compare(lhs.begin(), lhs.end(),
                                            rhs.begin(), rhs.end(),
                                            std::less<typename A::value_type>());
    }

    template <class CP, class O, class A>
    inline bool operator<=(const xbuffer_adaptor<CP, O, A>& lhs,
                           const xbuffer_adaptor<CP, O, A>& rhs)
    {
        return std::lexicographical_compare(lhs.begin(), lhs.end(),
                                            rhs.begin(), rhs.end(),
                                            std::less_equal<typename A::value_type>());
    }

    template <class CP, class O, class A>
    inline bool operator>(const xbuffer_adaptor<CP, O, A>& lhs,
                          const xbuffer_adaptor<CP, O, A>& rhs)
    {
        return std::lexicographical_compare(lhs.begin(), lhs.end(),
                                            rhs.begin(), rhs.end(),
                                            std::greater<typename A::value_type>());
    }

    template <class CP, class O, class A>
    inline bool operator>=(const xbuffer_adaptor<CP, O, A>& lhs,
                           const xbuffer_adaptor<CP, O, A>& rhs)
    {
        return std::lexicographical_compare(lhs.begin(), lhs.end(),
                                            rhs.begin(), rhs.end(),
                                            std::greater_equal<typename A::value_type>());
    }

    template <class CP, class O, class A>
    inline void swap(xbuffer_adaptor<CP, O, A>& lhs,
                     xbuffer_adaptor<CP, O, A>& rhs) noexcept
    {
        lhs.swap(rhs);
    }
}

#endif
