/***************************************************************************
* Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XTENSOR_STORAGE_HPP
#define XTENSOR_STORAGE_HPP

#include <algorithm>
#include <functional>
#include <initializer_list>
#include <iterator>
#include <memory>
#include <type_traits>

#include "xexception.hpp"
#include "xutils.hpp"

#ifndef XSHAPE_ALIGNMENT
    #ifdef XTENSOR_USE_XSIMD
        #include "xsimd/xsimd.hpp"
        #define XSHAPE_ALIGNMENT XSIMD_DEFAULT_ALIGNMENT
    #else
        #define XSHAPE_ALIGNMENT T
    #endif
#endif

namespace xt
{

    namespace detail
    {
        template <class It>
        using require_input_iter = typename std::enable_if<std::is_convertible<typename std::iterator_traits<It>::iterator_category,
                                                                               std::input_iterator_tag>::value>::type;
    }

    template <class T, class Allocator = std::allocator<T>>
    class uvector
    {
    public:

        using allocator_type = Allocator;

        using value_type = typename allocator_type::value_type;
        using reference = typename allocator_type::reference;
        using const_reference = typename allocator_type::const_reference;
        using pointer = typename allocator_type::pointer;
        using const_pointer = typename allocator_type::const_pointer;

        using size_type = typename allocator_type::size_type;
        using difference_type = typename allocator_type::difference_type;

        using iterator = pointer;
        using const_iterator = const_pointer;
        using reverse_iterator = std::reverse_iterator<iterator>;
        using const_reverse_iterator = std::reverse_iterator<const_iterator>;

        uvector() noexcept;
        explicit uvector(const allocator_type& alloc) noexcept;
        explicit uvector(size_type count, const allocator_type& alloc = allocator_type());
        uvector(size_type count, const_reference value, const allocator_type& alloc = allocator_type());

        template <class InputIt, class = detail::require_input_iter<InputIt>>
        uvector(InputIt first, InputIt last, const allocator_type& alloc = allocator_type());

        uvector(std::initializer_list<T> init, const allocator_type& alloc = allocator_type());

        ~uvector();

        uvector(const uvector& rhs);
        uvector(const uvector& rhs, const allocator_type& alloc);
        uvector& operator=(const uvector&);

        uvector(uvector&& rhs) noexcept;
        uvector(uvector&& rhs, const allocator_type& alloc) noexcept;
        uvector& operator=(uvector&& rhs) noexcept;

        allocator_type get_allocator() const noexcept;

        bool empty() const noexcept;
        size_type size() const noexcept;
        void resize(size_type size);

        reference operator[](size_type i);
        const_reference operator[](size_type i) const;

        reference front();
        const_reference front() const;

        reference back();
        const_reference back() const;

        pointer data() noexcept;
        const_pointer data() const noexcept;

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

        void swap(uvector& rhs) noexcept;

    private:

        template <class I>
        void init_data(I first, I last);

        void resize_impl(size_type new_size);

        allocator_type m_allocator;

        // Storing a pair of pointers is more efficient for iterating than
        // storing a pointer to the beginning and the size of the container
        pointer p_begin;
        pointer p_end;
    };

    template <class T, class A>
    bool operator==(const uvector<T, A>& lhs, const uvector<T, A>& rhs);

    template <class T, class A>
    bool operator!=(const uvector<T, A>& lhs, const uvector<T, A>& rhs);

    template <class T, class A>
    bool operator<(const uvector<T, A>& lhs, const uvector<T, A>& rhs);

    template <class T, class A>
    bool operator<=(const uvector<T, A>& lhs, const uvector<T, A>& rhs);

    template <class T, class A>
    bool operator>(const uvector<T, A>& lhs, const uvector<T, A>& rhs);

    template <class T, class A>
    bool operator>=(const uvector<T, A>& lhs, const uvector<T, A>& rhs);

    template <class T, class A>
    void swap(uvector<T, A>& lhs, uvector<T, A>& rhs) noexcept;

    /**************************
     * uvector implementation *
     **************************/

    namespace detail
    {
        template <class A>
        inline typename A::pointer safe_init_allocate(A& alloc, typename A::size_type size)
        {
            using pointer = typename A::pointer;
            using value_type = typename A::value_type;
            pointer res = alloc.allocate(size);
            if (!xtrivially_default_constructible<value_type>::value)
            {
                for (pointer p = res; p != res + size; ++p)
                {
                    alloc.construct(p, value_type());
                }
            }
            return res;
        }

        template <class A>
        inline void safe_destroy_deallocate(A& alloc, typename A::pointer ptr, typename A::size_type size)
        {
            using pointer = typename A::pointer;
            using value_type = typename A::value_type;
            if (ptr != nullptr)
            {
                if (!xtrivially_default_constructible<value_type>::value)
                {
                    for (pointer p = ptr; p != ptr + size; ++p)
                    {
                        alloc.destroy(p);
                    }
                }
                alloc.deallocate(ptr, size);
            }
        }
    }

    template <class T, class A>
    template <class I>
    inline void uvector<T, A>::init_data(I first, I last)
    {
        size_type size = static_cast<size_type>(std::distance(first, last));
        if (size != size_type(0))
        {
            p_begin = m_allocator.allocate(size);
            std::uninitialized_copy(first, last, p_begin);
            p_end = p_begin + size;
        }
    }

    template <class T, class A>
    inline void uvector<T, A>::resize_impl(size_type new_size)
    {
        size_type old_size = size();
        pointer old_begin = p_begin;
        if (new_size != old_size)
        {
            p_begin = detail::safe_init_allocate(m_allocator, new_size);
            p_end = p_begin + new_size;
            detail::safe_destroy_deallocate(m_allocator, old_begin, old_size);
        }
    }

    template <class T, class A>
    inline uvector<T, A>::uvector() noexcept
        : uvector(allocator_type())
    {
    }

    template <class T, class A>
    inline uvector<T, A>::uvector(const allocator_type& alloc) noexcept
        : m_allocator(alloc), p_begin(nullptr), p_end(nullptr)
    {
    }

    template <class T, class A>
    inline uvector<T, A>::uvector(size_type count, const allocator_type& alloc)
        : m_allocator(alloc), p_begin(nullptr), p_end(nullptr)
    {
        if (count != 0)
        {
            p_begin = detail::safe_init_allocate(m_allocator, count);
            p_end = p_begin + count;
        }
    }

    template <class T, class A>
    inline uvector<T, A>::uvector(size_type count, const_reference value, const allocator_type& alloc)
        : m_allocator(alloc), p_begin(nullptr), p_end(nullptr)
    {
        if (count != 0)
        {
            p_begin = m_allocator.allocate(count);
            p_end = p_begin + count;
            std::uninitialized_fill(p_begin, p_end, value);
        }
    }

    template <class T, class A>
    template <class InputIt, class>
    inline uvector<T, A>::uvector(InputIt first, InputIt last, const allocator_type& alloc)
        : m_allocator(alloc), p_begin(nullptr), p_end(nullptr)
    {
        init_data(first, last);
    }

    template <class T, class A>
    inline uvector<T, A>::uvector(std::initializer_list<T> init, const allocator_type& alloc)
        : m_allocator(alloc), p_begin(nullptr), p_end(nullptr)
    {
        init_data(init.begin(), init.end());
    }

    template <class T, class A>
    inline uvector<T, A>::~uvector()
    {
        detail::safe_destroy_deallocate(m_allocator, p_begin, size());
        p_begin = nullptr;
        p_end = nullptr;
    }

    template <class T, class A>
    inline uvector<T, A>::uvector(const uvector& rhs)
        : m_allocator(std::allocator_traits<allocator_type>::select_on_container_copy_construction(rhs.get_allocator())),
          p_begin(nullptr), p_end(nullptr)
    {
        init_data(rhs.p_begin, rhs.p_end);
    }

    template <class T, class A>
    inline uvector<T, A>::uvector(const uvector& rhs, const allocator_type& alloc)
        : m_allocator(alloc), p_begin(nullptr), p_end(nullptr)
    {
        init_data(rhs.p_begin, rhs.p_end);
    }

    template <class T, class A>
    inline uvector<T, A>& uvector<T, A>::operator=(const uvector& rhs)
    {
        // No copy and swap idiom here due to performance issues
        if (this != &rhs)
        {
            m_allocator = std::allocator_traits<allocator_type>::select_on_container_copy_construction(rhs.get_allocator());
            resize_impl(rhs.size());
            if (xtrivially_default_constructible<value_type>::value)
            {
                std::uninitialized_copy(rhs.p_begin, rhs.p_end, p_begin);
            }
            else
            {
                std::copy(rhs.p_begin, rhs.p_end, p_begin);
            }
        }
        return *this;
    }

    template <class T, class A>
    inline uvector<T, A>::uvector(uvector&& rhs) noexcept
        : m_allocator(std::move(rhs.m_allocator)), p_begin(rhs.p_begin), p_end(rhs.p_end)
    {
        rhs.p_begin = nullptr;
        rhs.p_end = nullptr;
    }

    template <class T, class A>
    inline uvector<T, A>::uvector(uvector&& rhs, const allocator_type& alloc) noexcept
        : m_allocator(alloc), p_begin(rhs.p_begin), p_end(rhs.p_end)
    {
        rhs.p_begin = nullptr;
        rhs.p_end = nullptr;
    }

    template <class T, class A>
    inline uvector<T, A>& uvector<T, A>::operator=(uvector&& rhs) noexcept
    {
        using std::swap;
        uvector tmp(std::move(rhs));
        swap(p_begin, tmp.p_begin);
        swap(p_end, tmp.p_end);
        return *this;
    }

    template <class T, class A>
    inline auto uvector<T, A>::get_allocator() const noexcept -> allocator_type
    {
        return allocator_type(m_allocator);
    }

    template <class T, class A>
    inline bool uvector<T, A>::empty() const noexcept
    {
        return size() == size_type(0);
    }

    template <class T, class A>
    inline auto uvector<T, A>::size() const noexcept -> size_type
    {
        return static_cast<size_type>(p_end - p_begin);
    }

    template <class T, class A>
    inline void uvector<T, A>::resize(size_type size)
    {
        resize_impl(size);
    }

    template <class T, class A>
    inline auto uvector<T, A>::operator[](size_type i) -> reference
    {
        return p_begin[i];
    }

    template <class T, class A>
    inline auto uvector<T, A>::operator[](size_type i) const -> const_reference
    {
        return p_begin[i];
    }

    template <class T, class A>
    inline auto uvector<T, A>::front() -> reference
    {
        return p_begin[0];
    }

    template <class T, class A>
    inline auto uvector<T, A>::front() const -> const_reference
    {
        return p_begin[0];
    }

    template <class T, class A>
    inline auto uvector<T, A>::back() -> reference
    {
        return *(p_end - 1);
    }

    template <class T, class A>
    inline auto uvector<T, A>::back() const -> const_reference
    {
        return *(p_end - 1);
    }

    template <class T, class A>
    inline auto uvector<T, A>::data() noexcept -> pointer
    {
        return p_begin;
    }

    template <class T, class A>
    inline auto uvector<T, A>::data() const noexcept -> const_pointer
    {
        return p_begin;
    }

    template <class T, class A>
    inline auto uvector<T, A>::begin() noexcept -> iterator
    {
        return p_begin;
    }

    template <class T, class A>
    inline auto uvector<T, A>::end() noexcept -> iterator
    {
        return p_end;
    }

    template <class T, class A>
    inline auto uvector<T, A>::begin() const noexcept -> const_iterator
    {
        return p_begin;
    }

    template <class T, class A>
    inline auto uvector<T, A>::end() const noexcept -> const_iterator
    {
        return p_end;
    }

    template <class T, class A>
    inline auto uvector<T, A>::cbegin() const noexcept -> const_iterator
    {
        return begin();
    }

    template <class T, class A>
    inline auto uvector<T, A>::cend() const noexcept -> const_iterator
    {
        return end();
    }

    template <class T, class A>
    inline auto uvector<T, A>::rbegin() noexcept -> reverse_iterator
    {
        return reverse_iterator(end());
    }

    template <class T, class A>
    inline auto uvector<T, A>::rend() noexcept -> reverse_iterator
    {
        return reverse_iterator(begin());
    }

    template <class T, class A>
    inline auto uvector<T, A>::rbegin() const noexcept -> const_reverse_iterator
    {
        return const_reverse_iterator(end());
    }

    template <class T, class A>
    inline auto uvector<T, A>::rend() const noexcept -> const_reverse_iterator
    {
        return const_reverse_iterator(begin());
    }

    template <class T, class A>
    inline auto uvector<T, A>::crbegin() const noexcept -> const_reverse_iterator
    {
        return rbegin();
    }

    template <class T, class A>
    inline auto uvector<T, A>::crend() const noexcept -> const_reverse_iterator
    {
        return rend();
    }

    template <class T, class A>
    inline void uvector<T, A>::swap(uvector<T, A>& rhs) noexcept
    {
        using std::swap;
        swap(m_allocator, rhs.m_allocator);
        swap(p_begin, rhs.p_begin);
        swap(p_end, rhs.p_end);
    }

    template <class T, class A>
    inline bool operator==(const uvector<T, A>& lhs, const uvector<T, A>& rhs)
    {
        return lhs.size() == rhs.size() && std::equal(lhs.begin(), lhs.end(), rhs.begin());
    }

    template <class T, class A>
    inline bool operator!=(const uvector<T, A>& lhs, const uvector<T, A>& rhs)
    {
        return !(lhs == rhs);
    }

    template <class T, class A>
    inline bool operator<(const uvector<T, A>& lhs, const uvector<T, A>& rhs)
    {
        return std::lexicographical_compare(lhs.begin(), lhs.end(),
                                            rhs.begin(), rhs.end());
    }

    template <class T, class A>
    inline bool operator<=(const uvector<T, A>& lhs, const uvector<T, A>& rhs)
    {
        return !(lhs > rhs);
    }

    template <class T, class A>
    inline bool operator>(const uvector<T, A>& lhs, const uvector<T, A>& rhs)
    {
        return rhs < lhs;
    }

    template <class T, class A>
    inline bool operator>=(const uvector<T, A>& lhs, const uvector<T, A>& rhs)
    {
        return !(lhs < rhs);
    }

    template <class T, class A>
    inline void swap(uvector<T, A>& lhs, uvector<T, A>& rhs) noexcept
    {
        lhs.swap(rhs);
    }

    /**************************
     * svector implementation *
     **************************/

    template <class T, std::size_t N, class A = std::allocator<T>, bool Init = true>
    class svector
    {
    public:

        using self_type = svector<T, N, A, Init>;
        using allocator_type = A;
        using size_type = typename A::size_type;
        using value_type = typename A::value_type;
        using pointer = typename A::pointer;
        using const_pointer = typename A::const_pointer;
        using reference = typename A::reference;
        using const_reference = typename A::const_reference;
        using difference_type = typename A::difference_type;

        using iterator = pointer;
        using const_iterator = const_pointer;
        using reverse_iterator = std::reverse_iterator<iterator>;
        using const_reverse_iterator = std::reverse_iterator<const_iterator>;

        svector() noexcept;
        ~svector();

        explicit svector(const allocator_type& alloc) noexcept;
        explicit svector(size_type n, const allocator_type& alloc = allocator_type());
        svector(size_type n, const value_type& v, const allocator_type& alloc = allocator_type());
        svector(std::initializer_list<T> il, const allocator_type& alloc = allocator_type());

        explicit svector(const std::vector<T>& vec);

        template <class IT, class = detail::require_input_iter<IT>>
        svector(IT begin, IT end, const allocator_type& alloc = allocator_type());

        svector& operator=(const svector& rhs);
        svector& operator=(svector&& rhs);
        svector& operator=(std::vector<T>& rhs);

        svector(const svector& other);
        svector(svector&& other);

        void assign(size_type n, const value_type& v);

        template <class V>
        void assign(std::initializer_list<V> il);

        template <class IT>
        void assign(IT other_begin, IT other_end);

        reference operator[](size_type idx);
        const_reference operator[](size_type idx) const;

        pointer data();
        const_pointer data() const;

        void resize(size_type n);

        size_type capacity() const;
        void push_back(const T& elt);

        void pop_back();

        iterator begin();
        const_iterator begin() const;
        const_iterator cbegin() const;
        iterator end();
        const_iterator end() const;
        const_iterator cend() const;

        reverse_iterator rbegin();
        const_reverse_iterator rbegin() const;
        const_reverse_iterator crbegin() const;
        reverse_iterator rend();
        const_reverse_iterator rend() const;
        const_reverse_iterator crend() const;

        size_type size() const;

        bool empty() const;

        reference front();
        const_reference front() const;
        reference back();
        const_reference back() const;

        bool on_stack();

        iterator erase(const_iterator cit);
        iterator erase(const_iterator cfirst, const_iterator clast);

        iterator insert(const_iterator it, const T& elt);

        template <std::size_t ON, class OA, bool InitA>
        void swap(svector<T, ON, OA, InitA>& rhs);

        allocator_type get_allocator() const noexcept;

    private:

        A m_allocator;

        T* m_begin = std::begin(m_data);
        T* m_end = std::begin(m_data);
        T* m_capacity = std::end(m_data);

        // stack allocated memory
        alignas(XSHAPE_ALIGNMENT) T m_data[N > 0 ? N : 1];

        void grow(size_type min_capacity = 0);
        void destroy_range(T* begin, T* end);
    };

    template <class T, std::size_t N, class A, bool Init>
    inline svector<T, N, A, Init>::~svector()
    {
        if (!on_stack())
        {
            detail::safe_destroy_deallocate(m_allocator, m_begin, static_cast<std::size_t>(m_capacity - m_begin));
        }
    }

    template <class T, std::size_t N, class A, bool Init>
    inline svector<T, N, A, Init>::svector() noexcept
        : svector(allocator_type())
    {
    }

    template <class T, std::size_t N, class A, bool Init>
    inline svector<T, N, A, Init>::svector(const allocator_type& alloc) noexcept
        : m_allocator(alloc)
    {
    }

    template <class T, std::size_t N, class A, bool Init>
    inline svector<T, N, A, Init>::svector(size_type n, const allocator_type& alloc)
        : m_allocator(alloc)
    {
        if (Init)
        {
            assign(n, T(0));
        }
        else
        {
            resize(n);
        }
    }

    template <class T, std::size_t N, class A, bool Init>
    template <class IT, class>
    inline svector<T, N, A, Init>::svector(IT begin, IT end, const allocator_type& alloc)
        : m_allocator(alloc)
    {
        assign(begin, end);
    }

    template <class T, std::size_t N, class A, bool Init>
    inline svector<T, N, A, Init>::svector(const std::vector<T>& vec)
    {
        assign(vec.begin(), vec.end());
    }

    template <class T, std::size_t N, class A, bool Init>
    inline svector<T, N, A, Init>::svector(size_type n, const value_type& v, const allocator_type& alloc)
        : m_allocator(alloc)
    {
        assign(n, v);
    }

    template <class T, std::size_t N, class A, bool Init>
    inline svector<T, N, A, Init>::svector(std::initializer_list<T> il, const allocator_type& alloc)
        : m_allocator(alloc)
    {
        assign(il.begin(), il.end());
    }

    template <class T, std::size_t N, class A, bool Init>
    inline svector<T, N, A, Init>& svector<T, N, A, Init>::operator=(const svector& rhs)
    {
        assign(rhs.begin(), rhs.end());
        return *this;
    }

    template <class T, std::size_t N, class A, bool Init>
    inline svector<T, N, A, Init>& svector<T, N, A, Init>::operator=(svector&& rhs)
    {
        assign(rhs.begin(), rhs.end());
        return *this;
    }

    template <class T, std::size_t N, class A, bool Init>
    inline svector<T, N, A, Init>& svector<T, N, A, Init>::operator=(std::vector<T>& rhs)
    {
        if (this != &rhs)
        {
            m_allocator = std::allocator_traits<allocator_type>::select_on_container_copy_construction(rhs.get_allocator());
            assign(rhs.begin(), rhs.end());
        }
        return *this;
    }

    template <class T, std::size_t N, class A, bool Init>
    inline svector<T, N, A, Init>::svector(const svector& rhs)
        : m_allocator(std::allocator_traits<allocator_type>::select_on_container_copy_construction(rhs.get_allocator()))
    {
        assign(rhs.begin(), rhs.end());
    }

    template <class T, std::size_t N, class A, bool Init>
    inline svector<T, N, A, Init>::svector(svector&& rhs)
    {
        this->swap(rhs);
    }

    template <class T, std::size_t N, class A, bool Init>
    inline void svector<T, N, A, Init>::assign(size_type n, const value_type& v)
    {
        if (n > N)
        {
            grow(n);
        }
        m_end = m_begin + n;
        std::fill(begin(), end(), v);
    }

    template <class T, std::size_t N, class A, bool Init>
    template <class V>
    inline void svector<T, N, A, Init>::assign(std::initializer_list<V> il)
    {
        assign(il.begin(), il.end());
    }

    template <class T, std::size_t N, class A, bool Init>
    template <class IT>
    inline void svector<T, N, A, Init>::assign(IT other_begin, IT other_end)
    {
        std::size_t size = static_cast<std::size_t>(other_end - other_begin);
        if (size > N)
        {
            grow(size);
        }
        std::uninitialized_copy(other_begin, other_end, m_begin);
        m_end = m_begin + size;
    }

    template <class T, std::size_t N, class A, bool Init>
    inline auto svector<T, N, A, Init>::operator[](size_type idx) -> reference
    {
        return m_begin[idx];
    }

    template <class T, std::size_t N, class A, bool Init>
    inline auto svector<T, N, A, Init>::operator[](size_type idx) const -> const_reference
    {
        return m_begin[idx];
    }

    template <class T, std::size_t N, class A, bool Init>
    inline auto svector<T, N, A, Init>::data() -> pointer
    {
        return m_begin;
    }

    template <class T, std::size_t N, class A, bool Init>
    inline auto svector<T, N, A, Init>::data() const -> const_pointer
    {
        return m_begin;
    }

    template <class T, std::size_t N, class A, bool Init>
    void svector<T, N, A, Init>::resize(size_type n)
    {
        if (n > N)
        {
            grow(n);
        }
        m_end = m_begin + n;
        if (Init)
        {
            std::fill(begin(), end(), T());
        }
    }

    template <class T, std::size_t N, class A, bool Init>
    inline auto svector<T, N, A, Init>::capacity() const -> size_type
    {
        return static_cast<std::size_t>(m_capacity - m_begin);
    }

    template <class T, std::size_t N, class A, bool Init>
    void svector<T, N, A, Init>::push_back(const T& elt)
    {
        if (m_end >= m_capacity)
        {
            grow();
        }
        *(m_end++) = elt;
    }

    template <class T, std::size_t N, class A, bool Init>
    void svector<T, N, A, Init>::pop_back()
    {
        --m_end;
    }

    template <class T, std::size_t N, class A, bool Init>
    inline auto svector<T, N, A, Init>::begin() -> iterator
    {
        return m_begin;
    }

    template <class T, std::size_t N, class A, bool Init>
    inline auto svector<T, N, A, Init>::begin() const -> const_iterator
    {
        return m_begin;
    }

    template <class T, std::size_t N, class A, bool Init>
    inline auto svector<T, N, A, Init>::cbegin() const -> const_iterator
    {
        return m_begin;
    }

    template <class T, std::size_t N, class A, bool Init>
    inline auto svector<T, N, A, Init>::end() -> iterator
    {
        return m_end;
    }

    template <class T, std::size_t N, class A, bool Init>
    inline auto svector<T, N, A, Init>::end() const -> const_iterator
    {
        return m_end;
    }

    template <class T, std::size_t N, class A, bool Init>
    inline auto svector<T, N, A, Init>::cend() const -> const_iterator
    {
        return m_end;
    }

    template <class T, std::size_t N, class A, bool Init>
    inline auto svector<T, N, A, Init>::rbegin() -> reverse_iterator
    {
        return reverse_iterator(m_end);
    }

    template <class T, std::size_t N, class A, bool Init>
    inline auto svector<T, N, A, Init>::rbegin() const -> const_reverse_iterator
    {
        return const_reverse_iterator(m_end);
    }

    template <class T, std::size_t N, class A, bool Init>
    inline auto svector<T, N, A, Init>::crbegin() const -> const_reverse_iterator
    {
        return const_reverse_iterator(m_end);
    }

    template <class T, std::size_t N, class A, bool Init>
    inline auto svector<T, N, A, Init>::rend() -> reverse_iterator
    {
        return reverse_iterator(m_begin);
    }

    template <class T, std::size_t N, class A, bool Init>
    inline auto svector<T, N, A, Init>::rend() const -> const_reverse_iterator
    {
        return const_reverse_iterator(m_begin);
    }

    template <class T, std::size_t N, class A, bool Init>
    inline auto svector<T, N, A, Init>::crend() const -> const_reverse_iterator
    {
        return const_reverse_iterator(m_begin);
    }

    template <class T, std::size_t N, class A, bool Init>
    inline auto svector<T, N, A, Init>::size() const -> size_type
    {
        return static_cast<size_type>(m_end - m_begin);
    }

    template <class T, std::size_t N, class A, bool Init>
    inline auto svector<T, N, A, Init>::empty() const -> bool
    {
        return m_begin == m_end;
    }

    template <class T, std::size_t N, class A, bool Init>
    inline auto svector<T, N, A, Init>::front() -> reference
    {
        XTENSOR_ASSERT(!empty());
        return m_begin[0];
    }

    template <class T, std::size_t N, class A, bool Init>
    inline auto svector<T, N, A, Init>::front() const -> const_reference
    {
        XTENSOR_ASSERT(!empty());
        return m_begin[0];
    }

    template <class T, std::size_t N, class A, bool Init>
    inline auto svector<T, N, A, Init>::back() -> reference
    {
        XTENSOR_ASSERT(!empty());
        return m_end[-1];
    }

    template <class T, std::size_t N, class A, bool Init>
    inline auto svector<T, N, A, Init>::back() const -> const_reference
    {
        XTENSOR_ASSERT(!empty());
        return m_end[-1];
    }

    template <class T, std::size_t N, class A, bool Init>
    inline auto svector<T, N, A, Init>::on_stack() -> bool
    {
        return m_begin == &m_data[0];
    }

    template <class T, std::size_t N, class A, bool Init>
    inline auto svector<T, N, A, Init>::get_allocator() const noexcept -> allocator_type
    {
        return m_allocator;
    }

    template <class T, std::size_t N, class A, bool Init>
    inline auto svector<T, N, A, Init>::erase(const_iterator cit) -> iterator
    {
        auto it = const_cast<pointer>(cit);
        iterator ret_val = it;
        std::move(it + 1, m_end, it);
        --m_end;
        return ret_val;
    }

    template <class T, std::size_t N, class A, bool Init>
    inline auto svector<T, N, A, Init>::erase(const_iterator cfirst, const_iterator clast) -> iterator
    {
        auto first = const_cast<pointer>(cfirst);
        auto last = const_cast<pointer>(clast);
        if (last == m_end)
        {
            m_end = first;
            return first;
        }

        iterator new_end = std::move(last, m_end, first);
        m_end = new_end;
        return first;
    }

    template <class T, std::size_t N, class A, bool Init>
    inline auto svector<T, N, A, Init>::insert(const_iterator cit, const T& elt) -> iterator
    {
        auto it = const_cast<pointer>(cit);
        if (it == m_end)
        {
            push_back(elt);
            return m_end - 1;
        }

        if (m_end >= m_capacity)
        {
            ptrdiff_t elt_no = it - m_begin;
            grow();
            it = m_begin + elt_no;
        }

        (*m_end) = back();
        std::move_backward(it, m_end - 1, m_end);
        ++m_end;

        // Update ref if element moved
        const T* elt_ptr = &elt;
        if (it <= elt_ptr && elt_ptr < m_end)
        {
            ++elt_ptr;
        }
        *it = *elt_ptr;
        return it;
    }

    template <class T, std::size_t N, class A, bool Init>
    inline void svector<T, N, A, Init>::destroy_range(T* begin, T* end)
    {
        if (!xtrivially_default_constructible<T>::value)
        {
            while (begin != end)
            {
                --end;
                end->~T();
            }
        }
    }

    template <class T, std::size_t N, class A, bool Init>
    template <std::size_t ON, class OA, bool InitA>
    inline void svector<T, N, A, Init>::swap(svector<T, ON, OA, InitA>& rhs)
    {
        if (this == &rhs)
        {
            return;
        }

        // We can only avoid copying elements if neither vector is small.
        if (!this->on_stack() && !rhs.on_stack()) {
            std::swap(this->m_begin, rhs.m_begin);
            std::swap(this->m_end, rhs.m_end);
            std::swap(this->m_capacity, rhs.m_capacity);
            return;
        }

        if (rhs.size() > this->capacity())
            this->resize(rhs.size());
        if (this->size() > rhs.capacity())
            rhs.resize(this->size());

        // Swap the shared elements.
        size_t num_shared = std::min(this->size(), rhs.size());

        for (size_type i = 0; i != num_shared; ++i)
        {
            std::swap((*this)[i], rhs[i]);
        }

        // Copy over the extra elts.
        if (this->size() > rhs.size())
        {
            size_t elements_diff = this->size() - rhs.size();
            std::copy(this->begin() + num_shared, this->end(), rhs.end());
            rhs.m_end = rhs.end() + elements_diff;
            this->destroy_range(this->begin() + num_shared, this->end());
            this->m_end = this->begin() + num_shared;
        }
        else if (rhs.size() > this->size())
        {
            size_t elements_diff = rhs.size() - this->size();
            std::uninitialized_copy(rhs.begin() + num_shared, rhs.end(), this->end());
            this->m_end = this->end() + elements_diff;
            this->destroy_range(rhs.begin() + num_shared, rhs.end());
            rhs.m_end = rhs.begin() + num_shared;
        }
    }

    template <class T, std::size_t N, class A, bool Init>
    inline void svector<T, N, A, Init>::grow(size_type min_capacity)
    {
        size_type current_size = size();
        size_type new_capacity = 2 * current_size + 1; // Always grow.
        if (new_capacity < min_capacity)
        {
            new_capacity = min_capacity;
        }

        T* new_alloc;
        // is data stack allocated?
        if (m_begin == &m_data[0])
        {
            new_alloc = m_allocator.allocate(new_capacity);
            std::uninitialized_copy(m_begin, m_end, new_alloc);
        }
        else
        {
            // If this wasn't grown from the inline copy, grow the allocated space.
            new_alloc = reinterpret_cast<pointer>(realloc(this->m_begin, new_capacity * sizeof(T)));
        }
        XTENSOR_ASSERT(new_alloc);

        m_end = new_alloc + current_size;
        m_begin = new_alloc;
        m_capacity = new_alloc + new_capacity;
    }

    template <class T, std::size_t N, class A, bool Init>
    inline bool operator==(const std::vector<T>& lhs, const svector<T, N, A, Init>& rhs)
    {
        return lhs.size() == rhs.size() && std::equal(lhs.begin(), lhs.end(), rhs.begin());
    }

    template <class T, std::size_t N, class A, bool Init>
    inline bool operator==(const svector<T, N, A, Init>& lhs, const std::vector<T>& rhs)
    {
        return lhs.size() == rhs.size() && std::equal(lhs.begin(), lhs.end(), rhs.begin());
    }

    template <class T, std::size_t N, class A, bool Init>
    inline bool operator==(const svector<T, N, A, Init>& lhs, const svector<T, N, A, Init>& rhs)
    {
        return lhs.size() == rhs.size() && std::equal(lhs.begin(), lhs.end(), rhs.begin());
    }

    template <class T, std::size_t N, class A, bool Init>
    inline bool operator!=(const svector<T, N, A, Init>& lhs, const svector<T, N, A, Init>& rhs)
    {
        return !(lhs == rhs);
    }

    template <class T, std::size_t N, class A, bool Init>
    inline bool operator<(const svector<T, N, A, Init>& lhs, const svector<T, N, A, Init>& rhs)
    {
        return std::lexicographical_compare(lhs.begin(), lhs.end(),
                                            rhs.begin(), rhs.end());
    }

    template <class T, std::size_t N, class A, bool Init>
    inline bool operator<=(const svector<T, N, A, Init>& lhs, const svector<T, N, A, Init>& rhs)
    {
        return !(lhs > rhs);
    }

    template <class T, std::size_t N, class A, bool Init>
    inline bool operator>(const svector<T, N, A, Init>& lhs, const svector<T, N, A, Init>& rhs)
    {
        return rhs < lhs;
    }

    template <class T, std::size_t N, class A, bool Init>
    inline bool operator>=(const svector<T, N, A, Init>& lhs, const svector<T, N, A, Init>& rhs)
    {
        return !(lhs < rhs);
    }

    template <class T, std::size_t N, class A, bool Init>
    inline void swap(svector<T, N, A, Init>& lhs, svector<T, N, A, Init>& rhs) noexcept
    {
        lhs.swap(rhs);
    }

}

#endif
