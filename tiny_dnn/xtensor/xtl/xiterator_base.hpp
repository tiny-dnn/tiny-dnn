/***************************************************************************
* Copyright (c) 2016, Sylvain Corlay and Johan Mabille                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XTL_XITERATOR_BASE_HPP
#define XTL_XITERATOR_BASE_HPP

#include <cstddef>
#include <iterator>

namespace xtl
{
    /**************************************
     * class xbidirectional_iterator_base *
     **************************************/

    template <class I, class T, class D = std::ptrdiff_t, class P = T*, class R = T&>
    class xbidirectional_iterator_base
    {
    public:

        using derived_type = I;
        using value_type = T;
        using reference = R;
        using pointer = P;
        using difference_type = D;
        using iterator_category = std::bidirectional_iterator_tag;

        inline friend derived_type operator++(derived_type& d, int)
        {
            derived_type tmp(d);
            ++d;
            return tmp;
        }

        inline friend derived_type operator--(derived_type& d, int)
        {
            derived_type tmp(d);
            --d;
            return tmp;

        }

        inline friend bool operator!=(const derived_type& lhs, const derived_type& rhs)
        {
            return !(lhs == rhs);
        }

        inline friend bool operator<=(const derived_type& lhs, const derived_type& rhs)
        {
            return !(rhs < lhs);
        }

        inline friend bool operator>=(const derived_type& lhs, const derived_type& rhs)
        {
            return !(lhs < rhs);
        }

        inline friend bool operator>(const derived_type& lhs, const derived_type& rhs)
        {
            return rhs < lhs;
        }
    };

    template <class T>
    using xbidirectional_iterator_base2 = xbidirectional_iterator_base<typename T::iterator_type,
                                                                       typename T::value_type,
                                                                       typename T::difference_type,
                                                                       typename T::pointer,
                                                                       typename T::reference>;

    /********************************
     * xrandom_access_iterator_base *
     ********************************/

    template <class I, class T, class D = std::ptrdiff_t, class P = T*, class R = T&>
    class xrandom_access_iterator_base : public xbidirectional_iterator_base<I, T, D, P, R>
    {
    public:

        using derived_type = I;
        using value_type = T;
        using reference = R;
        using pointer = P;
        using difference_type = D;
        using iterator_category = std::random_access_iterator_tag;

        inline reference operator[](difference_type n) const
        {
            return *(*static_cast<const derived_type*>(this) + n);
        }

        inline friend derived_type operator+(const derived_type& it, difference_type n)
        {
            derived_type tmp(it);
            return tmp += n;
        }

        inline friend derived_type operator+(difference_type n, const derived_type& it)
        {
            derived_type tmp(it);
            return tmp += n;
        }

        inline friend derived_type operator-(const derived_type& it, difference_type n)
        {
            derived_type tmp(it);
            return tmp -= n;
        }
    };

    template <class T>
    using xrandom_access_iterator_base2 = xrandom_access_iterator_base<typename T::iterator_type,
                                                                       typename T::value_type,
                                                                       typename T::difference_type,
                                                                       typename T::pointer,
                                                                       typename T::reference>;

    /*****************
     * xkey_iterator *
     *****************/

    template <class M>
    class xkey_iterator : public xbidirectional_iterator_base<xkey_iterator<M>, const typename M::key_type>
    {
    public:

        using self_type = xkey_iterator;
        using base_type = xbidirectional_iterator_base<self_type, const typename M::key_type>;
        using value_type = typename base_type::value_type;
        using reference = typename base_type::reference;
        using pointer = typename base_type::pointer;
        using difference_type = typename base_type::difference_type;
        using iterator_category = typename base_type::iterator_category;
        using subiterator = typename M::const_iterator;

        inline xkey_iterator(subiterator it) noexcept
            : m_it(it)
        {
        }

        inline self_type& operator++()
        {
            ++m_it;
            return *this;
        }

        inline self_type& operator--()
        {
            --m_it;
            return *this;
        }

        inline reference operator*() const
        {
            return m_it->first;
        }

        inline pointer operator->() const
        {
            return&(m_it->first);
        }

        inline bool operator==(const self_type& rhs) const
        {
            return m_it == rhs.m_it;
        }

        inline bool operator<(const self_type& rhs) const
        {
            return m_it < rhs.m_it;
        }

    private:

        subiterator m_it;
    };

    /***********************
     * common_iterator_tag *
     ***********************/

    template <class... Its>
    struct common_iterator_tag : std::common_type<typename std::iterator_traits<Its>::iterator_category...>
    {
    };

    template <class... Its>
    using common_iterator_tag_t = typename common_iterator_tag<Its...>::type;
}

#endif
