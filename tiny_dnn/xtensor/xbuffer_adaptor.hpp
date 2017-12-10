/***************************************************************************
* Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XBUFFER_ADAPTOR_HPP
#define XBUFFER_ADAPTOR_HPP

#include <algorithm>
#include <functional>
#include <iterator>
#include <memory>
#include <stdexcept>

#include "xstorage.hpp"

namespace xt {
struct no_ownership {};

struct acquire_ownership {};

namespace detail {

template <class T, class A>
class xbuffer_storage {
 public:
  using self_type       = xbuffer_storage<T, A>;
  using allocator_type  = A;
  using value_type      = typename allocator_type::value_type;
  using reference       = typename allocator_type::reference;
  using const_reference = typename allocator_type::const_reference;
  using pointer         = typename allocator_type::pointer;
  using const_pointer   = typename allocator_type::const_pointer;
  using size_type       = typename allocator_type::size_type;
  using difference_type = typename allocator_type::difference_type;

  xbuffer_storage();
  xbuffer_storage(T*& data,
                  size_type size,
                  const allocator_type& alloc = allocator_type());

  size_type size() const noexcept;
  void resize(size_type size);

  pointer data() noexcept;
  const_pointer data() const noexcept;

  void swap(self_type& rhs) noexcept;

 private:
  pointer p_data;
  size_type m_size;
};

template <class T, class A>
class xbuffer_owner_storage {
 public:
  using self_type       = xbuffer_owner_storage<T, A>;
  using allocator_type  = A;
  using value_type      = typename allocator_type::value_type;
  using reference       = typename allocator_type::reference;
  using const_reference = typename allocator_type::const_reference;
  using pointer         = typename allocator_type::pointer;
  using const_pointer   = typename allocator_type::const_pointer;
  using size_type       = typename allocator_type::size_type;
  using difference_type = typename allocator_type::difference_type;

  xbuffer_owner_storage();
  xbuffer_owner_storage(T*& data,
                        size_type size,
                        const allocator_type& alloc = allocator_type());
  ~xbuffer_owner_storage();

  xbuffer_owner_storage(const self_type&) = delete;
  self_type& operator                     =(const self_type&);

  xbuffer_owner_storage(self_type&&);
  self_type& operator=(self_type&&);

  size_type size() const noexcept;
  void resize(size_type size);

  pointer data() noexcept;
  const_pointer data() const noexcept;

  allocator_type get_allocator() const noexcept;

  void swap(self_type& rhs) noexcept;

 private:
  pointer* p_data;
  size_type m_size;
  allocator_type m_allocator;
};

template <class T, class A, class O>
struct get_buffer_storage {
  using type = xbuffer_storage<T, A>;
};

template <class T, class A>
struct get_buffer_storage<T, A, acquire_ownership> {
  using type = xbuffer_owner_storage<T, A>;
};

template <class T, class A, class O>
using buffer_storage_t = typename get_buffer_storage<T, A, O>::type;
}

template <class T, class O = no_ownership, class A = std::allocator<T>>
class xbuffer_adaptor : private detail::buffer_storage_t<T, A, O> {
 public:
  using base_type       = detail::buffer_storage_t<T, A, O>;
  using allocator_type  = typename base_type::allocator_type;
  using value_type      = typename base_type::value_type;
  using reference       = typename base_type::reference;
  using const_reference = typename base_type::const_reference;
  using pointer         = typename base_type::pointer;
  using const_pointer   = typename base_type::const_pointer;

  using size_type       = typename base_type::size_type;
  using difference_type = typename base_type::difference_type;

  using iterator               = pointer;
  using const_iterator         = const_pointer;
  using reverse_iterator       = std::reverse_iterator<iterator>;
  using const_reverse_iterator = std::reverse_iterator<const_iterator>;

  xbuffer_adaptor() = default;

  template <
    class OW = O,
    class    = std::enable_if_t<std::is_same<OW, acquire_ownership>::value>>
  xbuffer_adaptor(T*& data,
                  size_type size,
                  const allocator_type& alloc = allocator_type());

  template <class OW = O,
            class    = std::enable_if_t<std::is_same<OW, no_ownership>::value>>
  xbuffer_adaptor(T* data,
                  size_type size,
                  const allocator_type& alloc = allocator_type());

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

template <class T, class O, class A>
bool operator==(const xbuffer_adaptor<T, O, A>& lhs,
                const xbuffer_adaptor<T, O, A>& rhs);

template <class T, class O, class A>
bool operator!=(const xbuffer_adaptor<T, O, A>& lhs,
                const xbuffer_adaptor<T, O, A>& rhs);

template <class T, class O, class A>
bool operator<(const xbuffer_adaptor<T, O, A>& lhs,
               const xbuffer_adaptor<T, O, A>& rhs);

template <class T, class O, class A>
bool operator<=(const xbuffer_adaptor<T, O, A>& lhs,
                const xbuffer_adaptor<T, O, A>& rhs);

template <class T, class O, class A>
bool operator>(const xbuffer_adaptor<T, O, A>& lhs,
               const xbuffer_adaptor<T, O, A>& rhs);

template <class T, class O, class A>
bool operator>=(const xbuffer_adaptor<T, O, A>& lhs,
                const xbuffer_adaptor<T, O, A>& rhs);

template <class T, class O, class A>
void swap(xbuffer_adaptor<T, O, A>& lhs,
          xbuffer_adaptor<T, O, A>& rhs) noexcept;

/*******************
 * adaptor_closure *
 *******************/

namespace detail {
template <class C>
struct adaptor_closure_impl {
  using type = C&;
};

template <class T, class O, class A>
struct adaptor_closure_impl<xbuffer_adaptor<T, O, A>> {
  using type = xbuffer_adaptor<T, O, A>;
};
}

template <class C>
using adaptor_closure_t = typename detail::adaptor_closure_impl<C>::type;

/**********************************
 * xbuffer_storage implementation *
 **********************************/

namespace detail {
template <class T, class A>
inline xbuffer_storage<T, A>::xbuffer_storage() : p_data(nullptr), m_size(0) {}

template <class T, class A>
inline xbuffer_storage<T, A>::xbuffer_storage(T*& data,
                                              size_type size,
                                              const allocator_type&)
  : p_data(data), m_size(size) {}

template <class T, class A>
inline auto xbuffer_storage<T, A>::size() const noexcept -> size_type {
  return m_size;
}

template <class T, class A>
inline void xbuffer_storage<T, A>::resize(size_type size) {
  if (size != m_size) {
    throw std::runtime_error("xbuffer_storage not resizable");
  }
}

template <class T, class A>
inline auto xbuffer_storage<T, A>::data() noexcept -> pointer {
  return p_data;
}

template <class T, class A>
inline auto xbuffer_storage<T, A>::data() const noexcept -> const_pointer {
  return p_data;
}

template <class T, class A>
inline void xbuffer_storage<T, A>::swap(self_type& rhs) noexcept {
  using std::swap;
  swap(p_data, rhs.p_data);
  swap(m_size, rhs.m_size);
}
}

/****************************************
 * xbuffer_owner_storage implementation *
 ****************************************/

namespace detail {
template <class T, class A>
inline xbuffer_owner_storage<T, A>::xbuffer_owner_storage()
  : p_data(nullptr), m_size(0), m_allocator() {}

template <class T, class A>
inline xbuffer_owner_storage<T, A>::xbuffer_owner_storage(
  T*& data, size_type size, const allocator_type& alloc)
  : p_data(&data), m_size(size), m_allocator(alloc) {}

template <class T, class A>
inline xbuffer_owner_storage<T, A>::~xbuffer_owner_storage() {
  if (p_data != nullptr) {
    safe_destroy_deallocate(m_allocator, *p_data, m_size);
    p_data = nullptr;
    m_size = 0;
  }
}

template <class T, class A>
inline auto xbuffer_owner_storage<T, A>::operator=(const self_type& rhs)
  -> self_type& {
  using std::swap;
  if (this != &rhs) {
    allocator_type al = std::allocator_traits<allocator_type>::
      select_on_container_copy_construction(rhs.get_allocator());
    pointer tmp = safe_init_allocate(al, rhs.m_size);
    if (xtrivially_default_constructible<value_type>::value) {
      std::uninitialized_copy(*(rhs.p_data), *(rhs.p_data) + rhs.m_size, tmp);
    } else {
      std::copy(*(rhs.p_data), *(rhs.p_data) + rhs.m_size, tmp);
    }
    swap(*p_data, tmp);
    m_size = rhs.m_size;
    swap(m_allocator, al);
    safe_destroy_deallocate(al, tmp, m_size);
  }
  return *this;
}

template <class T, class A>
inline xbuffer_owner_storage<T, A>::xbuffer_owner_storage(self_type&& rhs)
  : p_data(rhs.p_data),
    m_size(rhs.m_size),
    m_allocator(std::move(rhs.m_allocator)) {
  rhs.p_data = nullptr;
  rhs.m_size = 0;
}

template <class T, class A>
inline auto xbuffer_owner_storage<T, A>::operator=(self_type&& rhs)
  -> self_type& {
  swap(rhs);
  return *this;
}

template <class T, class A>
inline auto xbuffer_owner_storage<T, A>::size() const noexcept -> size_type {
  return m_size;
}

template <class T, class A>
void xbuffer_owner_storage<T, A>::resize(size_type size) {
  using std::swap;
  if (size != m_size) {
    pointer tmp = safe_init_allocate(m_allocator, size);
    swap(*p_data, tmp);
    swap(m_size, size);
    safe_destroy_deallocate(m_allocator, tmp, size);
  }
}

template <class T, class A>
inline auto xbuffer_owner_storage<T, A>::data() noexcept -> pointer {
  return *p_data;
}

template <class T, class A>
inline auto xbuffer_owner_storage<T, A>::data() const noexcept
  -> const_pointer {
  return *p_data;
}

template <class T, class A>
inline auto xbuffer_owner_storage<T, A>::get_allocator() const noexcept
  -> allocator_type {
  return allocator_type(m_allocator);
}

template <class T, class A>
inline void xbuffer_owner_storage<T, A>::swap(self_type& rhs) noexcept {
  using std::swap;
  swap(p_data, rhs.p_data);
  swap(m_size, rhs.m_size);
  swap(m_allocator, rhs.m_allocator);
}
}

/**********************************
 * xbuffer_adaptor implementation *
 **********************************/

template <class T, class O, class A>
template <class OW, class>
inline xbuffer_adaptor<T, O, A>::xbuffer_adaptor(T*& data,
                                                 size_type size,
                                                 const allocator_type& alloc)
  : base_type(data, size, alloc) {}

template <class T, class O, class A>
template <class OW, class>
inline xbuffer_adaptor<T, O, A>::xbuffer_adaptor(T* data,
                                                 size_type size,
                                                 const allocator_type& alloc)
  : base_type(data, size, alloc) {}

template <class T, class O, class A>
bool xbuffer_adaptor<T, O, A>::empty() const noexcept {
  return size() == 0;
}

template <class T, class O, class A>
inline auto xbuffer_adaptor<T, O, A>::operator[](size_type i) -> reference {
  return data()[i];
}

template <class T, class O, class A>
inline auto xbuffer_adaptor<T, O, A>::operator[](size_type i) const
  -> const_reference {
  return data()[i];
}

template <class T, class O, class A>
inline auto xbuffer_adaptor<T, O, A>::front() -> reference {
  return data()[0];
}

template <class T, class O, class A>
inline auto xbuffer_adaptor<T, O, A>::front() const -> const_reference {
  return data()[0];
}

template <class T, class O, class A>
inline auto xbuffer_adaptor<T, O, A>::back() -> reference {
  return data()[size() - 1];
}

template <class T, class O, class A>
inline auto xbuffer_adaptor<T, O, A>::back() const -> const_reference {
  return data()[size() - 1];
}

template <class T, class O, class A>
inline auto xbuffer_adaptor<T, O, A>::begin() -> iterator {
  return data();
}

template <class T, class O, class A>
inline auto xbuffer_adaptor<T, O, A>::end() -> iterator {
  return data() + size();
}

template <class T, class O, class A>
inline auto xbuffer_adaptor<T, O, A>::begin() const -> const_iterator {
  return data();
}

template <class T, class O, class A>
inline auto xbuffer_adaptor<T, O, A>::end() const -> const_iterator {
  return data() + size();
}

template <class T, class O, class A>
inline auto xbuffer_adaptor<T, O, A>::cbegin() const -> const_iterator {
  return begin();
}

template <class T, class O, class A>
inline auto xbuffer_adaptor<T, O, A>::cend() const -> const_iterator {
  return end();
}

template <class T, class O, class A>
inline auto xbuffer_adaptor<T, O, A>::rbegin() -> reverse_iterator {
  return reverse_iterator(end());
}

template <class T, class O, class A>
inline auto xbuffer_adaptor<T, O, A>::rend() -> reverse_iterator {
  return reverse_iterator(begin());
}

template <class T, class O, class A>
inline auto xbuffer_adaptor<T, O, A>::rbegin() const -> const_reverse_iterator {
  return const_reverse_iterator(end());
}

template <class T, class O, class A>
inline auto xbuffer_adaptor<T, O, A>::rend() const -> const_reverse_iterator {
  return const_reverse_iterator(begin());
}

template <class T, class O, class A>
inline auto xbuffer_adaptor<T, O, A>::crbegin() const
  -> const_reverse_iterator {
  return rbegin();
}

template <class T, class O, class A>
inline auto xbuffer_adaptor<T, O, A>::crend() const -> const_reverse_iterator {
  return rend();
}

template <class T, class O, class A>
inline bool operator==(const xbuffer_adaptor<T, O, A>& lhs,
                       const xbuffer_adaptor<T, O, A>& rhs) {
  return lhs.size() == rhs.size() &&
         std::equal(lhs.begin(), lhs.end(), rhs.begin());
}

template <class T, class O, class A>
inline bool operator!=(const xbuffer_adaptor<T, O, A>& lhs,
                       const xbuffer_adaptor<T, O, A>& rhs) {
  return !(lhs == rhs);
}

template <class T, class O, class A>
inline bool operator<(const xbuffer_adaptor<T, O, A>& lhs,
                      const xbuffer_adaptor<T, O, A>& rhs) {
  return std::lexicographical_compare(lhs.begin(), lhs.end(), rhs.begin(),
                                      rhs.end(), std::less<T>());
}

template <class T, class O, class A>
inline bool operator<=(const xbuffer_adaptor<T, O, A>& lhs,
                       const xbuffer_adaptor<T, O, A>& rhs) {
  return std::lexicographical_compare(lhs.begin(), lhs.end(), rhs.begin(),
                                      rhs.end(), std::less_equal<T>());
}

template <class T, class O, class A>
inline bool operator>(const xbuffer_adaptor<T, O, A>& lhs,
                      const xbuffer_adaptor<T, O, A>& rhs) {
  return std::lexicographical_compare(lhs.begin(), lhs.end(), rhs.begin(),
                                      rhs.end(), std::greater<T>());
}

template <class T, class O, class A>
inline bool operator>=(const xbuffer_adaptor<T, O, A>& lhs,
                       const xbuffer_adaptor<T, O, A>& rhs) {
  return std::lexicographical_compare(lhs.begin(), lhs.end(), rhs.begin(),
                                      rhs.end(), std::greater_equal<T>());
}

template <class T, class O, class A>
inline void swap(xbuffer_adaptor<T, O, A>& lhs,
                 xbuffer_adaptor<T, O, A>& rhs) noexcept {
  lhs.swap(rhs);
}
}

#endif
