/***************************************************************************
* Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XSTORAGE_HPP
#define XSTORAGE_HPP

#include <algorithm>
#include <functional>
#include <initializer_list>
#include <iterator>
#include <memory>
#include <type_traits>

#include "xutils.hpp"

namespace xt {

namespace detail {
template <class It>
using require_input_iter = typename std::enable_if<
  std::is_convertible<typename std::iterator_traits<It>::iterator_category,
                      std::input_iterator_tag>::value>::type;
}

template <class T, class Allocator = std::allocator<T>>
class uvector {
 public:
  using allocator_type = Allocator;

  using value_type      = typename allocator_type::value_type;
  using reference       = typename allocator_type::reference;
  using const_reference = typename allocator_type::const_reference;
  using pointer         = typename allocator_type::pointer;
  using const_pointer   = typename allocator_type::const_pointer;

  using size_type       = typename allocator_type::size_type;
  using difference_type = typename allocator_type::difference_type;

  using iterator               = pointer;
  using const_iterator         = const_pointer;
  using reverse_iterator       = std::reverse_iterator<iterator>;
  using const_reverse_iterator = std::reverse_iterator<const_iterator>;

  uvector() noexcept;
  explicit uvector(const allocator_type& alloc) noexcept;
  explicit uvector(size_type count,
                   const allocator_type& alloc = allocator_type());
  uvector(size_type count,
          const_reference value,
          const allocator_type& alloc = allocator_type());

  template <class InputIt, class = detail::require_input_iter<InputIt>>
  uvector(InputIt first,
          InputIt last,
          const allocator_type& alloc = allocator_type());

  uvector(std::initializer_list<T> init,
          const allocator_type& alloc = allocator_type());

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

namespace detail {
template <class A>
inline typename A::pointer safe_init_allocate(A& alloc,
                                              typename A::size_type size) {
  using pointer    = typename A::pointer;
  using value_type = typename A::value_type;
  pointer res      = alloc.allocate(size);
  if (!xtrivially_default_constructible<value_type>::value) {
    for (pointer p = res; p != res + size; ++p) {
      alloc.construct(p, value_type());
    }
  }
  return res;
}

template <class A>
inline void safe_destroy_deallocate(A& alloc,
                                    typename A::pointer ptr,
                                    typename A::size_type size) {
  using pointer    = typename A::pointer;
  using value_type = typename A::value_type;
  if (ptr != nullptr) {
    if (!xtrivially_default_constructible<value_type>::value) {
      for (pointer p = ptr; p != ptr + size; ++p) {
        alloc.destroy(p);
      }
    }
    alloc.deallocate(ptr, size);
  }
}
}

template <class T, class A>
template <class I>
inline void uvector<T, A>::init_data(I first, I last) {
  size_type size = static_cast<size_type>(std::distance(first, last));
  if (size != size_type(0)) {
    p_begin = m_allocator.allocate(size);
    std::uninitialized_copy(first, last, p_begin);
    p_end = p_begin + size;
  }
}

template <class T, class A>
inline void uvector<T, A>::resize_impl(size_type new_size) {
  size_type old_size = size();
  pointer old_begin  = p_begin;
  if (new_size != old_size) {
    p_begin = detail::safe_init_allocate(m_allocator, new_size);
    p_end   = p_begin + new_size;
    detail::safe_destroy_deallocate(m_allocator, old_begin, old_size);
  }
}

template <class T, class A>
inline uvector<T, A>::uvector() noexcept : uvector(allocator_type()) {}

template <class T, class A>
inline uvector<T, A>::uvector(const allocator_type& alloc) noexcept
  : m_allocator(alloc), p_begin(nullptr), p_end(nullptr) {}

template <class T, class A>
inline uvector<T, A>::uvector(size_type count, const allocator_type& alloc)
  : m_allocator(alloc), p_begin(nullptr), p_end(nullptr) {
  if (count != 0) {
    p_begin = detail::safe_init_allocate(m_allocator, count);
    p_end   = p_begin + count;
  }
}

template <class T, class A>
inline uvector<T, A>::uvector(size_type count,
                              const_reference value,
                              const allocator_type& alloc)
  : m_allocator(alloc), p_begin(nullptr), p_end(nullptr) {
  if (count != 0) {
    p_begin = m_allocator.allocate(count);
    p_end   = p_begin + count;
    std::uninitialized_fill(p_begin, p_end, value);
  }
}

template <class T, class A>
template <class InputIt, class>
inline uvector<T, A>::uvector(InputIt first,
                              InputIt last,
                              const allocator_type& alloc)
  : m_allocator(alloc), p_begin(nullptr), p_end(nullptr) {
  init_data(first, last);
}

template <class T, class A>
inline uvector<T, A>::uvector(std::initializer_list<T> init,
                              const allocator_type& alloc)
  : m_allocator(alloc), p_begin(nullptr), p_end(nullptr) {
  init_data(init.begin(), init.end());
}

template <class T, class A>
inline uvector<T, A>::~uvector() {
  detail::safe_destroy_deallocate(m_allocator, p_begin, size());
  p_begin = nullptr;
  p_end   = nullptr;
}

template <class T, class A>
inline uvector<T, A>::uvector(const uvector& rhs)
  : m_allocator(std::allocator_traits<allocator_type>::
                  select_on_container_copy_construction(rhs.get_allocator())),
    p_begin(nullptr),
    p_end(nullptr) {
  init_data(rhs.p_begin, rhs.p_end);
}

template <class T, class A>
inline uvector<T, A>::uvector(const uvector& rhs, const allocator_type& alloc)
  : m_allocator(alloc), p_begin(nullptr), p_end(nullptr) {
  init_data(rhs.p_begin, rhs.p_end);
}

template <class T, class A>
inline uvector<T, A>& uvector<T, A>::operator=(const uvector& rhs) {
  // No copy and swap idiom here due to performance issues
  if (this != &rhs) {
    m_allocator = std::allocator_traits<allocator_type>::
      select_on_container_copy_construction(rhs.get_allocator());
    resize_impl(rhs.size());
    if (xtrivially_default_constructible<value_type>::value) {
      std::uninitialized_copy(rhs.p_begin, rhs.p_end, p_begin);
    } else {
      std::copy(rhs.p_begin, rhs.p_end, p_begin);
    }
  }
  return *this;
}

template <class T, class A>
inline uvector<T, A>::uvector(uvector&& rhs) noexcept
  : m_allocator(std::move(rhs.m_allocator)),
    p_begin(rhs.p_begin),
    p_end(rhs.p_end) {
  rhs.p_begin = nullptr;
  rhs.p_end   = nullptr;
}

template <class T, class A>
inline uvector<T, A>::uvector(uvector&& rhs,
                              const allocator_type& alloc) noexcept
  : m_allocator(alloc), p_begin(rhs.p_begin), p_end(rhs.p_end) {
  rhs.p_begin = nullptr;
  rhs.p_end   = nullptr;
}

template <class T, class A>
inline uvector<T, A>& uvector<T, A>::operator=(uvector&& rhs) noexcept {
  using std::swap;
  uvector tmp(std::move(rhs));
  swap(p_begin, tmp.p_begin);
  swap(p_end, tmp.p_end);
  return *this;
}

template <class T, class A>
inline auto uvector<T, A>::get_allocator() const noexcept -> allocator_type {
  return allocator_type(m_allocator);
}

template <class T, class A>
inline bool uvector<T, A>::empty() const noexcept {
  return size() == size_type(0);
}

template <class T, class A>
inline auto uvector<T, A>::size() const noexcept -> size_type {
  return static_cast<size_type>(p_end - p_begin);
}

template <class T, class A>
inline void uvector<T, A>::resize(size_type size) {
  resize_impl(size);
}

template <class T, class A>
inline auto uvector<T, A>::operator[](size_type i) -> reference {
  return p_begin[i];
}

template <class T, class A>
inline auto uvector<T, A>::operator[](size_type i) const -> const_reference {
  return p_begin[i];
}

template <class T, class A>
inline auto uvector<T, A>::front() -> reference {
  return p_begin[0];
}

template <class T, class A>
inline auto uvector<T, A>::front() const -> const_reference {
  return p_begin[0];
}

template <class T, class A>
inline auto uvector<T, A>::back() -> reference {
  return *(p_end - 1);
}

template <class T, class A>
inline auto uvector<T, A>::back() const -> const_reference {
  return *(p_end - 1);
}

template <class T, class A>
inline auto uvector<T, A>::data() noexcept -> pointer {
  return p_begin;
}

template <class T, class A>
inline auto uvector<T, A>::data() const noexcept -> const_pointer {
  return p_begin;
}

template <class T, class A>
inline auto uvector<T, A>::begin() noexcept -> iterator {
  return p_begin;
}

template <class T, class A>
inline auto uvector<T, A>::end() noexcept -> iterator {
  return p_end;
}

template <class T, class A>
inline auto uvector<T, A>::begin() const noexcept -> const_iterator {
  return p_begin;
}

template <class T, class A>
inline auto uvector<T, A>::end() const noexcept -> const_iterator {
  return p_end;
}

template <class T, class A>
inline auto uvector<T, A>::cbegin() const noexcept -> const_iterator {
  return begin();
}

template <class T, class A>
inline auto uvector<T, A>::cend() const noexcept -> const_iterator {
  return end();
}

template <class T, class A>
inline auto uvector<T, A>::rbegin() noexcept -> reverse_iterator {
  return reverse_iterator(end());
}

template <class T, class A>
inline auto uvector<T, A>::rend() noexcept -> reverse_iterator {
  return reverse_iterator(begin());
}

template <class T, class A>
inline auto uvector<T, A>::rbegin() const noexcept -> const_reverse_iterator {
  return const_reverse_iterator(end());
}

template <class T, class A>
inline auto uvector<T, A>::rend() const noexcept -> const_reverse_iterator {
  return const_reverse_iterator(begin());
}

template <class T, class A>
inline auto uvector<T, A>::crbegin() const noexcept -> const_reverse_iterator {
  return rbegin();
}

template <class T, class A>
inline auto uvector<T, A>::crend() const noexcept -> const_reverse_iterator {
  return rend();
}

template <class T, class A>
inline void uvector<T, A>::swap(uvector<T, A>& rhs) noexcept {
  using std::swap;
  swap(m_allocator, rhs.m_allocator);
  swap(p_begin, rhs.p_begin);
  swap(p_end, rhs.p_end);
}

template <class T, class A>
inline bool operator==(const uvector<T, A>& lhs, const uvector<T, A>& rhs) {
  return lhs.size() == rhs.size() &&
         std::equal(lhs.begin(), lhs.end(), rhs.begin());
}

template <class T, class A>
inline bool operator!=(const uvector<T, A>& lhs, const uvector<T, A>& rhs) {
  return !(lhs == rhs);
}

template <class T, class A>
inline bool operator<(const uvector<T, A>& lhs, const uvector<T, A>& rhs) {
  return std::lexicographical_compare(lhs.begin(), lhs.end(), rhs.begin(),
                                      rhs.end(), std::less<T>());
}

template <class T, class A>
inline bool operator<=(const uvector<T, A>& lhs, const uvector<T, A>& rhs) {
  return std::lexicographical_compare(lhs.begin(), lhs.end(), rhs.begin(),
                                      rhs.end(), std::less_equal<T>());
}

template <class T, class A>
inline bool operator>(const uvector<T, A>& lhs, const uvector<T, A>& rhs) {
  return std::lexicographical_compare(lhs.begin(), lhs.end(), rhs.begin(),
                                      rhs.end(), std::greater<T>());
}

template <class T, class A>
inline bool operator>=(const uvector<T, A>& lhs, const uvector<T, A>& rhs) {
  return std::lexicographical_compare(lhs.begin(), lhs.end(), rhs.begin(),
                                      rhs.end(), std::greater_equal<T>());
}

template <class T, class A>
inline void swap(uvector<T, A>& lhs, uvector<T, A>& rhs) noexcept {
  lhs.swap(rhs);
}
}

#endif
