/*
    Copyright (c) 2016, Taiga Nomi
    All rights reserved.
    
    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in the
    documentation and/or other materials provided with the distribution.
    * Neither the name of the <organization> nor the
    names of its contributors may be used to endorse or promote products
    derived from this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY 
    EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED 
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE 
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY 
    DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES 
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; 
    LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND 
    ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT 
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS 
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
#pragma once
#include <stdlib.h>
#ifdef _WIN32
#include <malloc.h>
#endif
#ifdef __MINGW32__
#include <mm_malloc.h>
#endif
#include "nn_error.h"

namespace tiny_cnn {

template <typename T, std::size_t alignment>
class aligned_allocator {
public:
    typedef T value_type;
    typedef T* pointer;
    typedef std::size_t size_type;
    typedef std::ptrdiff_t difference_type;
    typedef T& reference;
    typedef const T& const_reference;
    typedef const T* const_pointer;

    template <typename U>
    struct rebind {
        typedef aligned_allocator<U, alignment> other;
    };

    aligned_allocator() {}

    template <typename U>
    aligned_allocator(const aligned_allocator<U, alignment>&) {}

    const_pointer address(const_reference value) const {
        return std::addressof(value);
    }

    pointer address(reference value) const {
        return std::addressof(value);
    }

    pointer allocate(size_type size, const void* = nullptr) {
        void* p = aligned_alloc(alignment, sizeof(T) * size);
        if (!p && size > 0)
            throw nn_error("failed to allocate");
        return static_cast<pointer>(p);
    }
    
    size_type max_size() const {
        return ~static_cast<std::size_t>(0) / sizeof(T);
    }

    void deallocate(pointer ptr, size_type) {
        aligned_free(ptr);
    }

    template<class U, class V>
    void construct(U* ptr, const V& value) {
        void* p = ptr;
        ::new(p) U(value);
    }

#if defined(_MSC_VER) && _MSC_VER <= 1800
    // -vc2013 doesn't support variadic templates
#else
    template<class U, class... Args>
    void construct(U* ptr, Args&&... args) {
        void* p = ptr;
        ::new(p) U(std::forward<Args>(args)...);
    }
#endif

    template<class U>
    void construct(U* ptr) {
        void* p = ptr;
        ::new(p) U();
    }

    template<class U>
    void destroy(U* ptr) {
        ptr->~U();
    }

private:
    void* aligned_alloc(size_type align, size_type size) const {
#if defined(_MSC_VER)
        return ::_aligned_malloc(size, align);
#elif defined (__ANDROID__)
        return ::memalign(align, size);
#elif defined (__MINGW32__)
        return _mm_malloc(size, align);
#else // posix assumed
        void* p;
        if (::posix_memalign(&p, align, size) != 0) {
            p = 0;
        }
        return p;
#endif
    }

    void aligned_free(pointer ptr) {
#if defined(_MSC_VER)
        ::_aligned_free(ptr);
#elif defined(__MINGW32__)
        ::free(ptr);
#else
        ::free(ptr);
#endif
    }
};

template<typename T1, typename T2, std::size_t alignment>
inline bool operator==(const aligned_allocator<T1, alignment>&, const aligned_allocator<T2, alignment>&)
{
    return true;
}

template<typename T1, typename T2, std::size_t alignment>
inline bool operator!=(const aligned_allocator<T1, alignment>&, const aligned_allocator<T2, alignment>&)
{
    return false;
}
}