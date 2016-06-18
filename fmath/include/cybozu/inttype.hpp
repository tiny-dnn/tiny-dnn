#pragma once
/**
	@file
	@brief int type definition and macros
	Copyright (C) 2008 Cybozu Labs, Inc., all rights reserved.
*/

#if defined(_MSC_VER) && (MSC_VER <= 1500)
	typedef __int64 int64_t;
	typedef unsigned __int64 uint64_t;
	typedef unsigned int uint32_t;
	typedef int int32_t;
	typedef unsigned short uint16_t;
	typedef short int16_t;
	typedef unsigned char uint8_t;
	typedef signed char int8_t;
#else
	#include <stdint.h>
#endif

#ifdef _MSC_VER
	#ifndef CYBOZU_DEFINED_SSIZE_T
		#define CYBOZU_DEFINED_SSIZE_T
		#ifdef _WIN64
			typedef int64_t ssize_t;
		#else
			typedef int32_t ssize_t;
		#endif
	#endif
#else
	#include <unistd.h> // for ssize_t
#endif

#ifndef CYBOZU_ALIGN
	#ifdef _MSC_VER
		#define CYBOZU_ALIGN(x) __declspec(align(x))
	#else
		#define CYBOZU_ALIGN(x) __attribute__((aligned(x)))
	#endif
#endif
#ifndef CYBOZU_ALLOCA
	#ifdef _MSC_VER
		#include <malloc.h>
		#define CYBOZU_ALLOCA(x) _malloca(x)
	#else
		#define CYBOZU_ALLOCA_(x) __builtin_alloca(x)
	#endif
#endif
#ifndef CYBOZU_FOREACH
	// std::vector<int> v; CYBOZU_FOREACH(auto x, v) {...}
	#if defined(_MSC_VER) && (_MSC_VER >= 1400)
		#define CYBOZU_FOREACH(type_x, xs) for each (type_x in xs)
	#elif defined(__GNUC__)
		#define CYBOZU_FOREACH(type_x, xs) for (type_x : xs)
	#endif
#endif
#ifndef CYBOZU_NUM_OF_ARRAY
	#define CYBOZU_NUM_OF_ARRAY(x) (sizeof(x) / sizeof(*x))
#endif
#ifndef CYBOZU_SNPRINTF
	#ifdef _MSC_VER
		#define CYBOZU_SNPRINTF(x, len, ...) _snprintf_s(x, len - 1, __VA_ARGS__)
	#else
		#define CYBOZU_SNPRINTF(x, len, ...) snprintf(x, len, __VA_ARGS__)
	#endif
#endif

#define CYBOZU_CPP_VERSION_CPP03 0
#define CYBOZU_CPP_VERSION_TR1 1
#define CYBOZU_CPP_VERSION_CPP11 2

#if (__cplusplus >= 201103) || (_MSC_VER >= 1500) || defined(__GXX_EXPERIMENTAL_CXX0X__)
	#if defined(_MSC_VER) && (_MSC_VER <= 1600)
		#define CYBOZU_CPP_VERSION CYBOZU_CPP_VERSION_TR1
	#else
		#define CYBOZU_CPP_VERSION CYBOZU_CPP_VERSION_CPP11
	#endif
#elif (__GNUC__ >= 4 && __GNUC_MINOR__ >= 5) || (__clang_major__ >= 3)
	#define CYBOZU_CPP_VERSION CYBOZU_CPP_VERSION_TR1
#else
	#define CYBOZU_CPP_VERSION CYBOZU_CPP_VERSION_CPP03
#endif

#if (CYBOZU_CPP_VERSION == CYBOZU_CPP_VERSION_TR1)
	#define CYBOZU_NAMESPACE_STD std::tr1
	#define CYBOZU_NAMESPACE_TR1_BEGIN namespace tr1 {
	#define CYBOZU_NAMESPACE_TR1_END }
#else
	#define CYBOZU_NAMESPACE_STD std
	#define CYBOZU_NAMESPACE_TR1_BEGIN
	#define CYBOZU_NAMESPACE_TR1_END
#endif

#ifndef CYBOZU_OS_BIT
	#if defined(_WIN64) || defined(__x86_64__)
		#define CYBOZU_OS_BIT 64
	#else
		#define CYBOZU_OS_BIT 32
	#endif
#endif

namespace cybozu {
template<class T>
void disable_warning_unused_variable(const T&) { }
template<class T, class S>
T cast(const S* ptr) { return static_cast<T>(static_cast<const void*>(ptr)); }
template<class T, class S>
T cast(S* ptr) { return static_cast<T>(static_cast<void*>(ptr)); }
} // cybozu
