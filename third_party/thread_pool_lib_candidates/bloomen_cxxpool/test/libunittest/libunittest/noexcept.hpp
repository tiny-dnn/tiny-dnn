/**
 * @brief Compiler specific definitions of noexcept and noexcept(false)
 * @file noexcept.hpp
 */
#pragma once

#if defined(_MSC_VER) && _MSC_VER < 1900
/**
 * @brief Compiler specific definitions of noexcept
 */
#define UNITTEST_NOEXCEPT throw()
/**
 * @brief Compiler specific definitions of noexcept(false)
 */
#define UNITTEST_NOEXCEPT_FALSE
#else
/**
 * @brief Compiler specific definitions of noexcept
 */
#define UNITTEST_NOEXCEPT noexcept
/**
 * @brief Compiler specific definitions of noexcept(false)
 */
#define UNITTEST_NOEXCEPT_FALSE noexcept(false)
#endif
