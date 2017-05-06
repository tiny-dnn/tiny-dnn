/**
 * @brief A portable C++ library for unit testing, http://libunittest.net
 * @file all.hpp
 */
#pragma once
#include "unittest.hpp"

/**
 * @brief Control switch for including an automatic main function
 */
#ifndef UNITTEST_MAIN_FUNC
#define UNITTEST_MAIN_FUNC 1
#endif
/**
 * @brief Control switch for assert shortcuts
 */
#ifndef UNITTEST_ASSERT_SHORT
#define UNITTEST_ASSERT_SHORT 1
#endif
/**
 * @brief Control switch for miscellaneous macro shortcuts
 */
#ifndef UNITTEST_MACRO_SHORT
#define UNITTEST_MACRO_SHORT 1
#endif

#if !defined(UNITTEST_NO_MAIN_FUNC) && UNITTEST_MAIN_FUNC==1
#include "main.hpp"
#endif

#if !defined(UNITTEST_NO_ASSERT_SHORT) && UNITTEST_ASSERT_SHORT==1
#include "assertshortcuts.hpp"
#endif

#if !defined(UNITTEST_NO_MACRO_SHORT) && UNITTEST_MACRO_SHORT==1
#include "shortcuts.hpp"
#endif
