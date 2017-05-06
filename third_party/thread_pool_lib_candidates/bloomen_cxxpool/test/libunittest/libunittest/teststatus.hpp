/**
 * @brief An enumeration for the test status
 * @file teststatus.hpp
 */
#pragma once
/**
 * @brief Unit testing in C++
 */
namespace unittest {
/**
 * @brief Internal functionality, not relevant for most users
 */
namespace core {
/**
 * @brief An enumeration for the test status
 */
enum class teststatus : unsigned int {
    success = 0,
    failure = 1,
    error = 2,
    skipped = 3
};

} // core
} // unittest
