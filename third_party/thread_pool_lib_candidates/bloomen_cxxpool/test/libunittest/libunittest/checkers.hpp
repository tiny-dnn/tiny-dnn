/**
 * @brief A collection of check functions to be used with assertions
 * @file checkers.hpp
 */
#pragma once
#include "testfailure.hpp"
#include "formatting.hpp"
#include <string>
#include <iostream>
#include <type_traits>
/**
 * @brief Unit testing in C++
 */
namespace unittest {
/**
 * @brief Internal functionality, not relevant for most users
 */
namespace core {
/**
 * Checks if the value passed is finite. Performs an actual check only on
 * values of arithmetic type
 * @param value The value
 * @param argument The name of the argument
 * @param caller The name of the caller
 * @param message If given, is appended to the default fail message
 */
template<typename T,
         typename... Args>
void
check_isfinite(const T& value,
               const std::string& argument,
               const std::string& caller,
               const Args&... message)
{
    if (!unittest::core::isfinite(value)) {
        const std::string text = argument + " is not finite" + unittest::core::str_if(": ", value);
        unittest::fail(caller, text, message...);
    }
}
/**
 * Checks if the container passed contains only finite elements. Performs an
 * actual check only on values of arithmetic type
 * @param container The container
 * @param argument The name of the argument
 * @param caller The name of the caller
 * @param message If given, is appended to the default fail message
 */
template<typename T,
         typename... Args>
void
check_isfinite_container(const T& container,
                         const std::string& argument,
                         const std::string& caller,
                         const Args&... message)
{
    auto begin = std::begin(container);
    auto end = std::end(container);
    if (!std::all_of(begin, end, [](decltype(*begin) v) { return unittest::core::isfinite(v); })) {
        const std::string text = argument + " contains non-finite elements";
        unittest::fail(caller, text, message...);
    }
}
/**
 * Checks if the value passed is nan. Performs an actual check only on
 * values of arithmetic type
 * @param value The value
 * @param argument The name of the argument
 * @param caller The name of the caller
 * @param message If given, is appended to the default fail message
 */
template<typename T,
         typename... Args>
void
check_isnan(const T& value,
            const std::string& argument,
            const std::string& caller,
            const Args&... message)
{
    if (unittest::core::isnan(value)) {
        const std::string text = argument + " is not a number" + unittest::core::str_if(": ", value);
        unittest::fail(caller, text, message...);
    }
}
/**
 * @brief Checks if an epsilon is valid
 * @param epsilon The epsilon
 * @param caller The name of the caller
 * @param message If given, is appended to the default fail message
 */
template<typename T,
         typename... Args>
void
check_epsilon(const T& epsilon,
              const std::string& caller,
              const Args&... message)
{
    unittest::core::check_isfinite(epsilon, "epsilon", caller, message...);
    if (!(epsilon > epsilon - epsilon)) {
        const std::string text = "epsilon not greater than zero" + unittest::core::str_if(": ", epsilon);
        unittest::fail(caller, text, message...);
    }
}
/**
 * @brief Checks if the range bounds are valid
 * @param lower The lower bound
 * @param upper The upper bound
 * @param caller The name of the caller
 * @param message If given, is appended to the default fail message
 */
template<typename T,
         typename U,
         typename... Args>
void
check_range_bounds(const T& lower,
                   const U& upper,
                   const std::string& caller,
                   const Args&... message)
{
    unittest::core::check_isfinite(lower, "lower bound", caller, message...);
    unittest::core::check_isfinite(upper, "upper bound", caller, message...);
    if (lower > upper) {
        const std::string text = "lower bound cannot be greater than upper bound"  + unittest::core::str_if(": ", lower, ": lower")  + unittest::core::str_if(" > ", upper, " > upper");
        unittest::fail(caller, text, message...);
    }
}

} // core
} // unittest
