/**
 * @brief A collection of test assertions
 * @file assertions.hpp
 */
#pragma once
#include "utilities.hpp"
#include "formatting.hpp"
#include "testsuite.hpp"
#include "testfailure.hpp"
#include "checkers.hpp"
#include "func.hpp"
#include <string>
#include <regex>
#include <typeinfo>
#include <utility>
/**
 * @brief Unit testing in C++
 */
namespace unittest {
/**
 * @brief A collection of assertions
 */
namespace assertions {
/**
 * @brief Asserts that a value is true.
 * @param value A value
 * @param message If given, is appended to the default fail message
 */
template<typename T,
         typename... Args>
void
assert_true(const T& value,
            const Args&... message)
{
    unittest::core::check_isfinite(value, "value", UNITTEST_FUNC, message...);
    if (!value) {
        const std::string text = "value is not true" + unittest::core::str_if(": ", value);
        unittest::fail(UNITTEST_FUNC, text, message...);
    }
}
/**
 * @brief Asserts that a value is false.
 * @param value A value
 * @param message If given, is appended to the default fail message
 */
template<typename T,
         typename... Args>
void
assert_false(const T& value,
             const Args&... message)
{
    unittest::core::check_isfinite(value, "value", UNITTEST_FUNC, message...);
    if (value) {
        const std::string text = "value is not false" + unittest::core::str_if(": ", value);
        unittest::fail(UNITTEST_FUNC, text, message...);
    }
}
/**
 * @brief Asserts that two values are equal.
 *  Required operators: ==, <<
 * @param expected The expected value
 * @param actual The actual value
 * @param message If given, is appended to the default fail message
 */
template<typename T,
         typename U,
         typename... Args>
void
assert_equal(const T& expected,
             const U& actual,
             const Args&... message)
{
    unittest::core::check_isfinite(expected, "expected", UNITTEST_FUNC, message...);
    unittest::core::check_isfinite(actual, "actual", UNITTEST_FUNC, message...);
    if (!(expected == actual)) {
        const std::string text = unittest::core::str(expected) + " not equal to " + unittest::core::str(actual);
        unittest::fail(UNITTEST_FUNC, text, message...);
    }
}
/**
 * @brief Asserts that two values are not equal.
 *  Required operators: ==, <<
 * @param first A value
 * @param second Another value
 * @param message If given, is appended to the default fail message
 */
template<typename T,
         typename U,
         typename... Args>
void
assert_not_equal(const T& first,
                 const U& second,
                 const Args&... message)
{
    unittest::core::check_isfinite(first, "first", UNITTEST_FUNC, message...);
    unittest::core::check_isfinite(second, "second", UNITTEST_FUNC, message...);
    if (first == second) {
        const std::string text = unittest::core::str(first) + " equal to " + unittest::core::str(second);
        unittest::fail(UNITTEST_FUNC, text, message...);
    }
}
/**
 * @brief Asserts that two values are approximately equal up to some
 *  epsilon. The assertion succeeds if |a - b| < epsilon.
 *  Required operators: <, >, -, <<
 * @param expected The expected value
 * @param actual The actual value
 * @param epsilon The epsilon
 * @param message If given, is appended to the default fail message
 */
template<typename T,
         typename U,
         typename V,
         typename... Args>
void
assert_approx_equal(const T& expected,
                    const U& actual,
                    const V& epsilon,
                    const Args&... message)
{
    unittest::core::check_isfinite(expected, "expected", UNITTEST_FUNC, message...);
    unittest::core::check_isfinite(actual, "actual", UNITTEST_FUNC, message...);
    unittest::core::check_epsilon(epsilon, UNITTEST_FUNC, message...);
    if (!unittest::core::is_approx_equal(expected, actual, epsilon)) {
        const std::string text = unittest::core::str(expected) + " not approx. equal to " + unittest::core::str(actual) + " with epsilon = " + unittest::core::str(epsilon);
        unittest::fail(UNITTEST_FUNC, text, message...);
    }
}
/**
 * @brief Asserts that two values are not approximately equal up to some
 *  epsilon. The assertion succeeds if |a - b| < epsilon is false.
 *  Required operators: <, >, -, <<
 * @param first A value
 * @param second Another value
 * @param epsilon The epsilon
 * @param message If given, is appended to the default fail message
 */
template<typename T,
         typename U,
         typename V,
         typename... Args>
void
assert_approx_not_equal(const T& first,
                        const U& second,
                        const V& epsilon,
                        const Args&... message)
{
    unittest::core::check_isfinite(first, "first", UNITTEST_FUNC, message...);
    unittest::core::check_isfinite(second, "second", UNITTEST_FUNC, message...);
    unittest::core::check_epsilon(epsilon, UNITTEST_FUNC, message...);
    if (unittest::core::is_approx_equal(first, second, epsilon)) {
        const std::string text = unittest::core::str(first) + " approx. equal to " + unittest::core::str(second) + " with epsilon = " + unittest::core::str(epsilon);
        unittest::fail(UNITTEST_FUNC, text, message...);
    }
}
/**
 * @brief Asserts that two values are relatively approx. equal up to some
 *  epsilon. The assertion succeeds if |a - b| < |a| * epsilon.
 *  Required operators: <, >, -, *, <<
 * @param expected The expected value
 * @param actual The actual value
 * @param epsilon The epsilon
 * @param message If given, is appended to the default fail message
 */
template<typename T,
         typename U,
         typename V,
         typename... Args>
void
assert_approxrel_equal(const T& expected,
                       const U& actual,
                       const V& epsilon,
                       const Args&... message)
{
    unittest::core::check_isfinite(expected, "expected", UNITTEST_FUNC, message...);
    unittest::core::check_isfinite(actual, "actual", UNITTEST_FUNC, message...);
    unittest::core::check_epsilon(epsilon, UNITTEST_FUNC, message...);
    if (!unittest::core::is_approxrel_equal(expected, actual, epsilon)) {
        const std::string text = unittest::core::str(expected) + " not relatively approx. equal to " + unittest::core::str(actual) + " with epsilon = " + unittest::core::str(epsilon);
        unittest::fail(UNITTEST_FUNC, text, message...);
    }
}
/**
 * @brief Asserts that two values are not relatively approx. equal up to some
 *  epsilon. The assertion succeeds if |a - b| < |a| * epsilon is false.
 *  Required operators: <, >, -, *, <<
 * @param first The first value
 * @param second The second value
 * @param epsilon The epsilon
 * @param message If given, is appended to the default fail message
 */
template<typename T,
         typename U,
         typename V,
         typename... Args>
void
assert_approxrel_not_equal(const T& first,
                           const U& second,
                           const V& epsilon,
                           const Args&... message)
{
    unittest::core::check_isfinite(first, "first", UNITTEST_FUNC, message...);
    unittest::core::check_isfinite(second, "second", UNITTEST_FUNC, message...);
    unittest::core::check_epsilon(epsilon, UNITTEST_FUNC, message...);
    if (unittest::core::is_approxrel_equal(first, second, epsilon)) {
        const std::string text = unittest::core::str(first) + " relatively approx. equal to " + unittest::core::str(second) + " with epsilon = " + unittest::core::str(epsilon);
        unittest::fail(UNITTEST_FUNC, text, message...);
    }
}
/**
 * @brief Asserts that the first value is greater than the second.
 *  Required operators: >, <<
 * @param first A value
 * @param second Another value
 * @param message If given, is appended to the default fail message
 */
template<typename T,
         typename U,
         typename... Args>
void
assert_greater(const T& first,
               const U& second,
               const Args&... message)
{
    unittest::core::check_isfinite(first, "first", UNITTEST_FUNC, message...);
    unittest::core::check_isfinite(second, "second", UNITTEST_FUNC, message...);
    if (!(first > second)) {
        const std::string text = unittest::core::str(first) + " not greater than " + unittest::core::str(second);
        unittest::fail(UNITTEST_FUNC, text, message...);
    }
}
/**
 * @brief Asserts that the first value is greater than or equal to the second.
 *  Required operators: <, <<
 * @param first A value
 * @param second Another value
 * @param message If given, is appended to the default fail message
 */
template<typename T,
         typename U,
         typename... Args>
void
assert_greater_equal(const T& first,
                     const U& second,
                     const Args&... message)
{
    unittest::core::check_isfinite(first, "first", UNITTEST_FUNC, message...);
    unittest::core::check_isfinite(second, "second", UNITTEST_FUNC, message...);
    if (first < second) {
        const std::string text = unittest::core::str(first) + " not greater than or equal to " + unittest::core::str(second);
        unittest::fail(UNITTEST_FUNC, text, message...);
    }
}
/**
 * @brief Asserts that the first value is lesser than the second.
 *  Required operators: <, <<
 * @param first A value
 * @param second Another value
 * @param message If given, is appended to the default fail message
 */
template<typename T,
         typename U,
         typename... Args>
void
assert_lesser(const T& first,
              const U& second,
              const Args&... message)
{
    unittest::core::check_isfinite(first, "first", UNITTEST_FUNC, message...);
    unittest::core::check_isfinite(second, "second", UNITTEST_FUNC, message...);
    if (!(first < second)) {
        const std::string text = unittest::core::str(first) + " not lesser than " + unittest::core::str(second);
        unittest::fail(UNITTEST_FUNC, text, message...);
    }
}
/**
 * @brief Asserts that the first value is lesser than or equal to the second.
 *  Required operators: >, <<
 * @param first A value
 * @param second Another value
 * @param message If given, is appended to the default fail message
 */
template<typename T,
         typename U,
         typename... Args>
void
assert_lesser_equal(const T& first,
                    const U& second,
                    const Args&... message)
{
    unittest::core::check_isfinite(first, "first", UNITTEST_FUNC, message...);
    unittest::core::check_isfinite(second, "second", UNITTEST_FUNC, message...);
    if (first > second) {
        const std::string text = unittest::core::str(first) + " not lesser than or equal to " + unittest::core::str(second);
        unittest::fail(UNITTEST_FUNC, text, message...);
    }
}
/**
 * @brief Asserts that a value is in a given range with included bounds.
 *  Required operators: <, >, <<
 * @param value A value
 * @param lower The lower bound
 * @param upper The upper bound
 * @param message If given, is appended to the default fail message
 */
template<typename T,
         typename U,
         typename V,
         typename... Args>
void
assert_in_range(const T& value,
                const U& lower,
                const V& upper,
                const Args&... message)
{
    unittest::core::check_isfinite(value, "value", UNITTEST_FUNC, message...);
    unittest::core::check_range_bounds(lower, upper, UNITTEST_FUNC, message...);
    if (!unittest::core::is_in_range(value, lower, upper)) {
        const std::string text = unittest::core::str(value) + " not in range [" + unittest::core::str(lower) + ", " + unittest::core::str(upper) + "]";
        unittest::fail(UNITTEST_FUNC, text, message...);
    }
}
/**
 * @brief Asserts that a value is not in a given range with included bounds.
 *  Required operators: <, >, <<
 * @param value A value
 * @param lower The lower bound
 * @param upper The upper bound
 * @param message If given, is appended to the default fail message
 */
template<typename T,
         typename U,
         typename V,
         typename... Args>
void
assert_not_in_range(const T& value,
                    const U& lower,
                    const V& upper,
                    const Args&... message)
{
    unittest::core::check_isfinite(value, "value", UNITTEST_FUNC, message...);
    unittest::core::check_range_bounds(lower, upper, UNITTEST_FUNC, message...);
    if (unittest::core::is_in_range(value, lower, upper)) {
        const std::string text = unittest::core::str(value) + " in range [" + unittest::core::str(lower) + ", " + unittest::core::str(upper) + "]";
        unittest::fail(UNITTEST_FUNC, text, message...);
    }
}
/**
 * @brief Asserts that a value is in a container.
 *  Required operators: ==
 * @param value A value
 * @param container A container
 * @param message If given, is appended to the default fail message
 */
template<typename T,
         typename Container,
         typename... Args>
void
assert_in_container(const T& value,
                    const Container& container,
                    const Args&... message)
{
    unittest::core::check_isfinite(value, "value", UNITTEST_FUNC, message...);
    unittest::core::check_isfinite_container(container, "container", UNITTEST_FUNC, message...);
    if (!unittest::core::is_contained(value, container)) {
        const std::string text = "value not in container" + unittest::core::str_if(": ", value);
        unittest::fail(UNITTEST_FUNC, text, message...);
    }
}
/**
 * @brief Asserts that a value is not in a container.
 *  Required operators: ==
 * @param value A value
 * @param container A container
 * @param message If given, is appended to the default fail message
 */
template<typename T,
         typename Container,
         typename... Args>
void
assert_not_in_container(const T& value,
                        const Container& container,
                        const Args&... message)
{
    unittest::core::check_isfinite(value, "value", UNITTEST_FUNC, message...);
    unittest::core::check_isfinite_container(container, "container", UNITTEST_FUNC, message...);
    if (unittest::core::is_contained(value, container)) {
        const std::string text = "value in container" + unittest::core::str_if(": ", value);
        unittest::fail(UNITTEST_FUNC, text, message...);
    }
}
/**
 * @brief Asserts that a value is approx. in a container up to some epsilon.
 *  The assertion succeeds if |a - b| < epsilon for at least one element.
 *  Required operators: <, >, -
 * @param value A value
 * @param container A container
 * @param epsilon The epsilon
 * @param message If given, is appended to the default fail message
 */
template<typename T,
         typename Container,
         typename U,
         typename... Args>
void
assert_approx_in_container(const T& value,
                           const Container& container,
                           const U& epsilon,
                           const Args&... message)
{
    unittest::core::check_isfinite(value, "value", UNITTEST_FUNC, message...);
    unittest::core::check_isfinite_container(container, "container", UNITTEST_FUNC, message...);
    unittest::core::check_epsilon(epsilon, UNITTEST_FUNC, message...);
    if (!unittest::core::is_approx_contained(value, container, epsilon)) {
        const std::string text = "value not approx. in container" + unittest::core::str_if(": ", value) + unittest::core::str_if(" with epsilon = ", epsilon);
        unittest::fail(UNITTEST_FUNC, text, message...);
    }
}
/**
 * @brief Asserts that a value is not approx. in a container up to some epsilon.
 *  The assertion succeeds if |a - b| < epsilon is false for all elements.
 *  Required operators: <, >, -
 * @param value A value
 * @param container A container
 * @param epsilon The epsilon
 * @param message If given, is appended to the default fail message
 */
template<typename T,
         typename Container,
         typename U,
         typename... Args>
void
assert_approx_not_in_container(const T& value,
                               const Container& container,
                               const U& epsilon,
                               const Args&... message)
{
    unittest::core::check_isfinite(value, "value", UNITTEST_FUNC, message...);
    unittest::core::check_isfinite_container(container, "container", UNITTEST_FUNC, message...);
    unittest::core::check_epsilon(epsilon, UNITTEST_FUNC, message...);
    if (unittest::core::is_approx_contained(value, container, epsilon)) {
        const std::string text = "value approx. in container" + unittest::core::str_if(": ", value) + unittest::core::str_if(" with epsilon = ", epsilon);
        unittest::fail(UNITTEST_FUNC, text, message...);
    }
}
/**
 * @brief Asserts that a value is relatively approx. in a container up to some
 *  epsilon. The assertion succeeds if |a - b| < |a| * epsilon for at least one
 *  element. Required operators: <, >, -, *
 * @param value A value
 * @param container A container
 * @param epsilon The epsilon
 * @param message If given, is appended to the default fail message
 */
template<typename T,
         typename Container,
         typename U,
         typename... Args>
void
assert_approxrel_in_container(const T& value,
                              const Container& container,
                              const U& epsilon,
                              const Args&... message)
{
    unittest::core::check_isfinite(value, "value", UNITTEST_FUNC, message...);
    unittest::core::check_isfinite_container(container, "container", UNITTEST_FUNC, message...);
    unittest::core::check_epsilon(epsilon, UNITTEST_FUNC, message...);
    if (!unittest::core::is_approxrel_contained(value, container, epsilon)) {
        const std::string text = "value not relatively approx. in container" + unittest::core::str_if(": ", value) + unittest::core::str_if(" with epsilon = ", epsilon);
        unittest::fail(UNITTEST_FUNC, text, message...);
    }
}
/**
* @brief Asserts that a value is not relatively approx. in a container up to
*  some epsilon. The assertion succeeds if |a - b| < |a| * epsilon is false for
*  all elements. Required operators: <, >, -, *
* @param value A value
* @param container A container
* @param epsilon The epsilon
* @param message If given, is appended to the default fail message
*/
template<typename T,
         typename Container,
         typename U,
         typename... Args>
void
assert_approxrel_not_in_container(const T& value,
                                  const Container& container,
                                  const U& epsilon,
                                  const Args&... message)
{
    unittest::core::check_isfinite(value, "value", UNITTEST_FUNC, message...);
    unittest::core::check_isfinite_container(container, "container", UNITTEST_FUNC, message...);
	unittest::core::check_epsilon(epsilon, UNITTEST_FUNC, message...);
	if (unittest::core::is_approxrel_contained(value, container, epsilon)) {
		const std::string text = "value relatively approx. in container" + unittest::core::str_if(": ", value) + unittest::core::str_if(" with epsilon = ", epsilon);
		unittest::fail(UNITTEST_FUNC, text, message...);
	}
}
/**
 * @brief Asserts that two containers are equal.
 *  Required operators: ==
 * @param expected The expected container
 * @param actual The actual container
 * @param message If given, is appended to the default fail message
 */
template<typename Container1,
         typename Container2,
         typename... Args>
void
assert_equal_containers(const Container1& expected,
                        const Container2& actual,
                        const Args&... message)
{
    unittest::core::check_isfinite_container(expected, "expected", UNITTEST_FUNC, message...);
    unittest::core::check_isfinite_container(actual, "actual", UNITTEST_FUNC, message...);
    if (!unittest::core::is_containers_equal(expected, actual)) {
        const std::string text = "containers are not equal";
        unittest::fail(UNITTEST_FUNC, text, message...);
    }
}
/**
 * @brief Asserts that two containers are not equal.
 *  Required operators: ==
 * @param first A container
 * @param second Another container
 * @param message If given, is appended to the default fail message
 */
template<typename Container1,
         typename Container2,
         typename... Args>
void
assert_not_equal_containers(const Container1& first,
                            const Container2& second,
                            const Args&... message)
{
    unittest::core::check_isfinite_container(first, "first", UNITTEST_FUNC, message...);
    unittest::core::check_isfinite_container(second, "second", UNITTEST_FUNC, message...);
    if (unittest::core::is_containers_equal(first, second)) {
        const std::string text = "containers are equal";
        unittest::fail(UNITTEST_FUNC, text, message...);
    }
}
/**
 * @brief Asserts that two containers are approx. equal up to some epsilon.
 *  The assertion succeeds if |a - b| < epsilon for all pairs of elements.
 *  Required operators: <, >, -
 * @param expected The expected container
 * @param actual The actual container
 * @param epsilon The epsilon
 * @param message If given, is appended to the default fail message
 */
template<typename Container1,
         typename Container2,
         typename V,
         typename... Args>
void
assert_approx_equal_containers(const Container1& expected,
                               const Container2& actual,
                               const V& epsilon,
                               const Args&... message)
{
    unittest::core::check_isfinite_container(expected, "expected", UNITTEST_FUNC, message...);
    unittest::core::check_isfinite_container(actual, "actual", UNITTEST_FUNC, message...);
    unittest::core::check_epsilon(epsilon, UNITTEST_FUNC, message...);
    if (!unittest::core::is_containers_approx_equal(expected, actual, epsilon)) {
        const std::string text = "containers are not approx. equal" + unittest::core::str_if(" with epsilon = ", epsilon);
        unittest::fail(UNITTEST_FUNC, text, message...);
    }
}
/**
 * @brief Asserts that two containers are not approx. equal up to some
 *  epsilon. The assertion succeeds if |a - b| < epsilon is false for at
 *  least one pair of elements.
 *  Required operators: <, >, -
 * @param first A container
 * @param second Another container
 * @param epsilon The epsilon
 * @param message If given, is appended to the default fail message
 */
template<typename Container1,
         typename Container2,
         typename V,
         typename... Args>
void
assert_approx_not_equal_containers(const Container1& first,
                                   const Container2& second,
                                   const V& epsilon,
                                   const Args&... message)
{
    unittest::core::check_isfinite_container(first, "first", UNITTEST_FUNC, message...);
    unittest::core::check_isfinite_container(second, "second", UNITTEST_FUNC, message...);
    unittest::core::check_epsilon(epsilon, UNITTEST_FUNC, message...);
    if (unittest::core::is_containers_approx_equal(first, second, epsilon)) {
        const std::string text = "containers are approx. equal" + unittest::core::str_if(" with epsilon = ", epsilon);
        unittest::fail(UNITTEST_FUNC, text, message...);
    }
}
/**
 * @brief Asserts that two containers are relatively approx. equal up to some
 *  epsilon. The assertion succeeds if |a - b| < |a| * epsilon for all pairs of
 *  elements.
 *  Required operators: <, >, -, *
 * @param expected The expected container
 * @param actual The actual container
 * @param epsilon The epsilon
 * @param message If given, is appended to the default fail message
 */
template<typename Container1,
         typename Container2,
         typename V,
         typename... Args>
void
assert_approxrel_equal_containers(const Container1& expected,
                                  const Container2& actual,
                                  const V& epsilon,
                                  const Args&... message)
{
    unittest::core::check_isfinite_container(expected, "expected", UNITTEST_FUNC, message...);
    unittest::core::check_isfinite_container(actual, "actual", UNITTEST_FUNC, message...);
    unittest::core::check_epsilon(epsilon, UNITTEST_FUNC, message...);
    if (!unittest::core::is_containers_approxrel_equal(expected, actual, epsilon)) {
        const std::string text = "containers are not relatively approx. equal" + unittest::core::str_if(" with epsilon = ", epsilon);
        unittest::fail(UNITTEST_FUNC, text, message...);
    }
}
/**
 * @brief Asserts that two containers are not relatively approx. equal up to
 *  some epsilon. The assertion succeeds if |a - b| < |a| * epsilon is false for
 *  at least one pair of elements.
 *  Required operators: <, >, -, *
 * @param first A container
 * @param second Another container
 * @param epsilon The epsilon
 * @param message If given, is appended to the default fail message
 */
template<typename Container1,
         typename Container2,
         typename V,
         typename... Args>
void
assert_approxrel_not_equal_containers(const Container1& first,
                                      const Container2& second,
                                      const V& epsilon,
                                      const Args&... message)
{
    unittest::core::check_isfinite_container(first, "first", UNITTEST_FUNC, message...);
    unittest::core::check_isfinite_container(second, "second", UNITTEST_FUNC, message...);
    unittest::core::check_epsilon(epsilon, UNITTEST_FUNC, message...);
    if (unittest::core::is_containers_approxrel_equal(first, second, epsilon)) {
        const std::string text = "containers are relatively approx. equal" + unittest::core::str_if(" with epsilon = ", epsilon);
        unittest::fail(UNITTEST_FUNC, text, message...);
    }
}
/**
 * @brief Asserts that all container elements match a given condition.
 * @param container A container
 * @param condition A condition returning a boolean that can be applied to
 *  each container element
 * @param message If given, is appended to the default fail message
 */
template<typename Container,
         typename Functor,
         typename... Args>
void
assert_all_of(const Container& container,
              Functor condition,
              const Args&... message)
{
    if (!std::all_of(std::begin(container), std::end(container), condition)) {
        const std::string text = "Not all elements match the condition";
        unittest::fail(UNITTEST_FUNC, text, message...);
    }
}
/**
 * @brief Asserts that not all container elements match a given condition.
 * @param container A container
 * @param condition A condition returning a boolean that can be applied to
 *  each container element
 * @param message If given, is appended to the default fail message
 */
template<typename Container,
         typename Functor,
         typename... Args>
void
assert_not_all_of(const Container& container,
                  Functor condition,
                  const Args&... message)
{
    if (std::all_of(std::begin(container), std::end(container), condition)) {
        const std::string text = "All elements match the condition";
        unittest::fail(UNITTEST_FUNC, text, message...);
    }
}
/**
 * @brief Asserts that any container element matches a given condition.
 * @param container A container
 * @param condition A condition returning a boolean that can be applied to
 *  each container element
 * @param message If given, is appended to the default fail message
 */
template<typename Container,
         typename Functor,
         typename... Args>
void
assert_any_of(const Container& container,
              Functor condition,
              const Args&... message)
{
    if (!std::any_of(std::begin(container), std::end(container), condition)) {
        const std::string text = "No element matches the condition";
        unittest::fail(UNITTEST_FUNC, text, message...);
    }
}
/**
 * @brief Asserts that none of the container elements match a given condition.
 * @param container A container
 * @param condition A condition returning a boolean that can be applied to
 *  each container element
 * @param message If given, is appended to the default fail message
 */
template<typename Container,
         typename Functor,
         typename... Args>
void
assert_none_of(const Container& container,
               Functor condition,
               const Args&... message)
{
    if (!std::none_of(std::begin(container), std::end(container), condition)) {
        const std::string text = "At least one element matches the condition";
        unittest::fail(UNITTEST_FUNC, text, message...);
    }
}
/**
 * @brief Asserts that a string matches a regular expression.
 * @param string A string
 * @param regex The regular expression
 * @param message If given, is appended to the default fail message
 */
template<typename... Args>
void
assert_regex_match(const std::string& string,
                   const std::string& regex,
                   const Args&... message)
{
    if (!unittest::core::is_regex_matched(string, regex)) {
        const std::string text = unittest::core::str(string) + " does not match the pattern " + unittest::core::str(regex);
        unittest::fail(UNITTEST_FUNC, text, message...);
    }
}
/**
 * @brief Asserts that a string matches a regular expression.
 * @param string A string
 * @param regex The regular expression
 * @param message If given, is appended to the default fail message
 */
template<typename... Args>
void
assert_regex_match(const std::string& string,
                   const std::regex& regex,
                   const Args&... message)
{
    if (!unittest::core::is_regex_matched(string, regex)) {
        const std::string text = unittest::core::str(string) + " does not match the given pattern";
        unittest::fail(UNITTEST_FUNC, text, message...);
    }
}
/**
 * @brief Asserts that a string does not match a regular expression.
 * @param string A string
 * @param regex The regular expression
 * @param message If given, is appended to the default fail message
 */
template<typename... Args>
void
assert_not_regex_match(const std::string& string,
                       const std::string& regex,
                       const Args&... message)
{
    if (unittest::core::is_regex_matched(string, regex)) {
        const std::string text = unittest::core::str(string) + " matches the pattern " + unittest::core::str(regex);
        unittest::fail(UNITTEST_FUNC, text, message...);
    }
}
/**
 * @brief Asserts that a string does not match a regular expression.
 * @param string A string
 * @param regex The regular expression
 * @param message If given, is appended to the default fail message
 */
template<typename... Args>
void
assert_not_regex_match(const std::string& string,
                       const std::regex& regex,
                       const Args&... message)
{
    if (unittest::core::is_regex_matched(string, regex)) {
        const std::string text = unittest::core::str(string) + " matches the given pattern";
        unittest::fail(UNITTEST_FUNC, text, message...);
    }
}
/**
 * @brief Asserts that a functor throws Exception.
 *  Required operators: ()
 * @param functor The functor
 * @param message If given, is appended to the default fail message
 */
template<typename Exception,
         typename Functor,
         typename... Args>
void
assert_throw(Functor functor,
             const Args&... message)
{
    bool caught = false;
    if (unittest::core::testsuite::instance()->get_arguments().handle_exceptions) {
        try {
            functor();
        } catch (const Exception&) {
            caught = true;
        } catch (const std::exception& e) {
            const std::string text = unittest::join("An unexpected exception was thrown: ", unittest::core::type_name(e), ": '", e.what(), "'");
            unittest::fail(UNITTEST_FUNC, text, message...);
        } catch (...) {
            const std::string text = "An unexpected, unknown exception was thrown";
            unittest::fail(UNITTEST_FUNC, text, message...);
        }
    } else {
        try {
            functor();
        } catch (const Exception&) {
            caught = true;
        }
    }
    if (!caught) {
        const std::string text = unittest::join("The exception was not thrown: ", unittest::core::type_name<Exception>());
        unittest::fail(UNITTEST_FUNC, text, message...);
    }
}
/**
 * @brief Asserts that a functor does not throw any exception.
 *  Required operators: ()
 * @param functor The functor
 * @param message If given, is appended to the default fail message
 */
template<typename Functor,
         typename... Args>
void
assert_no_throw(Functor functor,
                const Args&... message)
{
    if (unittest::core::testsuite::instance()->get_arguments().handle_exceptions) {
        try {
            functor();
        } catch (const std::exception& e) {
            const std::string text = unittest::join("An exception was thrown: ", unittest::core::type_name(e), ": '", e.what(), "'");
            unittest::fail(UNITTEST_FUNC, text, message...);
        } catch (...) {
            const std::string text = "An unknown exception was thrown";
            unittest::fail(UNITTEST_FUNC, text, message...);
        }
    } else {
        functor();
    }
}

} // assertions
} // unittest
