/**
 * @brief Short aliases for all assertion macros
 * @file assertshortcuts.hpp
 */
#pragma once
/**
 * @brief Asserts that a value is true.
 * @param value A value
 */
#define ASSERT_TRUE(value) \
UNITTEST_ASSERT_TRUE(value)
/**
 * @brief Asserts that a value is true.
 * @param value A value
 * @param ... A user defined message
 */
#define ASSERT_TRUE_MSG(value, ...) \
UNITTEST_ASSERT_TRUE_MSG(value, __VA_ARGS__)
/**
 * @brief Asserts that a value is true (non-deadly assertion).
 * @param value A value
 */
#define NDASSERT_TRUE(value) \
UNITTEST_NDASSERT_TRUE(value)
/**
 * @brief Asserts that a value is true (non-deadly assertion).
 * @param value A value
 * @param ... A user defined message
 */
#define NDASSERT_TRUE_MSG(value, ...) \
UNITTEST_NDASSERT_TRUE_MSG(value, __VA_ARGS__)
/**
 * @brief Asserts that a value is false.
 * @param value A value
 */
#define ASSERT_FALSE(value) \
UNITTEST_ASSERT_FALSE(value)
/**
 * @brief Asserts that a value is false.
 * @param value A value
 * @param ... A user defined message
 */
#define ASSERT_FALSE_MSG(value, ...) \
UNITTEST_ASSERT_FALSE_MSG(value, __VA_ARGS__)
/**
 * @brief Asserts that a value is false (non-deadly assertion).
 * @param value A value
 */
#define NDASSERT_FALSE(value) \
UNITTEST_NDASSERT_FALSE(value)
/**
 * @brief Asserts that a value is false (non-deadly assertion).
 * @param value A value
 * @param ... A user defined message
 */
#define NDASSERT_FALSE_MSG(value, ...) \
UNITTEST_NDASSERT_FALSE_MSG(value, __VA_ARGS__)
/**
 * @brief Asserts that two values are equal.
 *  Required operators: ==, <<
 * @param expected The expected value
 * @param actual The actual value
 */
#define ASSERT_EQUAL(expected, actual) \
UNITTEST_ASSERT_EQUAL(expected, actual)
/**
 * @brief Asserts that two values are equal.
 *  Required operators: ==, <<
 * @param expected The expected value
 * @param actual The actual value
 * @param ... A user defined message
 */
#define ASSERT_EQUAL_MSG(expected, actual, ...) \
UNITTEST_ASSERT_EQUAL_MSG(expected, actual, __VA_ARGS__)
/**
 * @brief Asserts that two values are equal (non-deadly assertion).
 *  Required operators: ==, <<
 * @param expected The expected value
 * @param actual The actual value
 */
#define NDASSERT_EQUAL(expected, actual) \
UNITTEST_NDASSERT_EQUAL(expected, actual)
/**
 * @brief Asserts that two values are equal (non-deadly assertion).
 *  Required operators: ==, <<
 * @param expected The expected value
 * @param actual The actual value
 * @param ... A user defined message
 */
#define NDASSERT_EQUAL_MSG(expected, actual, ...) \
UNITTEST_NDASSERT_EQUAL_MSG(expected, actual, __VA_ARGS__)
/**
 * @brief Asserts that two values are not equal.
 *  Required operators: ==, <<
 * @param first The first value
 * @param second The second value
 */
#define ASSERT_NOT_EQUAL(first, second) \
UNITTEST_ASSERT_NOT_EQUAL(first, second)
/**
 * @brief Asserts that two values are not equal.
 *  Required operators: ==, <<
 * @param first The first value
 * @param second The second value
 * @param ... A user defined message
 */
#define ASSERT_NOT_EQUAL_MSG(first, second, ...) \
UNITTEST_ASSERT_NOT_EQUAL_MSG(first, second, __VA_ARGS__)
/**
 * @brief Asserts that two values are not equal (non-deadly assertion).
 *  Required operators: ==, <<
 * @param first The first value
 * @param second The second value
 */
#define NDASSERT_NOT_EQUAL(first, second) \
UNITTEST_NDASSERT_NOT_EQUAL(first, second)
/**
 * @brief Asserts that two values are not equal (non-deadly assertion).
 *  Required operators: ==, <<
 * @param first The first value
 * @param second The second value
 * @param ... A user defined message
 */
#define NDASSERT_NOT_EQUAL_MSG(first, second, ...) \
UNITTEST_NDASSERT_NOT_EQUAL_MSG(first, second, __VA_ARGS__)
/**
 * @brief Asserts that two values are approximately equal up to some
 *  epsilon. The assertion succeeds if |a - b| < epsilon.
 *  Required operators: <, >, -, <<
 * @param expected The expected value
 * @param actual The actual value
 * @param epsilon The epsilon
 */
#define ASSERT_APPROX_EQUAL(expected, actual, epsilon) \
UNITTEST_ASSERT_APPROX_EQUAL(expected, actual, epsilon)
/**
 * @brief Asserts that two values are approximately equal up to some
 *  epsilon. The assertion succeeds if |a - b| < epsilon.
 *  Required operators: <, >, -, <<
 * @param expected The expected value
 * @param actual The actual value
 * @param epsilon The epsilon
 * @param ... A user defined message
 */
#define ASSERT_APPROX_EQUAL_MSG(expected, actual, epsilon, ...) \
UNITTEST_ASSERT_APPROX_EQUAL_MSG(expected, actual, epsilon, __VA_ARGS__)
/**
 * @brief Asserts that two values are approximately equal up to some
 *  epsilon. The assertion succeeds if |a - b| < epsilon (non-deadly assertions).
 *  Required operators: <, >, -, <<
 * @param expected The expected value
 * @param actual The actual value
 * @param epsilon The epsilon
 */
#define NDASSERT_APPROX_EQUAL(expected, actual, epsilon) \
UNITTEST_NDASSERT_APPROX_EQUAL(expected, actual, epsilon)
/**
 * @brief Asserts that two values are approximately equal up to some
 *  epsilon. The assertion succeeds if |a - b| < epsilon (non-deadly assertions).
 *  Required operators: <, >, -, <<
 * @param expected The expected value
 * @param actual The actual value
 * @param epsilon The epsilon
 * @param ... A user defined message
 */
#define NDASSERT_APPROX_EQUAL_MSG(expected, actual, epsilon, ...) \
UNITTEST_NDASSERT_APPROX_EQUAL_MSG(expected, actual, epsilon, __VA_ARGS__)
/**
 * @brief Asserts that two values are not approximately equal up to some
 *  epsilon. The assertion succeeds if |a - b| < epsilon is false.
 *  Required operators: <, >, -, <<
 * @param first The first value
 * @param second The second value
 * @param epsilon The epsilon
 */
#define ASSERT_APPROX_NOT_EQUAL(first, second, epsilon) \
UNITTEST_ASSERT_APPROX_NOT_EQUAL(first, second, epsilon)
/**
 * @brief Asserts that two values are not approximately equal up to some
 *  epsilon. The assertion succeeds if |a - b| < epsilon is false.
 *  Required operators: <, >, -, <<
 * @param first The first value
 * @param second The second value
 * @param epsilon The epsilon
 * @param ... A user defined message
 */
#define ASSERT_APPROX_NOT_EQUAL_MSG(first, second, epsilon, ...) \
UNITTEST_ASSERT_APPROX_NOT_EQUAL_MSG(first, second, epsilon, __VA_ARGS__)
/**
 * @brief Asserts that two values are not approximately equal up to some
 *  epsilon. The assertion succeeds if |a - b| < epsilon is false
 *  (non-deadly assertion).
 *  Required operators: <, >, -, <<
 * @param first The first value
 * @param second The second value
 * @param epsilon The epsilon
 */
#define NDASSERT_APPROX_NOT_EQUAL(first, second, epsilon) \
UNITTEST_NDASSERT_APPROX_NOT_EQUAL(first, second, epsilon)
/**
 * @brief Asserts that two values are not approximately equal up to some
 *  epsilon. The assertion succeeds if |a - b| < epsilon is false
 *  (non-deadly assertion).
 *  Required operators: <, >, -, <<
 * @param first The first value
 * @param second The second value
 * @param epsilon The epsilon
 * @param ... A user defined message
 */
#define NDASSERT_APPROX_NOT_EQUAL_MSG(first, second, epsilon, ...) \
UNITTEST_NDASSERT_APPROX_NOT_EQUAL_MSG(first, second, epsilon, __VA_ARGS__)
/**
 * @brief Asserts that two values are relatively approx. equal up to some
 *  epsilon. The assertion succeeds if |a - b| < |a| * epsilon.
 *  Required operators: <, >, -, *, <<
 * @param expected The expected value
 * @param actual The actual value
 * @param epsilon The epsilon
 */
#define ASSERT_APPROXREL_EQUAL(expected, actual, epsilon) \
UNITTEST_ASSERT_APPROXREL_EQUAL(expected, actual, epsilon)
/**
 * @brief Asserts that two values are relatively approx. equal up to some
 *  epsilon. The assertion succeeds if |a - b| < |a| * epsilon.
 *  Required operators: <, >, -, *, <<
 * @param expected The expected value
 * @param actual The actual value
 * @param epsilon The epsilon
 * @param ... A user defined message
 */
#define ASSERT_APPROXREL_EQUAL_MSG(expected, actual, epsilon, ...) \
UNITTEST_ASSERT_APPROXREL_EQUAL_MSG(expected, actual, epsilon, __VA_ARGS__)
/**
 * @brief Asserts that two values are relatively approx. equal up to some
 *  epsilon. The assertion succeeds if |a - b| < |a| * epsilon
 *  (non-deadly assertion).
 *  Required operators: <, >, -, *, <<
 * @param expected The expected value
 * @param actual The actual value
 * @param epsilon The epsilon
 */
#define NDASSERT_APPROXREL_EQUAL(expected, actual, epsilon) \
UNITTEST_NDASSERT_APPROXREL_EQUAL(expected, actual, epsilon)
/**
 * @brief Asserts that two values are relatively approx. equal up to some
 *  epsilon. The assertion succeeds if |a - b| < |a| * epsilon
 *  (non-deadly assertion).
 *  Required operators: <, >, -, *, <<
 * @param expected The expected value
 * @param actual The actual value
 * @param epsilon The epsilon
 * @param ... A user defined message
 */
#define NDASSERT_APPROXREL_EQUAL_MSG(expected, actual, epsilon, ...) \
UNITTEST_NDASSERT_APPROXREL_EQUAL_MSG(expected, actual, epsilon, __VA_ARGS__)
/**
 * @brief Asserts that two values are not relatively approx. equal up to some
 *  epsilon. The assertion succeeds if |a - b| < |a| * epsilon is false
 *  Required operators: <, >, -, *, <<
 * @param first The first value
 * @param second The second value
 * @param epsilon The epsilon
 */
#define ASSERT_APPROXREL_NOT_EQUAL(first, second, epsilon) \
UNITTEST_ASSERT_APPROXREL_NOT_EQUAL(first, second, epsilon)
/**
 * @brief Asserts that two values are not relatively approx. equal up to some
 *  epsilon. The assertion succeeds if |a - b| < |a| * epsilon is false
 *  Required operators: <, >, -, *, <<
 * @param first The first value
 * @param second The second value
 * @param epsilon The epsilon
 * @param ... A user defined message
 */
#define ASSERT_APPROXREL_NOT_EQUAL_MSG(first, second, epsilon, ...) \
UNITTEST_ASSERT_APPROXREL_NOT_EQUAL_MSG(first, second, epsilon, __VA_ARGS__)
/**
 * @brief Asserts that two values are not relatively approx. equal up to some
 *  epsilon. The assertion succeeds if |a - b| < |a| * epsilon is false
 *  (non-deadly assertion).
 *  Required operators: <, >, -, *, <<
 * @param first The first value
 * @param second The second value
 * @param epsilon The epsilon
 */
#define NDASSERT_APPROXREL_NOT_EQUAL(first, second, epsilon) \
UNITTEST_NDASSERT_APPROXREL_NOT_EQUAL(first, second, epsilon)
/**
 * @brief Asserts that two values are not relatively approx. equal up to some
 *  epsilon. The assertion succeeds if |a - b| < |a| * epsilon is false
 *  (non-deadly assertion).
 *  Required operators: <, >, -, *, <<
 * @param first The first value
 * @param second The second value
 * @param epsilon The epsilon
 * @param ... A user defined message
 */
#define NDASSERT_APPROXREL_NOT_EQUAL_MSG(first, second, epsilon, ...) \
UNITTEST_NDASSERT_APPROXREL_NOT_EQUAL_MSG(first, second, epsilon, __VA_ARGS__)
/**
 * @brief Asserts that the first value is greater than the second.
 *  Required operators: >, <<
 * @param first A value
 * @param second Another value
 */
#define ASSERT_GREATER(first, second) \
UNITTEST_ASSERT_GREATER(first, second)
/**
 * @brief Asserts that the first value is greater than the second.
 *  Required operators: >, <<
 * @param first A value
 * @param second Another value
 * @param ... A user defined message
 */
#define ASSERT_GREATER_MSG(first, second, ...) \
UNITTEST_ASSERT_GREATER_MSG(first, second, __VA_ARGS__)
/**
 * @brief Asserts that the first value is greater than the second
 *  (non-deadly assertion).
 *  Required operators: >, <<
 * @param first A value
 * @param second Another value
 */
#define NDASSERT_GREATER(first, second) \
UNITTEST_NDASSERT_GREATER(first, second)
/**
 * @brief Asserts that the first value is greater than the second
 *  (non-deadly assertion).
 *  Required operators: >, <<
 * @param first A value
 * @param second Another value
 * @param ... A user defined message
 */
#define NDASSERT_GREATER_MSG(first, second, ...) \
UNITTEST_NDASSERT_GREATER_MSG(first, second, __VA_ARGS__)
/**
 * @brief Asserts that the first value is greater than or equal to the second.
 *  Required operators: <, <<
 * @param first A value
 * @param second Another value
 */
#define ASSERT_GREATER_EQUAL(first, second) \
UNITTEST_ASSERT_GREATER_EQUAL(first, second)
/**
 * @brief Asserts that the first value is greater than or equal to the second.
 *  Required operators: <, <<
 * @param first A value
 * @param second Another value
 * @param ... A user defined message
 */
#define ASSERT_GREATER_EQUAL_MSG(first, second, ...) \
UNITTEST_ASSERT_GREATER_EQUAL_MSG(first, second, __VA_ARGS__)
/**
 * @brief Asserts that the first value is greater than or equal
 *  to the second (non-deadly assertion).
 *  Required operators: <, <<
 * @param first A value
 * @param second Another value
 */
#define NDASSERT_GREATER_EQUAL(first, second) \
UNITTEST_NDASSERT_GREATER_EQUAL(first, second)
/**
 * @brief Asserts that the first value is greater than or equal
 *  to the second (non-deadly assertion).
 *  Required operators: <, <<
 * @param first A value
 * @param second Another value
 * @param ... A user defined message
 */
#define NDASSERT_GREATER_EQUAL_MSG(first, second, ...) \
UNITTEST_NDASSERT_GREATER_EQUAL_MSG(first, second, __VA_ARGS__)
/**
 * @brief Asserts that the first value is lesser than the second.
 *  Required operators: <, <<
 * @param first A value
 * @param second Another value
 */
#define ASSERT_LESSER(first, second) \
UNITTEST_ASSERT_LESSER(first, second)
/**
 * @brief Asserts that the first value is lesser than the second.
 *  Required operators: <, <<
 * @param first A value
 * @param second Another value
 * @param ... A user defined message
 */
#define ASSERT_LESSER_MSG(first, second, ...) \
UNITTEST_ASSERT_LESSER_MSG(first, second, __VA_ARGS__)
/**
 * @brief Asserts that the first value is lesser than the second
 *  (non-deadly assertion).
 *  Required operators: <, <<
 * @param first A value
 * @param second Another value
 */
#define NDASSERT_LESSER(first, second) \
UNITTEST_NDASSERT_LESSER(first, second)
/**
 * @brief Asserts that the first value is lesser than the second
 *  (non-deadly assertion).
 *  Required operators: <, <<
 * @param first A value
 * @param second Another value
 * @param ... A user defined message
 */
#define NDASSERT_LESSER_MSG(first, second, ...) \
UNITTEST_NDASSERT_LESSER_MSG(first, second, __VA_ARGS__)
/**
 * @brief Asserts that the first value is lesser than or equal to the second.
 *  Required operators: >, <<
 * @param first A value
 * @param second Another value
 */
#define ASSERT_LESSER_EQUAL(first, second) \
UNITTEST_ASSERT_LESSER_EQUAL(first, second)
/**
 * @brief Asserts that the first value is lesser than or equal to the second.
 *  Required operators: >, <<
 * @param first A value
 * @param second Another value
 * @param ... A user defined message
 */
#define ASSERT_LESSER_EQUAL_MSG(first, second, ...) \
UNITTEST_ASSERT_LESSER_EQUAL_MSG(first, second, __VA_ARGS__)
/**
 * @brief Asserts that the first value is lesser than or equal
 *  to the second (non-deadly assertion).
 *  Required operators: >, <<
 * @param first A value
 * @param second Another value
 */
#define NDASSERT_LESSER_EQUAL(first, second) \
UNITTEST_NDASSERT_LESSER_EQUAL(first, second)
/**
 * @brief Asserts that the first value is lesser than or equal
 *  to the second (non-deadly assertion).
 *  Required operators: >, <<
 * @param first A value
 * @param second Another value
 * @param ... A user defined message
 */
#define NDASSERT_LESSER_EQUAL_MSG(first, second, ...) \
UNITTEST_NDASSERT_LESSER_EQUAL_MSG(first, second, __VA_ARGS__)
/**
 * @brief Asserts that a value is in a given range with included bounds.
 *  Required operators: <, >, <<
 * @param value A value
 * @param lower The lower bound
 * @param upper The upper bound
 */
#define ASSERT_IN_RANGE(value, lower, upper) \
UNITTEST_ASSERT_IN_RANGE(value, lower, upper)
/**
 * @brief Asserts that a value is in a given range with included bounds.
 *  Required operators: <, >, <<
 * @param value A value
 * @param lower The lower bound
 * @param upper The upper bound
 * @param ... A user defined message
 */
#define ASSERT_IN_RANGE_MSG(value, lower, upper, ...) \
UNITTEST_ASSERT_IN_RANGE_MSG(value, lower, upper, __VA_ARGS__)
/**
 * @brief Asserts that a value is in a given range with included bounds
 *  (non-deadly assertion).
 *  Required operators: <, >, <<
 * @param value A value
 * @param lower The lower bound
 * @param upper The upper bound
 */
#define NDASSERT_IN_RANGE(value, lower, upper) \
UNITTEST_NDASSERT_IN_RANGE(value, lower, upper)
/**
 * @brief Asserts that a value is in a given range with included bounds
 *  (non-deadly assertion).
 *  Required operators: <, >, <<
 * @param value A value
 * @param lower The lower bound
 * @param upper The upper bound
 * @param ... A user defined message
 */
#define NDASSERT_IN_RANGE_MSG(value, lower, upper, ...) \
UNITTEST_NDASSERT_IN_RANGE_MSG(value, lower, upper, __VA_ARGS__)
/**
 * @brief Asserts that a value is not in a given range with included bounds.
 *  Required operators: <, >, <<
 * @param value A value
 * @param lower The lower bound
 * @param upper The upper bound
 */
#define ASSERT_NOT_IN_RANGE(value, lower, upper) \
UNITTEST_ASSERT_NOT_IN_RANGE(value, lower, upper)
/**
 * @brief Asserts that a value is not in a given range with included bounds.
 *  Required operators: <, >, <<
 * @param value A value
 * @param lower The lower bound
 * @param upper The upper bound
 * @param ... A user defined message
 */
#define ASSERT_NOT_IN_RANGE_MSG(value, lower, upper, ...) \
UNITTEST_ASSERT_NOT_IN_RANGE_MSG(value, lower, upper, __VA_ARGS__)
/**
 * @brief Asserts that a value is not in a given range with included bounds
 *  (non-deadly assertion).
 *  Required operators: <, >, <<
 * @param value A value
 * @param lower The lower bound
 * @param upper The upper bound
 */
#define NDASSERT_NOT_IN_RANGE(value, lower, upper) \
UNITTEST_NDASSERT_NOT_IN_RANGE(value, lower, upper)
/**
 * @brief Asserts that a value is not in a given range with included bounds
 *  (non-deadly assertion).
 *  Required operators: <, >, <<
 * @param value A value
 * @param lower The lower bound
 * @param upper The upper bound
 * @param ... A user defined message
 */
#define NDASSERT_NOT_IN_RANGE_MSG(value, lower, upper, ...) \
UNITTEST_NDASSERT_NOT_IN_RANGE_MSG(value, lower, upper, __VA_ARGS__)
/**
 * @brief Asserts that a value is in a container.
 *  Required operators: ==
 * @param value A value
 * @param container A container
 */
#define ASSERT_IN_CONTAINER(value, container) \
UNITTEST_ASSERT_IN_CONTAINER(value, container)
/**
 * @brief Asserts that a value is in a container.
 *  Required operators: ==
 * @param value A value
 * @param container A container
 * @param ... A user defined message
 */
#define ASSERT_IN_CONTAINER_MSG(value, container, ...) \
UNITTEST_ASSERT_IN_CONTAINER_MSG(value, container, __VA_ARGS__)
/**
 * @brief Asserts that a value is in a container (non-deadly assertion).
 *  Required operators: ==
 * @param value A value
 * @param container A container
 */
#define NDASSERT_IN_CONTAINER(value, container) \
UNITTEST_NDASSERT_IN_CONTAINER(value, container)
/**
 * @brief Asserts that a value is in a container (non-deadly assertion).
 *  Required operators: ==
 * @param value A value
 * @param container A container
 * @param ... A user defined message
 */
#define NDASSERT_IN_CONTAINER_MSG(value, container, ...) \
UNITTEST_NDASSERT_IN_CONTAINER_MSG(value, container, __VA_ARGS__)
/**
 * @brief Asserts that a value is not in a container.
 *  Required operators: ==
 * @param value A value
 * @param container A container
 */
#define ASSERT_NOT_IN_CONTAINER(value, container) \
UNITTEST_ASSERT_NOT_IN_CONTAINER(value, container)
/**
 * @brief Asserts that a value is not in a container.
 *  Required operators: ==
 * @param value A value
 * @param container A container
 * @param ... A user defined message
 */
#define ASSERT_NOT_IN_CONTAINER_MSG(value, container, ...) \
UNITTEST_ASSERT_NOT_IN_CONTAINER_MSG(value, container, __VA_ARGS__)
/**
 * @brief Asserts that a value is not in a container (non-deadly assertion).
 *  Required operators: ==
 * @param value A value
 * @param container A container
 */
#define NDASSERT_NOT_IN_CONTAINER(value, container) \
UNITTEST_NDASSERT_NOT_IN_CONTAINER(value, container)
/**
 * @brief Asserts that a value is not in a container (non-deadly assertion).
 *  Required operators: ==
 * @param value A value
 * @param container A container
 * @param ... A user defined message
 */
#define NDASSERT_NOT_IN_CONTAINER_MSG(value, container, ...) \
UNITTEST_NDASSERT_NOT_IN_CONTAINER_MSG(value, container, __VA_ARGS__)
/**
 * @brief Asserts that a value is approx. in a container up to some epsilon.
 *  The assertion succeeds if |a - b| < epsilon for at least one element.
 *  Required operators: <, >, -
 * @param value A value
 * @param container A container
 * @param epsilon The epsilon
 */
#define ASSERT_APPROX_IN_CONTAINER(value, container, epsilon) \
UNITTEST_ASSERT_APPROX_IN_CONTAINER(value, container, epsilon)
/**
 * @brief Asserts that a value is approx. in a container up to some epsilon.
 *  The assertion succeeds if |a - b| < epsilon for at least one element.
 *  Required operators: <, >, -
 * @param value A value
 * @param container A container
 * @param epsilon The epsilon
 * @param ... A user defined message
 */
#define ASSERT_APPROX_IN_CONTAINER_MSG(value, container, epsilon, ...) \
UNITTEST_ASSERT_APPROX_IN_CONTAINER_MSG(value, container, epsilon, __VA_ARGS__)
/**
 * @brief Asserts that a value is approx. in a container up to some epsilon.
 *  The assertion succeeds if |a - b| < epsilon for at least one element (non-deadly assertion).
 *  Required operators: <, >, -
 * @param value A value
 * @param container A container
 * @param epsilon The epsilon
 */
#define NDASSERT_APPROX_IN_CONTAINER(value, container, epsilon) \
UNITTEST_NDASSERT_APPROX_IN_CONTAINER(value, container, epsilon)
/**
 * @brief Asserts that a value is approx. in a container up to some epsilon.
 *  The assertion succeeds if |a - b| < epsilon for at least one element (non-deadly assertion).
 *  Required operators: <, >, -
 * @param value A value
 * @param container A container
 * @param epsilon The epsilon
 * @param ... A user defined message
 */
#define NDASSERT_APPROX_IN_CONTAINER_MSG(value, container, epsilon, ...) \
UNITTEST_NDASSERT_APPROX_IN_CONTAINER_MSG(value, container, epsilon, __VA_ARGS__)
/**
 * @brief Asserts that a value is not approx. in a container up to some epsilon.
 *  The assertion succeeds if |a - b| < epsilon is false for all elements.
 *  Required operators: <, >, -
 * @param value A value
 * @param container A container
 * @param epsilon The epsilon
 */
#define ASSERT_APPROX_NOT_IN_CONTAINER(value, container, epsilon) \
UNITTEST_ASSERT_APPROX_NOT_IN_CONTAINER(value, container, epsilon)
/**
 * @brief Asserts that a value is not approx. in a container up to some epsilon.
 *  The assertion succeeds if |a - b| < epsilon is false for all elements.
 *  Required operators: <, >, -
 * @param value A value
 * @param container A container
 * @param epsilon The epsilon
 * @param ... A user defined message
 */
#define ASSERT_APPROX_NOT_IN_CONTAINER_MSG(value, container, epsilon, ...) \
UNITTEST_ASSERT_APPROX_NOT_IN_CONTAINER_MSG(value, container, epsilon, __VA_ARGS__)
/**
 * @brief Asserts that a value is not approx. in a container up to some epsilon.
 *  The assertion succeeds if |a - b| < epsilon is false for all elements (non-deadly assertion).
 *  Required operators: <, >, -
 * @param value A value
 * @param container A container
 * @param epsilon The epsilon
 */
#define NDASSERT_APPROX_NOT_IN_CONTAINER(value, container, epsilon) \
UNITTEST_NDASSERT_APPROX_NOT_IN_CONTAINER(value, container, epsilon)
/**
 * @brief Asserts that a value is not approx. in a container up to some epsilon.
 *  The assertion succeeds if |a - b| < epsilon is false for all elements (non-deadly assertion).
 *  Required operators: <, >, -
 * @param value A value
 * @param container A container
 * @param epsilon The epsilon
 * @param ... A user defined message
 */
#define NDASSERT_APPROX_NOT_IN_CONTAINER_MSG(value, container, epsilon, ...) \
UNITTEST_NDASSERT_APPROX_NOT_IN_CONTAINER_MSG(value, container, epsilon, __VA_ARGS__)
/**
 * @brief Asserts that a value is relatively approx. in a container up to some
 *  epsilon. The assertion succeeds if |a - b| < |a| * epsilon for at least one
 *  element. Required operators: <, >, -, *
 * @param value A value
 * @param container A container
 * @param epsilon The epsilon
 */
#define ASSERT_APPROXREL_IN_CONTAINER(value, container, epsilon) \
UNITTEST_ASSERT_APPROXREL_IN_CONTAINER(value, container, epsilon)
/**
 * @brief Asserts that a value is relatively approx. in a container up to some
 *  epsilon. The assertion succeeds if |a - b| < |a| * epsilon for at least one
 *  element. Required operators: <, >, -, *
 * @param value A value
 * @param container A container
 * @param epsilon The epsilon
 * @param ... A user defined message
 */
#define ASSERT_APPROXREL_IN_CONTAINER_MSG(value, container, epsilon, ...) \
UNITTEST_ASSERT_APPROXREL_IN_CONTAINER_MSG(value, container, epsilon, __VA_ARGS__)
/**
 * @brief Asserts that a value is relatively approx. in a container up to some
 *  epsilon. The assertion succeeds if |a - b| < |a| * epsilon for at least one
 *  element (non-deadly assertion). Required operators: <, >, -, *
 * @param value A value
 * @param container A container
 * @param epsilon The epsilon
 */
#define NDASSERT_APPROXREL_IN_CONTAINER(value, container, epsilon) \
UNITTEST_NDASSERT_APPROXREL_IN_CONTAINER(value, container, epsilon)
/**
 * @brief Asserts that a value is relatively approx. in a container up to some
 *  epsilon. The assertion succeeds if |a - b| < |a| * epsilon for at least one
 *  element (non-deadly assertion). Required operators: <, >, -, *
 * @param value A value
 * @param container A container
 * @param epsilon The epsilon
 * @param ... A user defined message
 */
#define NDASSERT_APPROXREL_IN_CONTAINER_MSG(value, container, epsilon, ...) \
UNITTEST_NDASSERT_APPROXREL_IN_CONTAINER_MSG(value, container, epsilon, __VA_ARGS__)
/**
 * @brief Asserts that a value is not relatively approx. in a container up to
 *  some epsilon. The assertion succeeds if |a - b| < |a| * epsilon is false for
 *  all elements. Required operators: <, >, -, *
 * @param value A value
 * @param container A container
 * @param epsilon The epsilon
 */
#define ASSERT_APPROXREL_NOT_IN_CONTAINER(value, container, epsilon) \
UNITTEST_ASSERT_APPROXREL_NOT_IN_CONTAINER(value, container, epsilon)
/**
 * @brief Asserts that a value is not relatively approx. in a container up to
 *  some epsilon. The assertion succeeds if |a - b| < |a| * epsilon is false for
 *  all elements. Required operators: <, >, -, *
 * @param value A value
 * @param container A container
 * @param epsilon The epsilon
 * @param ... A user defined message
 */
#define ASSERT_APPROXREL_NOT_IN_CONTAINER_MSG(value, container, epsilon, ...) \
UNITTEST_ASSERT_APPROXREL_NOT_IN_CONTAINER_MSG(value, container, epsilon, __VA_ARGS__)
/**
 * @brief Asserts that a value is not relatively approx. in a container up to
 *  some epsilon. The assertion succeeds if |a - b| < |a| * epsilon is false for
 *  all elements (non-deadly assertion). Required operators: <, >, -, *
 * @param value A value
 * @param container A container
 * @param epsilon The epsilon
 */
#define NDASSERT_APPROXREL_NOT_IN_CONTAINER(value, container, epsilon) \
UNITTEST_NDASSERT_APPROXREL_NOT_IN_CONTAINER(value, container, epsilon)
/**
 * @brief Asserts that a value is not relatively approx. in a container up to
 *  some epsilon. The assertion succeeds if |a - b| < |a| * epsilon is false for
 *  all elements (non-deadly assertion). Required operators: <, >, -, *
 * @param value A value
 * @param container A container
 * @param epsilon The epsilon
 * @param ... A user defined message
 */
#define NDASSERT_APPROXREL_NOT_IN_CONTAINER_MSG(value, container, epsilon, ...) \
UNITTEST_NDASSERT_APPROXREL_NOT_IN_CONTAINER_MSG(value, container, epsilon, __VA_ARGS__)
/**
 * @brief Asserts that two containers are equal.
 *  Required operators: ==
 * @param expected The expected container
 * @param actual The actual container
 */
#define ASSERT_EQUAL_CONTAINERS(expected, actual) \
UNITTEST_ASSERT_EQUAL_CONTAINERS(expected, actual)
/**
 * @brief Asserts that two containers are equal.
 *  Required operators: ==
 * @param expected The expected container
 * @param actual The actual container
 * @param ... A user defined message
 */
#define ASSERT_EQUAL_CONTAINERS_MSG(expected, actual, ...) \
UNITTEST_ASSERT_EQUAL_CONTAINERS_MSG(expected, actual, __VA_ARGS__)
/**
 * @brief Asserts that two containers are equal (non-deadly assertion).
 *  Required operators: ==
 * @param expected The expected container
 * @param actual The actual container
 */
#define NDASSERT_EQUAL_CONTAINERS(expected, actual) \
UNITTEST_NDASSERT_EQUAL_CONTAINERS(expected, actual)
/**
 * @brief Asserts that two containers are equal (non-deadly assertion).
 *  Required operators: ==
 * @param expected The expected container
 * @param actual The actual container
 * @param ... A user defined message
 */
#define NDASSERT_EQUAL_CONTAINERS_MSG(expected, actual, ...) \
UNITTEST_NDASSERT_EQUAL_CONTAINERS_MSG(expected, actual, __VA_ARGS__)
/**
 * @brief Asserts that two containers are not equal.
 *  Required operators: ==
 * @param first A container
 * @param second Another container
 */
#define ASSERT_NOT_EQUAL_CONTAINERS(first, second) \
UNITTEST_ASSERT_NOT_EQUAL_CONTAINERS(first, second)
/**
 * @brief Asserts that two containers are not equal.
 *  Required operators: ==
 * @param first A container
 * @param second Another container
 * @param ... A user defined message
 */
#define ASSERT_NOT_EQUAL_CONTAINERS_MSG(first, second, ...) \
UNITTEST_ASSERT_NOT_EQUAL_CONTAINERS_MSG(first, second, __VA_ARGS__)
/**
 * @brief Asserts that two containers are not equal (non-deadly assertion).
 *  Required operators: ==
 * @param first A container
 * @param second Another container
 */
#define NDASSERT_NOT_EQUAL_CONTAINERS(first, second) \
UNITTEST_NDASSERT_NOT_EQUAL_CONTAINERS(first, second)
/**
 * @brief Asserts that two containers are not equal (non-deadly assertion).
 *  Required operators: ==
 * @param first A container
 * @param second Another container
 * @param ... A user defined message
 */
#define NDASSERT_NOT_EQUAL_CONTAINERS_MSG(first, second, ...) \
UNITTEST_NDASSERT_NOT_EQUAL_CONTAINERS_MSG(first, second, __VA_ARGS__)
/**
 * @brief Asserts that two containers are approx. equal up to some epsilon.
 *  The assertion succeeds if |a - b| < epsilon for all pairs of elements.
 *  Required operators: <, >, -
 * @param expected The expected container
 * @param actual The actual container
 * @param epsilon The epsilon
 */
#define ASSERT_APPROX_EQUAL_CONTAINERS(expected, actual, epsilon) \
UNITTEST_ASSERT_APPROX_EQUAL_CONTAINERS(expected, actual, epsilon)
/**
 * @brief Asserts that two containers are approx. equal up to some epsilon.
 *  The assertion succeeds if |a - b| < epsilon for all pairs of elements.
 *  Required operators: <, >, -
 * @param expected The expected container
 * @param actual The actual container
 * @param epsilon The epsilon
 * @param ... A user defined message
 */
#define ASSERT_APPROX_EQUAL_CONTAINERS_MSG(expected, actual, epsilon, ...) \
UNITTEST_ASSERT_APPROX_EQUAL_CONTAINERS_MSG(expected, actual, epsilon, __VA_ARGS__)
/**
 * @brief Asserts that two containers are approx. equal up to some epsilon.
 *  The assertion succeeds if |a - b| < epsilon for all pairs of elements (non-deadly assertion).
 *  Required operators: <, >, -
 * @param expected The expected container
 * @param actual The actual container
 * @param epsilon The epsilon
 */
#define NDASSERT_APPROX_EQUAL_CONTAINERS(expected, actual, epsilon) \
UNITTEST_NDASSERT_APPROX_EQUAL_CONTAINERS(expected, actual, epsilon)
/**
 * @brief Asserts that two containers are approx. equal up to some epsilon.
 *  The assertion succeeds if |a - b| < epsilon for all pairs of elements (non-deadly assertion).
 *  Required operators: <, >, -
 * @param expected The expected container
 * @param actual The actual container
 * @param epsilon The epsilon
 * @param ... A user defined message
 */
#define NDASSERT_APPROX_EQUAL_CONTAINERS_MSG(expected, actual, epsilon, ...) \
UNITTEST_NDASSERT_APPROX_EQUAL_CONTAINERS_MSG(expected, actual, epsilon, __VA_ARGS__)
/**
 * @brief Asserts that two containers are not approx. equal up to some
 *  epsilon. The assertion succeeds if |a - b| < epsilon is false for at
 *  least one pair of elements.
 *  Required operators: <, >, -
 * @param first A container
 * @param second Another container
 * @param epsilon The epsilon
 */
#define ASSERT_APPROX_NOT_EQUAL_CONTAINERS(first, second, epsilon) \
UNITTEST_ASSERT_APPROX_NOT_EQUAL_CONTAINERS(first, second, epsilon)
/**
 * @brief Asserts that two containers are not approx. equal up to some
 *  epsilon. The assertion succeeds if |a - b| < epsilon is false for at
 *  least one pair of elements.
 *  Required operators: <, >, -
 * @param first A container
 * @param second Another container
 * @param epsilon The epsilon
 * @param ... A user defined message
 */
#define ASSERT_APPROX_NOT_EQUAL_CONTAINERS_MSG(first, second, epsilon, ...) \
UNITTEST_ASSERT_APPROX_NOT_EQUAL_CONTAINERS_MSG(first, second, epsilon, __VA_ARGS__)
/**
 * @brief Asserts that two containers are not approx. equal up to some
 *  epsilon. The assertion succeeds if |a - b| < epsilon is false for at
 *  least one pair of elements (non-deadly assertion).
 *  Required operators: <, >, -
 * @param first A container
 * @param second Another container
 * @param epsilon The epsilon
 */
#define NDASSERT_APPROX_NOT_EQUAL_CONTAINERS(first, second, epsilon) \
UNITTEST_NDASSERT_APPROX_NOT_EQUAL_CONTAINERS(first, second, epsilon)
/**
 * @brief Asserts that two containers are not approx. equal up to some
 *  epsilon. The assertion succeeds if |a - b| < epsilon is false for at
 *  least one pair of elements (non-deadly assertion).
 *  Required operators: <, >, -
 * @param first A container
 * @param second Another container
 * @param epsilon The epsilon
 * @param ... A user defined message
 */
#define NDASSERT_APPROX_NOT_EQUAL_CONTAINERS_MSG(first, second, epsilon, ...) \
UNITTEST_NDASSERT_APPROX_NOT_EQUAL_CONTAINERS_MSG(first, second, epsilon, __VA_ARGS__)
/**
 * @brief Asserts that two containers are relatively approx. equal up to some
 *  epsilon. The assertion succeeds if |a - b| < |a| * epsilon for all pairs of
 *  elements.
 *  Required operators: <, >, -, *
 * @param expected The expected container
 * @param actual The actual container
 * @param epsilon The epsilon
 */
#define ASSERT_APPROXREL_EQUAL_CONTAINERS(expected, actual, epsilon) \
UNITTEST_ASSERT_APPROXREL_EQUAL_CONTAINERS(expected, actual, epsilon)
/**
 * @brief Asserts that two containers are relatively approx. equal up to some
 *  epsilon. The assertion succeeds if |a - b| < |a| * epsilon for all pairs of
 *  elements.
 *  Required operators: <, >, -, *
 * @param expected The expected container
 * @param actual The actual container
 * @param epsilon The epsilon
 * @param ... A user defined message
 */
#define ASSERT_APPROXREL_EQUAL_CONTAINERS_MSG(expected, actual, epsilon, ...) \
UNITTEST_ASSERT_APPROXREL_EQUAL_CONTAINERS_MSG(expected, actual, epsilon, __VA_ARGS__)
/**
 * @brief Asserts that two containers are relatively approx. equal up to some
 *  epsilon. The assertion succeeds if |a - b| < |a| * epsilon for all pairs of
 *  elements (non-deadly assertion).
 *  Required operators: <, >, -, *
 * @param expected The expected container
 * @param actual The actual container
 * @param epsilon The epsilon
 */
#define NDASSERT_APPROXREL_EQUAL_CONTAINERS(expected, actual, epsilon) \
UNITTEST_NDASSERT_APPROXREL_EQUAL_CONTAINERS(expected, actual, epsilon)
/**
 * @brief Asserts that two containers are relatively approx. equal up to some
 *  epsilon. The assertion succeeds if |a - b| < |a| * epsilon for all pairs of
 *  elements (non-deadly assertion).
 *  Required operators: <, >, -, *
 * @param expected The expected container
 * @param actual The actual container
 * @param epsilon The epsilon
 * @param ... A user defined message
 */
#define NDASSERT_APPROXREL_EQUAL_CONTAINERS_MSG(expected, actual, epsilon, ...) \
UNITTEST_NDASSERT_APPROXREL_EQUAL_CONTAINERS_MSG(expected, actual, epsilon, __VA_ARGS__)
/**
 * @brief Asserts that two containers are not relatively approx. equal up to
 *  some epsilon. The assertion succeeds if |a - b| < |a| * epsilon is false for
 *  at least one pair of elements.
 *  Required operators: <, >, -, *
 * @param first A container
 * @param second Another container
 * @param epsilon The epsilon
 */
#define ASSERT_APPROXREL_NOT_EQUAL_CONTAINERS(first, second, epsilon) \
UNITTEST_ASSERT_APPROXREL_NOT_EQUAL_CONTAINERS(first, second, epsilon)
/**
 * @brief Asserts that two containers are not relatively approx. equal up to
 *  some epsilon. The assertion succeeds if |a - b| < |a| * epsilon is false for
 *  at least one pair of elements.
 *  Required operators: <, >, -, *
 * @param first A container
 * @param second Another container
 * @param epsilon The epsilon
 * @param ... A user defined message
 */
#define ASSERT_APPROXREL_NOT_EQUAL_CONTAINERS_MSG(first, second, epsilon, ...) \
UNITTEST_ASSERT_APPROXREL_NOT_EQUAL_CONTAINERS_MSG(first, second, epsilon, __VA_ARGS__)
/**
 * @brief Asserts that two containers are not relatively approx. equal up to
 *  some epsilon. The assertion succeeds if |a - b| < |a| * epsilon is false for
 *  at least one pair of elements (non-deadly assertion).
 *  Required operators: <, >, -, *
 * @param first A container
 * @param second Another container
 * @param epsilon The epsilon
 */
#define NDASSERT_APPROXREL_NOT_EQUAL_CONTAINERS(first, second, epsilon) \
UNITTEST_NDASSERT_APPROXREL_NOT_EQUAL_CONTAINERS(first, second, epsilon)
/**
 * @brief Asserts that two containers are not relatively approx. equal up to
 *  some epsilon. The assertion succeeds if |a - b| < |a| * epsilon is false for
 *  at least one pair of elements (non-deadly assertion).
 *  Required operators: <, >, -, *
 * @param first A container
 * @param second Another container
 * @param epsilon The epsilon
 * @param ... A user defined message
 */
#define NDASSERT_APPROXREL_NOT_EQUAL_CONTAINERS_MSG(first, second, epsilon, ...) \
UNITTEST_NDASSERT_APPROXREL_NOT_EQUAL_CONTAINERS_MSG(first, second, epsilon, __VA_ARGS__)
/**
 * @brief Asserts that all container elements match a given condition.
 * @param container A container
 * @param condition A condition returning a boolean that can be applied to
 *  each container element
 */
#define ASSERT_ALL_OF(container, condition) \
UNITTEST_ASSERT_ALL_OF(container, condition)
/**
 * @brief Asserts that all container elements match a given condition.
 * @param container A container
 * @param condition A condition returning a boolean that can be applied to
 *  each container element
 * @param ... A user defined message
 */
#define ASSERT_ALL_OF_MSG(container, condition, ...) \
UNITTEST_ASSERT_ALL_OF_MSG(container, condition, __VA_ARGS__)
/**
 * @brief Asserts that all container elements match a given condition (non-deadly assertion).
 * @param container A container
 * @param condition A condition returning a boolean that can be applied to
 *  each container element
 */
#define NDASSERT_ALL_OF(container, condition) \
UNITTEST_NDASSERT_ALL_OF(container, condition)
/**
 * @brief Asserts that all container elements match a given condition (non-deadly assertion).
 * @param container A container
 * @param condition A condition returning a boolean that can be applied to
 *  each container element
 * @param ... A user defined message
 */
#define NDASSERT_ALL_OF_MSG(container, condition, ...) \
UNITTEST_NDASSERT_ALL_OF_MSG(container, condition, __VA_ARGS__)
/**
 * @brief Asserts that not all container elements match a given condition.
 * @param container A container
 * @param condition A condition returning a boolean that can be applied to
 *  each container element
 */
#define ASSERT_NOT_ALL_OF(container, condition) \
UNITTEST_ASSERT_NOT_ALL_OF(container, condition)
/**
 * @brief Asserts that not all container elements match a given condition.
 * @param container A container
 * @param condition A condition returning a boolean that can be applied to
 *  each container element
 * @param ... A user defined message
 */
#define ASSERT_NOT_ALL_OF_MSG(container, condition, ...) \
UNITTEST_ASSERT_NOT_ALL_OF_MSG(container, condition, __VA_ARGS__)
/**
 * @brief Asserts that not all container elements match a given condition (non-deadly assertion).
 * @param container A container
 * @param condition A condition returning a boolean that can be applied to
 *  each container element
 */
#define NDASSERT_NOT_ALL_OF(container, condition) \
UNITTEST_NDASSERT_NOT_ALL_OF(container, condition)
/**
 * @brief Asserts that not all container elements match a given condition (non-deadly assertion).
 * @param container A container
 * @param condition A condition returning a boolean that can be applied to
 *  each container element
 * @param ... A user defined message
 */
#define NDASSERT_NOT_ALL_OF_MSG(container, condition, ...) \
UNITTEST_NDASSERT_NOT_ALL_OF_MSG(container, condition, __VA_ARGS__)
/**
 * @brief Asserts that any container elements match a given condition.
 * @param container A container
 * @param condition A condition returning a boolean that can be applied to
 *  each container element
 */
#define ASSERT_ANY_OF(container, condition) \
UNITTEST_ASSERT_ANY_OF(container, condition)
/**
 * @brief Asserts that any container elements match a given condition.
 * @param container A container
 * @param condition A condition returning a boolean that can be applied to
 *  each container element
 * @param ... A user defined message
 */
#define ASSERT_ANY_OF_MSG(container, condition, ...) \
UNITTEST_ASSERT_ANY_OF_MSG(container, condition, __VA_ARGS__)
/**
 * @brief Asserts that any container elements match a given condition (non-deadly assertion).
 * @param container A container
 * @param condition A condition returning a boolean that can be applied to
 *  each container element
 */
#define NDASSERT_ANY_OF(container, condition) \
UNITTEST_NDASSERT_ANY_OF(container, condition)
/**
 * @brief Asserts that any container elements match a given condition (non-deadly assertion).
 * @param container A container
 * @param condition A condition returning a boolean that can be applied to
 *  each container element
 * @param ... A user defined message
 */
#define NDASSERT_ANY_OF_MSG(container, condition, ...) \
UNITTEST_NDASSERT_ANY_OF_MSG(container, condition, __VA_ARGS__)
/**
 * @brief Asserts that none of the container elements match a given condition.
 * @param container A container
 * @param condition A condition returning a boolean that can be applied to
 *  each container element
 */
#define ASSERT_NONE_OF(container, condition) \
UNITTEST_ASSERT_NONE_OF(container, condition)
/**
 * @brief Asserts that none of the container elements match a given condition.
 * @param container A container
 * @param condition A condition returning a boolean that can be applied to
 *  each container element
 * @param ... A user defined message
 */
#define ASSERT_NONE_OF_MSG(container, condition, ...) \
UNITTEST_ASSERT_NONE_OF_MSG(container, condition, __VA_ARGS__)
/**
 * @brief Asserts that none of the container elements match a given condition (non-deadly assertion).
 * @param container A container
 * @param condition A condition returning a boolean that can be applied to
 *  each container element
 */
#define NDASSERT_NONE_OF(container, condition) \
UNITTEST_NDASSERT_NONE_OF(container, condition)
/**
 * @brief Asserts that none of the container elements match a given condition (non-deadly assertion).
 * @param container A container
 * @param condition A condition returning a boolean that can be applied to
 *  each container element
 * @param ... A user defined message
 */
#define NDASSERT_NONE_OF_MSG(container, condition, ...) \
UNITTEST_NDASSERT_NONE_OF_MSG(container, condition, __VA_ARGS__)
/**
 * @brief Asserts that a string matches a regular expression.
 * @param string A string
 * @param regex The regular expression
 */
#define ASSERT_REGEX_MATCH(string, regex) \
UNITTEST_ASSERT_REGEX_MATCH(string, regex)
/**
 * @brief Asserts that a string matches a regular expression.
 * @param string A string
 * @param regex The regular expression
 * @param ... A user defined message
 */
#define ASSERT_REGEX_MATCH_MSG(string, regex, ...) \
UNITTEST_ASSERT_REGEX_MATCH_MSG(string, regex, __VA_ARGS__)
/**
 * @brief Asserts that a string matches a regular expression (non-deadly assertion).
 * @param string A string
 * @param regex The regular expression
 */
#define NDASSERT_REGEX_MATCH(string, regex) \
UNITTEST_NDASSERT_REGEX_MATCH(string, regex)
/**
 * @brief Asserts that a string matches a regular expression (non-deadly assertion).
 * @param string A string
 * @param regex The regular expression
 * @param ... A user defined message
 */
#define NDASSERT_REGEX_MATCH_MSG(string, regex, ...) \
UNITTEST_NDASSERT_REGEX_MATCH_MSG(string, regex, __VA_ARGS__)
/**
 * @brief Asserts that a string does not match a regular expression.
 * @param string A string
 * @param regex The regular expression
 */
#define ASSERT_NOT_REGEX_MATCH(string, regex) \
UNITTEST_ASSERT_NOT_REGEX_MATCH(string, regex)
/**
 * @brief Asserts that a string does not match a regular expression.
 * @param string A string
 * @param regex The regular expression
 * @param ... A user defined message
 */
#define ASSERT_NOT_REGEX_MATCH_MSG(string, regex, ...) \
UNITTEST_ASSERT_NOT_REGEX_MATCH_MSG(string, regex, __VA_ARGS__)
/**
 * @brief Asserts that a string does not match a regular expression (non-deadly assertion).
 * @param string A string
 * @param regex The regular expression
 */
#define NDASSERT_NOT_REGEX_MATCH(string, regex) \
UNITTEST_NDASSERT_NOT_REGEX_MATCH(string, regex)
/**
 * @brief Asserts that a string does not match a regular expression (non-deadly assertion).
 * @param string A string
 * @param regex The regular expression
 * @param ... A user defined message
 */
#define NDASSERT_NOT_REGEX_MATCH_MSG(string, regex, ...) \
UNITTEST_NDASSERT_NOT_REGEX_MATCH_MSG(string, regex, __VA_ARGS__)
/**
 * @brief Asserts that a functor throws exception.
 *  Required operators: ()
 * @param exception The exception type
 * @param functor The functor
 */
#define ASSERT_THROW(exception, functor) \
UNITTEST_ASSERT_THROW(exception, functor)
/**
 * @brief Asserts that a functor throws exception.
 *  Required operators: ()
 * @param exception The exception type
 * @param functor The functor
 * @param ... A user defined message
 */
#define ASSERT_THROW_MSG(exception, functor, ...) \
UNITTEST_ASSERT_THROW_MSG(exception, functor, __VA_ARGS__)
/**
 * @brief Asserts that a functor throws exception (non-deadly assertion).
 *  Required operators: ()
 * @param exception The exception type
 * @param functor The functor
 */
#define NDASSERT_THROW(exception, functor) \
UNITTEST_NDASSERT_THROW(exception, functor)
/**
 * @brief Asserts that a functor throws exception (non-deadly assertion).
 *  Required operators: ()
 * @param exception The exception type
 * @param functor The functor
 * @param ... A user defined message
 */
#define NDASSERT_THROW_MSG(exception, functor, ...) \
UNITTEST_NDASSERT_THROW_MSG(exception, functor, __VA_ARGS__)
/**
 * @brief Asserts that a functor does not throw any exception.
 *  Required operators: ()
 * @param functor The functor
 */
#define ASSERT_NO_THROW(functor) \
UNITTEST_ASSERT_NO_THROW(functor)
/**
 * @brief Asserts that a functor does not throw any exception.
 *  Required operators: ()
 * @param functor The functor
 * @param ... A user defined message
 */
#define ASSERT_NO_THROW_MSG(functor, ...) \
UNITTEST_ASSERT_NO_THROW_MSG(functor, __VA_ARGS__)
/**
 * @brief Asserts that a functor does not throw any exception (non-deadly assertion).
 *  Required operators: ()
 * @param functor The functor
 */
#define NDASSERT_NO_THROW(functor) \
UNITTEST_NDASSERT_NO_THROW(functor)
/**
 * @brief Asserts that a functor does not throw any exception (non-deadly assertion).
 *  Required operators: ()
 * @param functor The functor
 * @param ... A user defined message
 */
#define NDASSERT_NO_THROW_MSG(functor, ...) \
UNITTEST_NDASSERT_NO_THROW_MSG(functor, __VA_ARGS__)
