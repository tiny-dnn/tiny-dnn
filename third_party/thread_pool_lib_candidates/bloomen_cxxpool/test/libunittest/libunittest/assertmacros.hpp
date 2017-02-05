/**
 * @brief Assertion macros for further convenience
 * @file assertmacros.hpp
 */
#pragma once
/**
 * @brief Asserts that a value is true.
 * @param value A value
 */
#define UNITTEST_ASSERT_TRUE(value) \
unittest::assertions::assert_true(value, UNITTEST_SPOT, UNITTEST_CALL(value));
/**
 * @brief Asserts that a value is true.
 * @param value A value
 * @param ... A user defined message
 */
#define UNITTEST_ASSERT_TRUE_MSG(value, ...) \
unittest::assertions::assert_true(value, UNITTEST_SPOT, UNITTEST_CALL(value), static_cast<const std::ostringstream&>(std::ostringstream{} << __VA_ARGS__).str());
/**
 * @brief Asserts that a value is true (non-deadly assertion).
 * @param value A value
 */
#define UNITTEST_NDASSERT_TRUE(value) \
unittest::assertions::assert_true(value, UNITTEST_SPOT, UNITTEST_CALL(value), UNITTEST_NDAS);
/**
 * @brief Asserts that a value is true (non-deadly assertion).
 * @param value A value
 * @param ... A user defined message
 */
#define UNITTEST_NDASSERT_TRUE_MSG(value, ...) \
unittest::assertions::assert_true(value, UNITTEST_SPOT, UNITTEST_CALL(value), UNITTEST_NDAS, static_cast<const std::ostringstream&>(std::ostringstream{} << __VA_ARGS__).str());
/**
 * @brief Asserts that a value is false.
 * @param value A value
 */
#define UNITTEST_ASSERT_FALSE(value) \
unittest::assertions::assert_false(value, UNITTEST_SPOT, UNITTEST_CALL(value));
/**
 * @brief Asserts that a value is false.
 * @param value A value
 * @param ... A user defined message
 */
#define UNITTEST_ASSERT_FALSE_MSG(value, ...) \
unittest::assertions::assert_false(value, UNITTEST_SPOT, UNITTEST_CALL(value), static_cast<const std::ostringstream&>(std::ostringstream{} << __VA_ARGS__).str());
/**
 * @brief Asserts that a value is false (non-deadly assertion).
 * @param value A value
 */
#define UNITTEST_NDASSERT_FALSE(value) \
unittest::assertions::assert_false(value, UNITTEST_SPOT, UNITTEST_CALL(value), UNITTEST_NDAS);
/**
 * @brief Asserts that a value is false (non-deadly assertion).
 * @param value A value
 * @param ... A user defined message
 */
#define UNITTEST_NDASSERT_FALSE_MSG(value, ...) \
unittest::assertions::assert_false(value, UNITTEST_SPOT, UNITTEST_CALL(value), UNITTEST_NDAS, static_cast<const std::ostringstream&>(std::ostringstream{} << __VA_ARGS__).str());
/**
 * @brief Asserts that two values are equal.
 *  Required operators: ==, <<
 * @param expected The expected value
 * @param actual The actual value
 */
#define UNITTEST_ASSERT_EQUAL(expected, actual) \
unittest::assertions::assert_equal(expected, actual, UNITTEST_SPOT, UNITTEST_CALL(expected, actual));
/**
 * @brief Asserts that two values are equal.
 *  Required operators: ==, <<
 * @param expected The expected value
 * @param actual The actual value
 * @param ... A user defined message
 */
#define UNITTEST_ASSERT_EQUAL_MSG(expected, actual, ...) \
unittest::assertions::assert_equal(expected, actual, UNITTEST_SPOT, UNITTEST_CALL(expected, actual), static_cast<const std::ostringstream&>(std::ostringstream{} << __VA_ARGS__).str());
/**
 * @brief Asserts that two values are equal (non-deadly assertion).
 *  Required operators: ==, <<
 * @param expected The expected value
 * @param actual The actual value
 */
#define UNITTEST_NDASSERT_EQUAL(expected, actual) \
unittest::assertions::assert_equal(expected, actual, UNITTEST_SPOT, UNITTEST_CALL(expected, actual), UNITTEST_NDAS);
/**
 * @brief Asserts that two values are equal (non-deadly assertion).
 *  Required operators: ==, <<
 * @param expected The expected value
 * @param actual The actual value
 * @param ... A user defined message
 */
#define UNITTEST_NDASSERT_EQUAL_MSG(expected, actual, ...) \
unittest::assertions::assert_equal(expected, actual, UNITTEST_SPOT, UNITTEST_CALL(expected, actual), UNITTEST_NDAS, static_cast<const std::ostringstream&>(std::ostringstream{} << __VA_ARGS__).str());
/**
 * @brief Asserts that two values are not equal.
 *  Required operators: ==, <<
 * @param first The first value
 * @param second The second value
 */
#define UNITTEST_ASSERT_NOT_EQUAL(first, second) \
unittest::assertions::assert_not_equal(first, second, UNITTEST_SPOT, UNITTEST_CALL(first, second));
/**
 * @brief Asserts that two values are not equal.
 *  Required operators: ==, <<
 * @param first The first value
 * @param second The second value
 * @param ... A user defined message
 */
#define UNITTEST_ASSERT_NOT_EQUAL_MSG(first, second, ...) \
unittest::assertions::assert_not_equal(first, second, UNITTEST_SPOT, UNITTEST_CALL(first, second), static_cast<const std::ostringstream&>(std::ostringstream{} << __VA_ARGS__).str());
/**
 * @brief Asserts that two values are not equal (non-deadly assertion).
 *  Required operators: ==, <<
 * @param first The first value
 * @param second The second value
 */
#define UNITTEST_NDASSERT_NOT_EQUAL(first, second) \
unittest::assertions::assert_not_equal(first, second, UNITTEST_SPOT, UNITTEST_CALL(first, second), UNITTEST_NDAS);
/**
 * @brief Asserts that two values are not equal (non-deadly assertion).
 *  Required operators: ==, <<
 * @param first The first value
 * @param second The second value
 * @param ... A user defined message
 */
#define UNITTEST_NDASSERT_NOT_EQUAL_MSG(first, second, ...) \
unittest::assertions::assert_not_equal(first, second, UNITTEST_SPOT, UNITTEST_CALL(first, second), UNITTEST_NDAS, static_cast<const std::ostringstream&>(std::ostringstream{} << __VA_ARGS__).str());
/**
 * @brief Asserts that two values are approximately equal up to some
 *  epsilon. The assertion succeeds if |a - b| < epsilon.
 *  Required operators: <, >, -, <<
 * @param expected The expected value
 * @param actual The actual value
 * @param epsilon The epsilon
 */
#define UNITTEST_ASSERT_APPROX_EQUAL(expected, actual, epsilon) \
unittest::assertions::assert_approx_equal(expected, actual, epsilon, UNITTEST_SPOT, UNITTEST_CALL(expected, actual, epsilon));
/**
 * @brief Asserts that two values are approximately equal up to some
 *  epsilon. The assertion succeeds if |a - b| < epsilon.
 *  Required operators: <, >, -, <<
 * @param expected The expected value
 * @param actual The actual value
 * @param epsilon The epsilon
 * @param ... A user defined message
 */
#define UNITTEST_ASSERT_APPROX_EQUAL_MSG(expected, actual, epsilon, ...) \
unittest::assertions::assert_approx_equal(expected, actual, epsilon, UNITTEST_SPOT, UNITTEST_CALL(expected, actual, epsilon), static_cast<const std::ostringstream&>(std::ostringstream{} << __VA_ARGS__).str());
/**
 * @brief Asserts that two values are approximately equal up to some
 *  epsilon. The assertion succeeds if |a - b| < epsilon (non-deadly assertions).
 *  Required operators: <, >, -, <<
 * @param expected The expected value
 * @param actual The actual value
 * @param epsilon The epsilon
 */
#define UNITTEST_NDASSERT_APPROX_EQUAL(expected, actual, epsilon) \
unittest::assertions::assert_approx_equal(expected, actual, epsilon, UNITTEST_SPOT, UNITTEST_CALL(expected, actual, epsilon), UNITTEST_NDAS);
/**
 * @brief Asserts that two values are approximately equal up to some
 *  epsilon. The assertion succeeds if |a - b| < epsilon (non-deadly assertions).
 *  Required operators: <, >, -, <<
 * @param expected The expected value
 * @param actual The actual value
 * @param epsilon The epsilon
 * @param ... A user defined message
 */
#define UNITTEST_NDASSERT_APPROX_EQUAL_MSG(expected, actual, epsilon, ...) \
unittest::assertions::assert_approx_equal(expected, actual, epsilon, UNITTEST_SPOT, UNITTEST_CALL(expected, actual, epsilon), UNITTEST_NDAS, static_cast<const std::ostringstream&>(std::ostringstream{} << __VA_ARGS__).str());
/**
 * @brief Asserts that two values are not approximately equal up to some
 *  epsilon. The assertion succeeds if |a - b| < epsilon is false.
 *  Required operators: <, >, -, <<
 * @param first The first value
 * @param second The second value
 * @param epsilon The epsilon
 */
#define UNITTEST_ASSERT_APPROX_NOT_EQUAL(first, second, epsilon) \
unittest::assertions::assert_approx_equal(first, second, epsilon, UNITTEST_SPOT, UNITTEST_CALL(first, second, epsilon));
/**
 * @brief Asserts that two values are not approximately equal up to some
 *  epsilon. The assertion succeeds if |a - b| < epsilon is false.
 *  Required operators: <, >, -, <<
 * @param first The first value
 * @param second The second value
 * @param epsilon The epsilon
 * @param ... A user defined message
 */
#define UNITTEST_ASSERT_APPROX_NOT_EQUAL_MSG(first, second, epsilon, ...) \
unittest::assertions::assert_approx_not_equal(first, second, epsilon, UNITTEST_SPOT, UNITTEST_CALL(first, second, epsilon), static_cast<const std::ostringstream&>(std::ostringstream{} << __VA_ARGS__).str());
/**
 * @brief Asserts that two values are not approximately equal up to some
 *  epsilon. The assertion succeeds if |a - b| < epsilon is false
 *  (non-deadly assertion).
 *  Required operators: <, >, -, <<
 * @param first The first value
 * @param second The second value
 * @param epsilon The epsilon
 */
#define UNITTEST_NDASSERT_APPROX_NOT_EQUAL(first, second, epsilon) \
unittest::assertions::assert_approx_not_equal(first, second, epsilon, UNITTEST_SPOT, UNITTEST_CALL(first, second, epsilon), UNITTEST_NDAS);
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
#define UNITTEST_NDASSERT_APPROX_NOT_EQUAL_MSG(first, second, epsilon, ...) \
unittest::assertions::assert_approx_not_equal(first, second, epsilon, UNITTEST_SPOT, UNITTEST_CALL(first, second, epsilon), UNITTEST_NDAS, static_cast<const std::ostringstream&>(std::ostringstream{} << __VA_ARGS__).str());
/**
 * @brief Asserts that two values are relatively approx. equal up to some
 *  epsilon. The assertion succeeds if |a - b| < |a| * epsilon.
 *  Required operators: <, >, -, *, <<
 * @param expected The expected value
 * @param actual The actual value
 * @param epsilon The epsilon
 */
#define UNITTEST_ASSERT_APPROXREL_EQUAL(expected, actual, epsilon) \
unittest::assertions::assert_approxrel_equal(expected, actual, epsilon, UNITTEST_SPOT, UNITTEST_CALL(expected, actual, epsilon));
/**
 * @brief Asserts that two values are relatively approx. equal up to some
 *  epsilon. The assertion succeeds if |a - b| < |a| * epsilon.
 *  Required operators: <, >, -, *, <<
 * @param expected The expected value
 * @param actual The actual value
 * @param epsilon The epsilon
 * @param ... A user defined message
 */
#define UNITTEST_ASSERT_APPROXREL_EQUAL_MSG(expected, actual, epsilon, ...) \
unittest::assertions::assert_approxrel_equal(expected, actual, epsilon, UNITTEST_SPOT, UNITTEST_CALL(expected, actual, epsilon), static_cast<const std::ostringstream&>(std::ostringstream{} << __VA_ARGS__).str());
/**
 * @brief Asserts that two values are relatively approx. equal up to some
 *  epsilon. The assertion succeeds if |a - b| < |a| * epsilon
 *  (non-deadly assertion).
 *  Required operators: <, >, -, *, <<
 * @param expected The expected value
 * @param actual The actual value
 * @param epsilon The epsilon
 */
#define UNITTEST_NDASSERT_APPROXREL_EQUAL(expected, actual, epsilon) \
unittest::assertions::assert_approxrel_equal(expected, actual, epsilon, UNITTEST_SPOT, UNITTEST_CALL(expected, actual, epsilon), UNITTEST_NDAS);
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
#define UNITTEST_NDASSERT_APPROXREL_EQUAL_MSG(expected, actual, epsilon, ...) \
unittest::assertions::assert_approxrel_equal(expected, actual, epsilon, UNITTEST_SPOT, UNITTEST_CALL(expected, actual, epsilon), UNITTEST_NDAS, static_cast<const std::ostringstream&>(std::ostringstream{} << __VA_ARGS__).str());
/**
 * @brief Asserts that two values are not relatively approx. equal up to some
 *  epsilon. The assertion succeeds if |a - b| < |a| * epsilon is false
 *  Required operators: <, >, -, *, <<
 * @param first The first value
 * @param second The second value
 * @param epsilon The epsilon
 */
#define UNITTEST_ASSERT_APPROXREL_NOT_EQUAL(first, second, epsilon) \
unittest::assertions::assert_approxrel_not_equal(first, second, epsilon, UNITTEST_SPOT, UNITTEST_CALL(first, second, epsilon));
/**
 * @brief Asserts that two values are not relatively approx. equal up to some
 *  epsilon. The assertion succeeds if |a - b| < |a| * epsilon is false
 *  Required operators: <, >, -, *, <<
 * @param first The first value
 * @param second The second value
 * @param epsilon The epsilon
 * @param ... A user defined message
 */
#define UNITTEST_ASSERT_APPROXREL_NOT_EQUAL_MSG(first, second, epsilon, ...) \
unittest::assertions::assert_approxrel_not_equal(first, second, epsilon, UNITTEST_SPOT, UNITTEST_CALL(first, second, epsilon), static_cast<const std::ostringstream&>(std::ostringstream{} << __VA_ARGS__).str());
/**
 * @brief Asserts that two values are not relatively approx. equal up to some
 *  epsilon. The assertion succeeds if |a - b| < |a| * epsilon is false
 *  (non-deadly assertion).
 *  Required operators: <, >, -, *, <<
 * @param first The first value
 * @param second The second value
 * @param epsilon The epsilon
 */
#define UNITTEST_NDASSERT_APPROXREL_NOT_EQUAL(first, second, epsilon) \
unittest::assertions::assert_approxrel_not_equal(first, second, epsilon, UNITTEST_SPOT, UNITTEST_CALL(first, second, epsilon), UNITTEST_NDAS);
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
#define UNITTEST_NDASSERT_APPROXREL_NOT_EQUAL_MSG(first, second, epsilon, ...) \
unittest::assertions::assert_approxrel_not_equal(first, second, epsilon, UNITTEST_SPOT, UNITTEST_CALL(first, second, epsilon), UNITTEST_NDAS, static_cast<const std::ostringstream&>(std::ostringstream{} << __VA_ARGS__).str());
/**
 * @brief Asserts that the first value is greater than the second.
 *  Required operators: >, <<
 * @param first A value
 * @param second Another value
 */
#define UNITTEST_ASSERT_GREATER(first, second) \
unittest::assertions::assert_greater(first, second, UNITTEST_SPOT, UNITTEST_CALL(first, second));
/**
 * @brief Asserts that the first value is greater than the second.
 *  Required operators: >, <<
 * @param first A value
 * @param second Another value
 * @param ... A user defined message
 */
#define UNITTEST_ASSERT_GREATER_MSG(first, second, ...) \
unittest::assertions::assert_greater(first, second, UNITTEST_SPOT, UNITTEST_CALL(first, second), static_cast<const std::ostringstream&>(std::ostringstream{} << __VA_ARGS__).str());
/**
 * @brief Asserts that the first value is greater than the second
 *  (non-deadly assertion).
 *  Required operators: >, <<
 * @param first A value
 * @param second Another value
 */
#define UNITTEST_NDASSERT_GREATER(first, second) \
unittest::assertions::assert_greater(first, second, UNITTEST_SPOT, UNITTEST_CALL(first, second), UNITTEST_NDAS);
/**
 * @brief Asserts that the first value is greater than the second
 *  (non-deadly assertion).
 *  Required operators: >, <<
 * @param first A value
 * @param second Another value
 * @param ... A user defined message
 */
#define UNITTEST_NDASSERT_GREATER_MSG(first, second, ...) \
unittest::assertions::assert_greater(first, second, UNITTEST_SPOT, UNITTEST_CALL(first, second), UNITTEST_NDAS, static_cast<const std::ostringstream&>(std::ostringstream{} << __VA_ARGS__).str());
/**
 * @brief Asserts that the first value is greater than or equal to the second.
 *  Required operators: <, <<
 * @param first A value
 * @param second Another value
 */
#define UNITTEST_ASSERT_GREATER_EQUAL(first, second) \
unittest::assertions::assert_greater_equal(first, second, UNITTEST_SPOT, UNITTEST_CALL(first, second));
/**
 * @brief Asserts that the first value is greater than or equal to the second.
 *  Required operators: <, <<
 * @param first A value
 * @param second Another value
 * @param ... A user defined message
 */
#define UNITTEST_ASSERT_GREATER_EQUAL_MSG(first, second, ...) \
unittest::assertions::assert_greater_equal(first, second, UNITTEST_SPOT, UNITTEST_CALL(first, second), static_cast<const std::ostringstream&>(std::ostringstream{} << __VA_ARGS__).str());
/**
 * @brief Asserts that the first value is greater than or equal
 *  to the second (non-deadly assertion).
 *  Required operators: <, <<
 * @param first A value
 * @param second Another value
 */
#define UNITTEST_NDASSERT_GREATER_EQUAL(first, second) \
unittest::assertions::assert_greater_equal(first, second, UNITTEST_SPOT, UNITTEST_CALL(first, second), UNITTEST_NDAS);
/**
 * @brief Asserts that the first value is greater than or equal
 *  to the second (non-deadly assertion).
 *  Required operators: <, <<
 * @param first A value
 * @param second Another value
 * @param ... A user defined message
 */
#define UNITTEST_NDASSERT_GREATER_EQUAL_MSG(first, second, ...) \
unittest::assertions::assert_greater_equal(first, second, UNITTEST_SPOT, UNITTEST_CALL(first, second), UNITTEST_NDAS, static_cast<const std::ostringstream&>(std::ostringstream{} << __VA_ARGS__).str());
/**
 * @brief Asserts that the first value is lesser than the second.
 *  Required operators: <, <<
 * @param first A value
 * @param second Another value
 */
#define UNITTEST_ASSERT_LESSER(first, second) \
unittest::assertions::assert_lesser(first, second, UNITTEST_SPOT, UNITTEST_CALL(first, second));
/**
 * @brief Asserts that the first value is lesser than the second.
 *  Required operators: <, <<
 * @param first A value
 * @param second Another value
 * @param ... A user defined message
 */
#define UNITTEST_ASSERT_LESSER_MSG(first, second, ...) \
unittest::assertions::assert_lesser(first, second, UNITTEST_SPOT, UNITTEST_CALL(first, second), static_cast<const std::ostringstream&>(std::ostringstream{} << __VA_ARGS__).str());
/**
 * @brief Asserts that the first value is lesser than the second
 *  (non-deadly assertion).
 *  Required operators: <, <<
 * @param first A value
 * @param second Another value
 */
#define UNITTEST_NDASSERT_LESSER(first, second) \
unittest::assertions::assert_lesser(first, second, UNITTEST_SPOT, UNITTEST_CALL(first, second), UNITTEST_NDAS);
/**
 * @brief Asserts that the first value is lesser than the second
 *  (non-deadly assertion).
 *  Required operators: <, <<
 * @param first A value
 * @param second Another value
 * @param ... A user defined message
 */
#define UNITTEST_NDASSERT_LESSER_MSG(first, second, ...) \
unittest::assertions::assert_lesser(first, second, UNITTEST_SPOT, UNITTEST_CALL(first, second), UNITTEST_NDAS, static_cast<const std::ostringstream&>(std::ostringstream{} << __VA_ARGS__).str());
/**
 * @brief Asserts that the first value is lesser than or equal to the second.
 *  Required operators: >, <<
 * @param first A value
 * @param second Another value
 */
#define UNITTEST_ASSERT_LESSER_EQUAL(first, second) \
unittest::assertions::assert_lesser_equal(first, second, UNITTEST_SPOT, UNITTEST_CALL(first, second));
/**
 * @brief Asserts that the first value is lesser than or equal to the second.
 *  Required operators: >, <<
 * @param first A value
 * @param second Another value
 * @param ... A user defined message
 */
#define UNITTEST_ASSERT_LESSER_EQUAL_MSG(first, second, ...) \
unittest::assertions::assert_lesser_equal(first, second, UNITTEST_SPOT, UNITTEST_CALL(first, second), static_cast<const std::ostringstream&>(std::ostringstream{} << __VA_ARGS__).str());
/**
 * @brief Asserts that the first value is lesser than or equal
 *  to the second (non-deadly assertion).
 *  Required operators: >, <<
 * @param first A value
 * @param second Another value
 */
#define UNITTEST_NDASSERT_LESSER_EQUAL(first, second) \
unittest::assertions::assert_lesser_equal(first, second, UNITTEST_SPOT, UNITTEST_CALL(first, second), UNITTEST_NDAS);
/**
 * @brief Asserts that the first value is lesser than or equal
 *  to the second (non-deadly assertion).
 *  Required operators: >, <<
 * @param first A value
 * @param second Another value
 * @param ... A user defined message
 */
#define UNITTEST_NDASSERT_LESSER_EQUAL_MSG(first, second, ...) \
unittest::assertions::assert_lesser_equal(first, second, UNITTEST_SPOT, UNITTEST_CALL(first, second), UNITTEST_NDAS, static_cast<const std::ostringstream&>(std::ostringstream{} << __VA_ARGS__).str());
/**
 * @brief Asserts that a value is in a given range with included bounds.
 *  Required operators: <, >, <<
 * @param value A value
 * @param lower The lower bound
 * @param upper The upper bound
 */
#define UNITTEST_ASSERT_IN_RANGE(value, lower, upper) \
unittest::assertions::assert_in_range(value, lower, upper, UNITTEST_SPOT, UNITTEST_CALL(value, lower, upper));
/**
 * @brief Asserts that a value is in a given range with included bounds.
 *  Required operators: <, >, <<
 * @param value A value
 * @param lower The lower bound
 * @param upper The upper bound
 * @param ... A user defined message
 */
#define UNITTEST_ASSERT_IN_RANGE_MSG(value, lower, upper, ...) \
unittest::assertions::assert_in_range(value, lower, upper, UNITTEST_SPOT, UNITTEST_CALL(value, lower, upper), static_cast<const std::ostringstream&>(std::ostringstream{} << __VA_ARGS__).str());
/**
 * @brief Asserts that a value is in a given range with included bounds
 *  (non-deadly assertion).
 *  Required operators: <, >, <<
 * @param value A value
 * @param lower The lower bound
 * @param upper The upper bound
 */
#define UNITTEST_NDASSERT_IN_RANGE(value, lower, upper) \
unittest::assertions::assert_in_range(value, lower, upper, UNITTEST_SPOT, UNITTEST_CALL(value, lower, upper), UNITTEST_NDAS);
/**
 * @brief Asserts that a value is in a given range with included bounds
 *  (non-deadly assertion).
 *  Required operators: <, >, <<
 * @param value A value
 * @param lower The lower bound
 * @param upper The upper bound
 * @param ... A user defined message
 */
#define UNITTEST_NDASSERT_IN_RANGE_MSG(value, lower, upper, ...) \
unittest::assertions::assert_in_range(value, lower, upper, UNITTEST_SPOT, UNITTEST_CALL(value, lower, upper), UNITTEST_NDAS, static_cast<const std::ostringstream&>(std::ostringstream{} << __VA_ARGS__).str());
/**
 * @brief Asserts that a value is not in a given range with included bounds.
 *  Required operators: <, >, <<
 * @param value A value
 * @param lower The lower bound
 * @param upper The upper bound
 */
#define UNITTEST_ASSERT_NOT_IN_RANGE(value, lower, upper) \
unittest::assertions::assert_not_in_range(value, lower, upper, UNITTEST_SPOT, UNITTEST_CALL(value, lower, upper));
/**
 * @brief Asserts that a value is not in a given range with included bounds.
 *  Required operators: <, >, <<
 * @param value A value
 * @param lower The lower bound
 * @param upper The upper bound
 * @param ... A user defined message
 */
#define UNITTEST_ASSERT_NOT_IN_RANGE_MSG(value, lower, upper, ...) \
unittest::assertions::assert_not_in_range(value, lower, upper, UNITTEST_SPOT, UNITTEST_CALL(value, lower, upper), static_cast<const std::ostringstream&>(std::ostringstream{} << __VA_ARGS__).str());
/**
 * @brief Asserts that a value is not in a given range with included bounds
 *  (non-deadly assertion).
 *  Required operators: <, >, <<
 * @param value A value
 * @param lower The lower bound
 * @param upper The upper bound
 */
#define UNITTEST_NDASSERT_NOT_IN_RANGE(value, lower, upper) \
unittest::assertions::assert_not_in_range(value, lower, upper, UNITTEST_SPOT, UNITTEST_CALL(value, lower, upper), UNITTEST_NDAS);
/**
 * @brief Asserts that a value is not in a given range with included bounds
 *  (non-deadly assertion).
 *  Required operators: <, >, <<
 * @param value A value
 * @param lower The lower bound
 * @param upper The upper bound
 * @param ... A user defined message
 */
#define UNITTEST_NDASSERT_NOT_IN_RANGE_MSG(value, lower, upper, ...) \
unittest::assertions::assert_not_in_range(value, lower, upper, UNITTEST_SPOT, UNITTEST_CALL(value, lower, upper), UNITTEST_NDAS, static_cast<const std::ostringstream&>(std::ostringstream{} << __VA_ARGS__).str());
/**
 * @brief Asserts that a value is in a container.
 *  Required operators: ==
 * @param value A value
 * @param container A container
 */
#define UNITTEST_ASSERT_IN_CONTAINER(value, container) \
unittest::assertions::assert_in_container(value, container, UNITTEST_SPOT, UNITTEST_CALL(value, container));
/**
 * @brief Asserts that a value is in a container.
 *  Required operators: ==
 * @param value A value
 * @param container A container
 * @param ... A user defined message
 */
#define UNITTEST_ASSERT_IN_CONTAINER_MSG(value, container, ...) \
unittest::assertions::assert_in_container(value, container, UNITTEST_SPOT, UNITTEST_CALL(value, container), static_cast<const std::ostringstream&>(std::ostringstream{} << __VA_ARGS__).str());
/**
 * @brief Asserts that a value is in a container (non-deadly assertion).
 *  Required operators: ==
 * @param value A value
 * @param container A container
 */
#define UNITTEST_NDASSERT_IN_CONTAINER(value, container) \
unittest::assertions::assert_in_container(value, container, UNITTEST_SPOT, UNITTEST_CALL(value, container), UNITTEST_NDAS);
/**
 * @brief Asserts that a value is in a container (non-deadly assertion).
 *  Required operators: ==
 * @param value A value
 * @param container A container
 * @param ... A user defined message
 */
#define UNITTEST_NDASSERT_IN_CONTAINER_MSG(value, container, ...) \
unittest::assertions::assert_in_container(value, container, UNITTEST_SPOT, UNITTEST_CALL(value, container), UNITTEST_NDAS, static_cast<const std::ostringstream&>(std::ostringstream{} << __VA_ARGS__).str());
/**
 * @brief Asserts that a value is not in a container.
 *  Required operators: ==
 * @param value A value
 * @param container A container
 */
#define UNITTEST_ASSERT_NOT_IN_CONTAINER(value, container) \
unittest::assertions::assert_not_in_container(value, container, UNITTEST_SPOT, UNITTEST_CALL(value, container));
/**
 * @brief Asserts that a value is not in a container.
 *  Required operators: ==
 * @param value A value
 * @param container A container
 * @param ... A user defined message
 */
#define UNITTEST_ASSERT_NOT_IN_CONTAINER_MSG(value, container, ...) \
unittest::assertions::assert_not_in_container(value, container, UNITTEST_SPOT, UNITTEST_CALL(value, container), static_cast<const std::ostringstream&>(std::ostringstream{} << __VA_ARGS__).str());
/**
 * @brief Asserts that a value is not in a container (non-deadly assertion).
 *  Required operators: ==
 * @param value A value
 * @param container A container
 */
#define UNITTEST_NDASSERT_NOT_IN_CONTAINER(value, container) \
unittest::assertions::assert_not_in_container(value, container, UNITTEST_SPOT, UNITTEST_CALL(value, container), UNITTEST_NDAS);
/**
 * @brief Asserts that a value is not in a container (non-deadly assertion).
 *  Required operators: ==
 * @param value A value
 * @param container A container
 * @param ... A user defined message
 */
#define UNITTEST_NDASSERT_NOT_IN_CONTAINER_MSG(value, container, ...) \
unittest::assertions::assert_not_in_container(value, container, UNITTEST_SPOT, UNITTEST_CALL(value, container), UNITTEST_NDAS, static_cast<const std::ostringstream&>(std::ostringstream{} << __VA_ARGS__).str());
/**
 * @brief Asserts that a value is approx. in a container up to some epsilon.
 *  The assertion succeeds if |a - b| < epsilon for at least one element.
 *  Required operators: <, >, -
 * @param value A value
 * @param container A container
 * @param epsilon The epsilon
 */
#define UNITTEST_ASSERT_APPROX_IN_CONTAINER(value, container, epsilon) \
unittest::assertions::assert_approx_in_container(value, container, epsilon, UNITTEST_SPOT, UNITTEST_CALL(value, container, epsilon));
/**
 * @brief Asserts that a value is approx. in a container up to some epsilon.
 *  The assertion succeeds if |a - b| < epsilon for at least one element.
 *  Required operators: <, >, -
 * @param value A value
 * @param container A container
 * @param epsilon The epsilon
 * @param ... A user defined message
 */
#define UNITTEST_ASSERT_APPROX_IN_CONTAINER_MSG(value, container, epsilon, ...) \
unittest::assertions::assert_approx_in_container(value, container, epsilon, UNITTEST_SPOT, UNITTEST_CALL(value, container, epsilon), static_cast<const std::ostringstream&>(std::ostringstream{} << __VA_ARGS__).str());
/**
 * @brief Asserts that a value is approx. in a container up to some epsilon.
 *  The assertion succeeds if |a - b| < epsilon for at least one element (non-deadly assertion).
 *  Required operators: <, >, -
 * @param value A value
 * @param container A container
 * @param epsilon The epsilon
 */
#define UNITTEST_NDASSERT_APPROX_IN_CONTAINER(value, container, epsilon) \
unittest::assertions::assert_approx_in_container(value, container, epsilon, UNITTEST_SPOT, UNITTEST_CALL(value, container, epsilon), UNITTEST_NDAS);
/**
 * @brief Asserts that a value is approx. in a container up to some epsilon.
 *  The assertion succeeds if |a - b| < epsilon for at least one element (non-deadly assertion).
 *  Required operators: <, >, -
 * @param value A value
 * @param container A container
 * @param epsilon The epsilon
 * @param ... A user defined message
 */
#define UNITTEST_NDASSERT_APPROX_IN_CONTAINER_MSG(value, container, epsilon, ...) \
unittest::assertions::assert_approx_in_container(value, container, epsilon, UNITTEST_SPOT, UNITTEST_CALL(value, container, epsilon), UNITTEST_NDAS, static_cast<const std::ostringstream&>(std::ostringstream{} << __VA_ARGS__).str());
/**
 * @brief Asserts that a value is not approx. in a container up to some epsilon.
 *  The assertion succeeds if |a - b| < epsilon is false for all elements.
 *  Required operators: <, >, -
 * @param value A value
 * @param container A container
 * @param epsilon The epsilon
 */
#define UNITTEST_ASSERT_APPROX_NOT_IN_CONTAINER(value, container, epsilon) \
unittest::assertions::assert_approx_not_in_container(value, container, epsilon, UNITTEST_SPOT, UNITTEST_CALL(value, container, epsilon));
/**
 * @brief Asserts that a value is not approx. in a container up to some epsilon.
 *  The assertion succeeds if |a - b| < epsilon is false for all elements.
 *  Required operators: <, >, -
 * @param value A value
 * @param container A container
 * @param epsilon The epsilon
 * @param ... A user defined message
 */
#define UNITTEST_ASSERT_APPROX_NOT_IN_CONTAINER_MSG(value, container, epsilon, ...) \
unittest::assertions::assert_approx_not_in_container(value, container, epsilon, UNITTEST_SPOT, UNITTEST_CALL(value, container, epsilon), static_cast<const std::ostringstream&>(std::ostringstream{} << __VA_ARGS__).str());
/**
 * @brief Asserts that a value is not approx. in a container up to some epsilon.
 *  The assertion succeeds if |a - b| < epsilon is false for all elements (non-deadly assertion).
 *  Required operators: <, >, -
 * @param value A value
 * @param container A container
 * @param epsilon The epsilon
 */
#define UNITTEST_NDASSERT_APPROX_NOT_IN_CONTAINER(value, container, epsilon) \
unittest::assertions::assert_approx_not_in_container(value, container, epsilon, UNITTEST_SPOT, UNITTEST_CALL(value, container, epsilon), UNITTEST_NDAS);
/**
 * @brief Asserts that a value is not approx. in a container up to some epsilon.
 *  The assertion succeeds if |a - b| < epsilon is false for all elements (non-deadly assertion).
 *  Required operators: <, >, -
 * @param value A value
 * @param container A container
 * @param epsilon The epsilon
 * @param ... A user defined message
 */
#define UNITTEST_NDASSERT_APPROX_NOT_IN_CONTAINER_MSG(value, container, epsilon, ...) \
unittest::assertions::assert_approx_not_in_container(value, container, epsilon, UNITTEST_SPOT, UNITTEST_CALL(value, container, epsilon), UNITTEST_NDAS, static_cast<const std::ostringstream&>(std::ostringstream{} << __VA_ARGS__).str());
/**
 * @brief Asserts that a value is relatively approx. in a container up to some
 *  epsilon. The assertion succeeds if |a - b| < |a| * epsilon for at least one
 *  element. Required operators: <, >, -, *
 * @param value A value
 * @param container A container
 * @param epsilon The epsilon
 */
#define UNITTEST_ASSERT_APPROXREL_IN_CONTAINER(value, container, epsilon) \
unittest::assertions::assert_approxrel_in_container(value, container, epsilon, UNITTEST_SPOT, UNITTEST_CALL(value, container, epsilon));
/**
 * @brief Asserts that a value is relatively approx. in a container up to some
 *  epsilon. The assertion succeeds if |a - b| < |a| * epsilon for at least one
 *  element. Required operators: <, >, -, *
 * @param value A value
 * @param container A container
 * @param epsilon The epsilon
 * @param ... A user defined message
 */
#define UNITTEST_ASSERT_APPROXREL_IN_CONTAINER_MSG(value, container, epsilon, ...) \
unittest::assertions::assert_approxrel_in_container(value, container, epsilon, UNITTEST_SPOT, UNITTEST_CALL(value, container, epsilon), static_cast<const std::ostringstream&>(std::ostringstream{} << __VA_ARGS__).str());
/**
 * @brief Asserts that a value is relatively approx. in a container up to some
 *  epsilon. The assertion succeeds if |a - b| < |a| * epsilon for at least one
 *  element (non-deadly assertion). Required operators: <, >, -, *
 * @param value A value
 * @param container A container
 * @param epsilon The epsilon
 */
#define UNITTEST_NDASSERT_APPROXREL_IN_CONTAINER(value, container, epsilon) \
unittest::assertions::assert_approxrel_in_container(value, container, epsilon, UNITTEST_SPOT, UNITTEST_CALL(value, container, epsilon), UNITTEST_NDAS);
/**
 * @brief Asserts that a value is relatively approx. in a container up to some
 *  epsilon. The assertion succeeds if |a - b| < |a| * epsilon for at least one
 *  element (non-deadly assertion). Required operators: <, >, -, *
 * @param value A value
 * @param container A container
 * @param epsilon The epsilon
 * @param ... A user defined message
 */
#define UNITTEST_NDASSERT_APPROXREL_IN_CONTAINER_MSG(value, container, epsilon, ...) \
unittest::assertions::assert_approxrel_in_container(value, container, epsilon, UNITTEST_SPOT, UNITTEST_CALL(value, container, epsilon), UNITTEST_NDAS, static_cast<const std::ostringstream&>(std::ostringstream{} << __VA_ARGS__).str());
/**
 * @brief Asserts that a value is not relatively approx. in a container up to
 *  some epsilon. The assertion succeeds if |a - b| < |a| * epsilon is false for
 *  all elements. Required operators: <, >, -, *
 * @param value A value
 * @param container A container
 * @param epsilon The epsilon
 */
#define UNITTEST_ASSERT_APPROXREL_NOT_IN_CONTAINER(value, container, epsilon) \
unittest::assertions::assert_approxrel_not_in_container(value, container, epsilon, UNITTEST_SPOT, UNITTEST_CALL(value, container, epsilon));
/**
 * @brief Asserts that a value is not relatively approx. in a container up to
 *  some epsilon. The assertion succeeds if |a - b| < |a| * epsilon is false for
 *  all elements. Required operators: <, >, -, *
 * @param value A value
 * @param container A container
 * @param epsilon The epsilon
 * @param ... A user defined message
 */
#define UNITTEST_ASSERT_APPROXREL_NOT_IN_CONTAINER_MSG(value, container, epsilon, ...) \
unittest::assertions::assert_approxrel_not_in_container(value, container, epsilon, UNITTEST_SPOT, UNITTEST_CALL(value, container, epsilon), static_cast<const std::ostringstream&>(std::ostringstream{} << __VA_ARGS__).str());
/**
 * @brief Asserts that a value is not relatively approx. in a container up to
 *  some epsilon. The assertion succeeds if |a - b| < |a| * epsilon is false for
 *  all elements (non-deadly assertion). Required operators: <, >, -, *
 * @param value A value
 * @param container A container
 * @param epsilon The epsilon
 */
#define UNITTEST_NDASSERT_APPROXREL_NOT_IN_CONTAINER(value, container, epsilon) \
unittest::assertions::assert_approxrel_not_in_container(value, container, epsilon, UNITTEST_SPOT, UNITTEST_CALL(value, container, epsilon), UNITTEST_NDAS);
/**
 * @brief Asserts that a value is not relatively approx. in a container up to
 *  some epsilon. The assertion succeeds if |a - b| < |a| * epsilon is false for
 *  all elements (non-deadly assertion). Required operators: <, >, -, *
 * @param value A value
 * @param container A container
 * @param epsilon The epsilon
 * @param ... A user defined message
 */
#define UNITTEST_NDASSERT_APPROXREL_NOT_IN_CONTAINER_MSG(value, container, epsilon, ...) \
unittest::assertions::assert_approxrel_not_in_container(value, container, epsilon, UNITTEST_SPOT, UNITTEST_CALL(value, container, epsilon), UNITTEST_NDAS, static_cast<const std::ostringstream&>(std::ostringstream{} << __VA_ARGS__).str());
/**
 * @brief Asserts that two containers are equal.
 *  Required operators: ==
 * @param expected The expected container
 * @param actual The actual container
 */
#define UNITTEST_ASSERT_EQUAL_CONTAINERS(expected, actual) \
unittest::assertions::assert_equal_containers(expected, actual, UNITTEST_SPOT, UNITTEST_CALL(expected, actual));
/**
 * @brief Asserts that two containers are equal.
 *  Required operators: ==
 * @param expected The expected container
 * @param actual The actual container
 * @param ... A user defined message
 */
#define UNITTEST_ASSERT_EQUAL_CONTAINERS_MSG(expected, actual, ...) \
unittest::assertions::assert_equal_containers(expected, actual, UNITTEST_SPOT, UNITTEST_CALL(expected, actual), static_cast<const std::ostringstream&>(std::ostringstream{} << __VA_ARGS__).str());
/**
 * @brief Asserts that two containers are equal (non-deadly assertion).
 *  Required operators: ==
 * @param expected The expected container
 * @param actual The actual container
 */
#define UNITTEST_NDASSERT_EQUAL_CONTAINERS(expected, actual) \
unittest::assertions::assert_equal_containers(expected, actual, UNITTEST_SPOT, UNITTEST_CALL(expected, actual), UNITTEST_NDAS);
/**
 * @brief Asserts that two containers are equal (non-deadly assertion).
 *  Required operators: ==
 * @param expected The expected container
 * @param actual The actual container
 * @param ... A user defined message
 */
#define UNITTEST_NDASSERT_EQUAL_CONTAINERS_MSG(expected, actual, ...) \
unittest::assertions::assert_equal_containers(expected, actual, UNITTEST_SPOT, UNITTEST_CALL(expected, actual), UNITTEST_NDAS, static_cast<const std::ostringstream&>(std::ostringstream{} << __VA_ARGS__).str());
/**
 * @brief Asserts that two containers are not equal.
 *  Required operators: ==
 * @param first A container
 * @param second Another container
 */
#define UNITTEST_ASSERT_NOT_EQUAL_CONTAINERS(first, second) \
unittest::assertions::assert_not_equal_containers(first, second, UNITTEST_SPOT, UNITTEST_CALL(first, second));
/**
 * @brief Asserts that two containers are not equal.
 *  Required operators: ==
 * @param first A container
 * @param second Another container
 * @param ... A user defined message
 */
#define UNITTEST_ASSERT_NOT_EQUAL_CONTAINERS_MSG(first, second, ...) \
unittest::assertions::assert_not_equal_containers(first, second, UNITTEST_SPOT, UNITTEST_CALL(first, second), static_cast<const std::ostringstream&>(std::ostringstream{} << __VA_ARGS__).str());
/**
 * @brief Asserts that two containers are not equal (non-deadly assertion).
 *  Required operators: ==
 * @param first A container
 * @param second Another container
 */
#define UNITTEST_NDASSERT_NOT_EQUAL_CONTAINERS(first, second) \
unittest::assertions::assert_not_equal_containers(first, second, UNITTEST_SPOT, UNITTEST_CALL(first, second), UNITTEST_NDAS);
/**
 * @brief Asserts that two containers are not equal (non-deadly assertion).
 *  Required operators: ==
 * @param first A container
 * @param second Another container
 * @param ... A user defined message
 */
#define UNITTEST_NDASSERT_NOT_EQUAL_CONTAINERS_MSG(first, second, ...) \
unittest::assertions::assert_not_equal_containers(first, second, UNITTEST_SPOT, UNITTEST_CALL(first, second), UNITTEST_NDAS, static_cast<const std::ostringstream&>(std::ostringstream{} << __VA_ARGS__).str());
/**
 * @brief Asserts that two containers are approx. equal up to some epsilon.
 *  The assertion succeeds if |a - b| < epsilon for all pairs of elements.
 *  Required operators: <, >, -
 * @param expected The expected container
 * @param actual The actual container
 * @param epsilon The epsilon
 */
#define UNITTEST_ASSERT_APPROX_EQUAL_CONTAINERS(expected, actual, epsilon) \
unittest::assertions::assert_approx_equal_containers(expected, actual, epsilon, UNITTEST_SPOT, UNITTEST_CALL(expected, actual, epsilon));
/**
 * @brief Asserts that two containers are approx. equal up to some epsilon.
 *  The assertion succeeds if |a - b| < epsilon for all pairs of elements.
 *  Required operators: <, >, -
 * @param expected The expected container
 * @param actual The actual container
 * @param epsilon The epsilon
 * @param ... A user defined message
 */
#define UNITTEST_ASSERT_APPROX_EQUAL_CONTAINERS_MSG(expected, actual, epsilon, ...) \
unittest::assertions::assert_approx_equal_containers(expected, actual, epsilon, UNITTEST_SPOT, UNITTEST_CALL(expected, actual, epsilon), static_cast<const std::ostringstream&>(std::ostringstream{} << __VA_ARGS__).str());
/**
 * @brief Asserts that two containers are approx. equal up to some epsilon.
 *  The assertion succeeds if |a - b| < epsilon for all pairs of elements (non-deadly assertion).
 *  Required operators: <, >, -
 * @param expected The expected container
 * @param actual The actual container
 * @param epsilon The epsilon
 */
#define UNITTEST_NDASSERT_APPROX_EQUAL_CONTAINERS(expected, actual, epsilon) \
unittest::assertions::assert_approx_equal_containers(expected, actual, epsilon, UNITTEST_SPOT, UNITTEST_CALL(expected, actual, epsilon), UNITTEST_NDAS);
/**
 * @brief Asserts that two containers are approx. equal up to some epsilon.
 *  The assertion succeeds if |a - b| < epsilon for all pairs of elements (non-deadly assertion).
 *  Required operators: <, >, -
 * @param expected The expected container
 * @param actual The actual container
 * @param epsilon The epsilon
 * @param ... A user defined message
 */
#define UNITTEST_NDASSERT_APPROX_EQUAL_CONTAINERS_MSG(expected, actual, epsilon, ...) \
unittest::assertions::assert_approx_equal_containers(expected, actual, epsilon, UNITTEST_SPOT, UNITTEST_CALL(expected, actual, epsilon), UNITTEST_NDAS, static_cast<const std::ostringstream&>(std::ostringstream{} << __VA_ARGS__).str());
/**
 * @brief Asserts that two containers are not approx. equal up to some
 *  epsilon. The assertion succeeds if |a - b| < epsilon is false for at
 *  least one pair of elements.
 *  Required operators: <, >, -
 * @param first A container
 * @param second Another container
 * @param epsilon The epsilon
 */
#define UNITTEST_ASSERT_APPROX_NOT_EQUAL_CONTAINERS(first, second, epsilon) \
unittest::assertions::assert_approx_not_equal_containers(first, second, epsilon, UNITTEST_SPOT, UNITTEST_CALL(first, second, epsilon));
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
#define UNITTEST_ASSERT_APPROX_NOT_EQUAL_CONTAINERS_MSG(first, second, epsilon, ...) \
unittest::assertions::assert_approx_not_equal_containers(first, second, epsilon, UNITTEST_SPOT, UNITTEST_CALL(first, second, epsilon), static_cast<const std::ostringstream&>(std::ostringstream{} << __VA_ARGS__).str());
/**
 * @brief Asserts that two containers are not approx. equal up to some
 *  epsilon. The assertion succeeds if |a - b| < epsilon is false for at
 *  least one pair of elements (non-deadly assertion).
 *  Required operators: <, >, -
 * @param first A container
 * @param second Another container
 * @param epsilon The epsilon
 */
#define UNITTEST_NDASSERT_APPROX_NOT_EQUAL_CONTAINERS(first, second, epsilon) \
unittest::assertions::assert_approx_not_equal_containers(first, second, epsilon, UNITTEST_SPOT, UNITTEST_CALL(first, second, epsilon), UNITTEST_NDAS);
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
#define UNITTEST_NDASSERT_APPROX_NOT_EQUAL_CONTAINERS_MSG(first, second, epsilon, ...) \
unittest::assertions::assert_approx_not_equal_containers(first, second, epsilon, UNITTEST_SPOT, UNITTEST_CALL(first, second, epsilon), UNITTEST_NDAS, static_cast<const std::ostringstream&>(std::ostringstream{} << __VA_ARGS__).str());
/**
 * @brief Asserts that two containers are relatively approx. equal up to some
 *  epsilon. The assertion succeeds if |a - b| < |a| * epsilon for all pairs of
 *  elements.
 *  Required operators: <, >, -, *
 * @param expected The expected container
 * @param actual The actual container
 * @param epsilon The epsilon
 */
#define UNITTEST_ASSERT_APPROXREL_EQUAL_CONTAINERS(expected, actual, epsilon) \
unittest::assertions::assert_approxrel_equal_containers(expected, actual, epsilon, UNITTEST_SPOT, UNITTEST_CALL(expected, actual, epsilon));
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
#define UNITTEST_ASSERT_APPROXREL_EQUAL_CONTAINERS_MSG(expected, actual, epsilon, ...) \
unittest::assertions::assert_approxrel_equal_containers(expected, actual, epsilon, UNITTEST_SPOT, UNITTEST_CALL(expected, actual, epsilon), static_cast<const std::ostringstream&>(std::ostringstream{} << __VA_ARGS__).str());
/**
 * @brief Asserts that two containers are relatively approx. equal up to some
 *  epsilon. The assertion succeeds if |a - b| < |a| * epsilon for all pairs of
 *  elements (non-deadly assertion).
 *  Required operators: <, >, -, *
 * @param expected The expected container
 * @param actual The actual container
 * @param epsilon The epsilon
 */
#define UNITTEST_NDASSERT_APPROXREL_EQUAL_CONTAINERS(expected, actual, epsilon) \
unittest::assertions::assert_approxrel_equal_containers(expected, actual, epsilon, UNITTEST_SPOT, UNITTEST_CALL(expected, actual, epsilon), UNITTEST_NDAS);
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
#define UNITTEST_NDASSERT_APPROXREL_EQUAL_CONTAINERS_MSG(expected, actual, epsilon, ...) \
unittest::assertions::assert_approxrel_equal_containers(expected, actual, epsilon, UNITTEST_SPOT, UNITTEST_CALL(expected, actual, epsilon), UNITTEST_NDAS, static_cast<const std::ostringstream&>(std::ostringstream{} << __VA_ARGS__).str());
/**
 * @brief Asserts that two containers are not relatively approx. equal up to
 *  some epsilon. The assertion succeeds if |a - b| < |a| * epsilon is false for
 *  at least one pair of elements.
 *  Required operators: <, >, -, *
 * @param first A container
 * @param second Another container
 * @param epsilon The epsilon
 */
#define UNITTEST_ASSERT_APPROXREL_NOT_EQUAL_CONTAINERS(first, second, epsilon) \
unittest::assertions::assert_approxrel_not_equal_containers(first, second, epsilon, UNITTEST_SPOT, UNITTEST_CALL(first, second, epsilon));
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
#define UNITTEST_ASSERT_APPROXREL_NOT_EQUAL_CONTAINERS_MSG(first, second, epsilon, ...) \
unittest::assertions::assert_approxrel_not_equal_containers(first, second, epsilon, UNITTEST_SPOT, UNITTEST_CALL(first, second, epsilon), static_cast<const std::ostringstream&>(std::ostringstream{} << __VA_ARGS__).str());
/**
 * @brief Asserts that two containers are not relatively approx. equal up to
 *  some epsilon. The assertion succeeds if |a - b| < |a| * epsilon is false for
 *  at least one pair of elements (non-deadly assertion).
 *  Required operators: <, >, -, *
 * @param first A container
 * @param second Another container
 * @param epsilon The epsilon
 */
#define UNITTEST_NDASSERT_APPROXREL_NOT_EQUAL_CONTAINERS(first, second, epsilon) \
unittest::assertions::assert_approxrel_not_equal_containers(first, second, epsilon, UNITTEST_SPOT, UNITTEST_CALL(first, second, epsilon), UNITTEST_NDAS);
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
#define UNITTEST_NDASSERT_APPROXREL_NOT_EQUAL_CONTAINERS_MSG(first, second, epsilon, ...) \
unittest::assertions::assert_approxrel_not_equal_containers(first, second, epsilon, UNITTEST_SPOT, UNITTEST_CALL(first, second, epsilon), UNITTEST_NDAS, static_cast<const std::ostringstream&>(std::ostringstream{} << __VA_ARGS__).str());
/**
 * @brief Asserts that all container elements match a given condition.
 * @param container A container
 * @param condition A condition returning a boolean that can be applied to
 *  each container element
 */
#define UNITTEST_ASSERT_ALL_OF(container, condition) \
unittest::assertions::assert_all_of(container, condition, UNITTEST_SPOT, UNITTEST_CALL(container, condition));
/**
 * @brief Asserts that all container elements match a given condition.
 * @param container A container
 * @param condition A condition returning a boolean that can be applied to
 *  each container element
 * @param ... A user defined message
 */
#define UNITTEST_ASSERT_ALL_OF_MSG(container, condition, ...) \
unittest::assertions::assert_all_of(container, condition, UNITTEST_SPOT, UNITTEST_CALL(container, condition), static_cast<const std::ostringstream&>(std::ostringstream{} << __VA_ARGS__).str());
/**
 * @brief Asserts that all container elements match a given condition (non-deadly assertion).
 * @param container A container
 * @param condition A condition returning a boolean that can be applied to
 *  each container element
 */
#define UNITTEST_NDASSERT_ALL_OF(container, condition) \
unittest::assertions::assert_all_of(container, condition, UNITTEST_SPOT, UNITTEST_CALL(container, condition), UNITTEST_NDAS);
/**
 * @brief Asserts that all container elements match a given condition (non-deadly assertion).
 * @param container A container
 * @param condition A condition returning a boolean that can be applied to
 *  each container element
 * @param ... A user defined message
 */
#define UNITTEST_NDASSERT_ALL_OF_MSG(container, condition, ...) \
unittest::assertions::assert_all_of(container, condition, UNITTEST_SPOT, UNITTEST_CALL(container, condition), UNITTEST_NDAS, static_cast<const std::ostringstream&>(std::ostringstream{} << __VA_ARGS__).str());
/**
 * @brief Asserts that not all container elements match a given condition.
 * @param container A container
 * @param condition A condition returning a boolean that can be applied to
 *  each container element
 */
#define UNITTEST_ASSERT_NOT_ALL_OF(container, condition) \
unittest::assertions::assert_not_all_of(container, condition, UNITTEST_SPOT, UNITTEST_CALL(container, condition));
/**
 * @brief Asserts that not all container elements match a given condition.
 * @param container A container
 * @param condition A condition returning a boolean that can be applied to
 *  each container element
 * @param ... A user defined message
 */
#define UNITTEST_ASSERT_NOT_ALL_OF_MSG(container, condition, ...) \
unittest::assertions::assert_not_all_of(container, condition, UNITTEST_SPOT, UNITTEST_CALL(container, condition), static_cast<const std::ostringstream&>(std::ostringstream{} << __VA_ARGS__).str());
/**
 * @brief Asserts that not all container elements match a given condition (non-deadly assertion).
 * @param container A container
 * @param condition A condition returning a boolean that can be applied to
 *  each container element
 */
#define UNITTEST_NDASSERT_NOT_ALL_OF(container, condition) \
unittest::assertions::assert_not_all_of(container, condition, UNITTEST_SPOT, UNITTEST_CALL(container, condition), UNITTEST_NDAS);
/**
 * @brief Asserts that not all container elements match a given condition (non-deadly assertion).
 * @param container A container
 * @param condition A condition returning a boolean that can be applied to
 *  each container element
 * @param ... A user defined message
 */
#define UNITTEST_NDASSERT_NOT_ALL_OF_MSG(container, condition, ...) \
unittest::assertions::assert_not_all_of(container, condition, UNITTEST_SPOT, UNITTEST_CALL(container, condition), UNITTEST_NDAS, static_cast<const std::ostringstream&>(std::ostringstream{} << __VA_ARGS__).str());
/**
 * @brief Asserts that any container elements match a given condition.
 * @param container A container
 * @param condition A condition returning a boolean that can be applied to
 *  each container element
 */
#define UNITTEST_ASSERT_ANY_OF(container, condition) \
unittest::assertions::assert_any_of(container, condition, UNITTEST_SPOT, UNITTEST_CALL(container, condition));
/**
 * @brief Asserts that any container elements match a given condition.
 * @param container A container
 * @param condition A condition returning a boolean that can be applied to
 *  each container element
 * @param ... A user defined message
 */
#define UNITTEST_ASSERT_ANY_OF_MSG(container, condition, ...) \
unittest::assertions::assert_any_of(container, condition, UNITTEST_SPOT, UNITTEST_CALL(container, condition), static_cast<const std::ostringstream&>(std::ostringstream{} << __VA_ARGS__).str());
/**
 * @brief Asserts that any container elements match a given condition (non-deadly assertion).
 * @param container A container
 * @param condition A condition returning a boolean that can be applied to
 *  each container element
 */
#define UNITTEST_NDASSERT_ANY_OF(container, condition) \
unittest::assertions::assert_any_of(container, condition, UNITTEST_SPOT, UNITTEST_CALL(container, condition), UNITTEST_NDAS);
/**
 * @brief Asserts that any container elements match a given condition (non-deadly assertion).
 * @param container A container
 * @param condition A condition returning a boolean that can be applied to
 *  each container element
 * @param ... A user defined message
 */
#define UNITTEST_NDASSERT_ANY_OF_MSG(container, condition, ...) \
unittest::assertions::assert_any_of(container, condition, UNITTEST_SPOT, UNITTEST_CALL(container, condition), UNITTEST_NDAS, static_cast<const std::ostringstream&>(std::ostringstream{} << __VA_ARGS__).str());
/**
 * @brief Asserts that none of the container elements match a given condition.
 * @param container A container
 * @param condition A condition returning a boolean that can be applied to
 *  each container element
 */
#define UNITTEST_ASSERT_NONE_OF(container, condition) \
unittest::assertions::assert_none_of(container, condition, UNITTEST_SPOT, UNITTEST_CALL(container, condition));
/**
 * @brief Asserts that none of the container elements match a given condition.
 * @param container A container
 * @param condition A condition returning a boolean that can be applied to
 *  each container element
 * @param ... A user defined message
 */
#define UNITTEST_ASSERT_NONE_OF_MSG(container, condition, ...) \
unittest::assertions::assert_none_of(container, condition, UNITTEST_SPOT, UNITTEST_CALL(container, condition), static_cast<const std::ostringstream&>(std::ostringstream{} << __VA_ARGS__).str());
/**
 * @brief Asserts that none of the container elements match a given condition (non-deadly assertion).
 * @param container A container
 * @param condition A condition returning a boolean that can be applied to
 *  each container element
 */
#define UNITTEST_NDASSERT_NONE_OF(container, condition) \
unittest::assertions::assert_none_of(container, condition, UNITTEST_SPOT, UNITTEST_CALL(container, condition), UNITTEST_NDAS);
/**
 * @brief Asserts that none of the container elements match a given condition (non-deadly assertion).
 * @param container A container
 * @param condition A condition returning a boolean that can be applied to
 *  each container element
 * @param ... A user defined message
 */
#define UNITTEST_NDASSERT_NONE_OF_MSG(container, condition, ...) \
unittest::assertions::assert_none_of(container, condition, UNITTEST_SPOT, UNITTEST_CALL(container, condition), UNITTEST_NDAS, static_cast<const std::ostringstream&>(std::ostringstream{} << __VA_ARGS__).str());
/**
 * @brief Asserts that a string matches a regular expression.
 * @param string A string
 * @param regex The regular expression
 */
#define UNITTEST_ASSERT_REGEX_MATCH(string, regex) \
unittest::assertions::assert_regex_match(string, regex, UNITTEST_SPOT, UNITTEST_CALL(string, regex));
/**
 * @brief Asserts that a string matches a regular expression.
 * @param string A string
 * @param regex The regular expression
 * @param ... A user defined message
 */
#define UNITTEST_ASSERT_REGEX_MATCH_MSG(string, regex, ...) \
unittest::assertions::assert_regex_match(string, regex, UNITTEST_SPOT, UNITTEST_CALL(string, regex), static_cast<const std::ostringstream&>(std::ostringstream{} << __VA_ARGS__).str());
/**
 * @brief Asserts that a string matches a regular expression (non-deadly assertion).
 * @param string A string
 * @param regex The regular expression
 */
#define UNITTEST_NDASSERT_REGEX_MATCH(string, regex) \
unittest::assertions::assert_regex_match(string, regex, UNITTEST_SPOT, UNITTEST_CALL(string, regex), UNITTEST_NDAS);
/**
 * @brief Asserts that a string matches a regular expression (non-deadly assertion).
 * @param string A string
 * @param regex The regular expression
 * @param ... A user defined message
 */
#define UNITTEST_NDASSERT_REGEX_MATCH_MSG(string, regex, ...) \
unittest::assertions::assert_regex_match(string, regex, UNITTEST_SPOT, UNITTEST_CALL(string, regex), UNITTEST_NDAS, static_cast<const std::ostringstream&>(std::ostringstream{} << __VA_ARGS__).str());
/**
 * @brief Asserts that a string does not match a regular expression.
 * @param string A string
 * @param regex The regular expression
 */
#define UNITTEST_ASSERT_NOT_REGEX_MATCH(string, regex) \
unittest::assertions::assert_not_regex_match(string, regex, UNITTEST_SPOT, UNITTEST_CALL(string, regex));
/**
 * @brief Asserts that a string does not match a regular expression.
 * @param string A string
 * @param regex The regular expression
 * @param ... A user defined message
 */
#define UNITTEST_ASSERT_NOT_REGEX_MATCH_MSG(string, regex, ...) \
unittest::assertions::assert_not_regex_match(string, regex, UNITTEST_SPOT, UNITTEST_CALL(string, regex), static_cast<const std::ostringstream&>(std::ostringstream{} << __VA_ARGS__).str());
/**
 * @brief Asserts that a string does not match a regular expression (non-deadly assertion).
 * @param string A string
 * @param regex The regular expression
 */
#define UNITTEST_NDASSERT_NOT_REGEX_MATCH(string, regex) \
unittest::assertions::assert_not_regex_match(string, regex, UNITTEST_SPOT, UNITTEST_CALL(string, regex), UNITTEST_NDAS);
/**
 * @brief Asserts that a string does not match a regular expression (non-deadly assertion).
 * @param string A string
 * @param regex The regular expression
 * @param ... A user defined message
 */
#define UNITTEST_NDASSERT_NOT_REGEX_MATCH_MSG(string, regex, ...) \
unittest::assertions::assert_not_regex_match(string, regex, UNITTEST_SPOT, UNITTEST_CALL(string, regex), UNITTEST_NDAS, static_cast<const std::ostringstream&>(std::ostringstream{} << __VA_ARGS__).str());
/**
 * @brief Asserts that a functor throws exception.
 *  Required operators: ()
 * @param exception The exception type
 * @param functor The functor
 */
#define UNITTEST_ASSERT_THROW(exception, functor) \
unittest::assertions::assert_throw<exception>(functor, UNITTEST_SPOT, UNITTEST_CALL(exception, functor));
/**
 * @brief Asserts that a functor throws exception.
 *  Required operators: ()
 * @param exception The exception type
 * @param functor The functor
 * @param ... A user defined message
 */
#define UNITTEST_ASSERT_THROW_MSG(exception, functor, ...) \
unittest::assertions::assert_throw<exception>(functor, UNITTEST_SPOT, UNITTEST_CALL(exception, functor), static_cast<const std::ostringstream&>(std::ostringstream{} << __VA_ARGS__).str());
/**
 * @brief Asserts that a functor throws exception (non-deadly assertion).
 *  Required operators: ()
 * @param exception The exception type
 * @param functor The functor
 */
#define UNITTEST_NDASSERT_THROW(exception, functor) \
unittest::assertions::assert_throw<exception>(functor, UNITTEST_SPOT, UNITTEST_CALL(exception, functor), UNITTEST_NDAS);
/**
 * @brief Asserts that a functor throws exception (non-deadly assertion).
 *  Required operators: ()
 * @param exception The exception type
 * @param functor The functor
 * @param ... A user defined message
 */
#define UNITTEST_NDASSERT_THROW_MSG(exception, functor, ...) \
unittest::assertions::assert_throw<exception>(functor, UNITTEST_SPOT, UNITTEST_CALL(exception, functor), UNITTEST_NDAS, static_cast<const std::ostringstream&>(std::ostringstream{} << __VA_ARGS__).str());
/**
 * @brief Asserts that a functor does not throw any exception.
 *  Required operators: ()
 * @param functor The functor
 */
#define UNITTEST_ASSERT_NO_THROW(functor) \
unittest::assertions::assert_no_throw(functor, UNITTEST_SPOT, UNITTEST_CALL(functor));
/**
 * @brief Asserts that a functor does not throw any exception.
 *  Required operators: ()
 * @param functor The functor
 * @param ... A user defined message
 */
#define UNITTEST_ASSERT_NO_THROW_MSG(functor, ...) \
unittest::assertions::assert_no_throw(functor, UNITTEST_SPOT, UNITTEST_CALL(functor), static_cast<const std::ostringstream&>(std::ostringstream{} << __VA_ARGS__).str());
/**
 * @brief Asserts that a functor does not throw any exception (non-deadly assertion).
 *  Required operators: ()
 * @param functor The functor
 */
#define UNITTEST_NDASSERT_NO_THROW(functor) \
unittest::assertions::assert_no_throw(functor, UNITTEST_SPOT, UNITTEST_CALL(functor), UNITTEST_NDAS);
/**
 * @brief Asserts that a functor does not throw any exception (non-deadly assertion).
 *  Required operators: ()
 * @param functor The functor
 * @param ... A user defined message
 */
#define UNITTEST_NDASSERT_NO_THROW_MSG(functor, ...) \
unittest::assertions::assert_no_throw(functor, UNITTEST_SPOT, UNITTEST_CALL(functor), UNITTEST_NDAS, static_cast<const std::ostringstream&>(std::ostringstream{} << __VA_ARGS__).str());
