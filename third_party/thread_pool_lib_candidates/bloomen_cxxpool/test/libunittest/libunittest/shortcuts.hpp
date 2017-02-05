/**
 * @brief A collection of short aliases for selected macros
 * @file shortcuts.hpp
 */
#pragma once
/**
 * @brief Defines a string with info about the current file name
 *  and the current line number
 */
#define SPOT \
UNITTEST_SPOT
/**
 * @brief Defines a string indicating the use of non-deadly assertions
 */
#define NDAS \
UNITTEST_NDAS
/**
 * @brief Logs info for the current test
 */
#define TESTINFO(...) \
UNITTEST_TESTINFO(__VA_ARGS__)
/**
 * @brief Registers a test class at the global registry
 */
#define REGISTER(...) \
UNITTEST_REGISTER(__VA_ARGS__)
/**
 * @brief A test collection
 * @param name The name of the test collection
 */
#define COLLECTION(name) \
UNITTEST_COLLECTION(name)
/**
 * @brief Sets up a plain test
 * @param test_name The name of the test
 */
#define TEST(test_name) \
UNITTEST_TEST(test_name)
/**
 * @brief Sets up a skipped plain test
 * @param test_name The name of the test
 * @param skip_message A message explaining why the test is skipped
 */
#define TEST_SKIP(test_name, skip_message) \
UNITTEST_TEST_SKIP(test_name, skip_message)
/**
 * @brief Sets up a maybe plain test
 * @param test_name The name of the test
 * @param is_run Whether the test is run
 * @param skip_message A message explaining why the test is skipped
 */
#define TEST_MAYBE(test_name, is_run, skip_message) \
UNITTEST_TEST_MAYBE(test_name, is_run, skip_message)
/**
 * @brief Sets up a plain test with timeout
 * @param test_name The name of the test
 * @param timeout The maximum allowed run time in seconds (ignored if <= 0)
 */
#define TEST_TIME(test_name, timeout) \
UNITTEST_TEST_TIME(test_name, timeout)
/**
 * @brief Sets up a skipped plain test with timeout
 * @param test_name The name of the test
 * @param timeout The maximum allowed run time in seconds (ignored if <= 0)
 * @param skip_message A message explaining why the test is skipped
 */
#define TEST_TIME_SKIP(test_name, timeout, skip_message) \
UNITTEST_TEST_TIME_SKIP(test_name, timeout, skip_message)
/**
 * @brief Sets up a maybe plain test with timeout
 * @param test_name The name of the test
 * @param timeout The maximum allowed run time in seconds (ignored if <= 0)
 * @param is_run Whether the test is run
 * @param skip_message A message explaining why the test is skipped
 */
#define TEST_TIME_MAYBE(test_name, timeout, is_run, skip_message) \
UNITTEST_TEST_TIME_MAYBE(test_name, timeout, is_run, skip_message)
/**
 * @brief Sets up a plain test with a test fixture
 * @param fixture The test fixture
 * @param test_name The name of the test
 */
#define TEST_FIXTURE(fixture, test_name) \
UNITTEST_TEST_FIXTURE(fixture, test_name)
/**
 * @brief Sets up a skipped plain test with a test fixture
 * @param fixture The test fixture
 * @param test_name The name of the test
 * @param skip_message A message explaining why the test is skipped
 */
#define TEST_FIXTURE_SKIP(fixture, test_name, skip_message) \
UNITTEST_TEST_FIXTURE_SKIP(fixture, test_name, skip_message)
/**
 * @brief Sets up a maybe plain test with a test fixture
 * @param fixture The test fixture
 * @param test_name The name of the test
 * @param is_run Whether the test is run
 * @param skip_message A message explaining why the test is skipped
 */
#define TEST_FIXTURE_MAYBE(fixture, test_name, is_run, skip_message) \
UNITTEST_TEST_FIXTURE_MAYBE(fixture, test_name, is_run, skip_message)
/**
 * @brief Sets up a plain test with a test fixture and timeout
 * @param fixture The test fixture
 * @param test_name The name of the test
 * @param timeout The maximum allowed run time in seconds (ignored if <= 0)
 */
#define TEST_FIXTURE_TIME(fixture, test_name, timeout) \
UNITTEST_TEST_FIXTURE_TIME(fixture, test_name, timeout)
/**
 * @brief Sets up a skipped plain test with a test fixture and timeout
 * @param fixture The test fixture
 * @param test_name The name of the test
 * @param timeout The maximum allowed run time in seconds (ignored if <= 0)
 * @param skip_message A message explaining why the test is skipped
 */
#define TEST_FIXTURE_TIME_SKIP(fixture, test_name, timeout, skip_message) \
UNITTEST_TEST_FIXTURE_TIME_SKIP(fixture, test_name, timeout, skip_message)
/**
 * @brief Sets up a skipped plain test with a test fixture and timeout
 * @param fixture The test fixture
 * @param test_name The name of the test
 * @param timeout The maximum allowed run time in seconds (ignored if <= 0)
 * @param skip_message A message explaining why the test is skipped
 */
#define TEST_FIXTURE_TIME_SKIP(fixture, test_name, timeout, skip_message) \
UNITTEST_TEST_FIXTURE_TIME_SKIP(fixture, test_name, timeout, skip_message)
/**
 * @brief Sets up a maybe plain test with a test fixture and timeout
 * @param fixture The test fixture
 * @param test_name The name of the test
 * @param timeout The maximum allowed run time in seconds (ignored if <= 0)
 * @param is_run Whether the test is run
 * @param skip_message A message explaining why the test is skipped
 */
#define TEST_FIXTURE_TIME_MAYBE(fixture, test_name, timeout, is_run, skip_message) \
UNITTEST_TEST_FIXTURE_TIME_MAYBE(fixture, test_name, timeout, is_run, skip_message)
/**
 * @brief Sets up a templated test
 * @param test_name The name of the test
 */
#define TEST_TPL(test_name) \
UNITTEST_TEST_TPL(test_name)
/**
 * @brief Sets up a skipped templated test
 * @param test_name The name of the test
 * @param skip_message A message explaining why the test is skipped
 */
#define TEST_TPL_SKIP(test_name, skip_message) \
UNITTEST_TEST_TPL_SKIP(test_name, skip_message)
/**
 * @brief Sets up a maybe templated test
 * @param test_name The name of the test
 * @param is_run Whether the test is run
 * @param skip_message A message explaining why the test is skipped
 */
#define TEST_TPL_MAYBE(test_name, is_run, skip_message) \
UNITTEST_TEST_TPL_MAYBE(test_name, is_run, skip_message)
/**
 * @brief Sets up a templated test with a timeout
 * @param test_name The name of the test
 * @param timeout The maximum allowed run time in seconds (ignored if <= 0)
 */
#define TEST_TPL_TIME(test_name, timeout) \
UNITTEST_TEST_TPL_TIME(test_name, timeout)
/**
 * @brief Sets up a skipped templated test with a timeout
 * @param test_name The name of the test
 * @param timeout The maximum allowed run time in seconds (ignored if <= 0)
 * @param skip_message A message explaining why the test is skipped
 */
#define TEST_TPL_TIME_SKIP(test_name, timeout, skip_message) \
UNITTEST_TEST_TPL_TIME_SKIP(test_name, timeout, skip_message)
/**
 * @brief Sets up a maybe templated test with timeout
 * @param test_name The name of the test
 * @param timeout The maximum allowed run time in seconds (ignored if <= 0)
 * @param is_run Whether the test is run
 * @param skip_message A message explaining why the test is skipped
 */
#define TEST_TPL_TIME_MAYBE(test_name, timeout, is_run, skip_message) \
UNITTEST_TEST_TPL_TIME_MAYBE(test_name, timeout, is_run, skip_message)
/**
 * @brief Sets up a templated test with a test fixture
 * @param fixture The test fixture
 * @param test_name The name of the test
 */
#define TEST_TPL_FIXTURE(fixture, test_name) \
UNITTEST_TEST_TPL_FIXTURE(fixture, test_name)
/**
 * @brief Sets up a skipped templated test with a test fixture
 * @param fixture The test fixture
 * @param test_name The name of the test
 * @param skip_message A message explaining why the test is skipped
 */
#define TEST_TPL_FIXTURE_SKIP(fixture, test_name, skip_message) \
UNITTEST_TEST_TPL_FIXTURE_SKIP(fixture, test_name, skip_message)
/**
 * @brief Sets up a maybe templated test with a test fixture
 * @param fixture The test fixture
 * @param test_name The name of the test
 * @param is_run Whether the test is run
 * @param skip_message A message explaining why the test is skipped
 */
#define TEST_TPL_FIXTURE_MAYBE(fixture, test_name, is_run, skip_message) \
UNITTEST_TEST_TPL_FIXTURE_MAYBE(fixture, test_name, is_run, skip_message)
/**
 * @brief Sets up a templated test with a test fixture and timeout
 * @param fixture The test fixture
 * @param test_name The name of the test
 * @param timeout The maximum allowed run time in seconds (ignored if <= 0)
 */
#define TEST_TPL_FIXTURE_TIME(fixture, test_name, timeout) \
UNITTEST_TEST_TPL_FIXTURE_TIME(fixture, test_name, timeout)
/**
 * @brief Sets up a skipped templated test with a test fixture and timeout
 * @param fixture The test fixture
 * @param test_name The name of the test
 * @param timeout The maximum allowed run time in seconds (ignored if <= 0)
 * @param skip_message A message explaining why the test is skipped
 */
#define TEST_TPL_FIXTURE_TIME_SKIP(fixture, test_name, timeout, skip_message) \
UNITTEST_TEST_TPL_FIXTURE_TIME_SKIP(fixture, test_name, timeout, skip_message)
/**
 * @brief Sets up a maybe templated test with a test fixture and timeout
 * @param fixture The test fixture
 * @param test_name The name of the test
 * @param timeout The maximum allowed run time in seconds (ignored if <= 0)
 * @param is_run Whether the test is run
 * @param skip_message A message explaining why the test is skipped
 */
#define TEST_TPL_FIXTURE_TIME_MAYBE(fixture, test_name, timeout, is_run, skip_message) \
UNITTEST_TEST_TPL_FIXTURE_TIME_MAYBE(fixture, test_name, timeout, is_run, skip_message)
