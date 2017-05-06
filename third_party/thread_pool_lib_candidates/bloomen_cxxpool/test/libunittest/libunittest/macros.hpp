/**
 * @brief A collection of helper macros
 * @file macros.hpp
 */
#pragma once
/**
 * @brief Registers a test class within TestCase::run().
 *  To be called prior to UNITTEST_RUN
 * @param test_class The test class
 */
#define UNITTEST_CLASS(test_class) \
typedef test_class __test_class__;
/**
 * @brief A test run
 * @param test_method The test method
 */
#define UNITTEST_RUN(test_method) \
unittest::testrun(std::shared_ptr<typename __test_class__::context_type>(nullptr), &__test_class__::test_method, #test_method, false, "");
/**
 * @brief A skipped test run
 * @param test_method The test method
 * @param skip_message A message explaining why the test is skipped
 */
#define UNITTEST_RUN_SKIP(test_method, skip_message) \
unittest::testrun(std::shared_ptr<typename __test_class__::context_type>(nullptr), &__test_class__::test_method, #test_method, true, skip_message);
/**
 * @brief A maybe test run
 * @param test_method The test method
 * @param is_run Whether the test is run
 * @param skip_message A message explaining why the test is skipped
 */
#define UNITTEST_RUN_MAYBE(test_method, is_run, skip_message) \
unittest::testrun(std::shared_ptr<typename __test_class__::context_type>(nullptr), &__test_class__::test_method, #test_method, !is_run, skip_message);
/**
 * @brief A test run with timeout
 * @param test_method The test method
 * @param timeout The maximum allowed run time in seconds (ignored if <= 0)
 */
#define UNITTEST_RUN_TIME(test_method, timeout) \
unittest::testrun(std::shared_ptr<typename __test_class__::context_type>(nullptr), &__test_class__::test_method, #test_method, false, "", timeout);
/**
 * @brief A skipped test run with timeout
 * @param test_method The test method
 * @param timeout The maximum allowed run time in seconds (ignored if <= 0)
 * @param skip_message A message explaining why the test is skipped
 */
#define UNITTEST_RUN_TIME_SKIP(test_method, timeout, skip_message) \
unittest::testrun(std::shared_ptr<typename __test_class__::context_type>(nullptr), &__test_class__::test_method, #test_method, true, skip_message, timeout);
/**
 * @brief A maybe test run with timeout
 * @param test_method The test method
 * @param timeout The maximum allowed run time in seconds (ignored if <= 0)
 * @param is_run Whether the test is run
 * @param skip_message A message explaining why the test is skipped
 */
#define UNITTEST_RUN_TIME_MAYBE(test_method, timeout, is_run, skip_message) \
unittest::testrun(std::shared_ptr<typename __test_class__::context_type>(nullptr), &__test_class__::test_method, #test_method, !is_run, skip_message, timeout);
/**
 * @brief A test run with a test context
 * @param test_context The test context
 * @param test_method The test method
 */
#define UNITTEST_RUNCTX(test_context, test_method) \
unittest::testrun(test_context, &__test_class__::test_method, #test_method, false, "");
/**
 * @brief A skipped test run with a test context
 * @param test_context The test context
 * @param test_method The test method
 * @param skip_message A message explaining why the test is skipped
 */
#define UNITTEST_RUNCTX_SKIP(test_context, test_method, skip_message) \
unittest::testrun(test_context, &__test_class__::test_method, #test_method, true, skip_message);
/**
 * @brief A maybe test run with a test context
 * @param test_context The test context
 * @param test_method The test method
 * @param is_run Whether the test is run
 * @param skip_message A message explaining why the test is skipped
 */
#define UNITTEST_RUNCTX_MAYBE(test_context, test_method, is_run, skip_message) \
unittest::testrun(test_context, &__test_class__::test_method, #test_method, !is_run, skip_message);
/**
 * @brief A test run with a test context and timeout
 * @param test_context The test context
 * @param test_method The test method
 * @param timeout The maximum allowed run time in seconds (ignored if <= 0)
 */
#define UNITTEST_RUNCTX_TIME(test_context, test_method, timeout) \
unittest::testrun(test_context, &__test_class__::test_method, #test_method, false, "", timeout);
/**
 * @brief A skipped test run with a test context and timeout
 * @param test_context The test context
 * @param test_method The test method
 * @param timeout The maximum allowed run time in seconds (ignored if <= 0)
 * @param skip_message A message explaining why the test is skipped
 */
#define UNITTEST_RUNCTX_TIME_SKIP(test_context, test_method, timeout, skip_message) \
unittest::testrun(test_context, &__test_class__::test_method, #test_method, true, skip_message, timeout);
/**
 * @brief A maybe test run with a test context and timeout
 * @param test_context The test context
 * @param test_method The test method
 * @param timeout The maximum allowed run time in seconds (ignored if <= 0)
 * @param is_run Whether the test is run
 * @param skip_message A message explaining why the test is skipped
 */
#define UNITTEST_RUNCTX_TIME_MAYBE(test_context, test_method, timeout, is_run, skip_message) \
unittest::testrun(test_context, &__test_class__::test_method, #test_method, !is_run, skip_message, timeout);
/**
 * @brief Defines a string with info about the current file name
 *  and the current line number
 */
#define UNITTEST_SPOT \
unittest::core::string_of_file_and_line(__FILE__, __LINE__)
/**
 * @brief Defines a string indicating the use of non-deadly assertions
 */
#define UNITTEST_NDAS \
unittest::core::string_of_tagged_text(this->get_test_id(), "NDAS")
/**
 * @brief Defines a string of the call site
 */
#define UNITTEST_CALL(...) \
unittest::core::string_of_tagged_text(#__VA_ARGS__, "CALL")
/**
 * @brief Logs info for the current test
 */
#define UNITTEST_TESTINFO(...) \
unittest::core::testsuite::instance()->log_text(this->get_test_id(), unittest::join(__VA_ARGS__));
/**
 * @brief Joins two symbols. Just for internals
 * @param symbol1 A symbol
 * @param symbol2 Another symbol
 */
#define __UNITTEST_JOIN(symbol1, symbol2) \
__UNITTEST_DO_JOIN(symbol1, symbol2)
/**
 * @brief Joins two symbols. Just for internals
 * @param symbol1 A symbol
 * @param symbol2 Another symbol
 */
#define __UNITTEST_DO_JOIN(symbol1, symbol2) \
symbol1##symbol2
/**
 * @brief Registers a test class at the global registry
 */
#define UNITTEST_REGISTER(...) \
static unittest::core::testregistry<__VA_ARGS__> __UNITTEST_JOIN(__registered_at_, __LINE__)(__testcollection_type__(), #__VA_ARGS__);
/**
 * @brief A test collection
 * @param name The name of the test collection
 */
#define UNITTEST_COLLECTION(name) \
namespace name { \
    struct __testcollection_child__ : unittest::core::testcollection { \
        std::string \
        get_name() const \
        { \
            __testcollection_type__ prev_coll; \
            if (prev_coll.get_name() != testcollection::inactive_name()) \
                return prev_coll.get_name() + "::" + #name; \
            else \
                return #name; \
        } \
    }; \
    typedef __testcollection_child__ __testcollection_type__; \
} \
namespace name
/**
 * @brief Sets up a plain test
 * @param test_name The name of the test
 */
#define UNITTEST_TEST(test_name) \
__UNITTEST_TEST_PLAIN(unittest::sometype, test_name, false, "")
/**
 * @brief Sets up a skipped plain test
 * @param test_name The name of the test
 * @param skip_message A message explaining why the test is skipped
 */
#define UNITTEST_TEST_SKIP(test_name, skip_message) \
__UNITTEST_TEST_PLAIN(unittest::sometype, test_name, true, skip_message)
/**
 * @brief Sets up a maybe plain test
 * @param test_name The name of the test
 * @param is_run Whether the test is run
 * @param skip_message A message explaining why the test is skipped
 */
#define UNITTEST_TEST_MAYBE(test_name, is_run, skip_message) \
__UNITTEST_TEST_PLAIN(unittest::sometype, test_name, !is_run, skip_message)
/**
 * @brief Sets up a plain test with timeout
 * @param test_name The name of the test
 * @param timeout The maximum allowed run time in seconds (ignored if <= 0)
 */
#define UNITTEST_TEST_TIME(test_name, timeout) \
__UNITTEST_TEST_PLAIN_TIME(unittest::sometype, test_name, timeout, false, "")
/**
 * @brief Sets up a skipped plain test with timeout
 * @param test_name The name of the test
 * @param timeout The maximum allowed run time in seconds (ignored if <= 0)
 * @param skip_message A message explaining why the test is skipped
 */
#define UNITTEST_TEST_TIME_SKIP(test_name, timeout, skip_message) \
__UNITTEST_TEST_PLAIN_TIME(unittest::sometype, test_name, timeout, true, skip_message)
/**
 * @brief Sets up a maybe plain test with timeout
 * @param test_name The name of the test
 * @param timeout The maximum allowed run time in seconds (ignored if <= 0)
 * @param is_run Whether the test is run
 * @param skip_message A message explaining why the test is skipped
 */
#define UNITTEST_TEST_TIME_MAYBE(test_name, timeout, is_run, skip_message) \
__UNITTEST_TEST_PLAIN_TIME(unittest::sometype, test_name, timeout, !is_run, skip_message)
/**
 * @brief Sets up a plain test with a test fixture
 * @param fixture The test fixture
 * @param test_name The name of the test
 */
#define UNITTEST_TEST_FIXTURE(fixture, test_name) \
__UNITTEST_TEST_PLAIN(fixture, test_name, false, "")
/**
 * @brief Sets up a skipped plain test with a test fixture
 * @param fixture The test fixture
 * @param test_name The name of the test
 * @param skip_message A message explaining why the test is skipped
 */
#define UNITTEST_TEST_FIXTURE_SKIP(fixture, test_name, skip_message) \
__UNITTEST_TEST_PLAIN(fixture, test_name, true, skip_message)
/**
 * @brief Sets up a maybe plain test with a test fixture
 * @param fixture The test fixture
 * @param test_name The name of the test
 * @param is_run Whether the test is run
 * @param skip_message A message explaining why the test is skipped
 */
#define UNITTEST_TEST_FIXTURE_MAYBE(fixture, test_name, is_run, skip_message) \
__UNITTEST_TEST_PLAIN(fixture, test_name, !is_run, skip_message)
/**
 * @brief Sets up a plain test with a test fixture and timeout
 * @param fixture The test fixture
 * @param test_name The name of the test
 * @param timeout The maximum allowed run time in seconds (ignored if <= 0)
 */
#define UNITTEST_TEST_FIXTURE_TIME(fixture, test_name, timeout) \
__UNITTEST_TEST_PLAIN_TIME(fixture, test_name, timeout, false, "")
/**
 * @brief Sets up a skipped plain test with a test fixture and timeout
 * @param fixture The test fixture
 * @param test_name The name of the test
 * @param timeout The maximum allowed run time in seconds (ignored if <= 0)
 * @param skip_message A message explaining why the test is skipped
 */
#define UNITTEST_TEST_FIXTURE_TIME_SKIP(fixture, test_name, timeout, skip_message) \
__UNITTEST_TEST_PLAIN_TIME(fixture, test_name, timeout, true, skip_message)
/**
 * @brief Sets up a maybe plain test with a test fixture and timeout
 * @param fixture The test fixture
 * @param test_name The name of the test
 * @param timeout The maximum allowed run time in seconds (ignored if <= 0)
 * @param is_run Whether the test is run
 * @param skip_message A message explaining why the test is skipped
 */
#define UNITTEST_TEST_FIXTURE_TIME_MAYBE(fixture, test_name, timeout, is_run, skip_message) \
__UNITTEST_TEST_PLAIN_TIME(fixture, test_name, timeout, !is_run, skip_message)
/**
 * @brief Sets up a generic plain test. Only for internals
 * @param fixture The test fixture
 * @param test_name The name of the test
 * @param skipped Whether this test run is skipped
 * @param skip_message A message explaining why the test is skipped
 */
#define __UNITTEST_TEST_PLAIN(fixture, test_name, skipped, skip_message) \
struct test_name : unittest::testcase<>, fixture { \
    static void run() \
    { \
        UNITTEST_CLASS(test_name) \
        UNITTEST_RUN_MAYBE(test, !skipped, skip_message) \
    } \
    void test(); \
}; \
UNITTEST_REGISTER(test_name) \
void test_name::test()
/**
 * @brief Sets up a generic plain test with timeout. Only for internals
 * @param fixture The test fixture
 * @param test_name The name of the test
 * @param timeout The maximum allowed run time in seconds (ignored if <= 0)
 * @param skipped Whether this test run is skipped
 * @param skip_message A message explaining why the test is skipped
 */
#define __UNITTEST_TEST_PLAIN_TIME(fixture, test_name, timeout, skipped, skip_message) \
struct test_name : unittest::testcase<>, fixture { \
    static void run() \
    { \
        UNITTEST_CLASS(test_name) \
        UNITTEST_RUN_TIME_MAYBE(test, timeout, !skipped, skip_message) \
    } \
    void test(); \
}; \
UNITTEST_REGISTER(test_name) \
void test_name::test()
/**
 * @brief Sets up a templated test
 * @param test_name The name of the test
 */
#define UNITTEST_TEST_TPL(test_name) \
__UNITTEST_TEST_TPL(unittest::sometype, test_name, false, "")
/**
 * @brief Sets up a skipped templated test
 * @param test_name The name of the test
 * @param skip_message A message explaining why the test is skipped
 */
#define UNITTEST_TEST_TPL_SKIP(test_name, skip_message) \
__UNITTEST_TEST_TPL(unittest::sometype, test_name, true, skip_message)
/**
 * @brief Sets up a maybe templated test
 * @param test_name The name of the test
 * @param is_run Whether the test is run
 * @param skip_message A message explaining why the test is skipped
 */
#define UNITTEST_TEST_TPL_MAYBE(test_name, is_run, skip_message) \
__UNITTEST_TEST_TPL(unittest::sometype, test_name, !is_run, skip_message)
/**
 * @brief Sets up a templated test with timeout
 * @param test_name The name of the test
 * @param timeout The maximum allowed run time in seconds (ignored if <= 0)
 */
#define UNITTEST_TEST_TPL_TIME(test_name, timeout) \
__UNITTEST_TEST_TPL_TIME(unittest::sometype, test_name, timeout, false, "")
/**
 * @brief Sets up a skipped templated test with timeout
 * @param test_name The name of the test
 * @param timeout The maximum allowed run time in seconds (ignored if <= 0)
 * @param skip_message A message explaining why the test is skipped
 */
#define UNITTEST_TEST_TPL_TIME_SKIP(test_name, timeout, skip_message) \
__UNITTEST_TEST_TPL_TIME(unittest::sometype, test_name, timeout, true, skip_message)
/**
 * @brief Sets up a maybe templated test with timeout
 * @param test_name The name of the test
 * @param timeout The maximum allowed run time in seconds (ignored if <= 0)
 * @param is_run Whether the test is run
 * @param skip_message A message explaining why the test is skipped
 */
#define UNITTEST_TEST_TPL_TIME_MAYBE(test_name, timeout, is_run, skip_message) \
__UNITTEST_TEST_TPL_TIME(unittest::sometype, test_name, timeout, !is_run, skip_message)
/**
 * @brief Sets up a templated test with a test fixture
 * @param fixture The test fixture
 * @param test_name The name of the test
 */
#define UNITTEST_TEST_TPL_FIXTURE(fixture, test_name) \
__UNITTEST_TEST_TPL(fixture, test_name, false, "")
/**
 * @brief Sets up a skipped templated test with a test fixture
 * @param fixture The test fixture
 * @param test_name The name of the test
 * @param skip_message A message explaining why the test is skipped
 */
#define UNITTEST_TEST_TPL_FIXTURE_SKIP(fixture, test_name, skip_message) \
__UNITTEST_TEST_TPL(fixture, test_name, true, skip_message)
/**
 * @brief Sets up a maybe templated test with a test fixture
 * @param fixture The test fixture
 * @param test_name The name of the test
 * @param is_run Whether the test is run
 * @param skip_message A message explaining why the test is skipped
 */
#define UNITTEST_TEST_TPL_FIXTURE_MAYBE(fixture, test_name, is_run, skip_message) \
__UNITTEST_TEST_TPL(fixture, test_name, !is_run, skip_message)
/**
 * @brief Sets up a templated test with a test fixture and timeout
 * @param fixture The test fixture
 * @param test_name The name of the test
 * @param timeout The maximum allowed run time in seconds (ignored if <= 0)
 */
#define UNITTEST_TEST_TPL_FIXTURE_TIME(fixture, test_name, timeout) \
__UNITTEST_TEST_TPL_TIME(fixture, test_name, timeout, false, "")
/**
 * @brief Sets up a skipped templated test with a test fixture and timeout
 * @param fixture The test fixture
 * @param test_name The name of the test
 * @param timeout The maximum allowed run time in seconds (ignored if <= 0)
 * @param skip_message A message explaining why the test is skipped
 */
#define UNITTEST_TEST_TPL_FIXTURE_TIME_SKIP(fixture, test_name, timeout, skip_message) \
__UNITTEST_TEST_TPL_TIME(fixture, test_name, timeout, true, skip_message)
/**
 * @brief Sets up a maybe templated test with a test fixture and timeout
 * @param fixture The test fixture
 * @param test_name The name of the test
 * @param timeout The maximum allowed run time in seconds (ignored if <= 0)
 * @param is_run Whether the test is run
 * @param skip_message A message explaining why the test is skipped
 */
#define UNITTEST_TEST_TPL_FIXTURE_TIME_MAYBE(fixture, test_name, timeout, is_run, skip_message) \
__UNITTEST_TEST_TPL_TIME(fixture, test_name, timeout, !is_run, skip_message)
/**
 * @brief Sets up a generic templated test. Only for internals
 * @param fixture The test fixture
 * @param test_name The name of the test
 * @param skipped Whether this test run is skipped
 * @param skip_message A message explaining why the test is skipped
 */
#define __UNITTEST_TEST_TPL(fixture, test_name, skipped, skip_message) \
template<typename Type1, typename Type2=unittest::notype, typename Type3=unittest::notype> \
struct test_name : unittest::testcase<>, fixture { \
    static void run() \
    { \
        UNITTEST_CLASS(test_name) \
        UNITTEST_RUN_MAYBE(test, !skipped, skip_message) \
    } \
    void test(); \
}; \
template<typename Type1, typename Type2, typename Type3> \
void test_name<Type1,Type2,Type3>::test()
/**
 * @brief Sets up a generic templated test with timeout. Only for internals
 * @param fixture The test fixture
 * @param test_name The name of the test
 * @param timeout The maximum allowed run time in seconds (ignored if <= 0)
 * @param skipped Whether this test run is skipped
 * @param skip_message A message explaining why the test is skipped
 */
#define __UNITTEST_TEST_TPL_TIME(fixture, test_name, timeout, skipped, skip_message) \
template<typename Type1, typename Type2=unittest::notype, typename Type3=unittest::notype> \
struct test_name : unittest::testcase<>, fixture { \
    static void run() \
    { \
        UNITTEST_CLASS(test_name) \
        UNITTEST_RUN_TIME_MAYBE(test, timeout, !skipped, skip_message) \
    } \
    void test(); \
}; \
template<typename Type1, typename Type2, typename Type3> \
void test_name<Type1,Type2,Type3>::test()
