/**
 * @brief The testlog class and functions working on it
 * @file testlog.hpp
 */
#pragma once
#include "teststatus.hpp"
#include <string>
#include <ostream>
#include <vector>
/**
 * @brief Unit testing in C++
 */
namespace unittest {
/**
 * @brief The exception class to indicate test failures
 */
class testfailure;
/**
 * @brief Internal functionality, not relevant for most users
 */
namespace core {
/**
 * @brief Stores logging info about a single test
 */
struct testlog {
    /**
     * @brief Constructor
     */
    testlog();
    /**
     * @brief The name of the test class
     */
    std::string class_name;
    /**
     * @brief The name of the test method
     */
    std::string test_name;
    /**
     * @brief Whether the test ran successfully
     */
    bool successful;
    /**
     * @brief The test status
     */
    teststatus status;
    /**
     * @brief The name of the error type. Empty if test was successful
     */
    std::string error_type;
    /**
     * @brief The result message
     */
    std::string message;
    /**
     * @brief The test duration in seconds
     */
    double duration;
    /**
     * @brief Whether the test has timed out
     */
    bool has_timed_out;
    /**
     * @brief The test timeout so the maximum allowed run time
     */
    double timeout;
    /**
     * @brief The name of the assertion if applicable
     */
    std::string assertion;
    /**
     * @brief The method ID
     */
    std::string method_id;
    /**
     * @brief The text logged for this test
     */
    std::string text;
    /**
     * @brief The file name
     */
    std::string filename;
    /**
     * @brief The line number
     */
    int linenumber;
    /**
     * @brief A string of the call site
     */
    std::string callsite;
    /**
     * @brief The non-deadly failures
     */
    std::vector<unittest::testfailure> nd_failures;
};
/**
 * @brief Writes a test start message to the given output stream
 * @param stream The output stream
 * @param log A test log
 * @param verbose Whether to use verbose output
 */
void
write_test_start_message(std::ostream& stream,
                         const unittest::core::testlog& log,
                         bool verbose);
/**
 * @brief Writes a test end message to the given output stream
 * @param stream The output stream
 * @param log A test log
 * @param verbose Whether to use verbose output
 */
void
write_test_end_message(std::ostream& stream,
                       const unittest::core::testlog& log,
                       bool verbose);
/**
 * @brief Writes a test timeout message to the given output stream
 * @param stream The output stream
 * @param verbose Whether to use verbose output
 */
void
write_test_timeout_message(std::ostream& stream,
                           bool verbose);
/**
 * @brief Generates the full test name
 * @param class_name The name of the test class
 * @param test_name The name of the test method
 * @returns The full test name
 */
std::string
make_full_test_name(const std::string& class_name,
                    const std::string& test_name);
/**
 * @brief Returns whether a given test is executed
 * @param full_test_name The full test name
 * @param exact_name An exact name to be checked for, ignored if empty
 * @param filter_name A filter for the beginning of the test name, ignored if empty
 * @param regex_filter A regex filter, ignored if empty
 * @returns Whether a given test is executed
 */
bool
is_test_executed(const std::string& full_test_name,
                 const std::string& exact_name,
                 const std::string& filter_name,
				 const std::string& regex_filter);
/**
 * @brief Evaluates whether to keep running tests
 * @param log A test log
 * @param failure_stop Whether to stop after first fail
 * @returns Whether to keep running tests
 */
bool
keep_running(const unittest::core::testlog& log,
             bool failure_stop);

} // core
} // unittest
