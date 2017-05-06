/**
 * @brief The testfailure class to indicate test failures
 * @file testfailure.hpp
 */
#pragma once
#include "utilities.hpp"
#include "noexcept.hpp"
#include "testsuite.hpp"
#include <string>
#include <stdexcept>
/**
 * @brief Unit testing in C++
 */
namespace unittest {
/**
 * @brief The exception class to indicate test failures
 */
class testfailure : public std::runtime_error {
public:
    /**
     * @brief Constructor
     * @param assertion The name of the assertion
     * @param message The message of the test failure
     */
    testfailure(const std::string& assertion,
                const std::string& message);
    /**
     * @brief Constructor
     * @param assertion The name of the assertion
     * @param message The message of the test failure
     * @param user_msg The user defined message
     */
    testfailure(const std::string& assertion,
                const std::string& message,
                const std::string& user_msg);
    /**
     * @brief Copy constructor
     * @param other An instance of testfailure
     */
    testfailure(const testfailure& other);
    /**
     * @brief Copy assignment operator
     * @param other An instance of testfailure
     * @returns An testfailure instance
     */
    testfailure&
    operator=(const testfailure& other);
    /**
     * @brief Destructor
     */
    virtual
    ~testfailure() UNITTEST_NOEXCEPT;
    /**
     * @brief Returns the name of the assertion
     * @returns The name of the assertion
     */
    std::string
    assertion() const;
    /**
     * @brief Returns the name of the file in which the test failure occurred
     * @returns The name of the file in which the test failure occurred
     */
    std::string
    filename() const;
    /**
     * @brief Returns the line number in which the test failure occurred
     * @returns The line number in which the test failure occurred
     */
    int
    linenumber() const;
    /**
     * @brief Returns a string of the call site if applicable
     * @returns The call site of the failure
     */
    std::string
    callsite() const;

private:
    std::string error_msg_;
    std::string assertion_;
    std::pair<std::string, int> spot_;
    std::string callsite_;

    std::string make_error_msg(const std::string& message,
                               const std::string& user_msg);
};

namespace core {

/**
 * @brief Builds a fail message from the parameters passed and throws
 *  exception testfailure if assertion is deadly (the default)
 * @param assertion The name of the assertion
 * @param message The assertion message
 * @param usermsg The user message
 * @throws testfailure
 */
void
fail_impl(const std::string& assertion,
          const std::string& message,
          std::string usermsg);

} // core

/**
 * @brief Builds a fail message from the parameters passed and throws
 *  exception testfailure if assertion is deadly (the default)
 * @param assertion The name of the assertion
 * @param message The assertion message
 * @param args An arbitrary number of arguments that are concatenated
 *  to a single string and are appended to the assertion message
 * @throws testfailure
 */
template<typename... Args>
void
fail(const std::string& assertion,
     const std::string& message,
     const Args&... args)
{
    unittest::core::fail_impl(assertion, message, unittest::join("", args...));
}

} // unittest
