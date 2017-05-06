/**
 * @brief The testcase class from which to derive when implementing a test class
 * @file testcase.hpp
 */
#pragma once
#include "assertions.hpp"
#include "noexcept.hpp"
#include <string>
#include <memory>
/**
 * @brief Unit testing in C++
 */
namespace unittest {
/**
 * @brief Internal functionality, not relevant for most users
 */
namespace core {
/**
 * @brief Stores the test to be run and an optional test context.
 *  By using the ()-operator the test is executed.
 */
template<typename T>
struct testfunctor;

} // core

/**
 * @brief The class from which to derive when implementing a test case. The
 *  test context is optional
 */
template<typename TestContext=unittest::notype>
class testcase {
public:
    /**
     * @brief The type of the optional test context
     */
    typedef TestContext context_type;
    /**
     * @brief Constructor. Called before each test run.
     */
    testcase()
        : context_(nullptr), test_id_("")
    {}
    /**
     * @brief Destructor. Called after each test run.
     */
    virtual
    ~testcase() UNITTEST_NOEXCEPT_FALSE
    {}
    /**
     * @brief Copy constructor. Deleted
     * @param other An instance of testcase
     */
    testcase(const testcase& other) = delete;
    /**
     * @brief Copy assignment operator. Deleted
     * @param other An instance of testcase
     * @returns An testcase instance
     */
    testcase&
    operator=(const testcase& other) = delete;
    /**
     * @brief Move constructor. Deleted
     * @param other An instance of testcase
     */
    testcase(testcase&& other) = delete;
    /**
     * @brief Move assignment operator. Deleted
     * @param other An instance of testcase
     * @returns An testcase instance
     */
    testcase&
    operator=(testcase&& other) = delete;
    /**
     * @brief This is called before each test run.
     */
    virtual void
    set_up()
    {}
    /**
     * @brief This is called after each test run.
     */
    virtual void
    tear_down()
    {}
    /**
     * @brief Returns a pointer to the optional test context
     * @returns A pointer to the test context, nullptr if not defined
     */
    std::shared_ptr<context_type>
    get_test_context() const
    {
        return context_;
    }
    /**
     * @brief Returns the current test ID
     * @returns The current test ID
     */
    std::string
    get_test_id() const
    {
        return test_id_;
    }

private:

    template<typename T>
    friend struct unittest::core::testfunctor;

    void
    set_test_context(std::shared_ptr<context_type> context)
    {
        context_ = context;
    }

    void
    set_test_id(const std::string& test_id)
    {
        test_id_ = test_id;
    }

    std::shared_ptr<context_type> context_;
    std::string test_id_;
};

} // unittest
