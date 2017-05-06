/**
 * @brief The test suite collecting test information
 * @file testsuite.hpp
 */
#pragma once
#include "userargs.hpp"
#include "testresults.hpp"
#include "testlog.hpp"
#include <string>
#include <thread>
#include <map>
#include <atomic>
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
 * @brief The test suite collecting test information (singleton, thread-safe)
 */
class testsuite {
public:
    /**
     * @brief Returns a pointer to the instance of this class
     * @returns A pointer to the instance of this class
     */
    static testsuite*
    instance();
    /**
     * @brief Destructor
     */
    virtual
    ~testsuite();
    /**
     * @brief Copy constructor. Deleted
     * @param other An instance of testsuite
     */
    testsuite(const testsuite& other) = delete;
    /**
     * @brief Copy assignment operator. Deleted
     * @param other An instance of testsuite
     * @returns An testsuite instance
     */
    testsuite&
    operator=(const testsuite& other) = delete;
    /**
     * @brief Move constructor. Deleted
     * @param other An instance of testsuite
     */
    testsuite(testsuite&& other) = delete;
    /**
     * @brief Move assignment operator. Deleted
     * @param other An instance of testsuite
     * @returns An testsuite instance
     */
    testsuite&
    operator=(testsuite&& other) = delete;
    /**
     * @brief Assigns the user arguments
     * @param arguments The user arguments
     */
    void
    set_arguments(const unittest::core::userargs& arguments);
    /**
     * @brief Returns a reference to the user arguments
     * @returns A reference to the user arguments
     */
    const unittest::core::userargs&
    get_arguments() const;
    /**
     * @brief Returns a reference to the vector of registered class runs
     * @returns A reference to the vector of registered class runs
     */
    const std::vector<std::function<void()>>&
    get_class_runs() const;
    /**
     * @brief Returns a reference to the mappings from test class IDs to
     *  test class names
     * @returns A reference to mappings from test class IDs to test class names
     */
    const std::map<std::string, std::string>&
    get_class_maps() const;
    /**
     * @brief Returns a reference to the vector of lonely threads
     * @returns A reference to the vector of lonely threads
     */
    std::vector<std::pair<std::thread, std::shared_ptr<std::atomic_bool>>>&
    get_lonely_threads() const;
    /**
     * @brief Returns the accumulated test results
     * @returns The test results
     */
    unittest::core::testresults
    get_results() const;
    /**
     * @brief Logs text for a given method
     * @param method_id The method ID
     * @param text The text to be logged
     */
    void
    log_text(const std::string& method_id,
             const std::string& text);
    /**
     * @brief Logs a failure for a given method
     * @param method_id The method ID
     * @param failure The failure
     */
    void
    log_failure(const std::string& method_id,
                const unittest::testfailure& failure);

private:

    testsuite();

    friend class testmonitor;

    template<typename TestCase>
    friend void
    register_class(const std::string& class_name);

    friend void
    observe_and_wait(std::thread&& thread,
                     std::shared_ptr<std::atomic_bool> done,
                     std::shared_ptr<std::atomic_bool> has_timed_out,
                     double timeout);

    void
    make_keep_running(const unittest::core::testlog& log);

    void
    start_timing();

    void
    stop_timing();

    void
    collect(const unittest::core::testlog& log);

    bool
    is_test_run(const std::string& class_name,
                const std::string& test_name) const;

    void
    add_class_run(const std::function<void()>& class_run);

    void
    add_class_map(const std::string& typeid_name,
                  const std::string& class_name);

    void
    add_lonely_thread(std::thread&& thread,
                      std::shared_ptr<std::atomic_bool> done);

    std::vector<unittest::testfailure>
    get_failures(const std::string& method_id);

    struct impl;
    std::unique_ptr<impl> impl_;
};

} // core

/**
 * @brief Exception class to indicate errors emitted from the testsuite
 */
class testsuite_error : public std::runtime_error {
public:
    /**
     * @brief Constructor
     * @param message The exception message
     */
    testsuite_error(const std::string& message);
};

} // unittest
