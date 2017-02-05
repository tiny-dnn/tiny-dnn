/**
 * @brief Functionality to run tests
 * @file testrun.hpp
 */
#pragma once
#include "testfailure.hpp"
#include "testsuite.hpp"
#include "utilities.hpp"
#include "testcase.hpp"
#include <string>
#include <stdexcept>
#include <thread>
#include <functional>
/**
 * @brief Unit testing in C++
 */
namespace unittest {
/**
 * @brief Internal functionality, not relevant for most users
 */
namespace core {
/**
 * @brief The test monitor logs information about a single test
 */
class testmonitor {
public:
    /**
     * @brief Constructor
     * @param class_name The name of the test class
     * @param test_name The name of the current test method
     * @param method_id The ID of the current test method
     */
    testmonitor(const std::string& class_name,
                const std::string& test_name,
                const std::string& method_id);
    /**
     * @brief Destructor
     */
    virtual
    ~testmonitor();
    /**
     * @brief Copy constructor. Deleted
     * @param other An instance of testmonitor
     */
    testmonitor(const testmonitor& other) = delete;
    /**
     * @brief Copy assignment operator. Deleted
     * @param other An instance of testmonitor
     * @returns An testmonitor instance
     */
    testmonitor&
    operator=(const testmonitor& other) = delete;
    /**
     * @brief Move constructor. Deleted
     * @param other An instance of testmonitor
     */
    testmonitor(testmonitor&& other) = delete;
    /**
     * @brief Move assignment operator. Deleted
     * @param other An instance of testmonitor
     * @returns An testmonitor instance
     */
    testmonitor&
    operator=(testmonitor&& other) = delete;
    /**
     * @brief Returns whether to execute the current test
     * @returns Whether to execute the current test
     */
    bool
    is_executed();
    /**
     * @brief Logs success for the current test
     */
    void
    log_success();
    /**
     * @brief Logs skipped for the current test
     */
    void
    log_skipped(const std::string& message);
    /**
     * @brief Logs failure for the current test
     * @param e The failure exception
     */
    void
    log_failure(const unittest::testfailure& e);
    /**
     * @brief Logs error for the current test
     * @param e The error exception
     */
    void
    log_error(const std::exception& e);
    /**
     * @brief Logs unknown error for the current test
     */
    void
    log_unknown_error();
    /**
     * @brief Sets a certain timeout in seconds
     * @param timeout The timeout in seconds
     */
    void
    has_timed_out(double timeout);

private:
    struct impl;
    std::unique_ptr<impl> impl_;
};
/**
 * @brief Returns a unique method id
 * @param class_id The id of the test class
 * @param test_name The name of the test method
 * @returns A unique method id
 */
std::string
make_method_id(const std::string& class_id, const std::string& test_name);
/**
 * @brief The test info
 */
struct testinfo {
    /**
     * @brief The id of the test method
     */
    std::string method_id;
    /**
     * @brief The name of the test class
     */
    std::string class_name;
    /**
     * @brief The name of the current test method
     */
    std::string test_name;
    /**
     * @brief Whether a dry run is performed
     */
    bool dry_run;
    /**
     * @brief Whether to handle unexpected exceptions
     */
    bool handle_exceptions;
    /**
     * @brief Whether the test is done
     */
    std::shared_ptr<std::atomic_bool> done;
    /**
     * @brief @brief Whether the test has timed out
     */
    std::shared_ptr<std::atomic_bool> has_timed_out;
    /**
     * The test timeout
     */
    double timeout;
    /**
     * @brief Whether the current test is skipped
     */
    bool skipped;
    /**
     * @brief A message explaining why the test is skipped
     */
    std::string skip_message;
};
/**
 * @brief Creates the test info object
 * @param class_id The id of the test class
 * @param test_name The name of the current test method
 * @param skipped Whether this test run is skipped
 * @param skip_message A message explaining why the test is skipped
 * @param timeout The maximum allowed run time in seconds (ignored if <= 0)
 * @returns The test info object
 */
unittest::core::testinfo
make_testinfo(std::string class_id,
              std::string test_name,
              bool skipped,
              std::string skip_message,
              double timeout);
/**
 * @brief Observes the progress of an asynchronous operation and waits until
 *  the operation has finished or timed out
 * @param thread The asynchronous operation
 * @param done Whether the operation is finished
 * @param has_timed_out Whether the test has timed out
 * @param timeout The maximum allowed run time in seconds (ignored if <= 0)
 */
void
observe_and_wait(std::thread&& thread,
                 std::shared_ptr<std::atomic_bool> done,
                 std::shared_ptr<std::atomic_bool> has_timed_out,
                 double timeout);
/**
 * @brief Stores the test to be run and an optional test context.
 *  By using the ()-operator the test is executed.
 */
template<typename TestContext>
struct testfunctor {
    /**
     * @brief Constructor
     * @param context The test context, can be a nullptr
     * @param constructor A callback constructing the test class
     * @param caller A callback executing the test method given the test class
     * @param info The test info
     */
    testfunctor(std::shared_ptr<TestContext> context,
                std::function<unittest::testcase<TestContext>*()> constructor,
                std::function<void(unittest::testcase<TestContext>*)> caller,
                unittest::core::testinfo info)
        : context_(context),
          constructor_(constructor),
          caller_(caller),
          info_(std::move(info))
    {}
    /**
     * @brief Destructor
     */
    virtual
    ~testfunctor()
    {}
    /**
     * @brief Returns the test info
     * @returns The test info
     */
    const unittest::core::testinfo&
    info() const
    {
        return info_;
    }
    /**
     * @brief Executes the test
     */
    void
    operator()()
    {
        unittest::core::testmonitor monitor(info_.class_name, info_.test_name, info_.method_id);
        if (info_.skipped) {
            monitor.log_skipped(info_.skip_message);
        } else if (monitor.is_executed()) {
            if (info_.dry_run) {
                monitor.log_success();
            } else {
                unittest::testcase<TestContext>* test = nullptr;
                this->run(test, monitor);
                if (info_.has_timed_out->load())
                    monitor.has_timed_out(info_.timeout);
            }
        }
        info_.done->store(true);
    }

private:

    void
    run(unittest::testcase<TestContext>*& test,
        unittest::core::testmonitor& monitor)
    {
        if (this->construct(test, monitor)) {
            if (this->set_up(test, monitor)) {
                if (this->execute(test, monitor)) {
                    if (this->tear_down(test, monitor)) {
                    	this->destruct(test, monitor);
                    }
                }
            }
        }
    }

    bool
    do_nothing(unittest::testcase<TestContext>*&,
               unittest::core::testmonitor&)
    {
        return true;
    }

    bool
    handle(unittest::testcase<TestContext>*& test,
           unittest::core::testmonitor& monitor,
           bool (testfunctor::*function)(unittest::testcase<TestContext>*&, unittest::core::testmonitor&),
           bool (testfunctor::*error_callback)(unittest::testcase<TestContext>*&, unittest::core::testmonitor&))
    {
        if (info_.handle_exceptions) {
            try {
                (this->*function)(test, monitor);
                monitor.log_success();
                return true;
            } catch (const unittest::testfailure& e) {
                (this->*error_callback)(test, monitor);
                monitor.log_failure(e);
                return false;
            } catch (const std::exception& e) {
                (this->*error_callback)(test, monitor);
                monitor.log_error(e);
                return false;
            } catch (...) {
                (this->*error_callback)(test, monitor);
                monitor.log_unknown_error();
                return false;
            }
        } else {
            try {
                (this->*function)(test, monitor);
                monitor.log_success();
                return true;
            } catch (const unittest::testfailure& e) {
                (this->*error_callback)(test, monitor);
                monitor.log_failure(e);
                return false;
            }
        }
    }

    bool
    construct(unittest::testcase<TestContext>*& test,
              unittest::core::testmonitor& monitor)
    {
        return handle(test, monitor,
                      &testfunctor::_construct,
                      &testfunctor::destruct);
    }

    bool
    _construct(unittest::testcase<TestContext>*& test,
               unittest::core::testmonitor&)
    {
        test = constructor_();
        test->set_test_context(context_);
        test->set_test_id(info_.method_id);
        return true;
    }

    bool
    set_up(unittest::testcase<TestContext>*& test,
           unittest::core::testmonitor& monitor)
    {
        return this->handle(test, monitor,
                            &testfunctor::_set_up,
                            &testfunctor::tear_down);
    }

    bool
    _set_up(unittest::testcase<TestContext>*& test,
            unittest::core::testmonitor&)
    {
        test->set_up();
        return true;
    }

    bool
    execute(unittest::testcase<TestContext>*& test,
            unittest::core::testmonitor& monitor)
    {
        return this->handle(test, monitor,
                            &testfunctor::_execute,
                            &testfunctor::tear_down);
    }

    bool
    _execute(unittest::testcase<TestContext>*& test,
             unittest::core::testmonitor&)
    {
    	  caller_(test);
        return true;
    }

    bool
    tear_down(unittest::testcase<TestContext>*& test,
              unittest::core::testmonitor& monitor)
    {
        return this->handle(test, monitor,
                            &testfunctor::_tear_down,
                            &testfunctor::destruct);
    }

    bool
    _tear_down(unittest::testcase<TestContext>*& test,
               unittest::core::testmonitor&)
    {
        test->tear_down();
        return true;
    }

    bool
    destruct(unittest::testcase<TestContext>*& test,
             unittest::core::testmonitor& monitor)
    {
        return this->handle(test, monitor,
                            &testfunctor::_destruct,
                            &testfunctor::do_nothing);
    }

    bool
    _destruct(unittest::testcase<TestContext>*& test,
              unittest::core::testmonitor&)
    {
    	  delete test;
        return true;
    }

    std::shared_ptr<TestContext> context_;
    std::function<unittest::testcase<TestContext>*()> constructor_;
    std::function<void(unittest::testcase<TestContext>*)> caller_;
    const unittest::core::testinfo info_;
};
/**
 * @brief Runs the test functor (provides actual implementation)
 * @param context The test context, can be a nullptr
 * @param constructor A callback constructing the test class
 * @param caller A callback executing the test method given the test class
 * @param class_id The ID of the test class
 * @param test_name The name of the current test method
 * @param skipped Whether this test run is skipped
 * @param skip_message A message explaining why the test is skipped
 * @param timeout The maximum allowed run time in seconds (ignored if <= 0)
 */
template<typename TestContext>
void run_testfunctor_impl(std::shared_ptr<TestContext> context,
                          std::function<unittest::testcase<TestContext>*()> constructor,
                          std::function<void(unittest::testcase<TestContext>*)> caller,
                          const std::string& class_id,
                          const std::string& test_name,
                          bool skipped,
                          const std::string& skip_message,
                          double timeout)
{
    unittest::core::testfunctor<TestContext> functor(context, constructor, caller,
                    unittest::core::make_testinfo(class_id, test_name, skipped, skip_message, timeout));
    const double updated_timeout = functor.info().timeout;
    if (updated_timeout > 0) {
        std::shared_ptr<std::atomic_bool> done = functor.info().done;
        std::shared_ptr<std::atomic_bool> has_timed_out = functor.info().has_timed_out;
        std::thread thread(functor);
        unittest::core::observe_and_wait(std::move(thread), done, has_timed_out, updated_timeout);
    } else {
        functor();
    }
}
/**
 * @brief Runs the test functor
 * @param context The test context, can be a nullptr
 * @param constructor A callback constructing the test class
 * @param caller A callback executing the test method given the test class
 * @param class_id The ID of the test class
 * @param test_name The name of the current test method
 * @param skipped Whether this test run is skipped
 * @param skip_message A message explaining why the test is skipped
 * @param timeout The maximum allowed run time in seconds (ignored if <= 0)
 */
template<typename TestContext>
void run_testfunctor(std::shared_ptr<TestContext> context,
                     std::function<unittest::testcase<TestContext>*()> constructor,
                     std::function<void(unittest::testcase<TestContext>*)> caller,
                     const std::string& class_id,
                     const std::string& test_name,
                     bool skipped,
                     const std::string& skip_message,
                     double timeout)
{
    unittest::core::run_testfunctor_impl<TestContext>(context, constructor, caller, class_id, test_name, skipped, skip_message, timeout);
}
/**
 * @brief A typedef for the default context type
 */
typedef typename unittest::testcase<>::context_type def_context_type;
/**
 * @brief Runs the test functor. Spec. for the default context type
 * @param context The test context, can be a nullptr
 * @param constructor A callback constructing the test class
 * @param caller A callback executing the test method given the test class
 * @param class_id The ID of the test class
 * @param test_name The name of the current test method
 * @param skipped Whether this test run is skipped
 * @param skip_message A message explaining why the test is skipped
 * @param timeout The maximum allowed run time in seconds (ignored if <= 0)
 */
template<>
void run_testfunctor<def_context_type>(std::shared_ptr<def_context_type> context,
                                       std::function<unittest::testcase<def_context_type>*()> constructor,
                                       std::function<void(unittest::testcase<def_context_type>*)> caller,
                                       const std::string& class_id,
                                       const std::string& test_name,
                                       bool skipped,
                                       const std::string& skip_message,
                                       double timeout);
/**
 * @brief Updates the local timeout by assigning the global timeout
 *  from the test suite if the local one is not greater than zero
 * @param local_timeout The local timeout in seconds
 * @param global_timeout The global timeout in seconds
 */
void
update_local_timeout(double& local_timeout,
                     double global_timeout);
/**
 * @brief Updates information needed or a test run
 * @param class_id The current class' type ID
 * @param class_name The current class name
 * @param test_name The current test name
 * @param local_timeout The local timeout in seconds
 */
void
update_testrun_info(const std::string& class_id,
                    std::string& class_name,
                    std::string& test_name,
                    double& local_timeout);

} // core

/**
 * @brief A test run with a test context and with timeout measurement
 * @param context The test context, can be a nullptr
 * @param method A pointer to the method to be run
 * @param test_name The name of the current test method
 * @param skipped Whether this test run is skipped
 * @param skip_message A message explaining why the test is skipped
 * @param timeout The maximum allowed run time in seconds (ignored if <= 0)
 */
template<typename TestCase>
void
testrun(std::shared_ptr<typename TestCase::context_type> context,
        void (TestCase::*method)(),
        std::string test_name,
        bool skipped,
        std::string skip_message,
        double timeout)
{
    typedef typename TestCase::context_type context_type;
    auto constructor = []() -> unittest::testcase<context_type>* {
        return new TestCase;
    };
    auto caller = [&method](unittest::testcase<context_type>* test_class) {
        (dynamic_cast<TestCase*>(test_class)->*method)();
    };
    const std::string class_id = unittest::core::get_type_id<TestCase>();
    unittest::core::run_testfunctor<context_type>(context, constructor, caller, class_id, test_name, skipped, skip_message, timeout);
}
/**
 * @brief A test run with a test context and without timeout measurement
 * @param context The test context, can be a nullptr
 * @param method A pointer to the method to be run
 * @param test_name The name of the current test method
 * @param skipped Whether this test run is skipped
 * @param skip_message A message explaining why the test is skipped
 */
template<typename TestCase>
void
testrun(std::shared_ptr<typename TestCase::context_type> context,
        void (TestCase::*method)(),
        std::string test_name,
        bool skipped,
        std::string skip_message)
{
    unittest::testrun(context, method, test_name, skipped, skip_message, -1.);
}

} // unittest
