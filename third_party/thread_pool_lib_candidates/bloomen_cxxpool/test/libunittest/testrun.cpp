#include "libunittest/testrun.hpp"
#include "libunittest/utilities.hpp"
#include "libunittest/teststatus.hpp"
#include "libunittest/testlog.hpp"
#include "libunittest/testcollection.hpp"
#include <chrono>
#include <iostream>
#include <typeinfo>

namespace unittest {
namespace core {

void
observe_and_wait(std::thread&& thread,
                 std::shared_ptr<std::atomic_bool> done,
                 std::shared_ptr<std::atomic_bool> has_timed_out,
                 double timeout)
{
    if (!done->load() && timeout > 0) {
        const std::chrono::milliseconds min_resolution(2);
        volatile double overhead = 0.123;
        volatile double duration = -1.;

        // compute approx. overhead of instructions
        const auto start_instr = unittest::core::now();
        if (!done->load()) {
            if (duration < timeout) {
                std::this_thread::sleep_for(min_resolution);
                duration += timeout;
            }
        }
        overhead = duration_in_seconds(unittest::core::now() - start_instr);
        duration = overhead;

        while (!done->load()) {
            if (duration > timeout) {
                has_timed_out->store(true);
                auto suite = testsuite::instance();
                write_test_timeout_message(std::cout, suite->get_arguments().verbose);
                suite->add_lonely_thread(std::move(thread), done);
                break;
            }
            std::this_thread::sleep_for(min_resolution);
            duration += overhead;
        }
    }
    if (!has_timed_out->load())
        thread.join();
}

struct testmonitor::impl {

    testlog log_;
    std::chrono::microseconds start_;
    bool is_executed_;

    impl()
        : log_(),
          start_(std::chrono::microseconds::min()),
          is_executed_(true)
    {}

};

testmonitor::testmonitor(const std::string& class_name,
                         const std::string& test_name,
                         const std::string& method_id)
    : impl_(make_unique<impl>())
{
    auto suite = testsuite::instance();
    impl_->is_executed_ = suite->is_test_run(class_name, test_name);
    impl_->log_.class_name = class_name;
    impl_->log_.test_name = test_name;
    impl_->log_.method_id = method_id;
    if (impl_->is_executed_) {
        suite->start_timing();
        impl_->start_ = unittest::core::now();
        write_test_start_message(std::cout, impl_->log_, suite->get_arguments().verbose);
    }
}

testmonitor::~testmonitor()
{
    auto suite = testsuite::instance();
    if (impl_->is_executed_) {
        if (!suite->get_arguments().dry_run) {
            suite->stop_timing();
            if (impl_->log_.status!=teststatus::skipped) {
                const auto failures = suite->get_failures(impl_->log_.method_id);
                if (failures.size()) {
                    if (impl_->log_.assertion.empty()) {
                        impl_->log_.assertion = failures[failures.size()-1].assertion();
                        if (impl_->log_.filename.empty()) {
                            impl_->log_.filename = failures[failures.size()-1].filename();
                            impl_->log_.linenumber = failures[failures.size()-1].linenumber();
                        }
                    }
                    impl_->log_.nd_failures = failures;
                    impl_->log_.status = teststatus::failure;
                }
                impl_->log_.duration = duration_in_seconds(unittest::core::now() - impl_->start_);
            }
        }
        suite->make_keep_running(impl_->log_);
        impl_->log_.successful = impl_->log_.status==teststatus::success ||
                                 impl_->log_.status==teststatus::skipped;
        write_test_end_message(std::cout, impl_->log_, suite->get_arguments().verbose);
    }
    suite->collect(impl_->log_);
}

bool
testmonitor::is_executed()
{
    return impl_->is_executed_;
}

void
testmonitor::log_success()
{
    impl_->log_.status = teststatus::success;
    impl_->log_.message = "ok";
    impl_->log_.error_type = "";
}

void
testmonitor::log_skipped(const std::string& message)
{
    impl_->log_.status = teststatus::skipped;
    impl_->log_.text = message;
    impl_->log_.error_type = "";
}

void
testmonitor::log_failure(const testfailure& e)
{
    impl_->log_.status = teststatus::failure;
    impl_->log_.message = e.what();
    impl_->log_.error_type = "testfailure";
    impl_->log_.assertion = e.assertion();
    impl_->log_.filename = e.filename();
    impl_->log_.linenumber = e.linenumber();
    impl_->log_.callsite = e.callsite();
}

void
testmonitor::log_error(const std::exception& e)
{
    impl_->log_.status = teststatus::error;
    impl_->log_.message = e.what();
    impl_->log_.error_type = type_name(e);
}
void
testmonitor::log_unknown_error()
{
    impl_->log_.status = teststatus::error;
    impl_->log_.message = "Unknown message";
    impl_->log_.error_type = "Unknown exception";
}

void
testmonitor::has_timed_out(double timeout)
{
    impl_->log_.has_timed_out = true;
    impl_->log_.timeout = timeout;
}

void
update_local_timeout(double& local_timeout,
                     double global_timeout)
{
    if (!(local_timeout > 0))
        local_timeout = global_timeout;
}

void
update_testrun_info(const std::string& class_id,
                    std::string& class_name,
                    std::string& test_name,
                    double& local_timeout)
{
    auto suite = core::testsuite::instance();
    const auto& class_maps = suite->get_class_maps();
    class_name = get_from_map(class_maps, class_id);
    class_name = remove_white_spaces(class_name);
    test_name = remove_white_spaces(test_name);
    if (suite->get_arguments().disable_timeout) {
        local_timeout = -1;
    } else {
        const double global_timeout = suite->get_arguments().timeout;
        update_local_timeout(local_timeout, global_timeout);
    }
}

std::string
make_method_id(const std::string& class_id, const std::string& test_name)
{
    return class_id + test_name;
}

unittest::core::testinfo
make_testinfo(std::string class_id,
              std::string test_name,
              bool skipped,
              std::string skip_message,
              double timeout)
{
    std::string class_name;
    unittest::core::update_testrun_info(class_id, class_name, test_name, timeout);
    const unittest::core::userargs& args = unittest::core::testsuite::instance()->get_arguments();
    auto done = std::make_shared<std::atomic_bool>();
    done->store(false);
    auto has_timed_out = std::make_shared<std::atomic_bool>();
    has_timed_out->store(false);
    return {make_method_id(class_id, test_name), class_name,
            test_name, args.dry_run, args.handle_exceptions,
            done, has_timed_out, timeout, skipped && !args.ignore_skips, skip_message};
}

void run_testfunction(const unittest::core::testinfo& info,
                      unittest::core::testmonitor& monitor,
                      std::function<void()> function)
{
    if (info.skipped) {
        monitor.log_skipped(info.skip_message);
    } else if (monitor.is_executed()) {
        if (info.dry_run) {
            monitor.log_success();
        } else {
            function();
            if (info.has_timed_out->load())
                monitor.has_timed_out(info.timeout);
        }
    }
    info.done->store(true);
}

template<>
void run_testfunctor<def_context_type>(std::shared_ptr<def_context_type> context,
                                       std::function<unittest::testcase<def_context_type>*()> constructor,
                                       std::function<void(unittest::testcase<def_context_type>*)> caller,
                                       const std::string& class_id,
                                       const std::string& test_name,
                                       bool skipped,
                                       const std::string& skip_message,
                                       double timeout)
{
    run_testfunctor_impl<def_context_type>(context, constructor, caller, class_id, test_name, skipped, skip_message, timeout);
}

} // core
} // unittest
