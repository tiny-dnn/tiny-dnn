#include "libunittest/testresults.hpp"
#include "libunittest/teststatus.hpp"
#include "libunittest/utilities.hpp"
#include "libunittest/testfailure.hpp"
#include <mutex>

namespace unittest {
namespace core {

testresults::testresults()
    : successful(true), n_tests(0), n_successes(0), n_failures(0),
      n_errors(0), n_skipped(0), n_timeouts(0), duration(0), testlogs(0)
{}

void
write_failure_info(std::ostream& stream,
                   const std::string& assertion,
                   const std::string& filename,
                   int linenumber,
                   const std::string& callsite,
                   std::string indent="")
{
    if (!assertion.empty()) {
        stream << indent << "assertion: " << assertion;
        if (filename.size()) {
            stream << " in " << trim(filename);
            if (linenumber>-1)
                stream << " at line " << linenumber;
        }
    }
    if (!callsite.empty()) {
        stream << '\n';
        stream << indent << "calledwith: " << callsite;
    }
}

void
write_xml(std::ostream& stream,
          const testresults& results,
          const std::string& suite_name,
          long long shuffle_seed,
          const std::chrono::system_clock::time_point& time_point,
          bool local_time)
{
    static std::mutex write_xml_mutex_;
    std::lock_guard<std::mutex> lock(write_xml_mutex_);
    stream.setf(std::ios_base::fixed);
    stream << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>";
    stream << "\n";
    stream << "<testsuite name=\"" << xml_escape(trim(suite_name));
    stream << "\" timestamp=\"" << make_iso_timestamp(time_point, local_time);
    stream << "\" tests=\"" << results.n_tests + results.n_skipped;
    stream << "\" errors=\"" << results.n_errors;
    stream << "\" failures=\"" << results.n_failures;
    stream << "\" timeouts=\"" << results.n_timeouts;
    stream << "\" skipped=\"" << results.n_skipped;
    if (shuffle_seed >= 0)
        stream << "\" shuffseed=\"" << shuffle_seed;
    stream << "\" time=\"" << results.duration << "\">";
    stream << "\n";
    for (const auto& log : results.testlogs) {
        std::string system_out;
        if (log.text.size()) {
            system_out += "\t\t<system-out>\n";
            system_out += "\t\t\t" + xml_escape(trim(log.text)) + "\n";
            system_out += "\t\t</system-out>";
        }
        stream << "\t<testcase ";
        if (log.class_name.size())
            stream << "classname=\"" << xml_escape(log.class_name) << "\" ";
        stream << "name=\"" << xml_escape(log.test_name);
        if (log.assertion.size())
            stream << "\" assertions=\"" << xml_escape(log.assertion);
        if (log.filename.size())
            stream << "\" file=\"" << xml_escape(log.filename);
        if (log.linenumber>-1)
            stream << "\" line=\"" << log.linenumber;
        stream << "\" time=\"" << log.duration << "\"";
        if (log.has_timed_out)
            stream << " timeout=\"" << log.timeout << "\"";
        if (log.successful) {
            if (log.status==teststatus::skipped) {
                stream << ">";
                stream << "\n";
                stream << "\t\t<skipped/>";
                stream << "\n";
                stream << system_out;
                if (system_out.size())
                    stream << "\n";
                stream << "\t</testcase>";
            } else {
                if (system_out.size()) {
                    stream << ">\n";
                    stream << system_out << "\n";
                    stream << "\t</testcase>";
                } else {
                    stream << "/>";
                }
            }
            stream << "\n";
        } else {
            stream << ">";
            for (const auto& failure : log.nd_failures) {
                stream << "\n";
                stream << "\t\t<" << "failure" << " ";
                stream << "type=\"" << xml_escape("testfailure (non-deadly)");
                stream << "\" message=\"" << xml_escape(trim(failure.what())) << "\">";
                if (failure.assertion().size()) {
                    stream << "\n";
                    write_failure_info(stream, xml_escape(trim(failure.assertion())), xml_escape(trim(failure.filename())), failure.linenumber(), xml_escape(trim(failure.callsite())), "\t\t\t");
                }
                stream << "\n\t\t</" << "failure" << ">";
            }
            if (!log.error_type.empty()) {
                stream << "\n";
                std::string name("error");
                if (log.error_type=="testfailure")
                    name = "failure";
                stream << "\t\t<" << name << " ";
                stream << "type=\"" << xml_escape(log.error_type);
                stream << "\" message=\"" << xml_escape(trim(log.message)) << "\">";
                if (log.error_type=="testfailure" && !log.assertion.empty()) {
                    stream << "\n";
                    write_failure_info(stream, xml_escape(trim(log.assertion)), xml_escape(trim(log.filename)), log.linenumber, xml_escape(trim(log.callsite)), "\t\t\t");
                }
                stream << "\n";
                stream << "\t\t</" << name << ">";
                if (system_out.size())
                    stream << "\n";
                stream << system_out;
            }
            stream << "\n";
            stream << "\t</testcase>";
            stream << "\n";
        }
    }
    stream << "</testsuite>";
    stream << "\n";
    stream << std::flush;
    stream.unsetf(std::ios_base::fixed);
}

void
write_summary(std::ostream& stream,
              const testresults& results,
              long long shuffle_seed)
{
    static std::mutex write_summary_mutex_;
    std::lock_guard<std::mutex> lock(write_summary_mutex_);
    stream << "\n";
    write_horizontal_bar(stream, '-');
    stream << "\n";
    stream << "Ran " << results.n_tests << " tests in ";
    stream << results.duration << "s";
    if (shuffle_seed >= 0)
        stream << " (shuffseed: " << shuffle_seed << ")";
    stream << "\n\n";
    if (results.n_tests==results.n_successes) {
        stream << "OK";
        if (results.n_skipped>0 && results.n_timeouts>0) {
            stream << " (skipped=" << results.n_skipped;
            stream << ", timeouts=" << results.n_timeouts << ")";
        } else if (results.n_skipped>0) {
            stream << " (skipped=" << results.n_skipped << ")";
        } else if (results.n_timeouts>0) {
            stream << " (timeouts=" << results.n_timeouts << ")";
        }
        stream << "\n";
    } else {
        stream << "FAILED (";
        if (results.n_failures>0 && results.n_errors>0) {
            stream << "failures=" << results.n_failures;
            stream << ", errors=" << results.n_errors;
        } else if (results.n_failures>0) {
            stream << "failures=" << results.n_failures;
        } else if (results.n_errors>0) {
            stream << "errors=" << results.n_errors;
        }
        if (results.n_skipped>0)
            stream << ", skipped=" << results.n_skipped;
        if (results.n_timeouts>0)
            stream << ", timeouts=" << results.n_timeouts;
        stream << ")\n";
    }
    stream << std::flush;
}

void
write_error_info(std::ostream& stream,
                 const std::vector<testlog>& testlogs,
                 bool successful)
{
    static std::mutex write_error_info_mutex_;
    std::lock_guard<std::mutex> lock(write_error_info_mutex_);
    if (!successful) {
        stream << "\n";
        for (const auto& log : testlogs) {
            const auto status = log.status;
            if (status==teststatus::failure || status==teststatus::error) {
                write_horizontal_bar(stream, '=');
                stream << "\n";
                std::string flag("FAIL");
                if (status==teststatus::error)
                    flag = "ERROR";
                stream << flag << ": " << make_full_test_name(log.class_name, log.test_name);
                stream << " [" << log.duration << "s]";
                if (log.has_timed_out)
                    stream << " (TIMEOUT)";
                for (const auto& failure : log.nd_failures) {
                    stream << "\n";
                    write_horizontal_bar(stream, '-');
                    stream << "\n";
                    stream << "testfailure (non-deadly)" << ": " << trim(failure.what());
                    if (failure.assertion().size()) {
                        stream << "\n";
                        write_failure_info(stream, trim(failure.assertion()), trim(failure.filename()), failure.linenumber(), trim(failure.callsite()));
                    }
                }
                if (!log.error_type.empty()) {
                    stream << "\n";
                    write_horizontal_bar(stream, '-');
                    stream << "\n";
                    stream << log.error_type << ": " << trim(log.message);
                    if (log.error_type=="testfailure" && log.assertion.size()) {
                        stream << "\n";
                        write_failure_info(stream, trim(log.assertion), trim(log.filename), log.linenumber, trim(log.callsite));
                    }
                }
                if (log.text.size()) {
                    stream << "\n";
                    write_horizontal_bar(stream, '-');
                    stream << "\n";
                    stream << "INFO: " << trim(log.text);
                }
                stream << "\n\n";
            }
        }
        stream << std::flush;
    }
}

} // core
} // unittest
