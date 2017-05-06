#include "libunittest/testfailure.hpp"
#include "libunittest/utilities.hpp"


namespace unittest {

testfailure::testfailure(const std::string& assertion,
                         const std::string& message)
    : std::runtime_error(message),
      error_msg_(message),
      assertion_(assertion),
      spot_(std::make_pair("", -1)),
      callsite_()
{}

testfailure::testfailure(const std::string& assertion,
                         const std::string& message,
                         const std::string& user_msg)
    : std::runtime_error(make_error_msg(message, user_msg)),
      error_msg_(make_error_msg(message, user_msg)),
      assertion_(assertion),
      spot_(core::extract_file_and_line(user_msg)),
      callsite_(core::extract_tagged_text(user_msg, "CALL"))
{}

testfailure::testfailure(const testfailure& other)
    : std::runtime_error(other.error_msg_),
      error_msg_(other.error_msg_),
      assertion_(other.assertion_),
      spot_(other.spot_),
      callsite_(other.callsite_)
{}

testfailure&
testfailure::operator=(const testfailure& other)
{
    if (this!=&other) {
        error_msg_ = other.error_msg_;
        assertion_ = other.assertion_;
        spot_ = other.spot_;
        callsite_ = other.callsite_;
    }
    return *this;
}

testfailure::~testfailure() UNITTEST_NOEXCEPT
{}

std::string testfailure::make_error_msg(const std::string& message,
                                        const std::string& user_msg)
{
    auto msg = core::remove_tagged_text(user_msg, "SPOT");
    msg = core::remove_tagged_text(msg, "NDAS");
    msg = core::remove_tagged_text(msg, "CALL");
    return msg.size() ? message + " - " + msg : message;
}

std::string
testfailure::assertion() const
{
    return assertion_;
}

std::string
testfailure::filename() const
{
    return spot_.first;
}

int
testfailure::linenumber() const
{
    return spot_.second;
}

std::string
testfailure::callsite() const
{
    return callsite_;
}

namespace core {

void
fail_impl(const std::string& assertion,
          const std::string& message,
          std::string usermsg)
{
    const auto ndas_test_id = unittest::core::extract_tagged_text(usermsg, "NDAS");
    const unittest::testfailure failure(assertion, message, usermsg);
    if (ndas_test_id.empty()) {
        throw failure;
    } else {
        unittest::core::testsuite::instance()->log_failure(ndas_test_id, failure);
    }
}

} // core
} // unittest
