#include "libunittest/testcollection.hpp"

namespace unittest {
namespace core {

testcollection::testcollection()
{}

testcollection::~testcollection()
{}

std::string
testcollection::get_name() const
{
    return inactive_name();
}

std::string
testcollection::inactive_name()
{
    return "__inactive_collection__";
}

} // core
} // unittest
