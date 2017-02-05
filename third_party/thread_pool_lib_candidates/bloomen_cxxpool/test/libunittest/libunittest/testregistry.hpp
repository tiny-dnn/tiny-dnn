/**
 * @brief Registering test runs
 * @file testregistry.hpp
 */
#pragma once
#include "testsuite.hpp"
#include "utilities.hpp"
#include "testcollection.hpp"
/**
 * @brief Unit testing in C++
 */
namespace unittest {
/**
 * @brief Internal functionality, not relevant for most users
 */
namespace core {
/**
 * @brief Registers a test class at the testsuite
 * @param class_name The name of the test class
 */
template<typename TestCase>
void
register_class(const std::string& class_name)
{
    auto suite = unittest::core::testsuite::instance();
    suite->add_class_run(TestCase::run);
    suite->add_class_map(unittest::core::get_type_id<TestCase>(), class_name);
}
/**
 * @brief Registers a test class at the testsuite
 */
template<typename TestCase>
class testregistry {
public:
    /**
     * @brief Constructor
     * @param collection The test collection
     * @param class_name The name of the test class
     */
    testregistry(const unittest::core::testcollection& collection, std::string class_name)
    {
        if (collection.get_name() != testcollection::inactive_name())
            class_name = collection.get_name() + "::" + class_name;
        unittest::core::register_class<TestCase>(class_name);
    }
};

} // core
} // unittest
