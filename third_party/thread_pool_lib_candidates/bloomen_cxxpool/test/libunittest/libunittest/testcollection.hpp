/**
 * @brief A test collection
 * @file testcollection.hpp
 */
#pragma once
#include <string>
/**
 * @brief Unit testing in C++
 */
namespace unittest {
/**
 * @brief Internal functionality, not relevant for most users
 */
namespace core {
/**
 * @brief A test collection
 */
class testcollection {
public:
    /**
     * @brief Constructor
     */
    testcollection();
    /**
     * @brief Destructor
     */
    virtual
    ~testcollection();
    /**
     * @brief Copy constructor. Deleted
     * @param other An instance of testcollection
     */
    testcollection(const testcollection& other) = delete;
    /**
     * @brief Copy assignment operator. Deleted
     * @param other An instance of testcollection
     * @returns An testcollection instance
     */
    testcollection&
    operator=(const testcollection& other) = delete;
    /**
     * @brief Move constructor. Deleted
     * @param other An instance of testcollection
     */
    testcollection(testcollection&& other) = delete;
    /**
     * @brief Move assignment operator. Deleted
     * @param other An instance of testcollection
     * @returns An testcollection instance
     */
    testcollection&
    operator=(testcollection&& other) = delete;
    /**
     * @brief Returns the collection name
     * @returns The collection name
     */
    virtual std::string
    get_name() const;
    /**
     * @brief Returns the name that is returned by default by get_name()
     * @returns The name that is returned by default by get_name()
     */
    static std::string
    inactive_name();

};

} // core
} // unittest

/**
 * @brief The test collection type
 */
typedef unittest::core::testcollection __testcollection_type__;
