/**
 * @brief A provider of quotes
 * @file quote.hpp
 */
#pragma once
#include <string>
#include <random>
/**
 * @brief Unit testing in C++
 */
namespace unittest {
namespace core {
/**
 * @brief A quote
 */
struct quote {
    /**
     * @brief The quote message
     */
    std::string message;
    /**
     * @brief The author of the quote
     */
    std::string author;
};
/**
 * @brief Output stream operator for quote
 * @param os The output stream
 * @param q The quote
 * @return A reference to the output stream
 */
std::ostream&
operator<<(std::ostream& os, const unittest::core::quote& q);
/**
 * @brief Generates random quotes
 */
class quote_generator {
public:
    /**
     * @brief Constructor
     * @param seed A random seed
     */
    explicit
    quote_generator(int seed);
    /**
     * @brief Returns the next quote
     * @return The next quote
     */
    unittest::core::quote
    next() const;

private:

    mutable std::mt19937 generator_;
    std::vector<unittest::core::quote> quotes_;
};


} // core
} // unittest
