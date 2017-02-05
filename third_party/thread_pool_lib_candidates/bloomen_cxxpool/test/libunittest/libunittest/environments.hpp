/**
 * @brief A collection of functions to process test environments
 * @file environments.hpp
 */
#pragma once
/**
 * @brief Unit testing in C++
 */
namespace unittest {
/**
 * @brief Processes the default testing environment and runs the tests.
 *  This high-level function is intended to be used directly from the
 *  user's main() function
 * @param argc The number of user arguments
 * @param argv The array of user arguments
 * @returns A platform dependent program execution status: A success value if
 *  all tests succeed and an error value otherwise
 */
int
process(int argc, char **argv);

} // unittest
