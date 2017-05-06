/**
 * @brief A collection of useful strings
 * @file strings.hpp
 */
#pragma once
#include <string>
/**
 * @brief Unit testing in C++
 */
namespace unittest {
/**
 * @brief A collection of strings useful for testing
 */
namespace strings {
/**
 * @brief binary digits
 */
const std::string bin_digits = "01";
/**
 * @brief octal digits
 */
const std::string oct_digits = "01234567";
/**
 * @brief decimal digits
 */
const std::string dec_digits = "0123456789";
/**
 * @brief hexadecimal digits, mixed upper and lower case
 */
const std::string hex_digits = "0123456789abcdefABCDEF";
/**
 * @brief hexadecimal digits, lower case
 */
const std::string hex_digits_lower = "0123456789abcdef";
/**
 * @brief hexadecimal digits, upper case
 */
const std::string hex_digits_upper = "0123456789ABCDEF";
/**
 * @brief letters, lower case
 */
const std::string letters_lower = "abcdefghijklmnopqrstuvwxyz";
/**
 * @brief letters, upper case
 */
const std::string letters_upper = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
/**
 * @brief letters, mixed upper and lower case
 */
const std::string letters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";

} // strings
} // unittest
