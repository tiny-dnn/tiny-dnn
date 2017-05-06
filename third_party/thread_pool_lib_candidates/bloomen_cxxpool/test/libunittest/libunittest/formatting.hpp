/**
 * @brief Formatting functionality
 * @file formatting.hpp
 */
#pragma once
#include "testsuite.hpp"
#include <string>
#include <sstream>
#include <type_traits>
/**
 * @brief Unit testing in C++
 */
namespace unittest {
/**
 * @brief Internal functionality, not relevant for most users
 */
namespace core {
/**
 * @brief Converts an int to string
 */
struct tostr_converter_int {
    /**
     * @brief Converts an int to string
     * @param value The value to convert
     * @returns A string
     */
    template<typename T>
    std::string
    operator()(const T& value)
    {
        std::ostringstream stream;
        stream << value;
        return stream.str();
    }
};
/**
 * @brief Converts a float to string
 */
struct tostr_converter_float {
    /**
     * @brief Converts a float to string
     * @param value The value to convert
     * @returns A string
     */
    template<typename T>
    std::string
    operator()(const T& value)
    {
        std::ostringstream stream;
        auto precision = unittest::core::testsuite::instance()->get_arguments().max_value_precision;
        if (precision<=0) precision = std::numeric_limits<T>::max_digits10;
        stream.precision(precision);
        stream << value;
        return stream.str();
    }
};
/**
 * @brief Converts a value to string
 */
struct tostr_converter_other {
    /**
     * @brief Converts a value to string
     * @param value The value to convert
     * @returns A string
     */
    template<typename T>
    std::string
    operator()(const T& value)
    {
        std::ostringstream stream;
        stream << value;
        return "'" + limit_string_length(stream.str(), unittest::core::testsuite::instance()->get_arguments().max_string_length) + "'";
    }
};
/**
 * @brief Converts a value to string
 */
template<bool is_integral,
         bool is_float>
struct tostr_converter;
/**
 * @brief Converts a value to string. Spec. for int
 */
template<>
struct tostr_converter<true, false> {
    /**
     * @brief The actual converter type
     */
    typedef unittest::core::tostr_converter_int type;
};
/**
 * @brief Converts a value to string. Spec. for float
 */
template<>
struct tostr_converter<false, true> {
    /**
     * @brief The actual converter type
     */
    typedef unittest::core::tostr_converter_float type;
};
/**
 * @brief Converts a value to string. Spec. for other than int or float
 */
template<>
struct tostr_converter<false, false> {
    /**
     * @brief The actual converter type
     */
    typedef unittest::core::tostr_converter_other type;
};
/**
 * @brief Converts a given value to string by taking into account
 * 	the maximum string length and the maximum value precision
 * @param value The value
 * @returns A string
 */
template<typename T>
std::string
str(const T& value)
{
    typename unittest::core::tostr_converter<std::is_integral<T>::value, std::is_floating_point<T>::value>::type converter;
    return converter(value);
}
/**
 * @brief Converts a given value to string. Spec. for bool
 * @param value The value
 * @returns A string
 */
template<>
std::string
str<bool>(const bool& value);
/**
 * @brief Converts an arithmetic value to string
 */
struct tostr_if_converter_arithmetic {
    /**
     * @brief Converts an arithmetic value to string
     * @param prefix The prefix string
     * @param value The value
     * @return A string
     */
    template<typename T>
    std::string
    operator()(const std::string& prefix, const T& value, const std::string&)
    {
        return prefix + unittest::core::str(value);
    }
};
/**
 * @brief Converts a non-arithmetic value to string
 */
struct tostr_if_converter_other {
    /**
     * @brief Converts a non-arithmetic value to string
     * @param fallback
     * @return A string
     */
    template<typename T>
    std::string
    operator()(const std::string&, const T&, const std::string& fallback)
    {
        return fallback;
    }
};
/**
 * @brief Converts a value to string
 */
template<bool is_arithmetic>
struct tostr_if_converter;
/**
 * @brief Converts a value to string. Spec. for arithmetic
 */
template<>
struct tostr_if_converter<true> {
    /**
     * @brief Arithmetic value converter
     */
    typedef unittest::core::tostr_if_converter_arithmetic type;
};
/**
 * @brief Converts a value to string. Spec. for non-arithmetic
 */
template<>
struct tostr_if_converter<false> {
    /**
     * @brief Non-arithmetic value converter
     */
    typedef unittest::core::tostr_if_converter_other type;
};
/**
 * Converts the given value to string by appending the prefix if value is of
 * arithmetic type and if not, just returns the fallback string
 * @param prefix The prefix string
 * @param value The value
 * @param fallback The fallback string
 * @return A string
 */
template<typename T>
std::string
str_if(const std::string& prefix, const T& value, const std::string& fallback="")
{
    typename unittest::core::tostr_if_converter<std::is_arithmetic<T>::value>::type converter;
    return converter(prefix, value, fallback);
}

} // core
} // unittest
