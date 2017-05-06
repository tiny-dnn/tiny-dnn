/**
 * @brief The userargs class to control application behavior
 * @file userargs.hpp
 */
#pragma once
#include "argparser.hpp"
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
 * @brief High level user arguments to control application behavior
 */
struct userargs : argparser {
    /**
     * @brief Constructor
     */
    userargs();
    /**
     * @brief Whether the output is verbose (default: false)
     */
    bool verbose;
    /**
     * @brief Whether to stop running after first failure (default: false)
     */
    bool failure_stop;
    /**
     * @brief Whether to generate XML output (default: false)
     */
    bool generate_xml;
    /**
     * @brief Whether to handle unknown exceptions (default: true)
     */
    bool handle_exceptions;
    /**
     * @brief Whether to perform a dry run (default: false)
     */
    bool dry_run;
    /**
     * @brief The number of concurrent threads (default: 0)
     */
    int concurrent_threads;
    /**
     * @brief Sets a regex filter on the full test name (default: "")
     */
    std::string regex_filter;
    /**
     * @brief Sets a filter on the beginning of the full test name (default: "")
     */
    std::string name_filter;
    /**
     * @brief Sets a certain test to be run (default: "")
     * 	This supersedes the regex and the name filter
     */
    std::string test_name;
    /**
     * @brief The global test timeout (default: -1)
     */
    double timeout;
    /**
     * @brief The XML output file name (default: "libunittest.xml")
     */
    std::string xml_filename;
    /**
     * @brief Whether to disable timeout measurement (default: false)
     */
    bool disable_timeout;
    /**
     * @brief The maximum displayed value precision (default: -1)
     */
    int max_value_precision;
    /**
     * @brief The maximum displayed string length (default: 500)
     */
    int max_string_length;
    /**
     * @brief The name of the test suite (default: "libunittest")
     */
    std::string suite_name;
    /**
     * @brief The random seed of the optional shuffling (default: -1)
     */
    long long shuffle_seed;
    /**
     * @brief Whether to ignore all static skips (default: false)
     */
    bool ignore_skips;
    /**
     * @brief Whether to display a random quote and exit
     */
    bool display_quote;

private:

    std::string description();

    void assign_values();

    void post_parse();

};

} // core
} // unittest
