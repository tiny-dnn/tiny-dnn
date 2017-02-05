#include "libunittest/userargs.hpp"
#include "libunittest/utilities.hpp"
#include "libunittest/version.hpp"

namespace unittest {
namespace core {


userargs::userargs()
    : verbose(false), failure_stop(false), generate_xml(false),
      handle_exceptions(true), dry_run(false), concurrent_threads(0), regex_filter(""),
      name_filter(""), test_name(""), timeout(-1), xml_filename("libunittest.xml"),
      disable_timeout(false), max_value_precision(-1), max_string_length(500),
      suite_name("libunittest"), shuffle_seed(-1), ignore_skips(false), display_quote(false)
{
    register_trigger('v', "verbose", "Sets verbose output for running tests", verbose);
    register_trigger('d', "dry_run", "A dry run without actually executing any tests", dry_run);
    register_trigger('s', "fail_stop", "Stops running tests after the first test fails", failure_stop);
    register_trigger('x', "gen_xml", "Enables the generation of the XML output", generate_xml);
    register_trigger('e', "handle_exc", "Turns off handling of unexpected exceptions", handle_exceptions);
    register_trigger('k', "ignore_skips", "Runs tests regardless of whether they are skipped", ignore_skips);
    register_trigger('i', "no_timeouts", "Disables the measurement of any test timeouts", disable_timeout);
    register_trigger('q', "quote", "Displays a random quote and exits", display_quote);
    register_argument('p', "number", "Runs tests in parallel with a given number of threads", concurrent_threads, false);
    register_argument('n', "name", "A certain test to be run superseding any other run filter", test_name, false);
    register_argument('f', "string", "A run filter applied to the beginning of the test names", name_filter, false);
    register_argument('g', "regex", "A run filter based on regex applied to the test names", regex_filter, false);
    register_argument('z', "shuffseed", "A seed of >= 0 enables shuffling (0 means time based)", shuffle_seed, false);
    register_argument('t', "timeout", "A timeout in seconds for tests without static timeouts", timeout, false);
    register_argument('r', "precision", "The maximum displayed value precision", max_value_precision, false);
    register_argument('l', "length", "The maximum displayed string length", max_string_length, true);
    register_argument('o', "xmlfile", "The XML output file name", xml_filename, true);
    register_argument('u', "suite", "The name of the test suite", suite_name, true);
}

std::string
userargs::description()
{
    return "This is your test application using libunittest " + version();
}

void
userargs::assign_values()
{
    assign_value(verbose, 'v');
    assign_value(dry_run, 'd');
    assign_value(failure_stop, 's');
    assign_value(generate_xml, 'x');
    assign_value(handle_exceptions, 'e');
    assign_value(ignore_skips, 'k');
    assign_value(disable_timeout, 'i');
    assign_value(display_quote, 'q');
    assign_value(concurrent_threads, 'p');
    assign_value(test_name, 'n');
    assign_value(name_filter, 'f');
    assign_value(regex_filter, 'g');
    assign_value(shuffle_seed, 'z');
    assign_value(timeout, 't');
    assign_value(xml_filename, 'o');
    assign_value(max_string_length, 'l');
    assign_value(max_value_precision, 'r');
    assign_value(suite_name, 'u');
}

void
userargs::post_parse()
{
    if (was_used('o')) generate_xml = true;
    if (concurrent_threads<0) concurrent_threads = 0;
    if (max_string_length<10) max_string_length = 10;
    if (shuffle_seed==0) shuffle_seed = now().count() / 1000000;
}


} // core
} // unittest
