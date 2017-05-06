#include "libunittest/environments.hpp"
#include "libunittest/userargs.hpp"
#include "libunittest/testsuite.hpp"
#include "libunittest/testresults.hpp"
#include "libunittest/utilities.hpp"
#include "libunittest/testfailure.hpp"
#include "libunittest/quote.hpp"
#include <iostream>
#include <fstream>
#include <random>

namespace unittest {

int
process(int argc, char **argv)
{
    core::userargs arguments;
    try {
        arguments.parse(argc, argv);
    } catch (const core::argparser::exit_success& e) {
        std::cout << e.what();
        std::exit(EXIT_SUCCESS);
    } catch (const core::argparser::exit_error& e) {
        std::cout << "Error: " << e.what();
        std::exit(EXIT_FAILURE);
    }

    if (arguments.display_quote) {
        core::quote_generator quote_gen(0);
        std::cout << quote_gen.next() << std::endl;
        return EXIT_SUCCESS;
    }

    auto suite = core::testsuite::instance();
    suite->set_arguments(arguments);

    auto class_runs = suite->get_class_runs();
    if (arguments.shuffle_seed >= 0) {
        if (arguments.verbose)
            std::cout << "shuffseed = " << arguments.shuffle_seed << std::endl;
        std::mt19937 gen(static_cast<unsigned int>(arguments.shuffle_seed));
        std::shuffle(class_runs.begin(), class_runs.end(), gen);
    }
    core::call_functions(class_runs, arguments.concurrent_threads);

    const auto results = suite->get_results();
    write_error_info(std::cout, results.testlogs, results.successful);
    auto& threads = suite->get_lonely_threads();
    core::make_threads_happy(std::cout, threads, arguments.verbose);
    const auto full_results = suite->get_results();

    std::vector<core::testlog> delta_testlogs;
    bool successful = true;
    for (auto i=results.testlogs.size(); i<full_results.testlogs.size(); ++i) {
        delta_testlogs.push_back(full_results.testlogs[i]);
        if (!delta_testlogs.back().successful)
            successful = false;
    }
    write_error_info(std::cout, delta_testlogs, successful);

    write_summary(std::cout, full_results, arguments.shuffle_seed);
    if (arguments.generate_xml) {
        std::ofstream file(arguments.xml_filename);
        write_xml(file, full_results, arguments.suite_name, arguments.shuffle_seed);
    }

    return full_results.successful ? EXIT_SUCCESS : EXIT_FAILURE;
}

} // unittest
