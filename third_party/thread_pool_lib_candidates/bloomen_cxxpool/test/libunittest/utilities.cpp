#include "libunittest/utilities.hpp"
#include <thread>
#include <iostream>
#include <mutex>
#include <cstdlib>
#ifdef _MSC_VER
#include <windows.h>
#include <iomanip>
#else
#include "config.hpp"
#endif
#if HAVE_GCC_ABI_DEMANGLE==1
#include <cxxabi.h>
#endif

namespace unittest {
namespace core {

#if defined(_MSC_VER) && _MSC_VER < 1900
namespace {

const long long g_frequency = []() -> long long
{
    LARGE_INTEGER frequency;
    QueryPerformanceFrequency(&frequency);
    return frequency.QuadPart;
}();

struct windows_high_res_clock {
    typedef long long rep;
    typedef std::nano period;
    typedef std::chrono::duration<rep, period> duration;
    typedef std::chrono::time_point<windows_high_res_clock> tp;
    static const bool is_steady = true;
    static tp now()
    {
        LARGE_INTEGER count;
        QueryPerformanceCounter(&count);
        return tp(duration(count.QuadPart * static_cast<rep>(period::den) / g_frequency));
    }
};

}
#endif

std::chrono::microseconds
now()
{
#if defined(_MSC_VER) && _MSC_VER < 1900
    const auto now = windows_high_res_clock::now();
#else
    const auto now = std::chrono::high_resolution_clock::now();
#endif
    return std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch());
}

std::string
xml_escape(const std::string& data)
{
    std::string escaped;
    escaped.reserve(data.size());
    for (auto& character : data) {
        switch (character) {
            case '&':  escaped.append("&amp;");       break;
            case '\"': escaped.append("&quot;");      break;
            case '\'': escaped.append("&apos;");      break;
            case '<':  escaped.append("&lt;");        break;
            case '>':  escaped.append("&gt;");        break;
            default:   escaped.append(1, character);  break;
        }
    }
    return escaped;
}

std::string
make_iso_timestamp(const std::chrono::system_clock::time_point& time_point,
                   bool local_time)
{
    const auto rawtime = std::chrono::system_clock::to_time_t(time_point);
    const auto iso_format = "%Y-%m-%dT%H:%M:%S";
    struct std::tm timeinfo;
#ifdef _MSC_VER
    if (local_time)
        localtime_s(&timeinfo, &rawtime);
    else
        gmtime_s(&timeinfo, &rawtime);
    std::stringstream buffer;
    buffer << std::put_time(&timeinfo, iso_format);
    return buffer.str();
#else
    if (local_time)
        timeinfo = *std::localtime(&rawtime);
    else
        timeinfo = *std::gmtime(&rawtime);
    char buffer[20];
    std::strftime(buffer, sizeof(buffer), iso_format, &timeinfo);
    return buffer;
#endif
}

void
write_to_stream(std::ostream&)
{}

void
write_horizontal_bar(std::ostream& stream,
                     char character,
                     int length)
{
    const std::string bar(length, character);
    stream << bar << std::flush;
}

double
duration_in_seconds(const std::chrono::duration<double>& duration)
{
    return std::chrono::duration_cast<std::chrono::duration<double>>(duration).count();
}

bool
is_regex_matched(const std::string& value,
                 const std::string& regex)
{
    return std::regex_match(value, std::regex(regex));
}

bool
is_regex_matched(const std::string& value,
                 const std::regex& regex)
{
    return std::regex_match(value, regex);
}

int
call_functions(const std::vector<std::function<void()>>& functions,
               int n_threads)
{
    const int n_runs = functions.size();
    if (n_threads > n_runs)
        n_threads = n_runs;

    int counter = 0;
    if (n_threads > 1 && n_runs > 1) {
        while (counter < n_runs) {
            if (n_threads > n_runs - counter)
                n_threads = n_runs - counter;
            std::vector<std::thread> threads(n_threads);
            for (auto& thread : threads)
                thread = std::thread(functions[counter++]);
            for (auto& thread : threads)
                thread.join();
        }
    } else {
        for (auto& function : functions) {
            function();
            ++counter;
        }
    }

    return counter;
}

std::string
limit_string_length(const std::string& value,
                    int max_length)
{
    if (int(value.size()) > max_length) {
        return value.substr(0, max_length);
    } else {
        return value;
    }
}

std::string
string_of_file_and_line(const std::string& filename,
                        int linenumber)
{
    return string_of_tagged_text(join(filename, ":", linenumber), "SPOT");
}

std::string
string_of_tagged_text(const std::string& text, const std::string& tag)
{
    const std::string id = "@" + tag + "@";
    return id + text + id;
}

void
make_threads_happy(std::ostream& stream,
                   std::vector<std::pair<std::thread, std::shared_ptr<std::atomic_bool>>>& threads,
                   bool verbose)
{
    int n_unfinished = 0;
    for (auto& thread : threads)
        if (!thread.second->load())
            ++n_unfinished;
    if (n_unfinished && verbose)
    	stream << "\nWaiting for " << n_unfinished << " tests to finish ... " << std::endl;
    for (auto& thread : threads)
        thread.first.join();
}

bool
is_numeric(const std::string& value)
{
    bool result;
    try {
        to_number<double>(value);
        result = true;
    } catch (const std::invalid_argument&) {
        result = false;
    }
    return result;
}

template<>
bool
to_number<bool>(const std::string& value)
{
    std::istringstream stream(value);
    bool number;
    if (stream >> number) {
        return number;
    } else {
        throw std::invalid_argument("Not a boolean: " + value);
    }
}

std::string
trim(std::string value)
{
    const std::string chars = " \t\n";
    std::string check;
    while (check!=value) {
        check = value;
        for (auto& ch : chars) {
            value.erase(0, value.find_first_not_of(ch));
            value.erase(value.find_last_not_of(ch) + 1);
        }
    }
    return value;
}

std::string
remove_white_spaces(std::string value)
{
    value.erase(std::remove(value.begin(), value.end(), ' '), value.end());
    return value;
}

std::string
extract_tagged_text(const std::string& message, const std::string& tag)
{
    std::string text = "";
    const std::string id = "@" + tag + "@";
    auto index_start = message.find(id);
    if (index_start!=std::string::npos) {
        index_start += id.length();
        const auto substr = message.substr(index_start);
        const auto index_end = substr.find(id);
        if (index_end!=std::string::npos) {
            text = substr.substr(0, index_end);
        }
    }
    return text;
}

std::string
remove_tagged_text(std::string message, const std::string& tag)
{
    auto text = extract_tagged_text(message, tag);
    while (!text.empty()) {
        const auto token = string_of_tagged_text(text, tag);
        const auto index = message.find(token);
        message = message.substr(0, index) + message.substr(index+token.length());
        text = extract_tagged_text(message, tag);
    }
    return std::move(message);
}


std::pair<std::string, int>
extract_file_and_line(const std::string& message)
{
    std::string filename = "";
    int linenumber = -1;
    const std::string spot = extract_tagged_text(message, "SPOT");
    if (!spot.empty()) {
        const auto separator = spot.find_last_of(":");
        if (separator!=std::string::npos) {
            filename = spot.substr(0, separator);
            linenumber = to_number<int>(spot.substr(separator+1));
        }
    }
    return std::make_pair(filename, linenumber);
}

std::string
demangle_type(const std::string& type_name)
{
#if HAVE_GCC_ABI_DEMANGLE==1
    int status = -1;
    std::unique_ptr<char, void(*)(void*)> result{abi::__cxa_demangle(type_name.c_str(), NULL, NULL, &status), std::free};
    return status==0 ? result.get() : type_name;
#else
    return type_name;
#endif
}

} // core
} // unittest
