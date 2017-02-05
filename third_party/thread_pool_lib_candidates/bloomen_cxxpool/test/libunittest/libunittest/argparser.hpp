/**
 * @brief An argument parser
 * @file argparser.hpp
 */
#pragma once
#include "utilities.hpp"
#include "noexcept.hpp"
#include <string>
#include <sstream>
#include <vector>
#include <set>
/**
 * @brief Unit testing in C++
 */
namespace unittest {
/**
 * @brief Internal functionality, not relevant for most users
 */
namespace core {
/**
 * @brief An argument parser
 */
class argparser {
public:
    /**
     * @brief Returns the command help
     * @returns The command help
     */
    std::string
    get_help();
    /**
     * @brief Parses the arguments
     * @param argc The number of arguments
     * @param argv The array of arguments
     */
    void
    parse(int argc, char **argv);
    /**
     * @brief Returns the command line
     * @returns The command line
     */
    std::string
    command_line() const;
    /**
     * @brief The exception class to indicate exit success
     */
    class exit_success : public std::runtime_error {
    public:
        /**
         * @brief Constructor
         * @param message The exception message
         */
        explicit
        exit_success(const std::string& message);
        /**
         * @brief Destructor
         */
        ~exit_success() UNITTEST_NOEXCEPT;
    };
    /**
     * @brief The exception class to indicate parsing errors
     */
    class exit_error : public std::runtime_error {
    public:
        /**
         * @brief Constructor
         * @param message The exception message
         */
        explicit
        exit_error(const std::string& message);
        /**
         * @brief Destructor
         */
        ~exit_error() UNITTEST_NOEXCEPT;
    };
    /**
     * @brief Destructor
     */
    virtual
    ~argparser();

private:
    /**
     * @brief Override this to provide the name of the application
     * @returns The application name
     */
    virtual std::string
    app_name() { return ""; }
    /**
     * @brief Override this to provide a helpful command description
     * @returns The command description
     */
    virtual std::string
    description() { return ""; }
    /**
     * @brief Override this to assign values to variables through protected
     * 	method: assign_value()
     */
    virtual void
    assign_values() {}
    /**
     * @brief Override this to do stuff right after parsing, e.g., checking
     * 	assigned values
     */
    virtual void
    post_parse() {}

protected:
    /**
     * @brief Constructor
     */
    argparser();
    /**
     * @brief Registers an argument. Call this in the derivee's constructor
     * @param arg The argument flag
     * @param value_name The name of the argument
     * @param description A description
     * @param default_value The default value
     * @param display_default Whether to display the default value
     * @param required Whether this argument is required
     */
    template<typename T>
    void
    register_argument(char arg,
                      std::string value_name,
                      std::string description,
                      T default_value,
                      bool display_default,
                      bool required=false)
    {
        const argparser::argrow row = {registry_.size(), false, value_name, value_name, description, unittest::join(default_value), this->make_repr(default_value), display_default, required, false};
        this->add_to_registry(arg, std::move(row));
    }
    /**
     * @brief Registers a trigger. Call this in the derivee's constructor
     * @param arg The argument flag
     * @param value_name The name of the argument
     * @param description A description
     * @param default_value The default value
     */
    void
    register_trigger(char arg,
                     std::string value_name,
                     std::string description,
                     bool default_value);
    /**
     * @brief Assigns a value through the given argument flag. Call
     * 	this within the override of assign_values()
     * @param result The resulting value
     * @param arg The argument flag
     */
    template<typename T>
    void
    assign_value(T& result,
                 char arg)
    {
        assign_args_ += std::string(1, arg);
        auto& row = this->from_registry(arg);
        const std::string flag = this->make_arg_string(arg);
        result = this->get_value<T>(flag, row.default_value);
        bool found = false;
        std::vector<size_t> del_indices;
        for (size_t i=0; i<args_.size(); ++i) {
            if (args_[i]==flag) {
                if (++i<args_.size()) {
                    if (!row.is_used) {
                        result = this->get_value<T>(flag, args_[i]);
                        row.representation = this->make_repr(result);
                        row.is_used = true;
                    }
                    del_indices.push_back(i - 1);
                    del_indices.push_back(i);
                    found = true;
                } else {
                    this->error(unittest::join("Missing value to argument '", flag, "'"));
                }
            }
        }
        if (row.required && !found) {
            this->error(unittest::join("Missing required argument '", flag, "'"));
        }
        remove_indices_from_args(del_indices);
    }
    /**
     * @brief Returns whether the given argument flag was used by the client
     * @param arg The argument flag
     * @returns Whether the given argument flag was used
     */
    bool
    was_used(char arg);
    /**
     * @brief Writes help to screen and throws an exit_error
     * @param message The exception message
     */
    void
    error(const std::string& message);

private:

    struct argrow {
        argrow();
        argrow(size_t index,
               bool is_trigger,
               std::string value_name,
               std::string long_value_name,
               std::string description,
               std::string default_value,
               std::string representation,
               bool display_default,
               bool required,
               bool is_used);
        size_t index;
        bool is_trigger;
        std::string value_name;
        std::string long_value_name;
        std::string description;
        std::string default_value;
        std::string representation;
        bool display_default;
        bool required;
        bool is_used;
    };

    template<typename T>
    T
    get_value(std::string flag,
              std::string value)
    {
        T result;
        try {
            result = unittest::core::to_number<T>(value);
        } catch (const std::invalid_argument&) {
            this->error(unittest::join("The value to '", flag,"' must be numeric, not: ", value));
        }
        return result;
    }

    template<typename T>
    std::string
    make_repr(T value) const
    {
        return unittest::join(value);
    }

    std::string
    make_arg_string(char arg) const;

    std::vector<std::string>
    expand_arguments(int argc, char **argv);

    void
    add_to_registry(char arg, argparser::argrow row);

    bool
    in_registry(char arg);

    argparser::argrow&
    from_registry(char arg);

    void
    set_long_value_names();

    void
    check_assign_args();

    void
    remove_indices_from_args(std::vector<size_t> indices);

    friend std::ostream&
    operator<<(std::ostream& os, unittest::core::argparser& obj);

    std::string command_line_;
    std::string app_name_;
    std::vector<std::string> args_;
    std::string assign_args_;
    std::vector<std::pair<char,argparser::argrow>> registry_;
};
/**
 * @brief Registers an argument. Spec. for bool which registers a trigger
 * @param arg The argument flag
 * @param value_name The name of the argument
 * @param description A description
 * @param default_value The default value
 * @param display_default Whether to display the default value
 * @param required Whether this argument is required
 */
template<>
void
argparser::register_argument<bool>(char arg,
                                   std::string value_name,
                                   std::string description,
                                   bool default_value,
                                   bool display_default,
                                   bool required);
/**
 * @brief Assigns a value through the given argument flag. Spec. for bool
 * @param result The resulting value
 * @param arg The argument flag
 */
template<>
void
argparser::assign_value<bool>(bool& result,
                              char arg);
/**
 * @brief Returns a value for a given argument. Spec. for string
 * @param arg The argument
 * @param value The value
 * @returns The resulting value
 */
template<>
std::string
argparser::get_value<std::string>(std::string arg,
                                  std::string value);
/**
 * @brief Returns a string repr. for a given value. Spec. for bool
 * @param value The value
 * @returns A string repr.
 */
template<>
std::string
argparser::make_repr<bool>(bool value) const;
/**
 * @brief Returns a string repr. for a given value. Spec. for string
 * @param value The value
 * @returns A string repr.
 */
template<>
std::string
argparser::make_repr<std::string>(std::string value) const;
/**
 * @brief Output stream operator for argparser
 * @param os The output stream
 * @param obj An instance of argparser
 * @returns The output stream
 */
std::ostream&
operator<<(std::ostream& os, unittest::core::argparser& obj);


} // core
} // unittest
