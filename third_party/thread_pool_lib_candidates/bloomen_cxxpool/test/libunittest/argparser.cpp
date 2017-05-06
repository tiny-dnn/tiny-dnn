#include "libunittest/argparser.hpp"

namespace unittest {
namespace core {


argparser::argparser()
    : command_line_(), app_name_(), args_(), assign_args_(), registry_()
{
    register_trigger('h', "h", "Displays this help message and exits", false);
}

argparser::~argparser()
{}

std::string
argparser::command_line() const
{
    return command_line_;
}

std::string
argparser::make_arg_string(char arg) const
{
    return join('-', arg);
}

void
argparser::add_to_registry(char arg, argrow row)
{
    if (in_registry(arg))
        throw std::invalid_argument(join("Argument already registered: ", arg));
    registry_.push_back(std::make_pair(arg, row));
}

bool
argparser::in_registry(char arg)
{
    try {
        from_registry(arg);
        return true;
    } catch (const std::invalid_argument&) {
        return false;
    }
}

argparser::argrow&
argparser::from_registry(char arg)
{
    for (auto& pair : registry_) {
        if (pair.first==arg)
            return pair.second;
    }
    throw std::invalid_argument(join("Asking for unregistered argument: ", arg));
}

bool
argparser::was_used(char arg)
{
    return from_registry(arg).is_used;
}

template<>
void
argparser::register_argument<bool>(char arg,
                                   std::string value_name,
                                   std::string description,
                                   bool default_value,
                                   bool,
                                   bool)
{
    register_trigger(arg, value_name, description, default_value);
}

void
argparser::register_trigger(char arg,
                            std::string value_name,
                            std::string description,
                            bool default_value)
{
    const argrow row = {registry_.size(), true, value_name, value_name, description, join(default_value), make_repr(default_value), false, false, false};
    add_to_registry(arg, std::move(row));
}

std::string
argparser::get_help()
{
    set_long_value_names();
    std::ostringstream stream;
    const auto desc = description();
    if (desc.size()) stream << desc << "\n\n";
    if (!app_name_.size())
        app_name_ = app_name();
    stream << "Usage: " << (app_name_.size() ? app_name_ : "program") << " ";
    for (auto& row : registry_) {
        if (row.second.required) {
            stream << make_arg_string(row.first) << " " << row.second.value_name << " ";
        }
    }
    stream << "[Arguments]\n\n";
    stream << "Arguments:\n";
    for (auto& row : registry_) {
        std::string appendix;
        if (row.second.required)
            appendix = " (required)";
        else if (!row.second.is_trigger && row.second.display_default)
            appendix = " (default: " + row.second.default_value + ")";
        std::string value_name = row.second.long_value_name;
        if (row.second.is_trigger)
            value_name = std::string(value_name.size(), ' ');
        stream << make_arg_string(row.first) << " " << value_name << "  " << row.second.description << appendix << std::endl;
    }
    stream << std::flush;
    return stream.str();
}

void
argparser::error(const std::string& message)
{
    throw exit_error(join(message, '\n'));
}

std::vector<std::string>
argparser::expand_arguments(int argc, char **argv)
{
    const std::string bad_prefix = "--";
    const char flag = '-';
    const std::string sflag(1, flag);
    std::vector<std::string> args;
    for (int i=1; i<argc; ++i) {
        const std::string value = argv[i];
        if (value.substr(0, bad_prefix.size())==bad_prefix)
            error("Only options with a single '-' are supported");
        if (value.substr(0, 1)==sflag && !unittest::core::is_numeric(value)) {
            const std::string expanded(value.substr(1, value.size()));
            for (auto& val : expanded) {
                if (val != flag)
                    args.push_back(join(flag, val));
            }
        } else {
            args.push_back(value);
        }
    }
    return args;
}

template<>
void
argparser::assign_value<bool>(bool& result,
                              char arg)
{
    assign_args_ += std::string(1, arg);
    auto& row = from_registry(arg);
    const std::string flag = make_arg_string(arg);
    result = get_value<bool>(flag, row.default_value);
    std::vector<size_t> del_indices;
    for (size_t i=0; i<args_.size(); ++i) {
        if (args_[i]==flag) {
            if (!row.is_used) {
                result = !result;
                row.representation = make_repr(result);
                row.is_used = true;
            }
            del_indices.push_back(i);
        }
    }
    remove_indices_from_args(del_indices);
}

template<>
std::string
argparser::get_value<std::string>(std::string,
                                  std::string value)
{
    return value;
}

template<>
std::string
argparser::make_repr<bool>(bool value) const
{
    return value ? "true" : "false";
}

template<>
std::string
argparser::make_repr<std::string>(std::string value) const
{
    return join("\"", value, "\"");
}

void
argparser::set_long_value_names()
{
    size_t max_length = 0;
    for (auto& row : registry_) {
        if (row.second.long_value_name.length() > max_length)
            max_length = row.second.long_value_name.length();
    }
    for (auto& row : registry_) {
        row.second.long_value_name += std::string(max_length - row.second.long_value_name.length(), ' ');
    }
}

void
argparser::parse(int argc, char **argv)
{
    for (int i=0; i<argc; ++i) {
        command_line_ += std::string(argv[i]);
        if (i<argc-1) command_line_ += " ";
    }
    app_name_ = app_name();
    if (!app_name_.size())
        app_name_ = argv[0];
    args_ = expand_arguments(argc, argv);
    bool help;
    assign_value(help, 'h');
    if (help) {
        throw exit_success(get_help());
    }
    assign_values();
    check_assign_args();
    if (args_.size()) {
        error(join("Invalid argument: '", args_[0], "'"));
    }
    post_parse();
}

void
argparser::check_assign_args()
{
    bool good = true;
    if (assign_args_.size()!=registry_.size())
        good = false;
    std::set<char> set(assign_args_.begin(), assign_args_.end());
    if (set.size()!=registry_.size())
        good = false;
    for (auto value : set) {
        if (!in_registry(value))
            good = false;
    }
    if (!good)
        throw std::invalid_argument("Assign argument flags don't match those registered");
}

void
argparser::remove_indices_from_args(std::vector<size_t> indices)
{
    std::sort(indices.begin(), indices.end(), std::greater<size_t>());
    for (auto value : indices) {
        args_.erase(args_.begin() + value);
    }
}

std::ostream&
operator<<(std::ostream& os, argparser& obj)
{
    obj.set_long_value_names();
    for (auto& row : obj.registry_) {
        if (row.first!='h') {
            os << obj.make_arg_string(row.first) << " " << row.second.long_value_name << " = " << row.second.representation << std::endl;
        }
    }
    os << std::flush;
    return os;
}

argparser::argrow::argrow()
    : index(0), is_trigger(false), value_name(),
      long_value_name(), description(),
      default_value(), representation(),
      display_default(true), required(false),
      is_used(false)
{}

argparser::argrow::argrow(size_t index,
                          bool is_trigger,
                          std::string value_name,
                          std::string long_value_name,
                          std::string description,
                          std::string default_value,
                          std::string representation,
                          bool display_default,
                          bool required,
                          bool is_used)
    : index(index), is_trigger(is_trigger), value_name(value_name),
      long_value_name(long_value_name), description(description),
      default_value(default_value), representation(representation),
      display_default(display_default), required(required), is_used(is_used)
{}


argparser::exit_success::exit_success(const std::string& message)
    : std::runtime_error(message)
{}

argparser::exit_success::~exit_success() UNITTEST_NOEXCEPT
{}


argparser::exit_error::exit_error(const std::string& message)
    : std::runtime_error(message)
{}

argparser::exit_error::~exit_error() UNITTEST_NOEXCEPT
{}


} // core
} // unittest
