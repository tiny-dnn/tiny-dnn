#include "libunittest/random.hpp"

namespace unittest {

random_value<bool>::random_value()
    : random_object<bool>(), distribution_(0, 1)
{}

std::shared_ptr<random_object<bool>>
random_value<bool>::do_clone()
{
    return std::make_shared<random_value<bool>>(*this);
}

bool
random_value<bool>::do_get()
{
    return distribution_(this->gen()) == 1;
}

std::shared_ptr<random_object<bool>>
make_random_bool()
{
    return std::make_shared<random_value<bool>>();
}

} // unittest
