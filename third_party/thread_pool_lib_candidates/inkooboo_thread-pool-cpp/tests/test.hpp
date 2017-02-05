#include <stdexcept>
#include <iostream>
#include <sstream>

#define ASSERT(expr) \
    if (!(expr)) { \
        std::ostringstream ss; \
        ss << __FILE__ << ":" <<__LINE__ << " " << #expr; \
        throw std::runtime_error(ss.str()); \
    }


template <typename TestFunc>
inline void doTest(const char *name, TestFunc &&test) {
    std::cout << " - test ( " << name;
    try {
        test();
    } catch (const std::exception &e) {
        std::cout << " => failed with: " << e.what() << " )" << std::endl;
        throw;
    }
    std::cout << " => succeed )" << std::endl;
}
