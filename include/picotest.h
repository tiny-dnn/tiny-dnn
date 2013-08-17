/*
    Copyright (c) 2012, Taiga Nomi
    All rights reserved.
    
    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in the
    documentation and/or other materials provided with the distribution.
    * Neither the name of the <organization> nor the
    names of its contributors may be used to endorse or promote products
    derived from this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY 
    EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED 
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE 
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY 
    DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES 
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; 
    LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND 
    ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT 
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS 
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
#pragma once

#include <vector>
#include <iostream>
#include <functional>
#include <iomanip>
#include <string>
#include <sstream>
#include <algorithm>

#include <cstdio>
#include <cassert>
#include <stdint.h>
#include <cstdarg>
#include <cstring>

#if defined _WIN32 
#define PICOTEST_WINDOWS
#define NOMINMAX
#include <Windows.h>
#elif defined __linux__
#define PICOTEST_LINUX
#endif


/////////////////////////////////////////////////////////////////
// helper macros

#define PICOTEST_JOIN(X, Y)       PICOTEST_DO_JOIN( X, Y )
#define PICOTEST_DO_JOIN( X, Y )  PICOTEST_DO_JOIN2(X,Y)
#define PICOTEST_DO_JOIN2( X, Y ) X##Y
#define PICOTEST_STR(X) #X
#define PICOTEST_DISALLOW_COPY_AND_ASSIGN(typeName) \
void operator = (const typeName&);\
typeName(const typeName&)

/////////////////////////////////////////////////////////////////
// internal

namespace picotest {
namespace detail {
    /***** print with color *****/

    enum Color {
        COLOR_RED,
        COLOR_GREEN
    };

#ifdef PICOTEST_WINDOWS
    inline WORD getColorAttr(Color c) {
        switch (c) {
        case COLOR_RED:    return FOREGROUND_RED;
        case COLOR_GREEN:  return FOREGROUND_GREEN;
        default:           assert(0); return 0;
        }
    }
#else
    inline const char* getColorEscape(Color c) {
        switch (c) {
        case COLOR_RED:    return "\x1b[41m";
        case COLOR_GREEN:  return "\x1b[32m";
        default:           assert(0); return "";
        }
    }
#endif

    inline void coloredPrint(Color c, const char* fmt, ...) {
        va_list args;
        va_start(args, fmt);

#ifdef PICOTEST_WINDOWS
        const HANDLE std_handle = ::GetStdHandle(STD_OUTPUT_HANDLE);

        CONSOLE_SCREEN_BUFFER_INFO buffer_info;
        ::GetConsoleScreenBufferInfo(std_handle, &buffer_info);
        const WORD old_color = buffer_info.wAttributes;
        const WORD new_color  = getColorAttr(c) | FOREGROUND_INTENSITY;

        fflush(stdout);
        ::SetConsoleTextAttribute(std_handle, new_color);

        vprintf(fmt, args);

        fflush(stdout);
        ::SetConsoleTextAttribute(std_handle, old_color);
#else
        printf("%s", getColorEscape(c));
        vprintf(fmt, args);
        printf("\033[m");
#endif
        va_end(args);
    }

    /***** stringize *****/

    inline std::string toString(const void* addr, int size) {
        const unsigned char *p = reinterpret_cast<const unsigned char*>(addr);
        std::ostringstream os;
        int maxsize = size > 10 ? 10 : size;

        os << "[";
        for (int i = 0; i < maxsize; i++, p++)
            os << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(*p) << ((i == size - 1) ? "" : " ");
        if (maxsize > size)
            os << "...";
        os << "]";

        return os.str();
    }

    template<typename T>
    std::string toString(const T& v) {
        std::ostringstream os;
        os << v;
        return os.str();
    }

    inline std::string toString(bool b) {
        return b ? "true" : "false";
    }

    // fallback operator <<
    template <typename Char, typename CharTraits, typename T>
    ::std::basic_ostream<Char, CharTraits>& operator<<(
        ::std::basic_ostream<Char, CharTraits>& os, const T& v) {
            os << "(" << sizeof(v) << "-byte object)" << toString(reinterpret_cast<const void*>(&v), sizeof(v));
            return os;
    }

    // operator << for bool
    template <typename Char, typename CharTraits>
    ::std::basic_ostream<Char, CharTraits>& operator<<(
        ::std::basic_ostream<Char, CharTraits>& os, const bool& b) {
            os << b ? "true" : "false";
            return os;
    }

    template<typename T1, typename T2, typename OP>
    std::string makeExpressionStr(const T1& v1, const T2& v2, OP op) {
        return toString(v1) + " " + op.name() + " " + toString(v2);
    }

    /***** error message *****/
    inline std::string makeMessage(const std::string& expected, const std::string& actual) {
        return expected + " failed for: " + actual;
    }

    inline std::string makeMessage(const std::string& expression, bool expected) {
        return "(" + expression + ") == " + toString(expected) + 
            " failed for: (" + expression + ") == " + toString(!expected);
    }

    /***** comparing floating point numbers using ULP *****/

    struct Floating {
        static const size_t MIN_UPS = 4;

        union float_ {
            float value;
            uint32_t raw;
        };

        union double_ {
            double value;
            uint64_t raw;
        };

        static uint32_t sam (uint32_t bits) {
            return bits & 0x80000000 ? ~bits + 1 : bits | 0x80000000;
        }

        static uint64_t sam (uint64_t bits) {
            return bits & 0x8000000000000000LL ? ~bits + 1 : bits | 0x8000000000000000LL;
        }

        template<typename T>
        static T distance (const T& v1, const T& v2) {
            const T sam1 = sam(v1), sam2 = sam(v2);
            return sam1 >= sam2 ? (sam1 - sam2) : (sam2 - sam1);
        }

        static bool almostEqual(float v1, float v2) {
            float_ v1_, v2_;
            v1_.value = v1;
            v2_.value = v2;
            return distance(v1_.raw, v2_.raw) <= MIN_UPS;
        }

        static bool almostEqual(double v1, double v2) {
            double_ v1_, v2_;
            v1_.value = v1;
            v2_.value = v2;
            return distance(v1_.raw, v2_.raw) <= MIN_UPS;
        }
    };

    /***** stricmp/strcasecmp *****/
    inline int stricmp(const char* c1, const char* c2) {
#ifdef PICOTEST_WINDOWS
        return _stricmp(c1, c2);
#else
        return strcasecmp(c1, c2);
#endif
    }


}

/////////////////////////////////////////////////////////////////
// framework

namespace framework {

class  TestCase;
class  Test;

struct TestState {
    TestState() : testcase_(0), test_(0) {}

    static TestState& getInstance() {
        static TestState instance;
        return instance;
    }

    static TestCase* getCurrentTestCase() {
        return getInstance().testcase_;
    }

    static Test* getCurrentTest() {
        return getInstance().test_;
    }

    static void setCurrentTestCase(TestCase* testcase) {
        getInstance().testcase_ = testcase;
    }

    static void setCurrentTest(Test* test) {
        getInstance().test_ = test;
    }
private:
    PICOTEST_DISALLOW_COPY_AND_ASSIGN(TestState);

    TestCase* testcase_;
    Test* test_;
};

struct Failure {
    Failure(const std::string& file, int line, const std::string& expected, const std::string& actual)
        : file(file), line(line), message(detail::makeMessage(expected, actual)) {}

    Failure(const std::string& file, int line, const std::string& expression, bool expected)
        : file(file), line(line), message(detail::makeMessage(expression, expected)) {}

    Failure() {}
    std::string file;
    int line;
    std::string message;
};

class Test {
public:
    typedef std::vector<Failure> Failures;
    typedef void (*TestFunc)(void);

    Test (const std::string& name, TestFunc f) : executed_(false), name_(name), f_(f) {}

    void execute() {
        TestState::setCurrentTest(this);
        f_();
        executed_ = true;
    }

    const std::string& name() const {
        return name_;
    }

    bool success() const {
        return executed_ && failures_.empty();
    }

    void setFailure(const Failure& failure) {
        failures_.push_back(failure);
    }

    template<typename Char, typename CharTraits>
    void reportFailure(std::basic_ostream<Char, CharTraits>& os) const {
        for (Failures::const_iterator it = failures_.begin(), end = failures_.end(); it != end; ++it) 
            report(os, (*it));
    }

private:
    template<typename Char, typename CharTraits>
    void report(std::basic_ostream<Char, CharTraits>& os, const Failure& f) const {
        os << name_ << " : " << f.file << "(" << f.line << "): " << f.message << std::endl;
    }

    bool executed_;
    Failures failures_;
    std::string name_;
    TestFunc f_;
};

class TestCase {
public:
    typedef std::vector<Test> Tests;

    TestCase() {}
    TestCase(const std::string& name) : executed_(false), name_(name) {}

    void add(const Test&t) {
        tests_.push_back(t);
    }

    void execute() {
        TestState::setCurrentTestCase(this);
        
        std::for_each(tests_.begin(), tests_.end(), std::mem_fun_ref(&Test::execute));

        executed_ = true;
    }

    template<typename Char, typename CharTraits>
    void report(std::basic_ostream<Char, CharTraits>& os) const {
        coloredPrint(detail::COLOR_RED, "[ FAILED ] ");
        os << name_ << std::endl;

        for (Tests::const_iterator it = tests_.begin(), end = tests_.end(); it != end; ++it)
            if (!(*it).success()) (*it).reportFailure(os);
    }

    bool success() const {
        if (!executed_) return false;

        if (std::find_if(
                tests_.begin(), 
                tests_.end(), 
                std::not1(std::mem_fun_ref(&Test::success))) == tests_.end())
            return true;

        return false; // at least one test failed
    }

    const std::string& name() const {
        return name_;
    }

private:
    bool executed_;
    std::string name_;
    Tests tests_;
};

struct Registry {
public:
    typedef std::vector<TestCase> TestCases;

    static Registry& getInstance() {
        static Registry instance;
        return instance;
    }

    void add(const std::string& test_case_name, const Test& t) {
        TestCases::iterator found = find_by_name(test_case_name);

        if (found == tests_.end()) {
            tests_.push_back(TestCase(test_case_name));
            found = find_by_name(test_case_name);
        }
        (*found).add(t);
    }

    void testRun() {
        std::for_each(tests_.begin(), tests_.end(), std::mem_fun_ref(&TestCase::execute));
    }

    template<typename Char, typename CharTraits>
    void report(std::basic_ostream<Char, CharTraits>& os) const {
        int failed = numFailed();

        if (failed) {
            os << numFailed() << " of " << numTotal() << " tests failed." << std::endl;

            for (TestCases::const_iterator it = tests_.begin(), end = tests_.end(); it != end; ++it)
                if (!(*it).success()) (*it).report(os);
        } else {
            os << numSuccess() << "tests success." << std::endl;
        }
    }

    bool fail() const {
        return numTotal() > 0 && numFailed() > 0;
    }

    int numFailed() const {
        return numTotal() - numSuccess();
    }

    int numSuccess() const {
        return std::count_if(tests_.begin(), tests_.end(), std::mem_fun_ref(&TestCase::success));
    }

    int numTotal() const {
        return tests_.size();
    }

private:
    Registry(){}

    PICOTEST_DISALLOW_COPY_AND_ASSIGN(Registry);

    TestCases::iterator find_by_name(const std::string& test_case_name) {
        for (TestCases::iterator it = tests_.begin(), end = tests_.end(); it != end; ++it)
            if ((*it).name() == test_case_name) return it;

        return tests_.end();
    }

    TestCases tests_;
};

struct Registrar {
    Registrar (const std::string& test_case_name, const Test& t) {
        Registry::getInstance().add(test_case_name, t);
    }
};

} // namespace picotest::framework

/***** binary operators *****/

struct LT {
    template <typename T1, typename T2>
    bool operator()(const T1& lhs, const T2& rhs) { return lhs < rhs; }
    static std::string name() { return "<"; }
};

struct GT {
    template <typename T1, typename T2>
    bool operator()(const T1& lhs, const T2& rhs) { return lhs > rhs; }
    static std::string name() { return ">"; }
};

struct LE {
    template <typename T1, typename T2>
    bool operator()(const T1& lhs, const T2& rhs) { return lhs <= rhs; }
    static std::string name() { return "<="; }
};

struct GE {
    template <typename T1, typename T2>
    bool operator()(const T1& lhs, const T2& rhs) { return lhs >= rhs; }
    static std::string name() { return ">="; }
};

struct EQ {
    template <typename T1, typename T2>
    bool operator()(const T1& lhs, const T2& rhs) { return lhs == rhs; }

    // use when comparing against null
    template <typename T>
    bool operator()(int lhs, T* const rhs) {
        return reinterpret_cast<const int*>(lhs) == rhs;
    }

    template <typename T>
    bool operator()(T* const lhs, int rhs) {
        return lhs == reinterpret_cast<const int*>(rhs);
    }

    static std::string name() { return "=="; }
};

struct NE {
    template <class T1, class T2>
    bool operator()(const T1& lhs, const T2& rhs) { return !EQ()(lhs, rhs); }
    static std::string name() { return "!="; }
};

struct STREQ {
    bool operator()(const char* lhs, const char* rhs) { return strcmp(lhs, rhs) == 0; }
    static std::string name() { return "=="; }
};

struct STRNE {
    bool operator()(const char* lhs, const char* rhs) { return strcmp(lhs, rhs) != 0; }
    static std::string name() { return "!="; }
};

struct STRCASEEQ {
    bool operator()(const char* lhs, const char* rhs) { 
        return detail::stricmp(lhs, rhs) == 0; 
    }
    static std::string name() { return "=="; }
};

struct STRCASENE {
    bool operator()(const char* lhs, const char* rhs) { 
        return detail::stricmp(lhs, rhs) != 0; 

    }
    static std::string name() { return "!="; }
};

struct FLOATEQ {
    template<typename T>
    bool operator()(const T& lhs, const T& rhs) {
        return detail::Floating::almostEqual(lhs, rhs);
    }
    static std::string name() { return "=="; }
};

struct FLOATNE {
    template<typename T>
    bool operator()(const T& lhs, const T& rhs) {    
        return !FLOATEQ()(lhs, rhs);
    }
    static std::string name() { return "!="; }
};

/***** compare implementation *****/

template<typename T1, typename T2, typename OP>
bool compare(const T1& expected, const T2& actual, OP op, 
             const char* expected_str, const char* actual_str, const char* file, int line) {
    bool test_success = op(expected, actual);

    if (!test_success) {
        framework::TestState::getCurrentTest()->setFailure(
            framework::Failure(file, line,
            detail::makeExpressionStr(expected_str, actual_str, op),
            detail::makeExpressionStr(expected, actual, op)));        
    }
    return test_success;
}

template<typename T1, typename T2, typename T3>
bool compare_near(const T1& expected, const T2& actual, const T3& abs_error, 
             const char* expected_str, const char* actual_str, const char* file, int line) {
    bool test_success = std::abs(expected - actual) <= abs_error;

    if (!test_success) {
        framework::TestState::getCurrentTest()->setFailure(
            framework::Failure(file, line,
            detail::makeExpressionStr(expected_str, actual_str, picotest::FLOATEQ()),
            detail::makeExpressionStr(expected, actual, picotest::FLOATEQ())));        
    }
    return test_success;
}

inline bool evaluate(bool expected, bool actual, const char* expression, const char* file, int line) {
    bool test_success = expected == actual;

    if (!test_success) {
        framework::TestState::getCurrentTest()->setFailure(
            framework::Failure(file, line, expression, expected));    
    }

    return test_success;
}

} // namespace picotest

// using namespace testing for compatibility with google test
namespace testing {

class Test {
public:
    Test() {}
    virtual ~Test() {}
    void execute() {
        SetUp();
        test_method(); // template-method
        TearDown();
    }
protected:
    virtual void SetUp() {}
    virtual void TearDown() {}
    virtual void test_method() = 0;
};

} // namespace testing


/////////////////////////////////////////////////////////////////
// test with auto-registration

#define PICOTEST_IDENITY(test_case_name, test_name) PICOTEST_JOIN(test_case_name, test_name)

#define PICOTEST_TEST_CASE_INVOKER(test_case_name, test_name) \
PICOTEST_JOIN(PICOTEST_IDENITY(test_case_name, test_name), _invoker)

#define PICOTEST_TEST_CASE_REGISTRAR(test_case_name, test_name) \
static picotest::framework::Registrar PICOTEST_JOIN(PICOTEST_IDENITY(test_case_name, test_name), _registrar)

#define PICOTEST_MAKE_TEST(test_case_name, test_name) \
picotest::framework::Test(PICOTEST_STR(test_name), PICOTEST_TEST_CASE_INVOKER(test_case_name, test_name))

#define TEST(test_case_name, test_name) \
PICOTEST_TEST_CASE_AUTO_REGISTER(test_case_name, test_name, ::testing::Test)


#define TEST_F(test_fixture, test_name) \
PICOTEST_TEST_CASE_AUTO_REGISTER(test_fixture, test_name, test_fixture)


#define PICOTEST_TEST_CASE_AUTO_REGISTER(test_case_name, test_name, base_t) \
struct PICOTEST_IDENITY(test_case_name, test_name) : public base_t {        \
    void test_method();                                                     \
};                                                                          \
                                                                            \
void PICOTEST_TEST_CASE_INVOKER(test_case_name, test_name)() {              \
    PICOTEST_IDENITY(test_case_name, test_name) t;                          \
    t.execute();                                                            \
}                                                                           \
                                                                            \
PICOTEST_TEST_CASE_REGISTRAR(test_case_name, test_name)(                    \
    PICOTEST_STR(test_case_name),                                           \
    PICOTEST_MAKE_TEST(test_case_name, test_name));                         \
                                                                            \
void PICOTEST_IDENITY(test_case_name, test_name)::test_method()


/////////////////////////////////////////////////////////////////
// EXPECT_XX

#define EXPECT_BOOL(expected, expression) \
    picotest::evaluate(expected, expression, #expression, __FILE__, __LINE__)
#define EXPECT_BINARY(lhs, rhs, OP) \
    picotest::compare(lhs, rhs, OP(), #lhs, #rhs, __FILE__, __LINE__)
#define EXPECT_BINARY_NEAR(lhs, rhs, abs_error) \
    picotest::compare_near(lhs, rhs, abs_error, #lhs, #rhs, __FILE__, __LINE__)

#define EXPECT_TRUE(cond) EXPECT_BOOL(true, cond)
#define EXPECT_FALSE(cond) EXPECT_BOOL(false, cond)
#define EXPECT_EQ(expected, actual) EXPECT_BINARY(expected, actual, picotest::EQ)
#define EXPECT_NE(expected, actual) EXPECT_BINARY(expected, actual, picotest::NE)
#define EXPECT_LT(expected, actual) EXPECT_BINARY(expected, actual, picotest::LT)
#define EXPECT_GT(expected, actual) EXPECT_BINARY(expected, actual, picotest::GT)
#define EXPECT_LE(expected, actual) EXPECT_BINARY(expected, actual, picotest::LE)
#define EXPECT_GE(expected, actual) EXPECT_BINARY(expected, actual, picotest::GE)
#define EXPECT_STREQ(expected_str, actual_str) EXPECT_BINARY(expected_str, actual_str, picotest::STREQ)
#define EXPECT_STRNE(expected_str, actual_str) EXPECT_BINARY(expected_str, actual_str, picotest::STRNE)
#define EXPECT_STRCASEEQ(expected_str, actual_str) EXPECT_BINARY(expected_str, actual_str, picotest::STRCASEEQ)
#define EXPECT_STRCASENE(expected_str, actual_str) EXPECT_BINARY(expected_str, actual_str, picotest::STRCASENE)
#define EXPECT_FLOAT_EQ(expected, actual)  EXPECT_BINARY(expected, actual, picotest::FLOATEQ)
#define EXPECT_DOUBLE_EQ(expected, actual) EXPECT_BINARY(expected, actual, picotest::FLOATEQ)
#define EXPECT_FLOAT_NE(expected, actual)  EXPECT_BINARY(expected, actual, picotest::FLOATNE)
#define EXPECT_DOUBLE_NE(expected, actual) EXPECT_BINARY(expected, actual, picotest::FLOATNE)
#define EXPECT_NEAR(expected, actual, abs_error) EXPECT_BINARY_NEAR(expected, actual, abs_error)

/////////////////////////////////////////////////////////////////
// ASSERT_XX

#define ASSERT_BOOL(expected, expression) \
do {\
    if (!EXPECT_BOOL(expected, expression)){\
        return;\
    }\
} while(0)

#define ASSERT_BINARY(lhs, rhs, OP) \
do {\
    if (!EXPECT_BINARY(lhs, rhs, OP)){\
        return;\
    }\
} while(0)

#define ASSERT_TRUE(cond) ASSERT_BOOL(true, cond)
#define ASSERT_FALSE(cond) ASSERT_BOOL(false, cond)
#define ASSERT_EQ(expected, actual) ASSERT_BINARY(expected, actual, picotest::EQ)
#define ASSERT_NE(expected, actual) ASSERT_BINARY(expected, actual, picotest::NE)
#define ASSERT_LT(expected, actual) ASSERT_BINARY(expected, actual, picotest::LT)
#define ASSERT_GT(expected, actual) ASSERT_BINARY(expected, actual, picotest::GT)
#define ASSERT_LE(expected, actual) ASSERT_BINARY(expected, actual, picotest::LE)
#define ASSERT_GE(expected, actual) ASSERT_BINARY(expected, actual, picotest::GE)
#define ASSERT_STREQ(expected_str, actual_str) ASSERT_BINARY(expected_str, actual_str, picotest::STREQ)
#define ASSERT_STRNE(expected_str, actual_str) ASSERT_BINARY(expected_str, actual_str, picotest::STRNE)
#define ASSERT_STRCASEEQ(expected_str, actual_str) ASSERT_BINARY(expected_str, actual_str, picotest::STRCASEEQ)
#define ASSERT_STRCASENE(expected_str, actual_str) ASSERT_BINARY(expected_str, actual_str, picotest::STRCASENE)
#define ASSERT_FLOAT_EQ(expected, actual)  ASSERT_BINARY(expected, actual, picotest::FLOATEQ)
#define ASSERT_DOUBLE_EQ(expected, actual) ASSERT_BINARY(expected, actual, picotest::FLOATEQ)
#define ASSERT_FLOAT_NE(expected, actual)  ASSERT_BINARY(expected, actual, picotest::FLOATNE)
#define ASSERT_DOUBLE_NE(expected, actual) ASSERT_BINARY(expected, actual, picotest::FLOATNE)

/////////////////////////////////////////////////////////////////
// RUNNING ALL TESTS

#define RUN_ALL_TESTS() \
picotest::framework::Registry::getInstance().testRun(); \
picotest::framework::Registry::getInstance().report(std::cout); \
picotest::framework::Registry::getInstance().fail();
