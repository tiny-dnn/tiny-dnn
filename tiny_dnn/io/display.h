/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/

// addapted from boost progress.hpp, made c++11 only //

#pragma once

#include <chrono>    // for high_resolution_clock, NOLINT
#include <iostream>  // for ostream, cout, etc
#include <string>    // for string

namespace tiny_dnn {

class timer {
 public:
  timer() : t1(std::chrono::high_resolution_clock::now()) {}
  float_t elapsed() {
    return std::chrono::duration_cast<std::chrono::duration<float_t>>(
             std::chrono::high_resolution_clock::now() - t1)
      .count();
  }
  void restart() { t1 = std::chrono::high_resolution_clock::now(); }
  void start() { t1 = std::chrono::high_resolution_clock::now(); }
  void stop() { t2 = std::chrono::high_resolution_clock::now(); }
  float_t total() {
    stop();
    return std::chrono::duration_cast<std::chrono::duration<float_t>>(t2 - t1)
      .count();
  }
  ~timer() {}

 private:
  std::chrono::high_resolution_clock::time_point t1, t2;
};

//  progress_display  --------------------------------------------------------//

//  progress_display displays an appropriate indication of
//  progress at an appropriate place in an appropriate form.

class progress_display {
 public:
  explicit progress_display(size_t expected_count_,
                            std::ostream &os      = std::cout,
                            const std::string &s1 = "\n",  // leading strings
                            const std::string &s2 = "",
                            const std::string &s3 = "")
    // os is hint; implementation may ignore, particularly in embedded systems
    : m_os(os), m_s1(s1), m_s2(s2), m_s3(s3) {
    restart(expected_count_);
  }

  void restart(size_t expected_count_) {
    //  Effects: display appropriate scale
    //  Postconditions: count()==0, expected_count()==expected_count_
    _count = _next_tic_count = _tic = 0;
    _expected_count                 = expected_count_;

    m_os << m_s1 << "0%   10   20   30   40   50   60   70   80   90   100%\n"
         << m_s2 << "|----|----|----|----|----|----|----|----|----|----|"
         << std::endl  // endl implies flush, which ensures display
         << m_s3;
    if (!_expected_count) _expected_count = 1;  // prevent divide by zero
  }                                             // restart

  size_t operator+=(size_t increment) {
    //  Effects: Display appropriate progress tic if needed.
    //  Postconditions: count()== original count() + increment
    //  Returns: count().
    if ((_count += increment) >= _next_tic_count) {
      display_tic();
    }
    return _count;
  }

  size_t operator++() { return operator+=(1); }
  size_t count() const { return _count; }
  size_t expected_count() const { return _expected_count; }

 private:
  std::ostream &m_os;      // may not be present in all imps
  const std::string m_s1;  // string is more general, safer than
  const std::string m_s2;  //  const char *, and efficiency or size are
  const std::string m_s3;  //  not issues

  size_t _count, _expected_count, _next_tic_count;
  size_t _tic;
  void display_tic() {
    // use of floating point ensures that both large and small counts
    // work correctly.  static_cast<>() is also used several places
    // to suppress spurious compiler warnings.
    size_t tics_needed = static_cast<size_t>(
      (static_cast<double>(_count) / _expected_count) * 50.0);
    do {
      m_os << '*' << std::flush;
    } while (++_tic < tics_needed);
    _next_tic_count = static_cast<size_t>((_tic / 50.0) * _expected_count);
    if (_count == _expected_count) {
      if (_tic < 51) m_os << '*';
      m_os << std::endl;
    }
  }  // display_tic

  progress_display &operator=(const progress_display &) = delete;
};

}  // namespace tiny_dnn
