/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include <exception>
#include <string>
#include "tiny_dnn/util/colored_print.h"

namespace tiny_dnn {

/**
 * error exception class for tiny-dnn
 **/
class nn_error : public std::exception {
 public:
  explicit nn_error(const std::string &msg) : msg_(msg) {}
  const char *what() const throw() override { return msg_.c_str(); }

 private:
  std::string msg_;
};

/**
 * warning class for tiny-dnn (for debug)
 **/
class nn_warn {
 public:
  explicit nn_warn(const std::string &msg) : msg_(msg) {
#ifdef CNN_USE_STDOUT
    coloredPrint(Color::YELLOW, msg_h_ + msg_);
#endif
  }

 private:
  std::string msg_;
  std::string msg_h_ = std::string("[WARNING] ");
};

/**
 * info class for tiny-dnn (for debug)
 **/
class nn_info {
 public:
  explicit nn_info(const std::string &msg) : msg_(msg) {
#ifdef CNN_USE_STDOUT
    std::cout << msg_h + msg_ << std::endl;
#endif
  }

 private:
  std::string msg_;
  std::string msg_h = std::string("[INFO] ");
};

class nn_not_implemented_error : public nn_error {
 public:
  explicit nn_not_implemented_error(const std::string &msg = "not implemented")
    : nn_error(msg) {}
};

}  // namespace tiny_dnn
