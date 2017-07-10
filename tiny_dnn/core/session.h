/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "tiny_dnn/core/device.h"

namespace tiny_dnn {
namespace core {

class session {
 public:
  explicit session(const std::string name) : name_(name) {}

  std::string get_name() const { return name_; }
  size_t get_num_devices() const { return devices_.size(); }

  // will call construct graph
  // should we here specify the devices to use?
  void schedule_session(/* network<sequential>& net */);

  // will call forward or backward methods
  void run_session(/* data */);

 private:
  std::string name_;
  std::vector<std::shared_ptr<device>> devices_;
};

}  // namespace core
}  // namespace tiny_dnn
