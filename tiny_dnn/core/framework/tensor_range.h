/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

namespace tiny_dnn {

template <typename T = size_t>
class TensorRangeClass {
 public:
  TensorRangeClass(const T begin, const T end) : range(begin, end){};
  xt::xrange<T> get_range() { return range; }

 private:
  xt::xrange<T> range;
};

inline TensorRangeClass<unsigned long long> TensorRange(
  const unsigned long long begin, const unsigned long long end) {
  return TensorRangeClass<unsigned long long>(begin, end);
}

template <typename T = size_t>
class TensorSingleIndexClass {
 public:
  TensorSingleIndexClass(const T index) : range(index){};
  T get_range() { return range; }

 private:
  T range;
};

inline TensorSingleIndexClass<unsigned long long> TensorSingleIndex(
  const unsigned long long index) {
  return TensorSingleIndexClass<unsigned long long>(index);
}

}  // namespace tiny_dnn