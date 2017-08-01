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
  TensorRangeClass(const T begin, const T end) : range(begin, end) {}
  xt::xrange<T> get_range() { return range; }

 private:
  xt::xrange<T> range;
};

// TODO(Randnl): Use int16/int64/etc, rather than the C type long  [runtime/int]
// [4]
inline TensorRangeClass<unsigned long long>                 // NOLINT
  TensorRange(const unsigned long long begin,               // NOLINT
              const unsigned long long end) {               // NOLINT
  return TensorRangeClass<unsigned long long>(begin, end);  // NOLINT
}

template <typename T = size_t>
class TensorSingleIndexClass {
 public:
  explicit TensorSingleIndexClass(const T index) : range(index) {}
  T get_range() { return range; }

 private:
  T range;
};

// TODO(Randnl): Use int16/int64/etc, rather than the C type long  [runtime/int]
// [4]
inline TensorSingleIndexClass<unsigned long long>            // NOLINT
  TensorSingleIndex(const unsigned long long index) {        // NOLINT
  return TensorSingleIndexClass<unsigned long long>(index);  // NOLINT
}

}  // namespace tiny_dnn
