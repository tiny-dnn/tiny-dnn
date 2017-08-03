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

// TODO(Randl): Use int16/int64/etc, rather than the C type long  [runtime/int]
// [4]
inline TensorRangeClass<size_t>                 // NOLINT
  TensorRange(const size_t begin,               // NOLINT
              const size_t end) {               // NOLINT
  return TensorRangeClass<size_t>(begin, end);  // NOLINT
}

template <typename T = size_t>
class TensorSingleIndexClass {
 public:
  explicit TensorSingleIndexClass(const T index) : range(index) {}
  T get_range() { return range; }

 private:
  T range;
};

// TODO(Randl): Use int16/int64/etc, rather than the C type long  [runtime/int]
// [4]
inline TensorSingleIndexClass<size_t>            // NOLINT
  TensorSingleIndex(const size_t index) {        // NOLINT
  return TensorSingleIndexClass<size_t>(index);  // NOLINT
}

template <typename T = size_t>  // Just for uniformity, not really needed
class TensorAllClass {
 public:
  xt::xall_tag get_range() { return xt::all(); }
};

inline TensorAllClass<size_t> TensorAll() { return TensorAllClass<size_t>(); }
}  // namespace tiny_dnn
