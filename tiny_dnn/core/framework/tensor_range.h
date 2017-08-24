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

/**
 * Continuoius range of current axis
 * @param begin
 * @param end
 * @return
 */
inline TensorRangeClass<size_t> TensorRange(const size_t begin,
                                            const size_t end) {
  return TensorRangeClass<size_t>(begin, end);
}

template <typename T = size_t>
class TensorSingleIndexClass {
 public:
  explicit TensorSingleIndexClass(const T index) : range(index) {}
  T get_range() { return range; }

 private:
  T range;
};

/**
 * Single index of current axis. Reduces dimensions by 1
 * @param index
 * @return
 */
inline TensorSingleIndexClass<size_t> TensorSingleIndex(const size_t index) {
  return TensorSingleIndexClass<size_t>(index);
}

template <typename T = size_t>  // Just for uniformity, not really needed
class TensorAllClass {
 public:
  xt::xall_tag get_range() { return xt::all(); }
};

/**
 * New axis of dim 1
 * @return
 */
inline TensorAllClass<size_t> TensorAll() { return TensorAllClass<size_t>(); }

template <typename T = size_t>  // Just for uniformity, not really needed
class TensorNewAxisClass {
 public:
  xt::xnewaxis_tag get_range() { return xt::newaxis(); }
};

/**
 * New axis of dim 1
 * @return
 */
inline TensorNewAxisClass<size_t> TensorNewAxis() {
  return TensorNewAxisClass<size_t>();
}
}  // namespace tiny_dnn
