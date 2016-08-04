// Copyright 2016 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// iterator.h: Const forward iterators for VectorMap and VectorDup that help
// access data in architecture specific way, e.g. 4 elements at a time for NEON.

#ifndef GEMMLOWP_INTERNAL_ITERATOR_H_
#define GEMMLOWP_INTERNAL_ITERATOR_H_

namespace gemmlowp {

enum class VectorShape;

// ConstIterator is a forward only constant iterator that can be made
// architecture specific e.g. to return 4 values at once for NEON.
template <typename VectorType> class ConstIterator {
  // Unused default case.
};

template <typename tScalar, VectorShape tShape> class VectorMap;

template <typename tScalar, VectorShape tShape>
class ConstIterator<VectorMap<tScalar, tShape>> {
 public:
  typedef tScalar Scalar;
  ConstIterator(const VectorMap<tScalar, tShape>& vector_map,
                const int start_offset)
      : pointer_(vector_map.data() + start_offset) {}
  const Scalar operator*() const { return *pointer_; }
  const Scalar* get() const { return pointer_; }
  ConstIterator& operator+=(int inc) { pointer_ += inc; return *this; }
 private:
  const Scalar* pointer_;
};

template <typename tScalar, VectorShape tShape>
ConstIterator<VectorMap<tScalar, tShape>> const_iterator(
    const VectorMap<tScalar, tShape>& vector_map,
    const int start_offset) {
  return ConstIterator<VectorMap<tScalar, tShape>>(vector_map, start_offset);
}

template <typename tScalar, VectorShape tShape> class VectorDup;

template <typename tScalar, VectorShape tShape>
class ConstIterator<VectorDup<tScalar, tShape>> {
 public:
  typedef tScalar Scalar;
  ConstIterator(const VectorDup<tScalar, tShape>& vector_dup)
      : data_(vector_dup(0)) {}
  const Scalar operator*() const { return data_; }
  const Scalar* get() const { return &data_; }
  ConstIterator& operator+=(int inc) { return *this; }
 private:
  Scalar data_;
};

template <typename tScalar, VectorShape tShape>
ConstIterator<VectorDup<tScalar, tShape>> const_iterator(
    const VectorDup<tScalar, tShape>& vector_map,
    const int start_offset) {
  return ConstIterator<VectorDup<tScalar, tShape>>(vector_map);
}

}  // namespace gemmlowp

#endif  // GEMMLOWP_INTERNAL_ITERATOR_H_
