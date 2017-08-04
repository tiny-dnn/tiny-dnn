/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include "tiny_dnn/core/framework/device.fwd.h"
#include "tiny_dnn/core/framework/tensor_range.h"

#if defined(USE_OPENCL) || defined(USE_CUDA)
#ifdef USE_OPENCL
#include "third_party/CLCudaAPI/clpp11.h"
#else
#include "third_party/CLCudaAPI/cupp11.h"
#endif
#endif

namespace tiny_dnn {

/**
 * A tensor of the given dimension.
 *
 * U is type of the data stored
 * Storage is type of underlying storage
 */
template <typename U = float_t, typename Storage = xt::xarray<U>>
class Tensor {
  typedef U *UPtr;

 public:
  /**
   * Initializes an empty tensor.
   * @return
   */
  Tensor() {}

  /**
   * Initializes a tensor from storage.
   * @return
   */
  explicit Tensor(Storage &&s) : storage_(std::move(s)) {}

  /**
   * Constructor that accepts an initializer list of shape and create a
   * Tensor with that shape. For example, given shape = {2,3,4,5,6}, tensor
   * will be of size 2x3x4x5x6. Note: tensor isn't initialized by default
   * @param shape array containing N integers, sizes of dimensions
   * @return
   */
  explicit Tensor(std::vector<size_t> const &shape) : storage_(shape) {}

  /**
   * Constructor that accepts an initializer list of shape and create a
   * Tensor with that shape. For example, given shape = {2,3,4,5,6}, tensor
   * will be of size 2x3x4x5x6. Note: tensor isn't initialized by default
   * @param shape array containing N integers, sizes of dimensions
   * @return
   */
  explicit Tensor(std::initializer_list<size_t> const &shape)
    : storage_(shape) {}

  /**
   * Constructor that accepts an initializer list of shape and create a
   * Tensor with that shape and filling it with value. For example,
   * given shape = {2,3,4,5,6}, value = 0 tensor will be of size 2x3x4x5x6
   * filled with zeros
   * @param shape  shape array containing N integers, sizes of dimensions
   * @param value value to fill
   */
  explicit Tensor(std::initializer_list<size_t> const &shape, U value)
    : storage_(shape, value) {}

  /**
   * Temporal method to create a new Tensor from old tensor_t
   */
  explicit Tensor(const tensor_t &data) {
    std::vector<size_t> shape = {data.size(), data[0].size()};
    storage_                  = Storage(shape);

    // deep copy tensor data
    for (size_t i = 0; i < data.size(); ++i) {
      for (size_t j = 0; j < data[i].size(); ++j) {
        storage_(i, j) = data[i][j];
      }
    }
  }

  /**
   * Temporal method to create a new Tensor from old vec_t
   */
  explicit Tensor(const vec_t &data) {
    std::vector<size_t> shape = {data.size()};
    storage_                  = Storage(shape);

    // deep copy tensor data
    for (size_t i = 0; i < data.size(); ++i) {
      storage_(i) = data[i];
    }
  }

  Tensor(const Tensor &other) = default;         // copy ctor
  Tensor &operator=(const Tensor &T) = default;  // copy assign

  // ~Tensor() = default;

  Tensor(Tensor &&other) = default;             // move ctor
  Tensor &operator=(Tensor &&other) = default;  // move assign

// TODO(Randl): implement copy and move constructors
#if 0
  // TODO(Randl) :deep copy
    Tensor(const Tensor&other) {
        other.fromDevice();
        shape_ = other.shape_;
        storage_pointer_ = other.storage_pointer_;
        data_is_on_host_ = true;
        data_dirty_ = true;
        // device_data_ is intentionally left uninitialized.
    }
    // TODO(Randl): Move constructors for Tensor and TensorStorage

#endif

  /**
   *
   * @return the tensor shape
   */
  const std::vector<size_t> &shape() const { return storage_.shape(); }

  /**
   *
   * @return Tensor's number of dimensions
   */
  size_t dim() const { return storage_.dimension(); }

  /**
   *
   * @return is tensor is empty
   */
  bool empty() const { return size() == 0; }

  /**
   *
   * @return the total number of elements in Tensor
   */
  size_t size() const { return storage_.size(); }

  /**
   * Access to indexes in tensor
   * @param args indexes in tensor
   * @return the value of a specified index in the tensor
   */
  template <typename... Args>
  U &host_at(const Args... args) {
    return storage_(args...);
  }

  /**
   * Constant access to indexes in tensor
   * @param args indexes in tensor
   * @return the value of a specified index in the tensor
   */
  template <typename... Args>
  U host_at(const Args... args) const {
    return storage_(args...);
  }

  /**
   *
   * @return Iterator pointing to the beginning of Tensor
   */
  auto host_begin() { return storage_.begin(); }

  const auto host_begin() const { return storage_.cbegin(); }

  template <typename... Args>
  auto host_iter(const Args... args) {
    return std::next(storage_.begin(), host_offset(args...));
  }

  template <typename... Args>
  const auto host_iter(const Args... args) const {
    return std::next(storage_.cbegin(), host_offset(args...));
  }

  // TODO(Randl): rename?
  auto host_end() { return storage_.end(); }

  const auto host_end() const { return storage_.cend(); }

  U *host_pbegin() { return &*storage_.begin(); }

  const U *host_pbegin() const { return &*storage_.cbegin(); }

  U *host_pend() { return &*storage_.end(); }

  const U *host_pend() const { return &*storage_.cend(); }

  // TODO(Randl): check if strided.
  template <typename... Args>
  U *host_pointer(const Args... args) {
    return &*host_iter(args...);
  }

  template <typename... Args>
  const U *host_pointer(const Args... args) const {
    return &*host_iter(args...);
  }
  /**
   * Calculate an offset for last dimension.
   * @param d an index of last dimension
   * @return offest from the beginning of the dimesion
   */
  size_t host_offset(const size_t d) const { return d; }

  /**
   * Calculate an offest in 1D representation of nD Tensor. Parameters are
   * indexes of k last dimensions. If k is less than n, function returns an
   * offset from the first index of (n-k+1)th dimension. This allows recursive
   * call to acquire offset for generic number of dimensions
   * @param d index of (k-n)th dimension. For external call, n=k usually holds
   * @param args index of rest (k-1) dimensions.
   * @return offset from the first index of (n-k)th dimension
   */
  template <typename... Args>
  size_t host_offset(const size_t d, const Args... args) const {
    size_t dim = storage_.dimension() - sizeof...(args) - 1;
    /*if (d >= storage.shape()[dim]) {
      throw nn_error("Access tensor out of range.");
    }*/
    size_t shift = 1;
    for (size_t i = dim + 1; i < storage_.dimension(); ++i)
      shift *= storage_.shape()[i];  // TODO(Randl): optimize. Reverse argumets?

    return (d * shift + host_offset(args...));
  }

// TODO(Randl)
/*
const auto host_flatten() const {
  // fromDevice();
  return xt::broadcast(storage_, {size()});
}

auto host_data() {
  // fromDevice();
  return xt::broadcast(storage_, {size()});
}*/
// TODO(ּּRandl): should we enable this again?
#if 0
  U* mutable_host_data() {
        static_assert(!kConst, "Non-constant operation on constant Tensor");
        // fromDevice();
        // data_dirty_ = true;
        return storage_pointer_->data(offset);
    }
#endif

#if defined(USE_OPENCL) || defined(USE_CUDA)
  const void *device_data() const {
    /*storage_ptr_->toDevice();
    return (*storage_ptr_->device_data_)();*/
  }

  void *mutable_device_data() {
    /*storage_ptr_->toDevice();
    storage_ptr_->data_dirty_ = true;
    return (*storage_ptr_->device_data_)();*/
  }
#endif

  /**
   * Fill tensor with particular value
   * @param value
   * @return
   */
  Tensor &fill(U value) {
    // static_assert(!kConst, "Non-constant operation on constant Tensor");
    // data_is_on_host_ = true;
    // data_dirty_ = true;
    std::fill(storage_.begin(), storage_.end(), value);
    return *this;
  }

  /**
   * Fill tensor with another tensor repeated size times over first axis
   * @tparam S storage of parameter tensor
   * @param size
   * @param tensor
   * @return
   */
  template <typename S>
  Tensor &repeat(size_t size, Tensor<U, S> tensor) {
    auto shape = tensor.shape();
    shape.insert(shape.begin(), size);
    storage_.reshape(shape);
    for (size_t i = 0; i < size; ++i) storage_[i] = tensor.storage_;
    return *this;
  }

  template <typename S>
  Tensor &assign(Tensor<U, S> tensor) {
    auto in  = tensor.host_begin();
    auto out = host_begin();
    for (; in != tensor.host_end(); ++in, ++out) *out = *in;
    return *this;
  }

  // TODO(Randl): checked version
  /**
   * Reshape tensor
   * @param shape new shape
   */
  void reshape(const std::vector<size_t> &shape) { storage_.reshape(shape); }

  /**
   * Reshapes to shape {}
   */
  void clear() { storage_.reshape({}); }

  // TODO(Randl): checked version
  /**
   * Reshape tensor
   * @param shape new shape
   */
  void resize_axis(const size_t value, const size_t axis = 0) {
    auto curr  = shape();
    curr[axis] = value;
    storage_.reshape(curr);
  }

  Tensor operator[](size_t index) { return Tensor(storage_[index]); }
  const Tensor operator[](size_t index) const {
    return Tensor(storage_[index]);
  }
  // U host_at(size_t index) const { return storage_(index); }
  // U &host_at(size_t index) { return storage_(index); }

  /**
   * Returns view of current Tensor
   * @tparam Values index type
   * @tparam InputRanges
   * @param ranges
   * @return
   */
  template <typename... Values, template <typename> class... InputRanges>
  auto subView(InputRanges<Values>... ranges) {
    // TODO(Randl): stride
    using ViewType     = decltype(xt::view(storage_, ranges.get_range()...));
    using SharedTensor = Tensor<U, ViewType>;
    return SharedTensor(xt::view(storage_, ranges.get_range()...));
  }

  /**
   * Returns view of current Tensor
   * @tparam Values index type
   * @tparam InputRanges
   * @param ranges
   * @return
   */
  template <typename... Values, template <typename> class... InputRanges>
  auto subView(InputRanges<Values>... ranges) const {
    // TODO(Randl): stride
    using ViewType =
      decltype(xt::view((const Storage)storage_, ranges.get_range()...));
    using SharedTensor = Tensor<U, ViewType>;
    return SharedTensor(
      xt::view((const Storage)storage_, ranges.get_range()...));
  }

  /*
  // TODO: is needed?
  bool isSubView() const {
    return true; //std::is_same(Storage, xt::xview<U>);
  }
*/

  // TODO(Randl): base
  template <typename T, typename S>
  friend inline std::ostream &operator<<(std::ostream &os,
                                         const Tensor<T, S> &tensor);

  /**
   * Temporal method to convert new Tensor to tensor_t
   * @return
   */
  tensor_t toTensor() const {
    tensor_t tensor(storage_.shape()[0]);
    for (size_t i = 0; i < storage_.shape()[0]; ++i) {
      tensor[i].resize(storage_.shape()[1]);
      for (size_t j = 0; j < storage_.shape()[1]; ++j) {
        tensor[i][j] = storage_(i, j);
      }
    }
    return tensor;
  }

  /**
   * Temporary method to convert new Tensor to vec_t
   * @return
   */
  vec_t toVec() const {
    vec_t tensor(storage_.shape()[0]);
    auto vec = tensor.begin();
    auto ten = host_begin();
    for (; vec != tensor.end(); ++vec, ++ten) {
      *vec = *ten;
    }
    return tensor;
  }

  /**
   * Temporary method.
   * @return
   */
  Tensor &fromVec(vec_t vect) {
    for (size_t i = 0; i < storage_.shape()[0]; ++i) {
      storage_(i) = vect[i];
    }
    return *this;
  }

  /**
   * Temporal method to convert line of new Tensor to vec_t
   * @return
   */
  vec_t lineToVec(size_t line) const {
    vec_t tensor(storage_.shape()[0]);
    for (size_t i = 0; i < storage_.shape()[0]; ++i) {
      tensor[i] = storage_(line, i);
    }
    return tensor;
  }

  /**
   * Creates Tensor given the storage
   * @tparam T
   * @param storage
   */
  template <class T, class S, class... Args>
  explicit Tensor(T &storage, xt::xrange<S> r1, Args... args)
    : storage_(xt::view(storage, r1, args...)) {}

  template <typename T, typename S>
  friend inline std::ostream &operator<<(std::ostream &os,
                                         const Tensor<T, S> &tensor);

 private:
  Storage storage_;
};

using ViewTensor =
  decltype(Tensor<>({2, 2}).subView(TensorSingleIndex(1), TensorAll()));
using ConstViewTensor = decltype(((const Tensor<>)Tensor<>({2, 2}))
                                   .subView(TensorSingleIndex(1), TensorAll()));
}  // namespace tiny_dnn
