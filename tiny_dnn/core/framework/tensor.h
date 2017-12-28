/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include <memory>
#include <type_traits>
#include <vector>

#include "tiny_dnn/core/framework/device.fwd.h"

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
   *
   * @param T
   * @return
   */
  Tensor<U> &operator=(const Tensor<U> &T) {
    storage_ = T.storage_;
    return *this;
  }

// ~Tensor() = default;

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
    Tensor &operator = (const Tensor& other) {}

    Tensor(Tensor&& other) = default;         // move ctor
    Tensor &operator = (Tensor&&) = default;  // move assign
#endif

  /**
   *
   * @return the tensor shape
   */
  const auto shape() const { return storage_.shape(); }

  /**
   *
   * @return Tensor's number of dimensions
   */
  const auto dim() const { return storage_.dimension(); }

  /**
   *
   * @return the total number of elements in Tensor
   */
  const size_t size() const { return storage_.size(); }

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

  auto host_end() { return storage_.end(); }

  const auto host_end() const { return storage_.cend(); }

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
    storage_ptr_->toDevice();
    return (*storage_ptr_->device_data_)();
  }

  void *mutable_device_data() {
    static_assert(!kConst, "Non-constant operation on constant Tensor");
    storage_ptr_->toDevice();
    storage_ptr_->data_dirty_ = true;
    return (*storage_ptr_->device_data_)();
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

  // TODO(Randl): checked version
  /**
   * Reshape tensor
   * @param shape new shape
   */
  void reshape(const std::vector<size_t> &shape) { storage_.reshape(shape); }

  Tensor operator[](size_t index) { return Tensor(storage_[index]); }

  /**
   * Returns new Tensor which has shared storage with current, but different
   * shape
   *
   * Example:
   * @code
   * Tensor b({4,2});
   * a = b.subView({2,2,2};
   * @endcode
   * b is Tensor of shape 4x2 and a is Tensor of shape 2x2x2. Changing a(0,0,0)
   * will change b(0,0) too.
   * @param new_shape
   * @return
   */
  /*
 Tensor<U,xt::xbroadcast<Storage, std::vector<size_t>>>
 subView(std::initializer_list<size_t> const &new_shape) {
   auto res = Tensor<U,xt::xbroadcast<Storage,
 std::vector<size_t>>>(xt::broadcast(storage_, storage_.shape()));
   res.storage_.reshape(new_shape);
   return res;
 }*/

  // TODO(blackccpie): when upgrading to xtensor 0.14.0 and beyond,
  // the std::make_unsigned_t trick will be useless.
  // for further details see :
  // http://github.com/QuantStack/xtensor/issues/594
  /**
   * Returns view of current Tensor
   * @tparam Values index type
   */
  template <typename... Values>
  Tensor<U, xt::xview<Storage &, xt::xrange<std::make_unsigned_t<Values>>...>>
  subView(std::initializer_list<Values>... lists) {
    using SharedTensor =
      Tensor<U,
             xt::xview<Storage &, xt::xrange<std::make_unsigned_t<Values>>...>>;
    return SharedTensor(storage_,
                        xt::range(*(lists.begin()), *(lists.begin() + 1))...);
  }

  /*
  // TODO: is needed?
  bool isSubView() const {
    return true; //std::is_same(Storage, xt::xview<U>);
  }
*/

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

}  // namespace tiny_dnn
