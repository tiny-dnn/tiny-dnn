/*
    COPYRIGHT

    All contributions by Taiga Nomi
    Copyright (c) 2013, Taiga Nomi
    All rights reserved.

    All other contributions:
    Copyright (c) 2013-2016, the respective contributors.
    All rights reserved.

    Each contributor holds copyright over their respective contributions.
    The project versioning (Git) records all such contribution source
   information.

    LICENSE

    The BSD 3-Clause License


    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice,
   this
      list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright notice,
      this list of conditions and the following disclaimer in the documentation
      and/or other materials provided with the distribution.

    * Neither the name of tiny-dnn nor the names of its
      contributors may be used to endorse or promote products derived from
      this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
   ARE
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
    FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
    DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
    CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
   LIABILITY,
    OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
   USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
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

template <typename Container>
static inline size_t product(Container &c) {
  return std::accumulate(std::begin(c), std::end(c), size_t(1),
                         std::multiplies<size_t>());
}

template <typename C1, typename C2>
static inline size_t compute_offset(const C1 &start, const C2 &shape) {
  size_t res = 0;
  for (size_t i = 0; i < shape.size(); ++i) {
    res *= shape[i];
    res += (i < start.size()) ? *(start.begin() + i) : 0;
  }
  return res;
}

template <typename U = float_t, typename Allocator = aligned_allocator<U, 64>>
class TensorStorage {
  typedef typename std::vector<U, Allocator>::iterator DataIter;
  typedef typename std::vector<U, Allocator>::const_iterator ConstDataIter;

 public:
  /**
   * Initializes an empty tensor storage.
   * @return
   */
  TensorStorage() {}

  /**
   * Constructor that assepts a vector of shape and create a TensorStorage
   * with a size equivalent to that shape.
   * @param shape array containing N integers, sizes of dimensions
   * @return
   */
  explicit TensorStorage(const std::vector<size_t> &shape) { resize(shape); }

  /**
   * Constructor that assepts an initializer list  of shape and create a
   * TensorStorage with a size equivalent to that shape.
   * @param shape array containing N integers, sizes of dimensions
   * @return
   */
  explicit TensorStorage(std::initializer_list<size_t> const &shape) {
    resize(shape);
  }

  /**
   * Sychronizes data on host and device
   */
  void sync() {
    if (data_dirty_ && data_is_on_host_) {
      toDevice();
    } else {
      fromDevice();
    }
  }

  /**
   *
   * @param offset
   * @return iterator to an element at offset position
   */
  DataIter host_data(size_t offset) { return host_data_.begin() + offset; }

  /**
   *
   * @param offset
   * @return  constant iterator to an element at offset position
   */
  ConstDataIter host_data(size_t offset) const {
    return host_data_.begin() + offset;
  }

  void resize(const std::vector<size_t> &sz) { host_data_.resize(product(sz)); }

  void resize(std::initializer_list<size_t> const &sz) {
    host_data_.resize(product(sz), U(0));
  }

  size_t size() const { return host_data_.size(); }

 private:
  void toDevice() const {
#if defined(USE_OPENCL) || defined(USE_CUDA)
    CLCudaAPI::Queue queue = device_->queue();
    if (device_data_ && device_data_->GetSize() >= host_data_->size()) {
      device_data_->Write(queue, host_data_.size(), host_data_->data(), 0);
    } else {
      CLCudaAPI::Context ctx = device_->context();
      device_data_           = make_unique<CLCudaAPI::Buffer<U>>(
        ctx, queue, host_data_->begin(), host_data_->end());
    }
#endif

    data_is_on_host_ = false;
    data_dirty_      = false;
  }

  void fromDevice() const {
#if defined(USE_OPENCL) || defined(USE_CUDA)
    assert(device_);
    assert(device_data_);
    device_data_->Read(device_->queue(), host_data_->size(),
                       // using const_cast<> to avoid making host_data_
                       // entirely mutable
                       const_cast<U *>(host_data_->data()));
#endif
    data_is_on_host_ = true;
    data_dirty_      = false;
  }

  /** Vector containing the host tensor data in the stack */
  std::vector<U, Allocator> host_data_;

#if defined(USE_OPENCL) || defined(USE_CUDA)
  /* Pointer to the Tensor data in the device */
  std::unique_ptr<CLCudaAPI::Buffer<U>> device_data_;
#endif
  mutable bool data_is_on_host_;  // is current data is on host?
  mutable bool data_dirty_;       // have current data might been modified?
  mutable size_t dirty_from, dirty_to;  // range of indexes that can be modified

  /* Pointer to the current device where the data resides */
  Device *device_;
};

}  // namespace tiny_dnn
