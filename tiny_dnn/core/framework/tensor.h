/*
    COPYRIGHT

    All contributions by Taiga Nomi
    Copyright (c) 2013, Taiga Nomi
    All rights reserved.

    All other contributions:
    Copyright (c) 2013-2016, the respective contributors.
    All rights reserved.

    Each contributor holds copyright over their respective contributions.
    The project versioning (Git) records all such contribution source information.

    LICENSE

    The BSD 3-Clause License


    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice, this
      list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright notice,
      this list of conditions and the following disclaimer in the documentation
      and/or other materials provided with the distribution.

    * Neither the name of tiny-dnn nor the names of its
      contributors may be used to endorse or promote products derived from
      this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
    FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
    DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
    CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
    OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
#pragma once

#include <memory>
#include <type_traits>
#include <vector>

#include "tiny_dnn/core/framework/device.fwd.h"
#include "tiny_dnn/core/framework/tensor_utils.h"

#if defined(USE_OPENCL) || defined(USE_CUDA)
#ifdef USE_OPENCL
#include "third_party/CLCudaAPI/clpp11.h"
#else
#include "third_party/CLCudaAPI/cupp11.h"
#endif
#endif

namespace tiny_dnn {

template<typename U = float_t, typename Allocator = aligned_allocator<U, 64>>
class TensorStorage {
    typedef typename std::vector<U, aligned_allocator < U, 64>>::
    const_iterator ConstDataIter;
    typedef typename std::vector<U, aligned_allocator < U, 64>>::
    iterator DataIter;
 public:
    /**
     * Initializes an empty tensor storage.
     * @return
     */
    TensorStorage() {
    }

    /**
     * Constructor that assepts a vector of shape and create a TensorStorage
     * with a size equivalent to that shape.
     * @param shape array containing N integers, sizes of dimensions
     * @return
     */
    explicit TensorStorage(const std::vector<size_t> &shape) {
        resize(shape);
    }

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
        if (data_dirty_) {
            if (data_is_on_host_)
                toDevice();
            else
                fromDevice();
        }
    }

    /**
     *
     * @param offset
     * @return iterator to an element at offset position
     */
    DataIter host_data(size_t offset) {
        return host_data_.begin() + offset;
    }

    /**
     *
     * @param offset
     * @return  constant iterator to an element at offset position
     */
    ConstDataIter host_data(size_t offset) const {
        return host_data_.begin() + offset;
    }

    void resize(const std::vector<size_t> &sz) {
        host_data_.resize(product(sz));
    }
    
    void resize(std::initializer_list<size_t> const &sz) {
        host_data_.resize(product(sz), U(0));
    }

    size_t size() const {
        return host_data_.size();
    }
 private:

    void toDevice() const {
#if defined(USE_OPENCL) || defined(USE_CUDA)
            CLCudaAPI::Queue queue = device_->queue();
            if (device_data_ && device_data_->GetSize() >= host_data_->size()) {
                device_data_->Write(queue,
                                    host_data.size(),
                                    host_data_->data(),
                                    0);
            } else {
                CLCudaAPI::Context ctx = device_->context();
                device_data_ = make_unique<CLCudaAPI::Buffer < U> > (
                    ctx, queue, host_data_->begin(), host_data_->end());
            }
#endif

            data_is_on_host_ = false;
            data_dirty_ = false;
    }

    void fromDevice() const {
#if defined(USE_OPENCL) || defined(USE_CUDA)
            assert(device_);
            assert(device_data_);
            device_data_->Read(device_->queue(),
                               host_data_->size(),
                               // using const_cast<> to avoid making host_data_
                               // entirely mutable
                               const_cast<U *>(host_data_->data()));
#endif
            data_is_on_host_ = true;
            data_dirty_ = false;
    }

    std::vector<U, Allocator> host_data_;

#if defined(USE_OPENCL) || defined(USE_CUDA)
    /* Pointer to the Tensor data in the device */
    std::unique_ptr<CLCudaAPI::Buffer<U>> device_data_;
#endif
    mutable bool data_is_on_host_; // is current data is on host?
    mutable bool data_dirty_;      // have current data might been modified?
    mutable size_t dirty_from, dirty_to; // range of indexes that can be modified

    /* Pointer to the current device where the data resides */
    Device* device_;
};

/**
 * A tensor of the given dimension.
 * A tensor holds data in C-style nD array, i.e row-major order:
 * the rightmost index “varies the fastest”.
 *
 * Data is held by a std::vector with 64 bytes alignment.
 * Unmutable if kConst == true
 */
template<typename U, size_t kDimensions, bool kConst, typename Allocator>
class Tensor {
    // Define constant types for constant Tensor,
    // and mutable ones for mutable Tensor
    typedef typename std::conditional<kConst,
                                      const TensorStorage<U, Allocator>,
                                      TensorStorage<U, Allocator>>::type TensorStorageType;
    typedef typename std::conditional<kConst, const U *, U *>::type
        UPtr;
    typedef typename std::shared_ptr<TensorStorageType> TensorStoragePointer;
    typedef typename std::conditional<kConst,
                              typename std::vector<U, Allocator>
                                       ::const_iterator,
                              typename std::vector<U, Allocator>
                                       ::iterator>
                     ::type StorageIterator;

 public:

    /**
     * Initializes an empty tensor.
     * @return
     */
    Tensor() {
        storage_pointer_ = std::make_shared<TensorStorageType>();
        offset_ = 0;
        size_ = 0;
    }

    /**
     * Constructor that assepts an array of shape and create a Tensor with that
     * shape. For example, given shape = {2,3,4,5,6}, tensor
     * will be of size 2x3x4x5x6
     * @param shape array containing N integers, sizes of dimensions
     * @return
     */
    explicit Tensor(const std::array<size_t, kDimensions> &shape) {
        storage_pointer_ = std::make_shared<TensorStorageType>(shape);
        offset_ = 0;
        size_ = product(shape);
        shape_ = shape;
    }

    /**
     * Constructor that assepts a vector of shape and create a Tensor with that
     * shape. For example, given shape = {2,3,4,5,6}, tensor
     * will be of size 2x3x4x5x6
     * @param shape array containing N integers, sizes of dimensions
     * @return
     */
    explicit Tensor(const std::vector<size_t> &shape) {
        storage_pointer_ = std::make_shared<TensorStorageType>(shape);
        offset_ = 0;
        size_ = product(shape);
        shape_ = shape;
    }

    /**
     * Constructor that assepts an initializer list of shape and create a
     * Tensor with that shape. For example, given shape = {2,3,4,5,6}, tensor
     * will be of size 2x3x4x5x6
     * @param shape array containing N integers, sizes of dimensions
     * @return
     */
    explicit Tensor(std::initializer_list<size_t> const &shape) {

        storage_pointer_ = std::make_shared<TensorStorageType>(shape);
        offset_ = 0;
        size_ = product(shape);
        std::copy(shape.begin(), shape.end(), shape_.begin());
    }

    /**
     * Constructor that accepts a pointer to existing TensorStorage, together
     * with shape and offset.
     * @param storage pointer to TensorStorage
     * @param offset offset from first element of storage
     * @param shape shape of the Tensor
     * @return
     */
    explicit Tensor(const TensorStoragePointer storage,
                    const size_t offset,
                    const std::array<size_t, kDimensions> &shape) {
        storage_pointer_ = storage;
        offset_ = offset;
        size_ = product(shape);
        shape_ = shape;
    }

    ~Tensor() = default;

    Tensor(const Tensor&other) { //TODO(Randl):deep copy
        //TODO(Randl)
        /*other.fromDevice();
        shape_ = other.shape_;
        storage_pointer_ = other.storage_pointer_;
        data_is_on_host_ = true;
        data_dirty_ = true; */
        //device_data_ is intentionally left uninitialized.
    }
/*
 *
 * TODO(Randl): Move constructors for Tensor and TensorStorage
    Tensor &operator = (const Tensor& other) {
    }

#ifdef CNN_USE_DEFAULT_MOVE_CONSTRUCTORS
    Tensor(Tensor&& other) = default;        // move ctor
    Tensor &operator = (Tensor&&) = default; // move assign
#else

    Tensor(Tensor&& other) {
    }

    Tensor &operator = (Tensor&& other) {
    }
#endif
*/
    /**
     *
     * @return the tensor shape
     */
    const std::array<size_t, kDimensions>& shape() const { return shape_; }

    /**
     * Checked version of access to indexes in tensor (throw exceptions
     * for out-of-range error)
     * @param args indexes in tensor
     * @return the value of a specified index in the tensor
     */
    template<typename... Args>
    U& host_at(const Args... args) {
        static_assert(!kConst, "Non-constant operation on constant Tensor");
        return *host_ptr(args...);
    }

    /**
     * Checked version of access to indexes in tensor (throw exceptions
     * for out-of-range error)
     * @param args indexes in tensor
     * @return the value of a specified index in the tensor
     */
    template<typename... Args>
    U host_at(const Args... args) const {
        return *host_ptr(args...);
    }

    /**
     * Calculate an offset for last dimension.
     * @param d an index of last dimension
     * @return offest from the beginning of the dimesion
     */
    size_t host_pos(const size_t d) const { //TODO(Randl): unchecked version
        if (d >= shape_.back())  {
            throw nn_error("Access tensor out of range.");
        }
        return d;
    }

    /**
     * Calculate an offest in 1D representation of nD Tensor. Parameters are
     * indexes of k last dimensions. If k is less than n, function returns an
     * offset from the first index of (n-k+1)th dimension. This allows recursive
     * call to acquire offset for generic number of dimensions
     * @param d index of (k-n)th dimension. For external call, n=k usually holds
     * @param args index of rest (k-1) dimensions.
     * @return offset from the first index of (n-k)th dimension
     */
    template<typename... Args>
    size_t host_pos(const size_t  d,
                    const Args... args) const {
        static_assert(sizeof...(args) < kDimensions,
                      "Wrong number of dimensions");
        size_t dim = kDimensions - sizeof...(args) - 1;
        if (d >= shape_[dim]) {
            throw nn_error("Access tensor out of range.");
        }
        size_t shift = 1;
        for (size_t i = dim + 1; i < kDimensions; ++i)
            shift *= shape_[i]; //TODO(Randl): optimize. Reverse argumets?

        return (d * shift + host_pos(args...));
    }

    template<typename... Args>
    UPtr host_ptr(const Args... args) const {
    return &(*host_iter(args...));
    }

    template<typename... Args>
    StorageIterator host_iter (const Args... args) const {
        static_assert(!kConst, "Non-constant operation on constant Tensor");
        static_assert(sizeof...(args) == kDimensions,
                      "Wrong number of dimensions");
        return storage_pointer_->host_data(offset_) + host_pos(args...);
    }

    StorageIterator host_begin() const {
        return storage_pointer_->host_data(offset_);
    }

    StorageIterator host_data() const {
        //fromDevice();
        return storage_pointer_->host_data(offset_);
    }

    /*U* mutable_host_data() {
        static_assert(!kConst, "Non-constant operation on constant Tensor");
        //fromDevice();
        //data_dirty_ = true;
        return storage_pointer_->data(offset);
    }*/

#if defined(USE_OPENCL) || defined(USE_CUDA)
    const void *device_data() const {
        toDevice();
        return (*device_data_)();
    }

    void *mutable_device_data() {
        static_assert(!kConst, "Non-constant operation on constant Tensor");
        toDevice();
        data_dirty_ = true;
        return (*device_data_)();
    }
#endif

    void fill(U value) {
        static_assert(!kConst, "Non-constant operation on constant Tensor");
        //data_is_on_host_ = true;
        //data_dirty_ = true;
        std::fill(storage_pointer_->host_data(offset_),
                  storage_pointer_->host_data(offset_)+calcSize(),
                  value);
    }

    //TODO(Randl): variadic template version of reshape
    //TODO(Randl): non-checked version
    void reshape(const std::array<size_t, kDimensions> &sz) {
        static_assert(!kConst, "Non-constant operation on constant Tensor");
        //No size change for reshape
        if (calcSize() != product(sz))
            throw nn_error("Reshape to Tensor of different size.");
        shape_ = sz;

    }

    void resize(const std::array<size_t, kDimensions> &sz) {
        if (offset_ != 0 || size_ != storage_pointer_->size())
            throw nn_error("Resize of partial view is impossible.");
        storage_pointer_->resize(std::vector<size_t>(sz.begin(), sz.end()));
        shape_=sz;
    }
    
    size_t size() const {
        return size_;
    }

    Tensor operator[](size_t index) {
        return Tensor(storage_pointer_,
                      offset_ + index * size_ / shape_[0],
                      std::array<size_t, kDimensions - 1>(shape_.begin() + 1,
                                                          shape_.end()));
    }

private:
    size_t calcSize() const {
        return product(shape_);
    }

    /**
     * A tensor holds data in C-style nD array, i.e row-major order:
     * the rightmost index “varies the fastest”.
     */
    std::array<size_t, kDimensions> shape_;

    /* Offset from the beginning of TensorStorage */
    size_t offset_;
    size_t size_;

    /* pointer to TensorStorage */
    TensorStoragePointer storage_pointer_;

};

}  // namespace tiny_dnn
