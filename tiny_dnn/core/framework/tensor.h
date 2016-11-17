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

#include <cmath>     // sqrt
#include <algorithm> // std::fill, std::generate
#include <numeric>   // std::accumulate
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

template<typename U = float_t>
class Tensor {
 public:
    /*
     * Initializes an empty tensor.
     */
    Tensor() : shape_{ 0,0,0,0 } {}

    /*
     * Create a tensor of the given dimension.
     * It is assumed that a tensor will hold data in NxWxHxD order,
     * where:
     *  N the batch axis
     *  W the width axis
     *  H the heigth axis
     *  D the depth axis
     *
     *  Data will be hold by a std::vector with 64bytes alignment.
     */
    explicit Tensor(const cnn_size_t d0,
                    const cnn_size_t d1,
                    const cnn_size_t d2,
                    const cnn_size_t d3) {
        resize(d0, d1, d2, d3);
    }

    explicit Tensor(const std::array<cnn_size_t, 4>& shape) {
        resize(shape[0], shape[1], shape[2], shape[3]);
    }

    explicit Tensor(const std::vector<cnn_size_t>& shape) {
        assert(shape.size() == 4);
        resize(shape[0], shape[1], shape[2], shape[3]);
    }

    ~Tensor() = default;

    Tensor(const Tensor&) {
        other.fromDevice();
        shape_ = other.shape_;
        host_data_ = other.host_data_;
        data_is_on_host_ = true;
        data_dirty_ = true;       
        //device_data_ is intentionally left uninitialized.
    }

    Tensor &operator = (const Tensor& other) {
        other.fromDevice();
        shape_ = other.shape_;
        data_is_on_host_ = true;
        data_dirty_ = true;
        host_data_ = other.host_data_;

        //device_data_ is intentionally left as-is. It will be erased only if new tensor won't fit, and only when data gets moved to the GPU.
        return *this;
    }

#ifdef CNN_USE_DEFAULT_MOVE_CONSTRUCTORS
    Tensor(Tensor&& other) = default;        // move ctor
    Tensor &operator = (Tensor&&) = default; // move assign
#else
    Tensor(Tensor&& other) { // for VS2013 we need to manually implement these if we want to have move semantics
        shape_ = std::move(other.shape_);
        host_data_ = std::move(other.host_data_);
        device_data_ = std::move(other.device_data_);
        data_is_on_host_ = other.data_is_on_host_;
        data_dirty_ = other.data_dirty_;
        return *this;
    }
    Tensor &operator = (Tensor&& other) {
        shape_ = std::move(other.shape_);
        host_data_ = std::move(other.host_data_);
        device_data_ = std::move(other.device_data_);
        data_is_on_host_ = other.data_is_on_host_;
        data_dirty_ = other.data_dirty_;
}
#endif

    // Returns the tensor shape
    const std::array<cnn_size_t, 4>& shape() const { return shape_; }
    
    // Returns the value of a specified index in the tensor.
    // Checked version (throw exceptions for out-of-range error)
    U& host_at(const cnn_size_t d0,
          const cnn_size_t d1,
          const cnn_size_t d2,
          const cnn_size_t d3) {
        return *host_ptr(d0, d1, d2, d3);
    }

    U host_at(const cnn_size_t d0,
                const cnn_size_t d1,
                const cnn_size_t d2,
                const cnn_size_t d3) const {
        return *host_ptr(d0, d1, d2, d3);
    }

    // Returns the pointer to a specified index in the tensor
    // Checked version (throw exceptions for out-of-range error)
    const U* host_ptr(const cnn_size_t d0,
        const cnn_size_t d1,
        const cnn_size_t d2,
        const cnn_size_t d3) const {
        if (d0 >= shape_[0] || d1 >= shape_[1] ||
            d2 >= shape_[2] || d3 >= shape_[3]) {
            throw nn_error("Access tensor out of range.");
        }

        return host_data() + (
            shape_[1] * shape_[2] * shape_[3] * d0 +
            shape_[1] * shape_[2] * d3 +
            shape_[1] * d2 +
            d1
            );
    }

    U* host_ptr(const cnn_size_t d0,
        const cnn_size_t d1,
        const cnn_size_t d2,
        const cnn_size_t d3) {
        if (d0 >= shape_[0] || d1 >= shape_[1] ||
            d2 >= shape_[2] || d3 >= shape_[3]) {
            throw nn_error("Access tensor out of range.");
        }

        return mutable_host_data() + (
            shape_[1] * shape_[2] * shape_[3] * d0 +
            shape_[1] * shape_[2] * d3 +
            shape_[1] * d2 +
            d1
            );
    }

    // zero-overhead version (same performance to raw pointer access.
    // have an assertion for out-of-range error)
    U& operator[] (const size_t index) {
        return mutable_host_data()[index];
    }

    U operator[] (const size_t index) const {
        return host_data()[index];
    }

    const U* host_data() const {
        if (!data_is_on_host_ && data_dirty_) {
            fromDevice();
        }
        return host_data_.data();
    }
    
    U* mutable_host_data() {
        if (!data_is_on_host_ && data_dirty_) {
            fromDevice();
        }
        data_dirty_ = true;
        return host_data_.data();
    }

    const void *device_data() const {
        if (data_is_on_host_ && data_dirty_) const {
            toDevice();
        }
        return (*device_data_)();
    }

    void *mutable_device_data() {
        if (data_is_on_host_ && data_dirty_) {
            toDevice();
        }
        data_dirty_ = true;
        return (*device_data_)();
    }

    size_t size() const {
        return host_data_.size();
    }

    /* @brief Fills all the tensor values with a given value
     *
     * @param value The value to fill the tensor
     */
    void fill(const U value) {
        data_is_on_host_ = true;
        data_dirty_ = true;
        std::fill(host_data_.begin(), host_data_.end(), value);
    }

    // /* @brief Fills the tensor with evenly-spaced values in the interval
    //  *
    //  * @param from The lower bound of the interval
    //  * @param to The upper bound of the interval
    //  */
    // void linspace(const U from, const U to) {
    //     U start = from,
    //         step = (to - from) / (host_data_->end() - host_data_->begin() - 1);
    //     std::generate(host_data_->begin(),
    //                   host_data_->end(),
    //                   [&start, &step]() {
    //                     U tmp = start;
    //                     start += step;
    //                     return tmp;
    //                   });
    // }

    /* @brief Element-wise addition
     */
    Tensor add(const Tensor& src) const {
        return binary_element_wise_operation(src, std::plus<U>());
    }

    /* @brief Element-wise addition
     */
    Tensor add(const U scalar) const {
        return binary_scalar_operation(scalar, std::plus<U>());
    }

    /* @brief Element-wise subtraction
     */
    Tensor sub(const Tensor& src) const {
        return binary_element_wise_operation(src, std::minus<U>());
    }

    /* @brief Element-wise subtraction
     */
    Tensor sub(const U scalar) const {
        return add(-scalar);
    }

    /* @brief Element-wise multiplication
     */
    Tensor mul(const Tensor& src) const {
        return binary_element_wise_operation(src, std::multiplies<U>());
    }

    /* @brief Element-wise multiplication
     */
    Tensor mul(const U scalar) const {
        return binary_scalar_operation(scalar, std::multiplies<U>());
    }

    /* @brief Element-wise division
     */
    Tensor div(const Tensor& src) const {
        return binary_element_wise_operation(src, [](U a, U b)
        {
            return (b == U(0.0)) ? std::numeric_limits<U>::quiet_NaN() :
                a / b;

        });
    }

    /* @brief Element-wise division
     */
    Tensor div(const U scalar) const {
        if (scalar == U(0.0)) {
            Tensor<U> res(this->shape());
            res.fill(std::numeric_limits<U>::quiet_NaN());
            return std::move(res);
        } else {
            return binary_scalar_operation(src, std::divides<U>());
        }

    }

    /* @brief Element-wise square root
     */
    Tensor sqrt() const {
        return unary_element_wise_operation([](U v) {return std::sqrt(v); }); // lambda used because types cannot be deduced by simply specifying std::sqrt 
    }

    /* @brief Element-wise exponential
     */
    Tensor exp() const {
        return unary_element_wise_operation([](U v) {return std::exp(v); }); // lambda used because types cannot be deduced by simply specifying std::exp
    }

 private:
     void toDevice() const {
#if defined(USE_OPENCL) || defined(USE_CUDA)
         CLCudaAPI::Queue queue = device->queue();
         if (device_data_ && device_data_->GetSize() >= host_data_.size()) {
             device_data_->Write(queue, host_data.size(), host_data_.data(), 0);
         } else {
             CLCudaAPI::Context ctx = device->context();
             device_data_ = make_unique<CLCudaAPI::Buffer<U> >(
                 ctx, queue, host_data_.begin(), host_data_.end());
         }

#endif
         data_is_on_host_ = false;
         data_dirty_ = false;
     }

     void fromDevice() const {
#if defined(USE_OPENCL) || defined(USE_CUDA)
         assert(device_);
         assert(device_data_);
         device_data_->Read(device_->queue(), host_data_.size(), const_cast<U*>(host_data_.data())); // using const_cast<> to avoid making host_data_ entirely mutable
#endif
         data_is_on_host_ = true;
         data_dirty_ = false;
     }

    // Initializes the shape vector
    void resize(const cnn_size_t d0,
                 const cnn_size_t d1,
                 const cnn_size_t d2,
                 const cnn_size_t d3) {
	    shape_[0] = d0;
	    shape_[1] = d1;
	    shape_[2] = d2;
	    shape_[3] = d3;
        host_data_.resize(std::accumulate(std::begin(shape_), std::end(shape_), size_t(1), std::multiplies<size_t>()), U(0));
    }

    template<typename F> Tensor binary_element_wise_operation(const Tensor &src, F f) const
    {
        if (this->shape() != src.shape()) {
            throw nn_error("Tensor must have same shape");
        }

        Tensor<U> res(src.shape());

        const U* dst = res.mutable_host_data();
        const U* src1 = host_data();
        const U* src2 = src.host_data();

        for_i(true, res.size(), [dst, src1, src2, &f](size_t i) {
            dst[i] = f(src1[i], src2[i]);
        });

        return std::move(res);
    }

    template<typename F> Tensor unary_element_wise_operation(F f) const
    {
        if (this->shape() != src.shape()) {
            throw nn_error("Tensor must have same shape");
        }

        Tensor<U> res(src.shape());

        const U* dst = res.mutable_host_data();
        const U* src1 = host_data();

        for_i(true, res.size(), [dst, src1, &f](size_t i) {
            dst[i] = f(src1[i]);
        });

        return std::move(res);
    }

    template<typename F> Tensor binary_scalar_operation(U scalar, F f) const
    {
        if (this->shape() != src.shape()) {
            throw nn_error("Tensor must have same shape");
        }

        Tensor<U> res(src.shape());

        const U* dst = res.mutable_host_data();
        const U* src1 = host_data();

        for_i(true, res.size(), [dst, src1, scalar, &f](size_t i) {
            dst[i] = f(src1[i], scalar);
        });

        return std::move(res);
    }
private:
    /* Vector with the size of the tensor
     * shape_[0]: batch
     * shape_[1]: width
     * shape_[2]: height
     * shape_[3]: depth
     */
    std::array<cnn_size_t, 4> shape_;

    /* Pointer to the Tensor data in pure in the host device */
    std::vector<U, aligned_allocator<U, 64> > host_data_;

#if defined(USE_OPENCL) || defined(USE_CUDA)
    /* Pointer to the Tensor data in the device */
    std::unique_ptr<CLCudaAPI::Buffer<U> > device_data_;
#endif
    mutable bool data_is_on_host_;      //< current data is on host if true, on device if false.
    mutable bool data_dirty_;           //< set to true if current data might have been modified

    /* Pointer to the current device where the data resides */
    Device* device_;
};

// Overloaded method to print the Tensor class to the standard output
template<typename T>
inline std::ostream& operator<< (std::ostream &os,
                                 const Tensor<T>& tensor) {
    const std::vector<cnn_size_t>& shape = tensor.shape();
    for (cnn_size_t i = 0; i < shape[0]; ++i) {
        os << "-- Batch: " << i << "\n";
        for (cnn_size_t j = 0; j < shape[3]; ++j) {
            os << "-- Channel: " << j << "\n";
            os << "-- Data:\n";
            for (cnn_size_t k = 0; k < shape[1]; ++k) {
                for (cnn_size_t l = 0; l < shape[2]; ++l) {
                    os << "   " << tensor.at(i,k,l,j) << " ";
                }
                os << ";\n";
            }
        }
    }
    os << "----------------\n" 
       << "--> Tensor size: [ " 
       << shape[0] << " x " << shape[1] << " x "
       << shape[2] << " x " << shape[3] << " ]\n";
    return os;
}

}  // namespace tiny_dnn
