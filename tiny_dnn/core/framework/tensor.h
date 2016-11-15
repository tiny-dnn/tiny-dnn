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

#include <algorithm> // std::fill

#include "tiny_dnn/core/framework/device.fwd.h"

namespace tiny_dnn {

template<typename U = float_t>
class Tensor {
 public:
    /*
     * Initializes an empty tensor.
     */
    Tensor() {}

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
        reshape(d0, d1, d2, d3);
        resize();
    }

    explicit Tensor(const std::vector<cnn_size_t>& shape) {
        reshape(shape[0], shape[1], shape[2], shape[3]);
        resize();
    }

    // Move constructor
    Tensor(Tensor<U>&& other) = default;

    // Returns the tensor shape
    const std::vector<cnn_size_t>& shape() const { return shape_; }
    
    // Returns the value of a specified index in the tensor.
    // Checked version (throw exceptions for out-of-range error)
    template<typename T>
    T& at(const cnn_size_t d0,
          const cnn_size_t d1,
          const cnn_size_t d2,
          const cnn_size_t d3) {
        return *access_data<T>(d0, d1, d2, d3);
    }

    template<typename T>
    const T& at(const cnn_size_t d0,
                const cnn_size_t d1,
                const cnn_size_t d2,
                const cnn_size_t d3) const {
        return *access_data<T>(d0, d1, d2, d3);
    }

    // Returns the pointer to a specified index in the tensor
    // Checked version (throw exceptions for out-of-range error)
    template<typename T>
    T* ptr(const cnn_size_t d0,
           const cnn_size_t d1,
           const cnn_size_t d2,
           const cnn_size_t d3) {
        return access_data<T>(d0, d1, d2, d3);
    }
    
    template<typename T>
    const T* ptr(const cnn_size_t d0,
                 const cnn_size_t d1,
                 const cnn_size_t d2,
                 const cnn_size_t d3) const {
        return access_data<T>(d0, d1, d2, d3);
    }

    // zero-overhead version (same performance to raw pointer access.
    // have an assertion for out-of-range error)
    U& operator[] (const size_t index) {
        return *access_data<U>(index);
    }

    const U& operator[] (const size_t index) const {
        return *access_data<U>(index);
    }

    // this is only a proof of concept to copy data
    // from one device to another.
    void toDevice(const Device& device) {
#if defined(USE_OPENCL) || defined(USE_CUDA)
        CLCudaAPI::Context ctx = device.context();
        CLCudaAPI::Queue queue = device.queue();

        device_data_ = make_unique<CLCudaAPI::Buffer<U> >(
            ctx, queue, host_data_->begin(), host_data_->end());
#endif
    }

    // this is only a proof of concept to copy data
    // from one device to another.
    void fromDevice(const Device& device) {
#if defined(USE_OPENCL) || defined(USE_CUDA)
        CLCudaAPI::Queue queue = device.queue();

        device_data_->Read(
            queue, host_data_->size(), &host_data_->at(0));
#endif
    }

    /*template<typename T, typename U = float_t>
    T host_data() const {
        return static_cast<T>(*access_data(0));
    }

    template<typename T, typename U = float_t>
    T mutable_host_data() {
        return static_cast<T>(*access_data(0));
    }

    template<typename T, typename U = float_t>
    T device_data() const {
	fromDevice(device_);
        return static_cast<T>(*access_data(0));
    }

    template<typename T, typename U = float_t>
    T mutable_device_data() {
	fromDevice(device_);
        return static_cast<T>(*access_data(0));
    }*/

    size_t size() const {
        size_t new_size = 1;
        for (auto d : shape_) { new_size *= d; }
        return new_size;
    }

    /* @brief Fills all the tensor values with a given value
     *
     * @param value The value to fill the tensor
     */
    void fill(const U value) {
        std::fill(host_data_->begin(), host_data_->end(), value);
    }

    /* @brief Element-wise addition
     */
    Tensor<U> add(const Tensor<U>& src) const {
        if (this->shape() != src.shape()) {
            throw nn_error("Tensor must have same shape");
        }

        Tensor<U> res(src.shape());

        for_i(true, res.size(), [&](size_t i) {
            res[i] = this->operator[](i) + src[i];
        });

        return std::move(res);
    }

    /* @brief Element-wise addition
     */
    Tensor<U> add(const float_t scalar) const {
        Tensor<U> res(this->shape());

        for_i(true, res.size(), [&](size_t i) {
            res[i] = this->operator[](i) + scalar;
        });

        return std::move(res);
    }

    /* @brief Element-wise subtraction
     */
    Tensor<U> sub(const Tensor<U>& src) const {
        if (this->shape() != src.shape()) {
            throw nn_error("Tensor must have same shape");
        }

        Tensor<U> res(src.shape());

        for_i(true, res.size(), [&](size_t i) {
            res[i] = this->operator[](i) - src[i];
        });

        return std::move(res);
    }

    /* @brief Element-wise subtraction
     */
    Tensor<U> sub(const float_t scalar) const {
        Tensor<U> res(this->shape());

        for_i(true, res.size(), [&](size_t i) {
            res[i] = this->operator[](i) - scalar;
        });

        return std::move(res);
    }

    /* @brief Element-wise multiplication
     */
    Tensor<U> mul(const Tensor<U>& src) const {
        if (this->shape() != src.shape()) {
            throw nn_error("Tensor must have same shape");
        }

        Tensor<U> res(src.shape());

        for_i(true, res.size(), [&](size_t i) {
            res[i] = this->operator[](i) * src[i];
        });

        return std::move(res);
    }

    /* @brief Element-wise multiplication
     */
    Tensor<U> mul(const float_t scalar) const {
        Tensor<U> res(this->shape());

        for_i(true, res.size(), [&](size_t i) {
            res[i] = this->operator[](i) * scalar;
        });

        return std::move(res);
    }

    /* @brief Element-wise division
     */
    Tensor<U> div(const Tensor<>& src) const {
        if (this->shape() != src.shape()) {
            throw nn_error("Tensor must have same shape");
        }

        Tensor<U> res(src.shape());

        for_i(true, res.size(), [&](size_t i) {
            res[i] = this->operator[](i) / (src[i] + 1e-10);
        });

        return std::move(res);
    }

    /* @brief Element-wise division
     */
    Tensor<U> div(const float_t scalar) const {
        Tensor<U> res(this->shape());

        for_i(true, res.size(), [&](size_t i) {
            res[i] = this->operator[](i) / (scalar + 1e-10);
        });

        return std::move(res);
    }

 private:
    // Initializes the data buffer with the given value
    void resize(const U value = 0) {
        if (!host_data_) {
            host_data_ = make_unique<
                std::vector<U, aligned_allocator<U, 64> > >(size(), value);
        } else {
            host_data_->resize(size(), value);
        }
    }

    // Initializes the shape vector
    void reshape(const cnn_size_t d0,
                 const cnn_size_t d1,
                 const cnn_size_t d2,
                 const cnn_size_t d3) {
	    shape_.resize(4);
	    shape_[0] = d0;
	    shape_[1] = d1;
	    shape_[2] = d2;
	    shape_[3] = d3;
    }

    // Method to access to the tensor data.
    // It checks if the requested position is feasible or not.
    template<typename T>
    T* access_data(const cnn_size_t d0,
                   const cnn_size_t d1,
                   const cnn_size_t d2,
                   const cnn_size_t d3) {
        if (d0 >= shape_[0] || d1 >= shape_[1] ||
            d2 >= shape_[2] || d3 >= shape_[3]) {
            throw nn_error("Access tensor out of range.");
        }

        U* value = &host_data_->operator[](
            shape_[1] * shape_[2] * shape_[3] * d0 +
            shape_[1] * shape_[2] * d3 +
            shape_[1] * d2 +
            d1
        );

        // in case that requested type is not the same as
        // the specified during the tensor initilization
        // we cast the type.
        if (!std::is_same<T,U>::value) {
            return reinterpret_cast<T*>(value);
        }
        return value;
    }

    // Method to access to the tensor data.
    // It checks if the requested position is feasible or not.
    template<typename T>
    T* access_data(const cnn_size_t d0,
                   const cnn_size_t d1,
                   const cnn_size_t d2,
                   const cnn_size_t d3) const {
        if (d0 >= shape_[0] || d1 >= shape_[1] ||
            d2 >= shape_[2] || d3 >= shape_[3]) {
            throw nn_error("Access tensor out of range.");
        }

        U* value = &host_data_->operator[](
            shape_[1] * shape_[2] * shape_[3] * d0 +
            shape_[1] * shape_[2] * d3 +
            shape_[1] * d2 +
            d1
        );

        // in case that requested type is not the same as
        // the specified during the tensor initilization
        // we cast the type.
        if (!std::is_same<T,U>::value) {
            return reinterpret_cast<T*>(value);
        }
        return value;
    }

    template<typename T>
    T* access_data(const size_t index) {
        if (index >= host_data_->size()) {
            throw nn_error("Access tensor out of range.");
        }

        U* value = &host_data_->operator[](index);

        // in case that requested type is not the same as
        // the specified during the tensor initilization
        // we cast the type.
        if (!std::is_same<T,U>::value) {
            return reinterpret_cast<T*>(value);
        }
        return value;
    }

    template<typename T>
    T* access_data(const size_t index) const {
        if (index >= host_data_->size()) {
            throw nn_error("Access tensor out of range.");
        }

        U* value = &host_data_->operator[](index);

        // in case that requested type is not the same as
        // the specified during the tensor initilization
        // we cast the type.
        if (!std::is_same<T,U>::value) {
            return reinterpret_cast<T*>(value);
        }
        return value;
    }

 private:
    /* Vector with the size of the tensor
     * shape_[0]: batch
     * shape_[1]: width
     * shape_[2]: height
     * shape_[3]: depth
     */
    std::vector<cnn_size_t> shape_;

    /* Pointer to the Tensor data in pure CPU mode */
    std::unique_ptr<std::vector<U, aligned_allocator<U, 64> > > host_data_;

#if defined(USE_OPENCL) || defined(USE_CUDA)
    /* Pointer to the Tensor data in OpenCL mode */
    std::unique_ptr<CLCudaAPI::Buffer<U> > device_data_;
#endif

    /* Pointer to the current device where the data resides */
    Device* device_;
};

// Overloaded method to print the Tensor class to the standard output
inline std::ostream& operator<< (std::ostream &os,
		                 const Tensor<>& tensor) {
    const std::vector<cnn_size_t>& shape = tensor.shape();
    for (cnn_size_t i = 0; i < shape[0]; ++i) {
        os << "-- Batch: " << i << "\n";
        for (cnn_size_t j = 0; j < shape[3]; ++j) {
            os << "-- Channel: " << j << "\n";
            os << "-- Data:\n";
            for (cnn_size_t k = 0; k < shape[1]; ++k) {
                for (cnn_size_t l = 0; l < shape[2]; ++l) {
                    os << "   " << tensor.at<float_t>(i,k,l,j) << " ";
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
