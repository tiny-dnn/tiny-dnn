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

#include "tiny_dnn/core/framework/device.fwd.h"

namespace tiny_dnn {

/* Class modeling a Tensor
 * @param b The size of the batch
 * @param w The width of the tensor
 * @param h The height of the tensor
 * @param d The number of channels
 */
template<class U = float_t,
	 cnn_size_t b = 1,
	 cnn_size_t w = 1,
	 cnn_size_t h = 1,
	 cnn_size_t d = 1>
class Tensor {
 public:
    /* Default constructor for the Tensor class
     */
    Tensor() : shape_(4) {
        init_data (b, w, h, d);
        init_shape(b, w, h, d);
    }

#if defined(USE_OPENCL) || defined(USE_CUDA)
    typedef CLCudaAPI::Buffer<U>    buffer;
    typedef std::unique_ptr<buffer> buffer_ptr;
#endif

    typedef std::unique_ptr<std::vector<U> > data_ptr;
    
    // Returns the tensor shape
    std::vector<cnn_size_t> shape() const { return shape_; }
    
    // Returns the value of a specified index in the tensor.
    // Checked version (throw exceptions for out-of-range error)
    template<typename T>
    T& at(const cnn_size_t batch,
          const cnn_size_t width,
          const cnn_size_t height,
          const cnn_size_t depth) {
        return *access_data<T>(batch, width, height, depth);
    }

    template<typename T>
    const T& at(const cnn_size_t batch,
                const cnn_size_t width,
                const cnn_size_t height,
                const cnn_size_t depth) const {
        return *access_data<T>(batch, width, height, depth);
    }

    // Returns the pointer to a specified index in the tensor
    // Checked version (throw exceptions for out-of-range error)
    template<typename T>
    T* ptr(const cnn_size_t batch,
           const cnn_size_t width,
           const cnn_size_t height,
           const cnn_size_t depth) {
        return access_data<T>(batch, width, height, depth);
    }
    
    template<typename T>
    const T* ptr(const cnn_size_t batch,
                 const cnn_size_t width,
                 const cnn_size_t height,
                 const cnn_size_t depth) const {
        return access_data<T>(batch, width, height, depth);
    }

    // zero-overhead version (same performance to raw pointer access.
    // have an assertion for out-of-range error)
    template<typename T>
    T& operator[] (cnn_size_t index) {
        return *access_data<T>(index);
    }

    template<typename T>
    const T& operator[] (cnn_size_t index) const {
        return *access_data<T>(index);
    }

    // this is only a proof of concept to copy data
    // from one device to another.
    void toDevice(const Device& device) {
#if defined(USE_OPENCL) || defined(USE_CUDA)
        CLCudaAPI::Context ctx = device.context();
        CLCudaAPI::Queue queue = device.queue();

        device_data_ = buffer_ptr(new buffer(
            ctx, queue, host_data_->begin(), host_data_->end()));
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

 private:
    // Initializes the data buffer with zeroes
    void init_data(const cnn_size_t batch,  const cnn_size_t width,
                   const cnn_size_t height, const cnn_size_t depth) {
        host_data_ = data_ptr(new std::vector<U>(
            batch * width * height * depth, float_t(0.0)));
    }

    // Initializes the shape vector
    void init_shape(const cnn_size_t batch,  const cnn_size_t width,
                    const cnn_size_t height, const cnn_size_t depth) {
       shape_[0] = batch;  shape_[1] = width;
       shape_[2] = height; shape_[3] = depth;
    }

    // Method to access to the tensor data.
    // It checks if the requested position is feasible or not.
    template<typename T>
    T* access_data(const cnn_size_t batch,
                   const cnn_size_t width,
                   const cnn_size_t height,
                   const cnn_size_t depth) const {
        if (batch  > shape_[0] || width > shape_[1] ||
            height > shape_[2] || depth > shape_[3]) {
            nn_error("Access tensor out of range.");
        }

        U* value = &host_data_->at(shape_[1] * shape_[2] *
                ( shape_[3] * batch + depth ) + height + width);

        // in case that requested type is not the same as
        // the specified during the tensor initilization
        // we cast the type.
        if (!std::is_same<T,U>::value) {
	    return reinterpret_cast<T*>(value);
	}
	return value;
    }

    template<typename T>
    T* access_data(const cnn_size_t index) const {
        U* value = &host_data_->at(index);

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
    data_ptr host_data_;

#if defined(USE_OPENCL) || defined(USE_CUDA)
    /* Pointer to the Tensor data in OpenCL mode */
    buffer_ptr device_data_;
#endif

    /* Pointer to the current device where the data resides */
    Device* device_;
};

// Overloaded method to print the Tensor class to the standard output
inline std::ostream& operator<< (std::ostream &os,
		                 const Tensor<>& tensor) {
    std::vector<cnn_size_t> shape = tensor.shape();
    for (cnn_size_t i = 0; i < shape[0]; ++i) {
        os << "-- Batch: " << i << "\n";
        for (cnn_size_t j = 0; j < shape[3]; ++j) {
            os << "-- Channel: " << j << "\n";
            os << "-- Data:\n";
            for (cnn_size_t k = 0; k < shape[1]; ++k) {
                for (cnn_size_t l = 0; l < shape[2]; ++l) {
                    os << tensor.at<float_t>(i,k,l,j) << " ";
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
