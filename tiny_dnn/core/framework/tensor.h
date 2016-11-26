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
    Tensor()
    {
        reshape(0, 0, 0, 0);
    }

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
    explicit Tensor(const size_t d0,
                    const size_t d1,
                    const size_t d2,
                    const size_t d3) {
        reshape(d0, d1, d2, d3);
    }

    explicit Tensor(const std::array<size_t, 4>& shape) {
        reshape(shape[0], shape[1], shape[2], shape[3]);
    }

    explicit Tensor(const std::vector<size_t>& shape) {
        assert(shape.size() == 4);
        reshape(shape[0], shape[1], shape[2], shape[3]);
    }

    ~Tensor() = default;

    Tensor(const Tensor&other) {
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
#if defined(USE_OPENCL) || defined(USE_CUDA)
        device_data_ = std::move(other.device_data_);
#endif
        data_is_on_host_ = other.data_is_on_host_;
        data_dirty_ = other.data_dirty_;
    }

    Tensor &operator = (Tensor&& other) {
        shape_ = std::move(other.shape_);
        host_data_ = std::move(other.host_data_);
#if defined(USE_OPENCL) || defined(USE_CUDA)
        device_data_ = std::move(other.device_data_);
#endif
        data_is_on_host_ = other.data_is_on_host_;
        data_dirty_ = other.data_dirty_;
        return *this;
    }
#endif

    // Returns the tensor shape
    const std::array<size_t, 4>& shape() const { return shape_; }

    // Returns the value of a specified index in the tensor.
    // Checked version (throw exceptions for out-of-range error)
    U& host_at(const size_t d0,
               const size_t d1,
               const size_t d2,
               const size_t d3) {
        return *host_ptr(d0, d1, d2, d3);
    }

    U host_at(const size_t d0,
              const size_t d1,
              const size_t d2,
              const size_t d3) const {
        return *host_ptr(d0, d1, d2, d3);
    }

    // Returns the pointer to a specified index in the tensor
    // Checked version (throw exceptions for out-of-range error)
    const U* host_ptr(const size_t d0,
                      const size_t d1,
                      const size_t d2,
                      const size_t d3) const {
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

    U* host_ptr(const size_t d0,
                const size_t d1,
                const size_t d2,
                const size_t d3) {
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

    const U* host_data() const {
        fromDevice();
        return host_data_.data();
    }

    U* mutable_host_data() {
        fromDevice();
        data_dirty_ = true;
        return host_data_.data();
    }

#if defined(USE_OPENCL) || defined(USE_CUDA)
    const void *device_data() const {
        toDevice();
        return (*device_data_)();
    }

    void *mutable_device_data() {
        toDevice();
        data_dirty_ = true;
        return (*device_data_)();
    }
#endif

    size_t size() const {
        return host_data_.size();
    }

    void fill(U value) {
        data_is_on_host_ = true;
        data_dirty_ = true;
        std::fill(std::begin(host_data_), std::end(host_data_), value);
    }

    void reshape(const size_t d0,
                 const size_t d1,
                 const size_t d2,
                 const size_t d3) {
        shape_[0] = d0;
        shape_[1] = d1;
        shape_[2] = d2;
        shape_[3] = d3;
        host_data_.resize(calcSize(), U(0));
    }

    void reshape(const std::array<size_t, 4> &sz) {
        shape_ = sz;
        host_data_.resize(calcSize(), U(0));
    }

private:
    size_t calcSize() const {
        return std::accumulate(std::begin(shape_), std::end(shape_), size_t(1), std::multiplies<size_t>());
    }

    void toDevice() const {
        if (data_is_on_host_ && data_dirty_) {
#if defined(USE_OPENCL) || defined(USE_CUDA)
            CLCudaAPI::Queue queue = device->queue();
            if (device_data_ && device_data_->GetSize() >= host_data_.size()) {
                device_data_->Write(queue, host_data.size(), host_data_.data(), 0);
            }
            else {
                CLCudaAPI::Context ctx = device->context();
                device_data_ = make_unique<CLCudaAPI::Buffer<U> >(
                    ctx, queue, host_data_.begin(), host_data_.end());
            }
#endif
            data_is_on_host_ = false;
            data_dirty_ = false;
        }
    }

    void fromDevice() const {
        if (!data_is_on_host_ && data_dirty_) {
#if defined(USE_OPENCL) || defined(USE_CUDA)
            assert(device_);
            assert(device_data_);
            device_data_->Read(device_->queue(), host_data_.size(), const_cast<U*>(host_data_.data())); // using const_cast<> to avoid making host_data_ entirely mutable
#endif
            data_is_on_host_ = true;
            data_dirty_ = false;
        }
    }

private:
    /* Vector with the size of the tensor
     * shape_[0]: batch
     * shape_[1]: width
     * shape_[2]: height
     * shape_[3]: depth
     */
    std::array<size_t, 4> shape_;

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
    const std::vector<serial_size_t>& shape = tensor.shape();
    for (serial_size_t i = 0; i < shape[0]; ++i) {
        os << "-- Batch: " << i << "\n";
        for (serial_size_t j = 0; j < shape[3]; ++j) {
            os << "-- Channel: " << j << "\n";
            os << "-- Data:\n";
            for (serial_size_t k = 0; k < shape[1]; ++k) {
                for (serial_size_t l = 0; l < shape[2]; ++l) {
                    os << "   " << tensor.at(i, k, l, j) << " ";
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

// utilities for element-wise and tensor-scalar/scalar-tensor operations

template<typename TD, typename TS1, typename TS2, typename F> void binary_tensor_tensor_elementwise_operation(Tensor<TD> &dst, const Tensor<TS1> &src1, const Tensor<TS2> &src2, F f) {
    if (src1.shape() != src2.shape()) {
        throw nn_error("Tensor must have same shape");
    }

    dst.reshape(src1.shape());

    TD* pdst = dst.mutable_host_data();
    const TS1* psrc1 = src1.host_data();
    const TS2* psrc2 = src2.host_data();

    for_i(true, dst.size(), [pdst, psrc1, psrc2, &f](size_t i) {
        pdst[i] = f(psrc1[i], psrc2[i]);
    });
}

template<typename TD, typename TS, typename F> void unary_tensor_elementwise_operation(Tensor<TD> &dst, const Tensor<TS> &src, F f) {
    dst.reshape(src.shape());

    TD* pdst = dst.mutable_host_data();
    const TS* psrc = src.host_data();

    for_i(true, dst.size(), [pdst, psrc, &f](size_t i) {
        pdst[i] = f(psrc[i]);
    });
}

template<typename TD, typename TS1, typename TS2, typename F> void binary_tensor_scalar_operation(Tensor<TD> &dst, const Tensor<TS1> &src1, TS2 src2, F f) {
    dst.reshape(src1.shape());

    TD* pdst = dst.mutable_host_data();
    const TS1* psrc1 = src1.host_data();

    for_i(true, dst.size(), [pdst, psrc1, src2, &f](size_t i) {
        pdst[i] = f(psrc1[i], src2);
    });
}

template<typename TD, typename TS1, typename TS2, typename F> void binary_scalar_tensor_operation(Tensor<TD> &dst, TS1 src1, const Tensor<TS2> &src2, F f) {
    dst.reshape(src2.shape());

    TD* pdst = dst.mutable_host_data();
    const TS2* psrc2 = src2.host_data();

    for_i(true, dst.size(), [pdst, src1, psrc2, &f](size_t i) {
        pdst[i] = f(src1, psrc2[i]);
    });
}

// implementation of 

namespace details {
    template<typename TS1, typename TS2> auto plus(TS1 s1, TS2 s2) -> decltype(s1 + s2) { return s1 + s2; }

    template<typename TS1, typename TS2> auto minus(TS1 s1, TS2 s2) -> decltype(s1 - s2) { return s1 - s2; }

    template<typename TS1, typename TS2> auto multiplies(TS1 s1, TS2 s2) -> decltype(s1 * s2) { return s1 * s2; }

    template<typename TS1, typename TS2> auto divides_checked(TS1 s1, TS2 s2) -> decltype(s1 / s2) {
        typedef decltype(s1 / s2) result_type;
        return (s2 == result_type{}) ? std::numeric_limits<result_type>::quiet_NaN() : s1 / s2;
    }

    template<typename TS1, typename TS2> auto divides_unchecked(TS1 s1, TS2 s2) -> decltype(s1 / s2) {
        return s1 / s2;
    }

    template<typename T> T sqrt_checked(T s1) {
        return (s1 <= T{}) ? std::numeric_limits<T>::quiet_NaN() : sqrt(s1);
    }

    // do not inline - this function converts the std::exp overloadeds in a single templated function.
    template<typename T> T exp(T s1) {
        return std::exp(s1);
    }
}

template<typename TD, typename TS1, typename TS2> void layer_add(Tensor<TD> &dst, TS1 src1, const Tensor<TS2> &src2) {
    binary_scalar_tensor_operation(dst, src1, src2, details::plus<TS1, TS2>);
}

template<typename TD, typename TS1, typename TS2> void layer_add(Tensor<TD> &dst, const Tensor<TS1> &src1, TS2 src2) {
    binary_tensor_scalar_operation(dst, src1, src2, details::plus<TS1, TS2>);
}

template<typename TD, typename TS1, typename TS2> void layer_add(Tensor<TD> &dst, const Tensor<TS1> &src1, const Tensor<TS2> &src2) {
    binary_tensor_tensor_elementwise_operation(dst, src1, src2, details::plus<TS1, TS2>);
}

template<typename TD, typename TS1, typename TS2> void layer_sub(Tensor<TD> &dst, TS1 src1, const Tensor<TS2> &src2) {
    binary_scalar_tensor_operation(dst, src1, src2, details::minus<TS1, TS2>);
}

template<typename TD, typename TS1, typename TS2> void layer_sub(Tensor<TD> &dst, const Tensor<TS1> &src1, TS2 src2) {
    binary_tensor_scalar_operation(dst, src1, src2, details::minus<TS1, TS2>);
}

template<typename TD, typename TS1, typename TS2> void layer_sub(Tensor<TD> &dst, const Tensor<TS1> &src1, const Tensor<TS2> &src2) {
    binary_tensor_tensor_elementwise_operation(dst, src1, src2, details::minus<TS1, TS2>);
}

template<typename TD, typename TS1, typename TS2> void layer_mul(Tensor<TD> &dst, TS1 src1, const Tensor<TS2> &src2) {
    binary_scalar_tensor_operation(dst, src1, src2, details::multiplies<TS1, TS2>);
}

template<typename TD, typename TS1, typename TS2> void layer_mul(Tensor<TD> &dst, const Tensor<TS1> &src1, TS2 src2) {
    binary_tensor_scalar_operation(dst, src1, src2, details::multiplies<TS1, TS2>);
}

template<typename TD, typename TS1, typename TS2> void layer_mul(Tensor<TD> &dst, const Tensor<TS1> &src1, const Tensor<TS2> &src2) {
    binary_tensor_tensor_elementwise_operation(dst, src1, src2, details::multiplies<TS1, TS2>);
}

template<typename TD, typename TS1, typename TS2> void layer_div(Tensor<TD> &dst, TS1 src1, const Tensor<TS2> &src2) {
    binary_scalar_tensor_operation(dst, src1, src2, details::divides_checked<TS1, TS2>);
}

template<typename TD, typename TS1, typename TS2> void layer_div(Tensor<TD> &dst, const Tensor<TS1> &src1, TS2 src2) {
    if (src2 == TS2(0.0)) {
        dst.reshape(src1.shape());
        dst.fill(std::numeric_limits<TD>::quiet_NaN());
    } else {
        binary_tensor_scalar_operation(dst, src1, src2, details::divides_unchecked<TS1, TS2>);
    }
}

template<typename TD, typename TS1, typename TS2> void layer_div(Tensor<TD> &dst, const Tensor<TS1> &src1, const Tensor<TS2> &src2) {
    binary_tensor_tensor_elementwise_operation(dst, src1, src2, details::divides_checked<TS1, TS2>);
}

template<typename TD, typename TS> void layer_sqrt(Tensor<TD> &dst, const Tensor<TS> &src1) {
    return unary_tensor_elementwise_operation(dst, src1, details::sqrt_checked<TS>);
}

template<typename TD, typename TS> void layer_exp(Tensor<TD> &dst, const Tensor<TS> &src1) {
    return unary_tensor_elementwise_operation(dst, src1, details::exp<TS>);
}

}  // namespace tiny_dnn
