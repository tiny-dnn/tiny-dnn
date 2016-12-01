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

/**
 * A tensor of the given dimension.
 * A tensor holds data in C-style nD array, i.e row-major order:
 * the rightmost index “varies the fastest”.
 *
 *  Data is held by a std::vector with 64 bytes alignment.
 */
template<typename U = float_t, size_t kDimensions = 4>
class Tensor {
public:

    /**
     * Initializes an empty tensor.
     * @return
     */
    Tensor() {
        reshape(std::array<size_t, kDimensions>());
    }

    /**
     * Constructor that assepts an array of shape and create a Tensor with that
     * shape. For example, given shape = {2,3,4,5,6}, tensor will be of size
     * 2x3x4x5x6
     * @param shape array containing N integers, sizes of dimensions
     * @return
     */
    explicit Tensor(const std::array<size_t, kDimensions>& shape) {
        reshape(shape);
    }

    /**
     * Constructor that assepts a vector of shape and create a Tensor with that
     * shape. For example, given shape = {2,3,4,5,6}, tensor will be of size
     * 2x3x4x5x6
     * @param shape array containing N integers, sizes of dimensions
     * @return
     */
    explicit Tensor(const std::vector<size_t>& shape) {
        reshape(shape);
    }

    /**
     * Constructor that assepts an initializer list of shape and create a
     * Tensor with that shape. For example, given shape = {2,3,4,5,6}, tensor
     * will be of size 2x3x4x5x6
     * @param shape array containing N integers, sizes of dimensions
     * @return
     */
    explicit Tensor(std::initializer_list<size_t> const& shape)  {
        assert(shape.size() == kDimensions);
        std::array<size_t, kDimensions> tmp;
        std::copy(shape.begin(),shape.end(),tmp.begin());
        reshape(tmp);
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

        //device_data_ is intentionally left as-is. It will be erased only if
        // new tensor won't fit, and only when data gets moved to the GPU.
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
        static_assert(sizeof...(args) < kDimensions, "Wrong number of dimensions");
        size_t dim = kDimensions - sizeof...(args) - 1;
        if (d >= shape_[dim])  {
            throw nn_error("Access tensor out of range.");
        }
        size_t shift = 1;
        for (size_t i = dim + 1; i < kDimensions; ++i)
            shift *= shape_[i]; //TODO(Randl): optimize. Reverse argumets?

        return (d * shift + host_pos(args...) );
    }

    /**
     * Checked version (throw exceptions for out-of-range error)
     * @param args index of rest (k-1) dimensions.
     * @return the pointer to a specified index in the tensor
     */
    template<typename... Args>
    const U* host_ptr (const Args... args) const {
        static_assert(sizeof...(args) == kDimensions, "Wrong number of dimensions");
        return host_data() + host_pos(args...);
    }

    template<typename... Args>
    U* host_ptr(const Args... args) {
        static_assert(sizeof...(args) == kDimensions, "Wrong number of dimensions");
        return mutable_host_data() + host_pos(args...);
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


    //TODO(Randl): variadic template version of reshape
    void reshape(const std::array<size_t, kDimensions> &sz) {
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
            CLCudaAPI::Queue queue = device_->queue();
            if (device_data_ && device_data_->GetSize() >= host_data_.size()) {
                device_data_->Write(queue, host_data.size(), host_data_.data(), 0);
            }
            else {
                CLCudaAPI::Context ctx = device_->context();
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
    /**
     * A tensor holds data in C-style nD array, i.e row-major order:
     * the rightmost index “varies the fastest”.
     */
    std::array<size_t, kDimensions> shape_;

    /* Pointer to the Tensor data in pure in the host device */
    std::vector<U, aligned_allocator<U, 64>> host_data_;

#if defined(USE_OPENCL) || defined(USE_CUDA)
    /* Pointer to the Tensor data in the device */
    std::unique_ptr<CLCudaAPI::Buffer<U> > device_data_;
#endif
    mutable bool data_is_on_host_;    //< current data is on host if true, on device if false.
    mutable bool data_dirty_;         //< set to true if current data might have been modified

    /* Pointer to the current device where the data resides */
    Device* device_;
};

template<typename T>
void print_pack(std::ostream &out, T t) { //TODO: C++17 allows easier printing
    out << t;
}

template<typename T, typename U, typename... Args>
void print_pack(std::ostream &out, T t, U u, Args... args) {
    out << t << ',';
    print_pack(out, u, args...);
}

template<typename T, size_t kDim, typename... Args>
inline std::ostream& print_last_two_dimesions (std::ostream     &os,
                                               const Tensor<T, kDim>& tensor,
                                               const Args...    args) {

    const std::array<size_t, kDim>& shape = tensor.shape();
    for (size_t k = 0; k < shape[kDim-1]; ++k) {
        for (size_t l = 0; l < shape[kDim-2]; ++l) {
            os << " " << tensor.host_at(args..., l, k) << " ";
        }
        os << ";\n";
    }
    return os;
}

template<typename T, size_t kDim, typename... Args>
inline std::ostream& print_last_n_dimesions (std::ostream &os,
                                               const Tensor<T, kDim>& tensor,
                                               const int d,
                                               const Args... args) {
    const std::array<size_t, kDim>& shape = tensor.shape();
    const size_t n_dim = sizeof...(args);
    if (n_dim == shape.size() - 3) {
        os << "Tensor(";
        print_pack(os, d, args...);
        os << ",:,:):\n";
        print_last_two_dimesions (os, tensor, d, args...);
        return os;
    }
    for (size_t k = 0; k < shape[n_dim+1]; ++k) {
        print_last_n_dimesions(os, tensor, d, args..., k);
    }
    return os;
}
/**
 * Overloaded method to print the Tensor class to the standard output
 * @param os
 * @param tensor
 * @return
 */
//TODO(Randl): make to compile
template<typename T, size_t kDim>
inline std::ostream& operator<< (std::ostream &os,
                                 const Tensor<T, kDim>& tensor) {

    const std::array<size_t, kDim>& shape = tensor.shape();

    for (size_t i = 0 ; i < shape[0]; ++i)
        print_last_n_dimesions(os, tensor, i);
    return os;
}

// utilities for element-wise and tensor-scalar/scalar-tensor operations

template<typename TD, typename TS1, typename TS2, typename F, size_t kDim> void binary_tensor_tensor_elementwise_operation(Tensor<TD, kDim> &dst, const Tensor<TS1, kDim> &src1, const Tensor<TS2, kDim> &src2, F f) {
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

template<typename TD, typename TS, typename F, size_t kDim> void unary_tensor_elementwise_operation(Tensor<TD, kDim> &dst, const Tensor<TS, kDim> &src, F f) {
    dst.reshape(src.shape());

    TD* pdst = dst.mutable_host_data();
    const TS* psrc = src.host_data();

    for_i(true, dst.size(), [pdst, psrc, &f](size_t i) {
        pdst[i] = f(psrc[i]);
    });
}

template<typename TD, typename TS1, typename TS2, typename F, size_t kDim> void binary_tensor_scalar_operation(Tensor<TD, kDim> &dst, const Tensor<TS1, kDim> &src1, TS2 src2, F f) {
    dst.reshape(src1.shape());

    TD* pdst = dst.mutable_host_data();
    const TS1* psrc1 = src1.host_data();

    for_i(true, dst.size(), [pdst, psrc1, src2, &f](size_t i) {
        pdst[i] = f(psrc1[i], src2);
    });
}

template<typename TD, typename TS1, typename TS2, typename F, size_t kDim> void binary_scalar_tensor_operation(Tensor<TD, kDim> &dst, TS1 src1, const Tensor<TS2, kDim> &src2, F f) {
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

template<typename TD, typename TS1, typename TS2, size_t kDim> void layer_add(Tensor<TD, kDim> &dst, TS1 src1, const Tensor<TS2, kDim> &src2) {
    binary_scalar_tensor_operation(dst, src1, src2, details::plus<TS1, TS2>);
}

template<typename TD, typename TS1, typename TS2, size_t kDim> void layer_add(Tensor<TD, kDim> &dst, const Tensor<TS1, kDim> &src1, TS2 src2) {
    binary_tensor_scalar_operation(dst, src1, src2, details::plus<TS1, TS2>);
}

template<typename TD, typename TS1, typename TS2, size_t kDim> void layer_add(Tensor<TD, kDim> &dst, const Tensor<TS1, kDim> &src1, const Tensor<TS2, kDim> &src2) {
    binary_tensor_tensor_elementwise_operation(dst, src1, src2, details::plus<TS1, TS2>);
}

template<typename TD, typename TS1, typename TS2, size_t kDim> void layer_sub(Tensor<TD, kDim> &dst, TS1 src1, const Tensor<TS2, kDim> &src2) {
    binary_scalar_tensor_operation(dst, src1, src2, details::minus<TS1, TS2>);
}

template<typename TD, typename TS1, typename TS2, size_t kDim> void layer_sub(Tensor<TD, kDim> &dst, const Tensor<TS1, kDim> &src1, TS2 src2) {
    binary_tensor_scalar_operation(dst, src1, src2, details::minus<TS1, TS2>);
}

template<typename TD, typename TS1, typename TS2, size_t kDim> void layer_sub(Tensor<TD, kDim> &dst, const Tensor<TS1, kDim> &src1, const Tensor<TS2, kDim> &src2) {
    binary_tensor_tensor_elementwise_operation(dst, src1, src2, details::minus<TS1, TS2>);
}

template<typename TD, typename TS1, typename TS2, size_t kDim> void layer_mul(Tensor<TD, kDim> &dst, TS1 src1, const Tensor<TS2, kDim> &src2) {
    binary_scalar_tensor_operation(dst, src1, src2, details::multiplies<TS1, TS2>);
}

template<typename TD, typename TS1, typename TS2, size_t kDim> void layer_mul(Tensor<TD, kDim> &dst, const Tensor<TS1, kDim> &src1, TS2 src2) {
    binary_tensor_scalar_operation(dst, src1, src2, details::multiplies<TS1, TS2>);
}

template<typename TD, typename TS1, typename TS2, size_t kDim> void layer_mul(Tensor<TD, kDim> &dst, const Tensor<TS1, kDim> &src1, const Tensor<TS2, kDim> &src2) {
    binary_tensor_tensor_elementwise_operation(dst, src1, src2, details::multiplies<TS1, TS2>);
}

template<typename TD, typename TS1, typename TS2, size_t kDim> void layer_div(Tensor<TD, kDim> &dst, TS1 src1, const Tensor<TS2, kDim> &src2) {
    binary_scalar_tensor_operation(dst, src1, src2, details::divides_checked<TS1, TS2>);
}

template<typename TD, typename TS1, typename TS2, size_t kDim> void layer_div(Tensor<TD, kDim> &dst, const Tensor<TS1, kDim> &src1, TS2 src2) {
    if (src2 == TS2(0.0)) {
        dst.reshape(src1.shape());
        dst.fill(std::numeric_limits<TD>::quiet_NaN());
    } else {
        binary_tensor_scalar_operation(dst, src1, src2, details::divides_unchecked<TS1, TS2>);
    }
}

template<typename TD, typename TS1, typename TS2, size_t kDim> void layer_div(Tensor<TD, kDim> &dst, const Tensor<TS1, kDim> &src1, const Tensor<TS2, kDim> &src2) {
    binary_tensor_tensor_elementwise_operation(dst, src1, src2, details::divides_checked<TS1, TS2>);
}

template<typename TD, typename TS, size_t kDim> void layer_sqrt(Tensor<TD, kDim> &dst, const Tensor<TS, kDim> &src1) {
    return unary_tensor_elementwise_operation(dst, src1, details::sqrt_checked<TS>);
}

template<typename TD, typename TS, size_t kDim> void layer_exp(Tensor<TD, kDim> &dst, const Tensor<TS, kDim> &src1) {
    return unary_tensor_elementwise_operation(dst, src1, details::exp<TS>);
}

}  // namespace tiny_dnn
