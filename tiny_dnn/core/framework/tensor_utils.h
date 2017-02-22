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

#include <algorithm>  // std::fill, std::generate
#include <cmath>      // sqrt
#include <numeric>    // std::accumulate

#include "tiny_dnn/core/framework/tensor.h"

namespace tiny_dnn {

template <typename T>
std::ostream &print_vector(std::ostream &out, std::vector<T> indices) {
  for (auto &t : indices) out << t << ',';
  return out;
}

/*template <typename T, typename... Args>
inline std::ostream &print_last_two_dimesions(std::ostream &os,
                                              const Tensor<T> &tensor,
                                              const Args... args) {
  const std::array<size_t> &shape = tensor.shape();
  for (size_t k = 0; k < shape[kDim - 2]; ++k) {
    for (size_t l = 0; l < shape[kDim - 1]; ++l) {
      os << "\t" << tensor.host_at(args..., k, l);
    }
    os << "\n";
  }
  return os;
}*/

/*template <typename T,
          typename... Args,
          typename std::enable_if<sizeof...(Args) == kDim - 3, int>::type = 0>
inline std::ostream &print_last_n_dimesions(std::ostream &os,
                                            const Tensor<T> &tensor,
                                            const int d,
                                            const Args... args) {
  // const std::array<size_t>& shape = tensor.shape();
  // const size_t n_dim = sizeof...(args);
  os << "Tensor(";
  print_pack(os, d, args...);
  os << ",:,:):\n";
  print_last_two_dimesions(os, tensor, d, args...);
  return os;
}*/

// TODO(Ranld): static_if (C++17)
/*template <typename T>
inline std::ostream &operator<<(std::ostream &os, const Tensor<T, 1> &tensor) {
  const std::array<size_t, 1> &shape = tensor.shape();
  for (size_t i = 0; i < shape[0]; ++i) os << "\t" << tensor.host_at(i);
  os << "\n";
  return os;
}

template <typename T>
inline std::ostream &operator<<(std::ostream &os, const Tensor<T, 2> &tensor) {
  print_last_two_dimesions(os, tensor);
  return os;
}*/

template <typename T>
inline std::ostream &print_last_two_dimensions(std::ostream &os,
                                               const Tensor<T> &tensor,
                                               std::vector<size_t> indices) {
  const size_t dims = tensor.dims();
  const auto shape  = tensor.shape();

  assert(tensor.dims() == indices.size() + 2);

  os << "(";
  print_vector(os, indices);
  os << ":,:):\n";

  for (size_t k = 0; k < shape[dims - 2]; ++k) {
    for (size_t l = 0; l < shape[dims - 1]; ++l) {
      os << "\t" << tensor.host_at(indices, k, l);
    }
    os << "\n";
  }
  return os;
}

template <typename T>
inline std::ostream &print_last_dimension(std::ostream &os,
                                          const Tensor<T> &tensor,
                                          std::vector<size_t> indices) {
  for (size_t k = 0; k < tensor.shape()[0]; ++k) {
    os << "\t" << tensor.host_at(indices, k);
    os << "\n";
  }
  return os;
}

/**
 * Prints last n out of total m dimensions of Tensor, with first (m-n) indices
 * specified
 * @param os Stream to print to
 * @param tensor Tesor which is printed
 * @param indices vector of indices
 * @return same stream
 */
template <typename T>
inline std::ostream &print_last_n_dimensions(std::ostream &os,
                                             const Tensor<T> &tensor,
                                             std::vector<size_t> indices) {
  const size_t n_dim = indices.size();
  assert(n_dim < indices.size());
  if (tensor.dims() <= n_dim + 1) {
    return print_last_dimension(os, tensor, indices);
  } else if (tensor.dims() == n_dim + 2) {
    return print_last_two_dimensions(os, tensor, indices);
  } else {
    const std::vector<size_t> &shape = tensor.shape();
    for (size_t k = 0; k < shape[n_dim]; ++k) {
      indices.push_back(k);
      print_last_n_dimensions(os, tensor, indices);
      indices.pop_back();
    }
  }

  return os;
}

template <typename T>
inline std::ostream &print_tensor_footer(std::ostream &os,
                                         const Tensor<T> &tensor) {
  const auto shape = tensor.shape();
  os << "[tiny_dnn.Tensor of size " << shape[0];
  for (size_t i = 1; i < shape.size(); ++i) {
    os << "x" << shape[i];
  }
  os << "]" << std::endl;
  return os;
}

/**
 * Overloaded method to print the Tensor class to the standard output
 * @param os
 * @param tensor
 * @return
 */
template <typename T>
inline std::ostream &operator<<(std::ostream &os, const Tensor<T> &tensor) {
  for (size_t i = 0; i < tensor.shape()[0]; ++i) {
    print_last_n_dimensions(os, tensor, {i});
  }
  print_tensor_footer(os, tensor);
  return os;
}

// utilities for element-wise and tensor-scalar/scalar-tensor operations

template <typename TD, typename TS1, typename TS2, typename F>
void binary_tensor_tensor_elementwise_operation(Tensor<TD> &dst,
                                                const Tensor<TS1> &src1,
                                                const Tensor<TS2> &src2,
                                                F f) {
  if (src1.shape() != src2.shape()) {
    throw nn_error("Tensor must have same shape");
  }

  dst.resize(src1.shape());

  auto pdst  = dst.host_begin();
  auto psrc1 = src1.host_begin();
  auto psrc2 = src2.host_begin();

  for_i(true, dst.size(), [pdst, psrc1, psrc2, &f](size_t i) {
    pdst[i] = f(psrc1[i], psrc2[i]);
  });
}

template <typename TD, typename TS, typename F>
void unary_tensor_elementwise_operation(Tensor<TD> &dst,
                                        const Tensor<TS> &src,
                                        F f) {
  dst.resize(src.shape());

  auto pdst = dst.host_begin();
  auto psrc = src.host_begin();

  for_i(true, dst.size(), [pdst, psrc, &f](size_t i) { pdst[i] = f(psrc[i]); });
}

template <typename TD, typename TS1, typename TS2, typename F>
void binary_tensor_scalar_operation(Tensor<TD> &dst,
                                    const Tensor<TS1> &src1,
                                    TS2 src2,
                                    F f) {
  dst.resize(src1.shape());

  auto pdst  = dst.host_begin();
  auto psrc1 = src1.host_begin();

  for_i(true, dst.size(),
        [pdst, psrc1, src2, &f](size_t i) { pdst[i] = f(psrc1[i], src2); });
}

template <typename TD, typename TS1, typename TS2, typename F>
void binary_scalar_tensor_operation(Tensor<TD> &dst,
                                    TS1 src1,
                                    const Tensor<TS2> &src2,
                                    F f) {
  dst.resize(src2.shape());

  auto pdst  = dst.host_begin();
  auto psrc2 = src2.host_begin();

  for_i(true, dst.size(),
        [pdst, src1, psrc2, &f](size_t i) { pdst[i] = f(src1, psrc2[i]); });
}

// implementation of

namespace details {

template <typename TS1, typename TS2>
auto plus(TS1 s1, TS2 s2) -> decltype(s1 + s2) {
  return s1 + s2;
}

template <typename TS1, typename TS2>
auto minus(TS1 s1, TS2 s2) -> decltype(s1 - s2) {
  return s1 - s2;
}

template <typename TS1, typename TS2>
auto multiplies(TS1 s1, TS2 s2) -> decltype(s1 * s2) {
  return s1 * s2;
}

template <typename TS1, typename TS2>
auto divides_checked(TS1 s1, TS2 s2) -> decltype(s1 / s2) {
  typedef decltype(s1 / s2) result_type;
  return (s2 == result_type{}) ? std::numeric_limits<result_type>::quiet_NaN()
                               : s1 / s2;
}

template <typename TS1, typename TS2>
auto divides_unchecked(TS1 s1, TS2 s2) -> decltype(s1 / s2) {
  return s1 / s2;
}

template <typename T>
T sqrt_checked(T s1) {
  return (s1 <= T{}) ? std::numeric_limits<T>::quiet_NaN() : sqrt(s1);
}

// do not inline - this function converts the std::exp
// overloadeds in a single templated function.
template <typename T>
T exp(T s1) {
  return std::exp(s1);
}

}  // namespace details

template <typename TD, typename TS1, typename TS2>
void layer_add(Tensor<TD> &dst, TS1 src1, const Tensor<TS2> &src2) {
  binary_scalar_tensor_operation(dst, src1, src2, details::plus<TS1, TS2>);
}

template <typename TD, typename TS1, typename TS2>
void layer_add(Tensor<TD> &dst, const Tensor<TS1> &src1, TS2 src2) {
  binary_tensor_scalar_operation(dst, src1, src2, details::plus<TS1, TS2>);
}

template <typename TD, typename TS1, typename TS2>
void layer_add(Tensor<TD> &dst,
               const Tensor<TS1> &src1,
               const Tensor<TS2> &src2) {
  binary_tensor_tensor_elementwise_operation(dst, src1, src2,
                                             details::plus<TS1, TS2>);
}

template <typename TD, typename TS1, typename TS2>
void layer_sub(Tensor<TD> &dst, TS1 src1, const Tensor<TS2> &src2) {
  binary_scalar_tensor_operation(dst, src1, src2, details::minus<TS1, TS2>);
}

template <typename TD, typename TS1, typename TS2>
void layer_sub(Tensor<TD> &dst, const Tensor<TS1> &src1, TS2 src2) {
  binary_tensor_scalar_operation(dst, src1, src2, details::minus<TS1, TS2>);
}

template <typename TD, typename TS1, typename TS2>
void layer_sub(Tensor<TD> &dst,
               const Tensor<TS1> &src1,
               const Tensor<TS2> &src2) {
  binary_tensor_tensor_elementwise_operation(dst, src1, src2,
                                             details::minus<TS1, TS2>);
}

template <typename TD, typename TS1, typename TS2>
void layer_mul(Tensor<TD> &dst, TS1 src1, const Tensor<TS2> &src2) {
  binary_scalar_tensor_operation(dst, src1, src2,
                                 details::multiplies<TS1, TS2>);
}

template <typename TD, typename TS1, typename TS2>
void layer_mul(Tensor<TD> &dst, const Tensor<TS1> &src1, TS2 src2) {
  binary_tensor_scalar_operation(dst, src1, src2,
                                 details::multiplies<TS1, TS2>);
}

template <typename TD, typename TS1, typename TS2>
void layer_mul(Tensor<TD> &dst,
               const Tensor<TS1> &src1,
               const Tensor<TS2> &src2) {
  binary_tensor_tensor_elementwise_operation(dst, src1, src2,
                                             details::multiplies<TS1, TS2>);
}

template <typename TD, typename TS1, typename TS2>
void layer_div(Tensor<TD> &dst, TS1 src1, const Tensor<TS2> &src2) {
  binary_scalar_tensor_operation(dst, src1, src2,
                                 details::divides_checked<TS1, TS2>);
}

template <typename TD, typename TS1, typename TS2>
void layer_div(Tensor<TD> &dst, const Tensor<TS1> &src1, TS2 src2) {
  if (src2 == TS2(0.0)) {
    dst.resize(src1.shape());
    dst.fill(std::numeric_limits<TD>::quiet_NaN());
  } else {
    binary_tensor_scalar_operation(dst, src1, src2,
                                   details::divides_unchecked<TS1, TS2>);
  }
}

template <typename TD, typename TS1, typename TS2>
void layer_div(Tensor<TD> &dst,
               const Tensor<TS1> &src1,
               const Tensor<TS2> &src2) {
  binary_tensor_tensor_elementwise_operation(
    dst, src1, src2, details::divides_checked<TS1, TS2>);
}

template <typename TD, typename TS>
void layer_sqrt(Tensor<TD> &dst, const Tensor<TS> &src1) {
  return unary_tensor_elementwise_operation(dst, src1,
                                            details::sqrt_checked<TS>);
}

template <typename TD, typename TS>
void layer_exp(Tensor<TD> &dst, const Tensor<TS> &src1) {
  return unary_tensor_elementwise_operation(dst, src1, details::exp<TS>);
}

}  // namespace tiny_dnn
