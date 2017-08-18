/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include <algorithm>  // std::fill, std::generate
#include <cmath>      // sqrt
#include <functional>
#include <limits>
#include <numeric>  // std::accumulate

#include "tiny_dnn/xtensor/xio.hpp"

#include "tiny_dnn/core/framework/tensor.h"

namespace tiny_dnn {

/**
 *
 * @tparam T
 * @tparam S
 * @param os
 * @param tensor
 * @return
 */
template <typename T, typename S>
inline std::ostream &operator<<(std::ostream &os, const Tensor<T, S> &tensor) {
  os << "Tensor of dim " << tensor.dim();
  if (tensor.dim() == 0) return os;
  os << ", size " << tensor.size() << " and shape ";
  for (size_t i = 0; i < tensor.dim() - 1; ++i) {
    os << tensor.shape()[i] << "x";
  }
  os << tensor.shape().back() << ":" << std::endl;
  os << tensor.storage_;
  return os;
}

template <class T, class S1, class S2>
bool operator==(const Tensor<T, S1> &t1, const Tensor<T, S2> &t2) {
  if (t1.shape() != t2.shape()) return false;
  auto it1 = t1.host_begin(), it2 = t2.host_begin();
  for (; it1 != t1.host_end(); ++it1, ++it2) {
    if (*it1 != *it2) return false;
  }
  return true;
}

template <class T, class S1, class S2>
bool operator!=(const Tensor<T, S1> &t1, const Tensor<T, S2> &t2) {
  return !(t1 == t2);
}

// utilities for element-wise and tensor-scalar/scalar-tensor operations

template <typename TD,
          typename TDS,
          typename TS1,
          typename TS1S,
          typename TS2,
          typename TS2S,
          typename F>
void binary_tensor_tensor_elementwise_operation(Tensor<TD, TDS> &dst,
                                                const Tensor<TS1, TS1S> &src1,
                                                const Tensor<TS2, TS2S> &src2,
                                                F f) {
  if (src1.shape() != src2.shape()) {
    throw nn_error("Tensor must have same shape");
  }

  dst.reshape(src1.shape());

  auto pdst  = dst.host_begin();
  auto psrc1 = src1.host_begin();
  auto psrc2 = src2.host_begin();

  for_i(true, dst.size(), [pdst, psrc1, psrc2, &f](size_t i) {
    *std::next(pdst, i) = f(*std::next(psrc1, i), *std::next(psrc2, i));
  });
}

template <typename TD, typename TDS, typename TS, typename TSS, typename F>
void unary_tensor_elementwise_operation(Tensor<TD, TDS> &dst,
                                        const Tensor<TS, TSS> &src,
                                        F f) {
  dst.reshape(src.shape());

  auto pdst = dst.host_begin();
  auto psrc = src.host_begin();

  for_i(true, dst.size(), [pdst, psrc, &f](size_t i) {
    *std::next(pdst, i) = f(*std::next(psrc, i));
  });
}

template <typename TD,
          typename TDS,
          typename TS1,
          typename TS1S,
          typename TS2,
          typename F>
void binary_tensor_scalar_operation(Tensor<TD, TDS> &dst,
                                    const Tensor<TS1, TS1S> &src1,
                                    TS2 src2,
                                    F f) {
  dst.reshape(src1.shape());

  auto pdst  = dst.host_begin();
  auto psrc1 = src1.host_begin();

  for_i(true, dst.size(), [pdst, psrc1, src2, &f](size_t i) {
    *std::next(pdst, i) = f(*std::next(psrc1, i), src2);
  });
}

template <typename TD,
          typename TDS,
          typename TS1,
          typename TS2,
          typename TS2S,
          typename F>
void binary_scalar_tensor_operation(Tensor<TD, TDS> &dst,
                                    TS1 src1,
                                    const Tensor<TS2, TS2S> &src2,
                                    F f) {
  dst.reshape(src2.shape());

  auto pdst  = dst.host_begin();
  auto psrc2 = src2.host_begin();

  for_i(true, dst.size(), [pdst, src1, psrc2, &f](size_t i) {
    *std::next(pdst, i) = f(src1, *std::next(psrc2, i));
  });
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

template <typename TD, typename TDS, typename TS1, typename TS2, typename TS2S>
void layer_add(Tensor<TD, TDS> &dst, TS1 src1, const Tensor<TS2, TS2S> &src2) {
  binary_scalar_tensor_operation(dst, src1, src2, details::plus<TS1, TS2>);
}

template <typename TD, typename TDS, typename TS1, typename TS1S, typename TS2>
void layer_add(Tensor<TD, TDS> &dst, const Tensor<TS1, TS1S> &src1, TS2 src2) {
  binary_tensor_scalar_operation(dst, src1, src2, details::plus<TS1, TS2>);
}

template <typename TD,
          typename TDS,
          typename TS1,
          typename TS1S,
          typename TS2,
          typename TS2S>
void layer_add(Tensor<TD, TDS> &dst,
               const Tensor<TS1, TS1S> &src1,
               const Tensor<TS2, TS2S> &src2) {
  binary_tensor_tensor_elementwise_operation(dst, src1, src2,
                                             details::plus<TS1, TS2>);
}

template <typename TD, typename TDS, typename TS1, typename TS2, typename TS2S>
void layer_sub(Tensor<TD, TDS> &dst, TS1 src1, const Tensor<TS2, TS2S> &src2) {
  binary_scalar_tensor_operation(dst, src1, src2, details::minus<TS1, TS2>);
}

template <typename TD, typename TDS, typename TS1, typename TS1S, typename TS2>
void layer_sub(Tensor<TD, TDS> &dst, const Tensor<TS1, TS1S> &src1, TS2 src2) {
  binary_tensor_scalar_operation(dst, src1, src2, details::minus<TS1, TS2>);
}

template <typename TD,
          typename TDS,
          typename TS1,
          typename TS1S,
          typename TS2,
          typename TS2S>
void layer_sub(Tensor<TD, TDS> &dst,
               const Tensor<TS1, TS1S> &src1,
               const Tensor<TS2, TS2S> &src2) {
  binary_tensor_tensor_elementwise_operation(dst, src1, src2,
                                             details::minus<TS1, TS2>);
}

template <typename TD, typename TDS, typename TS1, typename TS2, typename TS2S>
void layer_mul(Tensor<TD, TDS> &dst, TS1 src1, const Tensor<TS2, TS2S> &src2) {
  binary_scalar_tensor_operation(dst, src1, src2,
                                 details::multiplies<TS1, TS2>);
}

template <typename TD, typename TDS, typename TS1, typename TS1S, typename TS2>
void layer_mul(Tensor<TD, TDS> &dst, const Tensor<TS1, TS1S> &src1, TS2 src2) {
  binary_tensor_scalar_operation(dst, src1, src2,
                                 details::multiplies<TS1, TS2>);
}

template <typename TD,
          typename TDS,
          typename TS1,
          typename TS1S,
          typename TS2,
          typename TS2S>
void layer_mul(Tensor<TD, TDS> &dst,
               const Tensor<TS1, TS1S> &src1,
               const Tensor<TS2, TS2S> &src2) {
  binary_tensor_tensor_elementwise_operation(dst, src1, src2,
                                             details::multiplies<TS1, TS2>);
}

template <typename TD, typename TDS, typename TS1, typename TS2, typename TS2S>
void layer_div(Tensor<TD, TDS> &dst, TS1 src1, const Tensor<TS2, TS2S> &src2) {
  binary_scalar_tensor_operation(dst, src1, src2,
                                 details::divides_checked<TS1, TS2>);
}

template <typename TD, typename TDS, typename TS1, typename TS1S, typename TS2>
void layer_div(Tensor<TD, TDS> &dst, const Tensor<TS1, TS1S> &src1, TS2 src2) {
  if (src2 == TS2(0.0)) {
    dst.reshape(src1.shape());
    dst.fill(std::numeric_limits<TD>::quiet_NaN());
  } else {
    binary_tensor_scalar_operation(dst, src1, src2,
                                   details::divides_unchecked<TS1, TS2>);
  }
}

template <typename TD,
          typename TDS,
          typename TS1,
          typename TS1S,
          typename TS2,
          typename TS2S>
void layer_div(Tensor<TD, TDS> &dst,
               const Tensor<TS1, TS1S> &src1,
               const Tensor<TS2, TS2S> &src2) {
  binary_tensor_tensor_elementwise_operation(
    dst, src1, src2, details::divides_checked<TS1, TS2>);
}

template <typename TD, typename TDS, typename TS, typename TSS>
void layer_sqrt(Tensor<TD, TDS> &dst, const Tensor<TS, TSS> &src1) {
  return unary_tensor_elementwise_operation(dst, src1,
                                            details::sqrt_checked<TS>);
}

template <typename TD, typename TDS, typename TS, typename TSS>
void layer_exp(Tensor<TD, TDS> &dst, const Tensor<TS, TSS> &src1) {
  return unary_tensor_elementwise_operation(dst, src1, details::exp<TS>);
}

}  // namespace tiny_dnn
