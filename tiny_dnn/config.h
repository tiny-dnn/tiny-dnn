// Copyright (c) 2013-2016, Taiga Nomi. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#pragma once
#include <cstddef>

/**
 * define if you want to use intel TBB library
 */
//#define CNN_USE_TBB

/**
 * define to enable avx vectorization
 */
//#define CNN_USE_AVX

/**
 * define to enable sse2 vectorization
 */
//#define CNN_USE_SSE

/**
 * define to enable OMP parallelization
 */
//#define CNN_USE_OMP

/**
 * define to use exceptions
 */
#define CNN_USE_EXCEPTIONS

/**
 * comment out if you want tiny-dnn to be quiet 
 */
#define CNN_USE_STDOUT


/**
 * disable serialization/deserialization function
 * You can uncomment this to speedup compilation&linking time,
 * if you don't use network::save / network::load functions.
 **/
//#define CNN_NO_SERIALIZATION

/**
 * number of task in batch-gradient-descent.
 * @todo automatic optimization
 */
#ifdef CNN_USE_OMP
#define CNN_TASK_SIZE 100
#else
#define CNN_TASK_SIZE 8
#endif

#if !defined(_MSC_VER) && !defined(_WIN32) && !defined(WIN32)
#define CNN_USE_GEMMLOWP // gemmlowp doesn't support MSVC/mingw
#endif

namespace tiny_dnn {

/**
 * calculation data type
 * you can change it to float, or user defined class (fixed point,etc)
 **/
typedef float float_t;

/**
 * size of layer, model, data etc.
 * change to smaller type if memory footprint is severe
 **/
typedef std::size_t cnn_size_t;

}
