/*
    Copyright (c) 2013, Taiga Nomi
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in the
    documentation and/or other materials provided with the distribution.
    * Neither the name of the <organization> nor the
    names of its contributors may be used to endorse or promote products
    derived from this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
    EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
    DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
    LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
    ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
#pragma once
#include <cstddef>
#include <cstdint>

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
#ifdef CNN_USE_DOUBLE
typedef double float_t;
#else
typedef float float_t;
#endif

/**
 * size of layer, model, data etc.
 * change to smaller type if memory footprint is severe
 **/
typedef std::uint32_t serial_size_t;

}
