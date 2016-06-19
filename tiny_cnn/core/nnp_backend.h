/*
    Copyright (c) 2016, Taiga Nomi, Edgar Riba
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

#include "tiny_cnn/core/math_backend.h"

#ifdef CNN_USE_NNPACK
#include "nnpack.h"
#endif

namespace tiny_cnn {
namespace core {

class nnp_backend : public math_backend {
 public:
    // context holds solution-dependent parameters
    // context should be able to hold any types of structures (like boost::any)
    nnp_backend(conv_params* params) : params_(params) {}

    // core math functions

    void conv2d(cnn_size_t                 index,
                const std::vector<vec_t*>& in_data,
                std::vector<vec_t*>&       out_data) {
#ifdef CNN_USE_NNPACK
      // TODO: auto configure algo
      //const auto algorithm = nnp_algorithm(algo);
      const nnp_convolution_algorithm algorithm = nnp_convolution_algorithm_auto;

      // TODO: auto configure kernel
      // const auto kernel_transform_strategy = nnp_kts(kts);
      const nnp_convolution_kernel_transform_strategy kernel_tf_strategy =
          nnp_convolution_kernel_transform_strategy_reuse;

      const cnn_size_t input_channels = params_->in.depth_;
      const cnn_size_t output_channels = params_->out.depth_;

      const nnp_size input_size = {
          static_cast<size_t>(params_->in.width_),
          static_cast<size_t>(params_->in.height_)
      };

      const nnp_size kernel_size = {
          static_cast<size_t>(params_->weight.width_),
          static_cast<size_t>(params_->weight.height_)
      };

      const float_t dx = params_->in_padded.width_  - params_->in.width_;
      const float_t dy = params_->in_padded.height_ - params_->in.height_;

      // we'll assume that padding is symmetric

      const nnp_padding padding = {
          static_cast<size_t>(dy/2),  // top
          static_cast<size_t>(dx/2),  // right
          static_cast<size_t>(dy/2),  // bottom
          static_cast<size_t>(dx/2)   // left
      };

      const float* input_pointer =
          reinterpret_cast<const float*>(&in_data[0]->at(0));

      const float* kernel_pointer =
          reinterpret_cast<const float*>(&in_data[1]->at(0));

      const float* bias =
          reinterpret_cast<const float*>(&in_data[2]->at(0));

      float* output_pointer =
          reinterpret_cast<float*>(&out_data[1]->at(0));

      // TODO: embed it into a class
      const size_t num_mkl_threads = 1;
      pthreadpool_t threadpool = pthreadpool_create(num_mkl_threads);

      nnp_profile* profile = 0;

      nnp_status status = nnp_convolution_inference(
          algorithm,
          kernel_tf_strategy,
          input_channels,
          output_channels,
          input_size,
          padding,
          kernel_size,
          input_pointer,
          kernel_pointer,
          bias,
          output_pointer,
          threadpool,
          profile);

      if (status != nnp_status_success) {
          throw nn_error("Could not succeed with nnp_convolution_inference");
      }

      // TODO: embed it into a class
      pthreadpool_destroy(threadpool);
#else
        throw nn_error("Tiny-cnn has to be compiled with NNPACK support.");
#endif
    }

    void deconv2d(cnn_size_t                 index,
                const std::vector<vec_t*>& in_data,
                std::vector<vec_t*>&       out_data) {
#ifdef CNN_USE_NNPACK
      // TODO: auto configure algo
      //const auto algorithm = nnp_algorithm(algo);
      const nnp_convolution_algorithm algorithm = nnp_convolution_algorithm_auto;

      // TODO: auto configure kernel
      // const auto kernel_transform_strategy = nnp_kts(kts);
      const nnp_convolution_kernel_transform_strategy kernel_tf_strategy =
          nnp_convolution_kernel_transform_strategy_reuse;

      const cnn_size_t input_channels = params_->in.depth_;
      const cnn_size_t output_channels = params_->out.depth_;

      const nnp_size input_size = {
          static_cast<size_t>(params_->in.width_),
          static_cast<size_t>(params_->in.height_)
      };

      const nnp_size kernel_size = {
          static_cast<size_t>(params_->weight.width_),
          static_cast<size_t>(params_->weight.height_)
      };

      const float_t dx = params_->in_padded.width_  - params_->in.width_;
      const float_t dy = params_->in_padded.height_ - params_->in.height_;

      // we'll assume that padding is symmetric

      const nnp_padding padding = {
          static_cast<size_t>(dy/2),  // top
          static_cast<size_t>(dx/2),  // right
          static_cast<size_t>(dy/2),  // bottom
          static_cast<size_t>(dx/2)   // left
      };

      const float* input_pointer =
          reinterpret_cast<const float*>(&in_data[0]->at(0));

      const float* kernel_pointer =
          reinterpret_cast<const float*>(&in_data[1]->at(0));

      const float* bias =
          reinterpret_cast<const float*>(&in_data[2]->at(0));

      float* output_pointer =
          reinterpret_cast<float*>(&out_data[1]->at(0));

      // TODO: embed it into a class
      const size_t num_mkl_threads = 1;
      pthreadpool_t threadpool = pthreadpool_create(num_mkl_threads);

      nnp_profile* profile = 0;

      nnp_status status = nnp_convolution_inference(
          algorithm,
          kernel_tf_strategy,
          input_channels,
          output_channels,
          input_size,
          padding,
          kernel_size,
          input_pointer,
          kernel_pointer,
          bias,
          output_pointer,
          threadpool,
          profile);

      if (status != nnp_status_success) {
          throw nn_error("Could not succeed with nnp_convolution_inference");
      }

      // TODO: embed it into a class
      pthreadpool_destroy(threadpool);
#else
        throw nn_error("Tiny-cnn has to be compiled with NNPACK support.");
#endif
    }

    void conv2d_back(cnn_size_t                 index,
                     const std::vector<vec_t*>& in_data,
                     const std::vector<vec_t*>& out_data,
                     std::vector<vec_t*>&       out_grad,
                     std::vector<vec_t*>&       in_grad) {
        throw nn_error("NNPACK does not support back propagation.");
    }

    void deconv2d_back(cnn_size_t                 index,
                     const std::vector<vec_t*>& in_data,
                     const std::vector<vec_t*>& out_data,
                     std::vector<vec_t*>&       out_grad,
                     std::vector<vec_t*>&       in_grad) {
        throw nn_error("NNPACK does not support back propagation.");
    }

    void matmul() {
        throw nn_error("not implemented yet.");
    }

    void maxpool() {
        throw nn_error("not implemented yet.");
    }

 private:
    /* Pointer to the convolution parameters */
    conv_params* params_;
};

}  // namespace core
}  // namespace tiny_cnn
