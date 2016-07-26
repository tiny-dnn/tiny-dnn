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

#include "tiny_cnn/core/backend.h"
#include "tiny_cnn/core/kernels/libdnn_conv2d_kernel.h"

#ifdef USE_OPENCL
#include "third_party/CLCudaAPI/clpp11.h"
#endif

namespace tiny_cnn {
namespace core {

class dnn_backend : public backend {
 public:
    // context holds solution-dependent parameters
    // context should be able to hold any types of structures (like boost::any)
    dnn_backend() {}

    // convolution
    dnn_backend(conv_params* params,
                std::function<void(const tensor_t&)> f,
                conv_layer_worker_specific_storage* ptr)
        : params_c_(params),
          conv_layer_worker_storage_(ptr),
          copy_and_pad_input(f) {}

    // core math functions

    void conv2d(const std::vector<tensor_t*>& in_data,
                std::vector<tensor_t*>&       out_data) override {
/*
#ifdef CNN_USE_LIBDNN
        copy_and_pad_input(*in_data[0]);
        const vec_t& W    = (*in_data[1])[0];
        const vec_t& bias = (*in_data[2])[0];
        tensor_t&     a    = *out_data[1];
        const std::vector<const vec_t*> &in = (*conv_layer_worker_storage_).prev_out_padded_; // input // NOLINT
 
        fill_tensor(a, float_t(0));

        for (cnn_size_t i = 0; i < in.size(); i++) {
            kernels::libdnn_conv2d_kernel(*params_c_,
                *in[i], W, bias, a[i]);
        }
        // kernels::libdnn_conv2d_kernel(*params_c_, in, W, bias, a);
#else
        throw nn_error("Tiny-cnn has not been compiled with LibDNN support.");
#endif
*/
#ifdef USE_OPENCL

        // Platform/device settings
        constexpr auto platform_id = size_t{0};
        constexpr auto device_id = size_t{0};

        // Initializes the CLCudaAPI platform and device. This initializes the OpenCL/CUDA back-end and
        // selects a specific device on the platform. The device class has methods to retrieve properties
        // such as the device name and vendor. More examples of device properties are given in the
        // `device_info.cc` sample program. 
        printf("\n## Initializing...\n");
        auto platform = CLCudaAPI::Platform(platform_id);
        auto device = CLCudaAPI::Device(platform, device_id);
        printf(" > Running on device '%s' of '%s'\n", device.Name().c_str(), device.Vendor().c_str());

        // Creates a new CLCudaAPI context and queue for this device. The queue can be used to schedule
        // commands such as launching a kernel or performing a device-host memory copy.
        auto context = CLCudaAPI::Context(device);
        auto queue = CLCudaAPI::Queue(context, device);

        // Creates a new CLCudaAPI event to be able to time kernels
        auto event = CLCudaAPI::Event();

		// Creates a new program based on the kernel string. Note that the kernel string is moved-out when
		// constructing the program to save copying: it should no longer be used in the remainder of this
		// function.
		auto program_string = std::string{}; //TODO(edgar): load from Caffe
		auto compiler_options = std::vector<std::string>{};
		auto program = CLCudaAPI::Program(context, std::move(program_string));

		// Builds this program and checks for any compilation errors. If there are any, they are printed
  		// and execution is halted.
  		printf("## Compiling the kernel...\n");
  		auto build_status = program.Build(device, compiler_options);
  		if (build_status != CLCudaAPI::BuildStatus::kSuccess) {
    		auto message = program.GetBuildInfo(device);
    		printf(" > Compiler error(s)/warning(s) found:\n%s\n", message.c_str());
    		return;
		}
    
        // get convoltion data
        copy_and_pad_input(*in_data[0]);
        const vec_t& W    = (*in_data[1])[0];
        const vec_t& bias = (*in_data[2])[0];
        tensor_t&     a   = *out_data[1];
        const std::vector<const vec_t*> &in = (*conv_layer_worker_storage_).prev_out_padded_; // input // NOLINT
 
        fill_tensor(a, float_t(0));

        // Creates two new device buffers and copies the host data to these device buffers.
        printf("## Allocating device memory...\n");
        auto dev_in   = CLCudaAPI::Buffer<float>(context, queue, in.begin(), in.end());
        auto dev_W    = CLCudaAPI::Buffer<float>(context, queue, W.begin(), W.end());
        auto dev_bias = CLCudaAPI::Buffer<float>(context, queue, bias.begin(), bias.end());
        auto dev_a    = CLCudaAPI::Buffer<float>(context, queue, a.begin(), a.end());

        printf(" > Size of buffer in is %zu bytes\n",   dev_in.GetSize());
        printf(" > Size of buffer W is %zu bytes\n",    dev_W.GetSize());
        printf(" > Size of buffer bias is %zu bytes\n", dev_bias.GetSize());
        printf(" > Size of buffer a is %zu bytes\n",    dev_a.GetSize());

        // Creates the 'convolution' kernel from the compiled program and sets the four arguments. Note
        // that this uses the direct form instead of setting each argument separately.
        auto kernel = CLCudaAPI::Kernel(program, "convolution");
        // auto size_x_int = static_cast<int>(size_x);
        // auto size_y_int = static_cast<int>(size_y);
        // kernel.SetArguments(dev_a, dev_b, size_x_int, size_y_int);
 
        // Creates a 1-dimensional thread configuration with thread-blocks/work-groups of 256 threads
        // and a total number of threads equal to the number of elements in the input/output vectors.
        constexpr auto size = static_cast<size_t>(2048 * 2048); // TODO: revise this parameter
        constexpr auto kWorkGroupSize = size_t{256};
        auto global = std::vector<size_t>{size};
        auto local = std::vector<size_t>{kWorkGroupSize};

        // Enqueues the kernel and waits for the result. Note that launching the kernel is always
        // a-synchronous and thus requires finishing the queue in order to complete the operation.
        printf("## Running the kernel...\n");
        kernel.Launch(queue, global, local, event.pointer());
        queue.Finish(event);
        printf(" > Took %.3lf ms\n", event.GetElapsedTime());

        // Reads the results back to the host memory
        //dev_a.Read(queue, size, a[0]);

#else
        throw nn_error("Tiny-DNN has not been compiled with OpenCL support");
#endif
    }

    void conv2d_q(const std::vector<tensor_t*>& in_data,
                  std::vector<tensor_t*>&       out_data) override {
        throw nn_error("not implemented yet.");
    }

    void conv2d_eq(const std::vector<tensor_t*>& in_data,
                   std::vector<tensor_t*>&       out_data) override {
        throw nn_error("not implemented yet.");
    }

    void conv2d(const std::vector<tensor_t*>& in_data,
                const std::vector<tensor_t*>& out_data,
                std::vector<tensor_t*>&       out_grad,
                std::vector<tensor_t*>&       in_grad) override {
        throw nn_error("not implemented yet.");
    }
 
    void conv2d_q(const std::vector<tensor_t*>& in_data,
                  const std::vector<tensor_t*>& out_data,
                  std::vector<tensor_t*>&       out_grad,
                  std::vector<tensor_t*>&       in_grad) override {
        throw nn_error("not implemented yet.");
    }

    void deconv2d(const std::vector<tensor_t*>& in_data,
                  std::vector<tensor_t*>&       out_data) override {
        throw nn_error("not implemented yet.");
    }

    void deconv2d_q(const std::vector<tensor_t*>& in_data,
                    std::vector<tensor_t*>&       out_data) override {
        throw nn_error("not implemented yet.");
    }

    void deconv2d_eq(const std::vector<tensor_t*>& in_data,
                     std::vector<tensor_t*>&       out_data) override {
        throw nn_error("not implemented yet.");
    }

    void deconv2d(const std::vector<tensor_t*>& in_data,
                  const std::vector<tensor_t*>& out_data,
                  std::vector<tensor_t*>&       out_grad,
                  std::vector<tensor_t*>&       in_grad) override {
        throw nn_error("not implemented yet.");
    }

    void deconv2d_q(const std::vector<tensor_t*>& in_data,
                    const std::vector<tensor_t*>& out_data,
                    std::vector<tensor_t*>&       out_grad,
                    std::vector<tensor_t*>&       in_grad) override {
        throw nn_error("not implemented yet.");
    }

    void maxpool(const std::vector<tensor_t*>& in_data,
                 std::vector<tensor_t*>&       out_data) override {
        throw nn_error("not implemented yet.");
    }

    void maxpool(const std::vector<tensor_t*>& in_data,
                 const std::vector<tensor_t*>& out_data,
                 std::vector<tensor_t*>&       out_grad,
                 std::vector<tensor_t*>&       in_grad) override {
        throw nn_error("not implemented yet.");
    }

    void fully(const std::vector<tensor_t*>& in_data,
               std::vector<tensor_t*>&       out_data) override {
        throw nn_error("not implemented yet.");
    }

    void fully_q(const std::vector<tensor_t*>& in_data,
                 std::vector<tensor_t*>&       out_data) override {
        throw nn_error("not implemented yet.");
    }

    void fully_eq(const std::vector<tensor_t*>& in_data,
                  std::vector<tensor_t*>&       out_data) override {
        throw nn_error("not implemented yet.");
    }

    void fully(const std::vector<tensor_t*>& in_data,
               const std::vector<tensor_t*>& out_data,
               std::vector<tensor_t*>&       out_grad,
               std::vector<tensor_t*>&       in_grad) override {
        throw nn_error("not implemented yet.");

    }

    void fully_q(const std::vector<tensor_t*>& in_data,
                 const std::vector<tensor_t*>& out_data,
                 std::vector<tensor_t*>&       out_grad,
                 std::vector<tensor_t*>&       in_grad) override {
        throw nn_error("not implemented yet.");
    }
 
    backend_t get_type() const { return backend_t::libdnn; }

 private:

    /* Pointer to the convolution parameters */
    conv_params* params_c_;

    /* Pointer to the convolution workers */
    conv_layer_worker_specific_storage* conv_layer_worker_storage_;

    /* Pointers to parent class functions */
    std::function<void(const tensor_t&)> copy_and_pad_input;
};

}  // namespace core
}  // namespace tiny_cnn
