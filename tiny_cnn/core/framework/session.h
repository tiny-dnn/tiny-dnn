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

#include <string>
#include <vector>

#ifdef CNN_HAVE_OPENCL
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.hpp>
#include <CL/opencl.h>
#endif
#endif

#include "tiny_cnn/core/core.h"

namespace tiny_cnn {

class session {
 public:
    session(const std::string name) : name_(name) {}

    // will call construct graph
    // should we here specify the devices to use?
    void schedule_session(/* network<sequential>& net */);

    // will call forward or backward methods
    void run() {
        tune_kernels();
    }

    // Returns the session name
    std::string name() const { return name_; }

    // Returns the number of registered devices
    cnn_size_t num_devices() const { return devices_.size(); }

    // Registers a device and an operation to the current session
    void register_op(const device& dev, const layer& op) {
        // Given a device and an operation/layer we create
        // a device pointer in order to be registered.
        device* device_ptr = device::create(
                dev.type(), op.backend_type(), dev.id());

        if (!device_ptr) {
            throw nn_error("Could not allocate device with op");
        }

        // TODO(edgar): check if device was already registered
        // if device/op is suitable we register them
        device_ptr->register_op(op);
        devices_.push_back(device_ptr);
    }

    // Print the all available devices info
    void print_all_available_devices() {
#ifdef CNN_HAVE_OPENCL
        std::vector<cl::Device>   devices;
        std::vector<cl::Platform> platforms;

        // create platforms
        cl::Platform::get(&platforms);
        
        if (platforms.size() == 0) {
            nn_error("OpenCL could not find any available platform.");
        }

        // get all the available platforms
        for (size_t i = 0; i < platforms.size(); i++) {
            // get all the available devices
            platforms[i].getDevices(CL_DEVICE_TYPE_ALL, &devices);

            if (devices.size() == 0) {
                nn_error("OpenCL could not find any available device.");
            }

            for (size_t j = 0; j < devices.size(); j++) {
                cl::Device d = devices[j];
                
                std::cout << " ,--[ Device (" << id << ")  "          << "\n";
                std::cout << " |`-> Hardware name:         "; 
                std::cout << d.getInfo<CL_DEVICE_NAME>()              << "\n";
                std::cout << " |`-> Hardware vendor:       "; 
                std::cout << d.getInfo<CL_DEVICE_VENDOR>()            << "\n";
                std::cout << " |`-> Hardware version:      "; 
                std::cout << d.getInfo<CL_DEVICE_VERSION>()           << "\n";
                std::cout << " |`-> Software version:      ";
                std::cout << d.getInfo<CL_DRIVER_VERSION>()           << "\n";
                std::cout << " |`-> OpenCL version:        ";
                std::cout << d.getInfo<CL_DEVICE_OPENCL_C_VERSION>()  << "\n";
                std::cout << "  `-> Parallel compute unit: ";
                std::cout << d.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << "\n\n";
            }
        }
#else
        nn_error("Tiny-dnn was not build with OpenCL support.");
    }
#endif

 private:
    void tune_kernels() {
        for (auto d: devices_) {
            for (auto op: d->ops()) {
                // TODO(edgar): decide what to do here
                if (op->layer_type() == "conv" &&
                    tiny_cnn::have_libdnn()) {
                        op->tune_kernel(d->context(),
                                        d->device(),
                                        d->queue(),
                                        d->id(),
                                        d->id_list(),
                                        op->params());
                }/* else {
                    op->tune_kernel(o->program_string(),
                                    o->compiler_options(),
                                    d->context(),
                                    d->device());
                }*/
            }
        }
    }

    /* The session name */
    std::string name_;

    /* A vector of pointers to registered devices.
     * The data is not owned by the current class.
     * */
    std::vector<device*> devices_;

    // TODO(edgar): add a map to avoid multiple instances of
    // a certain type of device.
    // std::unordered_map<int, device*> devices_;
};

}  // namespace tiny_cnn
