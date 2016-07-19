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

//#include "tiny_cnn/core/device.h"
#include "tiny_cnn/core/framework/device_base.h"

namespace tiny_cnn {
namespace core {

class session {
 public:
    session(const std::string name) : name_(name) {}

    // will call construct graph
    // should we here specify the devices to use?
    void schedule_session(/* network<sequential>& net */);

    // will call forward or backward methods
    void run_session(/* data */);
    
    // Returns the session name
    std::string name() const { return name_; }

    // Returns the number of available devices
    cnn_size_t num_devices() const { return devices_.size(); }

    // Return the number of available platforms
    cnn_size_t num_platforms() const { return platforms_.size(); }

    void init_session() {
#ifdef CNN_HAVE_OPENCL
        // create platform
        cl::Platform::get(&platforms_);
        
        // no found devices
        if (platforms_.size() == 0) {
            nn_warn("OpenCL cannot find any available device");
            return;
        }

        // get all the available devices
        // by default we are getting GPU fromfirst device
        platforms_[0].getDevices(CL_DEVICE_TYPE_ALL, &devices_);

        std::cout << "OpenCL has found " << num_devices() << " devices" << std::endl; 
#else
        nn_warn("TinyCNN has not been compiled with OpenCL support.");
#endif
    }

#ifdef CNN_HAVE_OPENCL
    // Returns an specified device
    cl::Device get_device(const int id) {
         if (id < 0 || id > static_cast<int>(num_devices())) {
            nn_error("Cannot get device");
        }
        return devices_[id];    
    }
#endif

    // Print the specified device info
    void print_device_info(const int id) {
#ifdef CNN_HAVE_OPENCL
        cl::Device d = get_device(id);

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
#else
        nn_warn("TinyCNN has not been compiled with OpenCL support.");
#endif
    }

 private:
    std::string name_;
 
#ifdef CNN_HAVE_OPENCL
    std::vector<cl::Device> devices_;
    std::vector<cl::Platform> platforms_;
#else
    // TODO: Just for quick test
    std::vector<int> platforms_;
    std::vector<std::shared_ptr<tiny_cnn::device_base> > devices_;
#endif
};

}  // namespace core
}  // namespace tiny_cnn
