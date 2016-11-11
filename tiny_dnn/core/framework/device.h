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

#include "tiny_dnn/core/framework/device.fwd.h"
#include "tiny_dnn/core/framework/program_manager.h"

namespace tiny_dnn {
 
inline Device::Device(device_t type)
        : type_(type), has_clcuda_api_(false) {
    nn_info("Initializing Non-OpenCL device ...");
    if (type == device_t::GPU) {
        throw nn_error("Bad GPU device initialization. "
                       "Please provide platform_id and device_id");
    }
    nn_info("Initializing Non-OpenCL device ... OK");
}

inline Device::Device(device_t type,
                      const int platform_id,
                      const int device_id)
        : type_(type)
        , has_clcuda_api_(true)
        , platform_id_(platform_id)
        , device_id_(device_id) {
#if defined(USE_OPENCL) || defined(USE_CUDA)
    // Instantiate Platform and Device
    nn_info("Initializing OpenCL platform ...");
    auto platform = CLCudaAPI::Platform(platform_id);

    // Print short pltform info
    nn_info("Initializing OpenCL platform ... OK");
    nn_info("-- Running on platform " 
                       + to_string(platform_id) +
            ". Found " + to_string(platform.NumDevices()) + " devices.");

    // Create and retain device object
    nn_info("Initializing OpenCL device ...");
    device_.reset(new CLCudaAPI::Device(platform, device_id));

    // Print short device info
    nn_info("Initializing OpenCL device ... OK");
    nn_info("-- Running on device " + to_string(device_->Name()) +
                             " of " + to_string(device_->Vendor()));
    nn_info("-- Device type: "  + to_string(device_->Type()));
    nn_info("-- Capabilities: " + to_string(device_->Capabilities()));

    // check device type
    if (type == device_t::CPU && !device_->IsCPU()) {
        throw nn_error("Not found a CPU device. You are on: " +
                       to_string(device_->Type()));
    }
    else if (type == device_t::GPU && !device_->IsGPU()) {
        throw nn_error("Not found a GPU device. You are on: " +
                       to_string(device_->Type()));
    }

    // Create and retain device context
    nn_info("Initializing OpenCL device context ...");

    context_.reset(new CLCudaAPI::Context(*device_));
    queue_.reset(new CLCudaAPI::Queue(*context_, *device_));

    nn_info("Initializing OpenCL device context ... OK");
#else 
    nn_error("TinyDNN has not been compiled with OpenCL or CUDA support.");
#endif
}

inline void Device::registerOp(layer &l) {
    // TODO(egdar/nyanp): Should we raise an error here?
    if (!hasCLCudaAPI()) {
        throw nn_error("Cannot register layer: " + l.layer_type()
                           + ". Device has disabled OpenCL support. Please "
                               "specify platform and device in "
                               "Device constructor");
    }

    if (l.engine() != core::backend_t::opencl &&
        l.engine() != core::backend_t::libdnn) {
        throw nn_error("Cannot register layer: " + l.layer_type() +
            ". Enabled engine: " + to_string(l.engine()) + ". OpenCL engine "
            "(backend_t::opencl) should be used.");
    }

    // Register the op to this device
    ProgramManager::getInstance().registerOp(*this, l);
}

}  // namespace tiny_dnn
