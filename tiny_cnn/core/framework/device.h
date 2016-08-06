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

    * Neither the name of tiny-cnn nor the names of its
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

#include "tiny_cnn/core/framework/device.fwd.h"
#include "tiny_cnn/core/framework/program_manager.h"

namespace tiny_cnn {
 
Device::Device(device_t type)
        : type_(type), has_clcuda_api_(false) {
    if (type == device_t::GPU) {
        nn_error("Bad GPU device initialization. "
                     "Please provide platform_id and device_id");
    }
}

Device::Device(device_t type,
                const int platform_id,
                const int device_id)
        : type_(type)
        , has_clcuda_api_(true)
        , platform_id_(platform_id)
        , device_id_(device_id) {
#if defined(USE_OPENCL) || defined(USE_CUDA)
    // Instantiate Platform and Device
    auto platform = CLCudaAPI::Platform(platform_id);

    // Create and retain device object
    device_ = std::make_shared<CLCudaAPI::Device>(platform, device_id);

    // Create and retain device context
    context_ = std::make_shared<CLCudaAPI::Context>(*device_);

    // Print short device info
    nn_info("Initializing ...");
    nn_info("Running on device " + to_string(device_->Name().c_str()) +
            " of " + to_string(device_->Vendor().c_str()));
#else 
    nn_error("TinyDNN has not been compiled with OpenCL or CUDA support.");
#endif
}

void Device::registerOp(const layer& l) {
    // TODO(egdar/nyanp): Should we raise an error here?
    if (!hasCLCudaAPI()) {
        nn_warn("Cannot register layer: " + l.layer_type() + "."
                "Device has disabled OpenCL support.");
        return;
    }

    if (l.backend_type() != core::backend_t::OpenCL) {
        nn_warn("Cannot register layer: " + l.layer_type() +
                ". Enabled engine: " + to_string(l.backend_type()));
        return;
    }

    // Register the op to this device
    ProgramManager::getInstance().registerOp(*this, l);
}

}  // namespace tiny_cnn
