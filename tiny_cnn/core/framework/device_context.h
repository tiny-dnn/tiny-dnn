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

#ifdef USE_OPENCL
#include "third_party/CLCudaAPI/clpp11.h"
#endif

namespace tiny_cnn {

class DeviceContext {
 public:
    // Initializes the CLCudaAPI platform and device.
    // This initializes the OpenCL/CUDA back-end and selects a specific device
    // on the platform. The device class has methods to retrieve properties such
    // as the device name and vendor.
    // Creates a new CLCudaAPI context and queue for this device. The queue can
    // be used to schedule commands such as launching a kernel or performing a
    // device-host memory copy.
    explicit DeviceContext(const int platform_id, const int device_id)
            : platform_(CLCudaAPI::Platform(platform_id))
            , device_(CLCudaAPI::Device(platform_, device_id))
            , context_(CLCudaAPI::Context(device_))
            , queue_(CLCudaAPI::Queue(context_, device_)) {
        printf("\n## Initializing...\n");
        printf(" > Running on device '%s' of '%s'\n",
                device_.Name().c_str(), device_.Vendor().c_str());
    }

    // Returns C++11 device platform
    CLCudaAPI::Platform platform() const { return platform_; }

    // Returns C++11 device
    CLCudaAPI::Device device_ptr() const { return device_; }

    // Returns C++11 context
    CLCudaAPI::Context context() const { return context_; }

    // Returns C++11 queue
    CLCudaAPI::Queue queue() const { return queue_; }

 private:
    CLCudaAPI::Platform platform_;
    CLCudaAPI::Device device_;
    CLCudaAPI::Context context_;
    CLCudaAPI::Queue queue_;
};

}  // namespace tiny_cnn

