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

#if defined(USE_OPENCL) || defined(USE_CUDA)
#ifdef USE_OPENCL
#include "third_party/CLCudaAPI/clpp11.h"
#else
#include "third_party/CLCudaAPI/cupp11.h"
#endif
#endif

namespace tiny_dnn {

enum class device_t { NONE, CPU, GPU /*, FPGA */ };

inline std::ostream& operator << (std::ostream& os, device_t type) {
    switch (type) {
        case device_t::NONE: os << "NONE"; break;
        case device_t::CPU:  os << "CPU"; break;
        case device_t::GPU:  os << "GPU"; break;
        default:
            throw nn_error("Not supported ostream enum: " +
                    to_string(static_cast<int>(type)));
            break;
    }
    return os;
}

/* The class models a physical device */
class Device {
 public:
    /* Custom CPU constructor
     *
     * @param type The device type. Can be only CPU.
     */
    inline explicit Device(device_t type);

    /* CPU/GPU OpenCL constructor.
     * Device context is initialized in constructor.
     *
     * @param type The device type. Can be both CPU and GPU.
     * @param platform_id The platform identification number.
     * @param device_id The device identification number.
     */
    inline explicit Device(device_t type,
                           const int platform_id,
                           const int device_id);

    // Returns the device type
    device_t type() const { return type_; }

    // Returns true if CLCudaAPI is enabled to this device 
    bool hasCLCudaAPI() const { return has_clcuda_api_; }

    // Returns the platform id
    int platformId() const { return platform_id_; }
    
    // Returns the device id
    int deviceId() const { return device_id_; }

#if defined(USE_OPENCL) || defined(USE_CUDA)
    // Returns the CLCudaAPI Device object
    CLCudaAPI::Device device() const { return *device_;  }

    // Returns the CLCudaAPI Context object
    CLCudaAPI::Context context() const { return *context_; }

    // Returns the CLCudaAPI Queue object
    CLCudaAPI::Queue queue() const { return *queue_; }
#endif

    bool operator==(const Device& d) const {
        if (d.type() == this->type() &&
            d.hasCLCudaAPI() == this->hasCLCudaAPI() &&
            d.platformId() == this->platformId() &&
            d.deviceId() == this->deviceId()) {
            return true;
        }
        return false;
    }

    /* Registers and create an OpenCL program per Operation type.
     *
     * @param l The layer to be registered
     */
    inline void registerOp(layer& l);

 private:
    /* The device type */
    device_t type_;
    /* Boolean to check if device has OpenCL */
    bool has_clcuda_api_;
    /* The platform identification number */
    int platform_id_;
    /* The device identification number */
    int device_id_;
#if defined(USE_OPENCL) || defined(USE_CUDA)
    /* The CLCudaAPI device */
    std::shared_ptr<CLCudaAPI::Device> device_;
    /* The CLCudaAPI device context */
    std::shared_ptr<CLCudaAPI::Context> context_;
    /* The CLCudaAPI device queue */
    std::shared_ptr<CLCudaAPI::Queue> queue_;
#endif
};

}  // namespace tiny_dnn
