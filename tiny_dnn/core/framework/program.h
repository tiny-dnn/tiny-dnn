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

#include "tiny_dnn/layers/layer.h"
#include "tiny_dnn/core/framework/device.fwd.h"

#if defined(USE_OPENCL) || defined(USE_CUDA)
#ifdef USE_OPENCL
#include "third_party/CLCudaAPI/clpp11.h"
#else
#include "third_party/CLCudaAPI/cupp11.h"
#endif
#endif

namespace tiny_dnn {

/* The class models a program to be stored in the register.
 * Each instance of this class will be used as key in the register.
 */
class Program {
 public:
    explicit Program(const Device* device, const layer* op)
        : device_(device), op_(op) {}

    // Returns the device associated to the program
    const Device* device() const { return device_; }

    // Return the layer pointer
    const layer* op() const { return op_; }

    bool operator==(const Program& p) const {
        if (p.device() == this->device() &&
            p.op()->layer_type() == this->op()->layer_type()) {
            return true;
        }
        return false;
    }

 private:
    const Device* device_;
    const layer* op_;
};

/* Hash function to store Programs in the register.
 */
class ProgramHash {
 public:
    size_t operator()(const Program& p) const {
        // check there is a device and an op assigned
        // to the input program.
        if (p.device() == nullptr || p.op() == nullptr) {
            throw nn_error("No Op or Device in Program.");
        }

        // Compute individual hash values for data members and combine
        // them using XOR and bit shifting.
        return (std::hash<int>()(static_cast<int>(p.device()->type())) ^
                std::hash<bool>()(p.device()->hasCLCudaAPI()) ^
                std::hash<int>()(p.device()->platformId()) ^
                std::hash<int>()(p.device()->deviceId()) ^
                std::hash<std::string>()(p.op()->layer_type()));
    }
};

}  // namespace tiny_dnn
