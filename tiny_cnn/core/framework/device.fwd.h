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

#include "tiny_cnn/layers/layer.fwd.h"

namespace tiny_cnn {

/* Supported devices type
 *
 * */
enum device_t { CPU, GPU /*, FPGA*/ };

/* Base class modeling a device 
 *
 * @param type The type of the device
 * @param id The identification number
 *
 * */
class device {
 public:
    explicit device(const device_t type, const int id);

    // Register an ops to the current device
    void register_op(const std::vector<layer*>& ops);

    // Returns the device type
    device_t type() const { return type_; }

    // Returns the device id
    int id() const { return id_; }

    // Returns the ids list
    // TODO(edgar/naibaf7): What does it really mean
    //  this values?
    int id_list() const { return id_; }

    // Returns the device linked ops
    std::vector<layer*> ops() const { return ops_; }
 
 protected:
    /* The type of the device */
    device_t type_;
    
    /* The id of the current device */
    int id_;

    /* A vector of pointers to registered ops.
     * The data is not owned by the current class.
     * */
    std::vector<layer*> ops_;
 
 private:
    bool check_availability(layer* layer);
};

}  // namespace tiny_cnn
