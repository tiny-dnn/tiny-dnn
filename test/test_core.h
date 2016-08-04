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
#include "picotest/picotest.h"
#include "testhelper.h"
#include "tiny_cnn/tiny_cnn.h"

using namespace tiny_cnn;

namespace tiny_cnn {

TEST(core, device) {
    // CPU and GPU devices are instantiated

    Device my_cpu_device(device_t::CPU);
    Device my_gpu_device(device_t::GPU, 0, 0);
}

TEST(core, add_bad_device) {
    // A simple CPU device cannot register an op.
    // A warning is expected telling the user to use
    // more parameters when device is created.

    Device my_gpu_device(device_t::CPU);

    convolutional_layer<sigmoid> l(5, 5, 3, 1, 2,
        padding::valid, true, 1, 1, backend_t::OpenCL);

    my_gpu_device.registerOp(l);
}

TEST(core, add_bad_layer) {
    // A GPU device cannot register an op with non-OpenCL engine.
    // A warning is expected telling the user to redefine the op engine.
 
    Device my_gpu_device(device_t::GPU, 0, 0);

    convolutional_layer<sigmoid> l(5, 5, 3, 1, 2,
        padding::valid, true, 1, 1, backend_t::tiny_cnn);

    my_gpu_device.registerOp(l);
}

TEST(core, device_add_op) {
    // An Op with OpenCL engine is registeres to
    // a GPU device which will compile its program, and
    // will place it to the general register.

    Device my_gpu_device(device_t::GPU, 0, 0);

    convolutional_layer<sigmoid> l(5, 5, 3, 1, 2,
        padding::valid, true, 1, 1, backend_t::OpenCL);

    ASSERT_EQ(ProgramManager::getInstance().num_programs(), 0);

    // first time op registration: OK
    my_gpu_device.registerOp(l);

    ASSERT_EQ(ProgramManager::getInstance().num_programs(), 1);

    // second time op registraion: we expect that Op it's not
    // registrated since it's already there.
    my_gpu_device.registerOp(l);

    ASSERT_EQ(ProgramManager::getInstance().num_programs(), 1);
}

} // namespace tiny-cnn
