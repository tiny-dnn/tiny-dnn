/*
    Copyright (c) 2013, Taiga Nomi
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
#include "util.h"
#include <fstream>
#include <cstdint>
#include <algorithm>

#define CIFAR10_IMAGE_SIZE (3072)


namespace tiny_cnn {

inline void parse_cifar10(const std::string& filename, std::vector<vec_t>& train_images, std::vector<label_t>& train_labels)
{
    tiny_cnn::float_t scale_min = -1.0;
    tiny_cnn::float_t scale_max = 1.0;
    std::ifstream ifs(filename.c_str(), std::ios::in | std::ios::binary);
    if (ifs.fail() || ifs.bad())
        throw nn_error("failed to open file");

    uint8_t label;
    std::vector<unsigned char> buf(CIFAR10_IMAGE_SIZE);

    while (ifs.read((char*) &label, 1)) {
        vec_t img;

        if (!ifs.read((char*) &buf[0], CIFAR10_IMAGE_SIZE)) break;
        std::transform(buf.begin(), buf.end(), std::back_inserter(img),
            [=](unsigned char c) { return scale_min + (scale_max - scale_min) * c / 255; });

        train_images.push_back(img);
        train_labels.push_back(label);
    }
}

} // namespace tiny_cnn
