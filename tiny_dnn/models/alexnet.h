/*
    Copyright (c) 2013, Taiga Nomi
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

//#include "tiny_dnn/tiny_dnn.h"

using namespace tiny_dnn::activation;
using namespace tiny_dnn::layers;

namespace models {

// Based on: https://github.com/DeepMark/deepmark/blob/master/torch/image%2Bvideo/alexnet.lua
class alexnet : public network<sequential> {
 public:
    explicit alexnet(const std::string& name = "")
            : network<sequential>(name) {
        *this << conv<relu>(224, 224, 11, 11, 3, 64, padding::valid, true, 4, 4);
        *this << max_pool<identity>(54, 54, 64, 2);
        *this << conv<relu>(27, 27, 5, 5, 64, 192, padding::valid, true, 1, 1);
        *this << max_pool<identity>(23, 23, 192, 1);
        *this << conv<relu>(23, 23, 3, 3, 192, 384, padding::valid, true, 1, 1);
        *this << conv<relu>(21, 21, 3, 3, 384, 256, padding::valid, true, 1, 1);
        *this << conv<relu>(19, 19, 3, 3, 256, 256, padding::valid, true, 1, 1);
        *this << max_pool<identity>(17, 17, 256, 1);
    }
};

}  // namespace models
