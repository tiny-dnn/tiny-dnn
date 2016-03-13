/*
    Copyright (c) 2016, Taiga Nomi
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
#include <iostream>
#include <cstdio>
#include "picotest/picotest.h"
#include "tiny_cnn/tiny_cnn.h"

namespace tiny_cnn {

inline bool exists(const std::string& path) {
    if (FILE *file = std::fopen(path.c_str(), "r")) {
        fclose(file);
        return true;
    } else {
        return false;
    }
}

inline std::string unique_path() {
    std::string pattern = "%%%%-%%%%-%%%%-%%%%";

    for (auto p = pattern.begin(); p != pattern.end(); ++p) {
        if (*p == '%') *p = (rand()%10)+'0';
    }
    return exists(pattern) ? unique_path() : pattern;
}

template <typename T>
void serialization_test(const T& src, T& dst)
{
    EXPECT_FALSE(src.has_same_weights(dst, 1E-5));

    std::string tmp_file_path = unique_path();

    // write
    {
        std::ofstream ofs(tmp_file_path.c_str());
        ofs << src;
    }

    // read
    {
        std::ifstream ifs(tmp_file_path.c_str());
        ifs >> dst;
    }

    std::remove(tmp_file_path.c_str());

    EXPECT_TRUE(src.has_same_weights(dst, 1E-5));
}

} // namespace tiny_cnn
