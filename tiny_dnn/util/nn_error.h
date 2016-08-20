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
#include <exception>
#include <string>
#include "tiny_dnn/util/colored_print.h"

namespace tiny_dnn {

/**
 * error exception class for tiny-dnn
 **/
class nn_error : public std::exception {
public:
    explicit nn_error(const std::string& msg) : msg_(msg) {}
    const char* what() const throw() override {
        return msg_.c_str();
    }

private:
    std::string msg_;
};


/**
 * warning class for tiny-dnn (for debug)
 **/
class nn_warn {
public:
    explicit nn_warn(const std::string& msg) : msg_(msg) {
#ifdef CNN_USE_STDOUT
        coloredPrint(Color::YELLOW, msg_h_ + msg_);
#endif
    }

private:
    std::string msg_;
    std::string msg_h_ = std::string("[WARNING] ");
};


/**
 * info class for tiny-dnn (for debug)
 **/
class nn_info {
public:
    nn_info(const std::string& msg) : msg_(msg) {
#ifdef CNN_USE_STDOUT
        std::cout << msg_h + msg_ << std::endl;
#endif
    }

private:
    std::string msg_;
    std::string msg_h = std::string("[INFO] ");
};


class nn_not_implemented_error : public nn_error {
public:
    explicit nn_not_implemented_error(const std::string& msg = "not implemented") : nn_error(msg) {}
};

} // namespace tiny_dnn
