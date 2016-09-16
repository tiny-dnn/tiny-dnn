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
#include "tiny_dnn/config.h"

#ifdef CNN_WINDOWS
#ifndef NOMINMAX
#define NOMINMAX
#endif // ifdef NOMINMAX
#include <Windows.h>
#endif

namespace tiny_dnn {

enum class Color {
    RED,
    GREEN,
    BLUE,
    YELLOW
};

#ifdef CNN_WINDOWS
inline WORD getColorAttr(Color c) {
    switch (c) {
    case Color::RED:    return FOREGROUND_RED;
    case Color::GREEN:  return FOREGROUND_GREEN;
    case Color::BLUE:   return FOREGROUND_BLUE;
    case Color::YELLOW: return FOREGROUND_GREEN|FOREGROUND_RED;
    default:            assert(0); return 0;
    }
}
#else
inline const char* getColorEscape(Color c) {
    switch (c) {
    case Color::RED:    return "\033[31m";
    case Color::GREEN:  return "\033[32m";
    case Color::BLUE:   return "\033[34m";
    case Color::YELLOW: return "\033[33m";
    default:           assert(0); return "";
    }
}
#endif

inline void coloredPrint(Color c, const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);

#ifdef CNN_WINDOWS
    const HANDLE std_handle = ::GetStdHandle(STD_OUTPUT_HANDLE);

    CONSOLE_SCREEN_BUFFER_INFO buffer_info;
    ::GetConsoleScreenBufferInfo(std_handle, &buffer_info);
    const WORD old_color = buffer_info.wAttributes;
    const WORD new_color = getColorAttr(c) | FOREGROUND_INTENSITY;

    fflush(stdout);
    ::SetConsoleTextAttribute(std_handle, new_color);

    vprintf(fmt, args);

    fflush(stdout);
    ::SetConsoleTextAttribute(std_handle, old_color);
#else
    printf("%s", getColorEscape(c));
    vprintf(fmt, args);
    printf("\033[m");
#endif
    va_end(args);
}

inline void coloredPrint(Color c, const std::string& msg) {
    coloredPrint(c, msg.c_str());
}

} // tiny_dnn
