/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include <string>

#include "tiny_dnn/config.h"

#ifdef CNN_WINDOWS
#ifndef NOMINMAX
#define NOMINMAX
#endif  // ifdef NOMINMAX
#include <Windows.h>
#endif

namespace tiny_dnn {

enum class Color { RED, GREEN, BLUE, YELLOW };

#ifdef CNN_WINDOWS
inline WORD getColorAttr(Color c) {
  switch (c) {
    case Color::RED: return FOREGROUND_RED;
    case Color::GREEN: return FOREGROUND_GREEN;
    case Color::BLUE: return FOREGROUND_BLUE;
    case Color::YELLOW: return FOREGROUND_GREEN | FOREGROUND_RED;
    default: assert(0); return 0;
  }
}
#else
inline const char *getColorEscape(Color c) {
  switch (c) {
    case Color::RED: return "\033[31m";
    case Color::GREEN: return "\033[32m";
    case Color::BLUE: return "\033[34m";
    case Color::YELLOW: return "\033[33m";
    default: assert(0); return "";
  }
}
#endif

inline void coloredPrint(Color c, const char *fmt, ...) {
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

inline void coloredPrint(Color c, const std::string &msg) {
  coloredPrint(c, msg.c_str());
}

}  // namespace tiny_dnn
