/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#define CNN_UNREFERENCED_PARAMETER(x) (void)(x)

#if defined _WIN32 && !defined(__MINGW32__)
#define CNN_WINDOWS
#endif
