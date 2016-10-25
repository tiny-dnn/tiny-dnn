// Copyright (c) 2013-2016, Taiga Nomi. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#pragma once

#define CNN_UNREFERENCED_PARAMETER(x) (void)(x)

#if defined(_MSC_VER) && (_MSC_VER <= 1800)
// msvc2013 doesn't have move constructor
#define CNN_DEFAULT_MOVE_CONSTRUCTOR_UNAVAILABLE
#define CNN_DEFAULT_ASSIGNMENT_OPERATOR_UNAVAILABLE
#endif

#if defined(_MSC_VER) && (_MSC_VER <= 1800)
// msvc2013 doesn't have alignof operator
#define CNN_ALIGNOF(x) __alignof(x)
#else
#define CNN_ALIGNOF(x) alignof(x)
#endif

#if !defined(_MSC_VER) || (_MSC_VER >= 1900) // default generation of move constructor is unsupported in VS2013
#define CNN_USE_DEFAULT_MOVE_CONSTRUCTORS
#endif

#if defined _WIN32
#define CNN_WINDOWS
#endif

