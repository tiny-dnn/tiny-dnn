/*
    Copyright (c) 2013, Taiga Nomi
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#endif

#include "gtest/gtest.h"
#include "tiny_dnn/tiny_dnn.h"

TEST(no_duplicate_symbols, no_duplicate_symbols) {
  // The real test is that the tests link without errors due to duplicate
  // symbols
  // typically caused by missing inline keywords in headers.
  // This is why this test is placed in a separate .cpp file.
  EXPECT_TRUE(true);
}
