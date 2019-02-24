/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include <vector>

namespace tiny_dnn {

TEST(l2norm, forward) {
  l2_normalization_layer l2_norm(4, 3, float_t(1e-10), 20);
  
  // clang-format off
  tensor_t in = {
    {
       0.0,  0.0,  0.0,  0.0,  // ch-0 of data#0
      -4.0,  0.0, -1.0,  2.0,  // ch-1 of data#0
       1.0,  0.0,  1.0,  3.0,  // ch-2 of data#0
    }, {
       0.0,  0.0,  0.0,  0.0,  // ch-0 of data#1
       2.0,  0.0, -4.0, -3.0,  // ch-1 of data#1
       2.0,  5.0,  1.0, 10.0   // ch-2 of data#1
    }
  };

  tensor_t expect = {
      {
          0.0000f,   0.0000f,    0.0000f,    0.0000f,  // ch-0 of data#0
        -19.4029f,   0.0000f,  -14.1421f,   11.0940f,  // ch-1 of data#0
          4.8507f,   0.0000f,   14.1421f,   16.6410f,  // ch-2 of data#0
      }, {                            
          0.0000f,   0.0000f,    0.0000f,    0.0000f,   // ch-0 of data#1
         14.1421f,   0.0000f,  -19.4029f,   -5.7470f,   // ch-1 of data#1
         14.1421f,  20.0000f,    4.8507f,   19.1565f    // ch-2 of data#1
      }
  };
  // clang-format on

  std::vector<const tensor_t *> result;
  l2_norm.forward({in}, result);

  for (size_t i = 0; i < 2; i++) {
    for (size_t j = 0; j < 3 * 4; j++) {
      EXPECT_NEAR(expect[i][j], (*result[0])[i][j], 1e-3);
    }
  }
}

}  // namespace tiny_dnn
