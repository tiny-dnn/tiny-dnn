/*
    Copyright (c) 2013, Taiga Nomi
    Copyright (c) 2016, Taiga Nomi, Edgar Riba
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/

#include <iostream>

#include "tiny_dnn/tiny_dnn.h"

using namespace tiny_dnn;
using namespace std;

int main(int argc, char **argv) {
  CNN_UNREFERENCED_PARAMETER(argc);
  CNN_UNREFERENCED_PARAMETER(argv);
  models::alexnet nn;

  // change all layers at once
  nn.weight_init(weight_init::constant(2.0));
  nn.bias_init(weight_init::constant(2.0));
  nn.init_weight();

  vec_t in(224 * 224 * 3);

  // generate random variables
  uniform_rand(in.begin(), in.end(), 0, 1);

  timer t;  // start the timer

  // predict
  auto res = nn.predict(in);

  double elapsed_s = t.elapsed();
  t.stop();

  cout << "Elapsed time(s): " << elapsed_s << endl;
}
