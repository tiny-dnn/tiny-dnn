/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.
    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/

// this example shows how to use tiny-dnn library to fit data, by learning a
// sinus function.

// please also see:
// https://github.com/tiny-dnn/tiny-dnn/blob/master/docs/how_tos/How-Tos.md

#include <iostream>

#include "tiny_dnn/tiny_dnn.h"

int main() {
  // create a simple network with 2 layer of 10 neurons each
  // input is x, output is sin(x)
  tiny_dnn::network<tiny_dnn::sequential> net;
  net << tiny_dnn::fully_connected_layer(1, 10);
  net << tiny_dnn::tanh_layer();
  net << tiny_dnn::fully_connected_layer(10, 10);
  net << tiny_dnn::tanh_layer();
  net << tiny_dnn::fully_connected_layer(10, 1);

  // create input and desired output on a period
  std::vector<tiny_dnn::vec_t> X;
  std::vector<tiny_dnn::vec_t> sinusX;
  for (float x = -3.1416f; x < 3.1416f; x += 0.2f) {
    tiny_dnn::vec_t vx    = {x};
    tiny_dnn::vec_t vsinx = {sinf(x)};

    X.push_back(vx);
    sinusX.push_back(vsinx);
  }

  // set learning parameters
  size_t batch_size = 16;    // 16 samples for each network weight update
  int epochs        = 2000;  // 2000 presentation of all samples
  tiny_dnn::adamax opt;

  // this lambda function will be called after each epoch
  int iEpoch              = 0;
  auto on_enumerate_epoch = [&]() {
    // compute loss and disp 1/100 of the time
    iEpoch++;
    if (iEpoch % 100) return;

    double loss = net.get_loss<tiny_dnn::mse>(X, sinusX);
    std::cout << "epoch=" << iEpoch << "/" << epochs << " loss=" << loss
              << std::endl;
  };

  // learn
  std::cout << "learning the sinus function with 2000 epochs:" << std::endl;
  net.fit<tiny_dnn::mse>(opt, X, sinusX, batch_size, epochs, []() {},
                         on_enumerate_epoch);

  std::cout << std::endl
            << "Training finished, now computing prediction results:"
            << std::endl;

  // compare prediction and desired output
  float fMaxError = 0.f;
  for (float x = -3.1416f; x < 3.1416f; x += 0.2f) {
    tiny_dnn::vec_t xv = {x};
    float fPredicted   = net.predict(xv)[0];
    float fDesired     = sinf(x);

    std::cout << "x=" << x << " sinX=" << fDesired
              << " predicted=" << fPredicted << std::endl;

    // update max error
    float fError = fabs(fPredicted - fDesired);

    if (fMaxError < fError) fMaxError = fError;
  }

  std::cout << std::endl << "max_error=" << fMaxError << std::endl;

  return 0;
}
