// this example show how to use tiny-dnn library to fit data, by learning a
// sinus function

// please see also at:
// https://github.com/tiny-dnn/tiny-dnn/blob/master/docs/how_tos/How-Tos.md

#include <iostream>
using namespace std;

#include "tiny_dnn/tiny_dnn.h"
using namespace tiny_dnn;

int main() {
  // create a simple network with 2 layer of 10 neurons each
  // input is x, output is sin(x)
  network<sequential> net;
  net << fully_connected_layer(1, 10);
  net << tanh_layer();
  net << fully_connected_layer(10, 10);
  net << tanh_layer();
  net << fully_connected_layer(10, 1);

  // create input and desired output on a period
  vector<vec_t> X;
  vector<vec_t> sinusX;
  for (float x = -3.1416f; x < 3.1416f; x += 0.2f) {
    vec_t vx    = {x};
    vec_t vsinx = {sinf(x)};

    X.push_back(vx);
    sinusX.push_back(vsinx);
  }

  // set learning parameters
  size_t batch_size = 16;    // 16 samples for each network weight update
  int epochs        = 2000;  // 2000 presentation of all samples
  adamax opt;

  // this lambda function will be called after each epoch
  int iEpoch              = 0;
  auto on_enumerate_epoch = [&]() {
    // compute loss and disp 1/100 of the time
    iEpoch++;
    if (iEpoch % 100) return;

    double loss = net.get_loss<mse>(X, sinusX);
    cout << "epoch=" << iEpoch << "/" << epochs << " loss=" << loss << endl;
  };

  // learn
  cout << "learning the sinus function with 2000 epochs:" << endl;
  net.fit<mse>(opt, X, sinusX, batch_size, epochs, []() {}, on_enumerate_epoch);

  // compare prediction and desired output
  float dMaxError = 0.f;
  for (float x = -3.1416f; x < 3.1416f; x += 0.2f) {
    vec_t xv         = {x};
    float fPredicted = net.predict(xv)[0];
    float fDesired   = sinf(x);

    cout << "x=" << x << " sinX=" << fDesired << " predicted=" << fPredicted
         << endl;

    // update max error
    float dError = abs(fPredicted - fDesired);

    if (dMaxError < dError) dMaxError = dError;
  }

  cout << endl << "max_error=" << dMaxError << endl;

  return 0;
}
