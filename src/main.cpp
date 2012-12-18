#include <iostream>
#include "cnn.h"
#include "fully_connected_layer.h"
#include "convolutional_layer.h"

using namespace nn;

int main(void) {
    cnn nn(0.3, 0.3);
    fully_connected_layer<3, 2> layer;
    nn.add(&layer);

    convolutional_prop in(1, 32, 32);
    convolutional_prop out(6, 28, 28);
    convolutional_layer<> cn(in, out, 5);

    vec_t a(3, 0.0), t(2, 0.0);

    vec_t inv(32*32, 0);
    cn.forward_propagation(inv);

    std::vector<vec_t> data, train;

    a[0] = 3.0; a[2] = -1.0;
    t[0] = 0.3; t[1] = 0.7;

    data.push_back(a);
    train.push_back(t);
    nn.check(data, train);
    for (int i = 0; i < 100; i++)
        nn.train(data, train);
    nn.check(data, train);
    vec_t predicted;
    nn.predict(a, &predicted);


}