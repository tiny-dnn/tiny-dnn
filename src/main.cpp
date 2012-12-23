#include <iostream>
#include "network.h"
#include "fully_connected_layer.h"
#include "convolutional_layer.h"
#include "pooling_layer.h"

using namespace nn;

int main(void) {
    network<cross_entropy, gradient_descent> nn;
    convolutional_layer<sigmoid_activation> C1(32, 32, 5, 1, 6);
    average_pooling_layer<sigmoid_activation> S2(28, 28, 6, 2);
    convolutional_layer<sigmoid_activation> C3(14, 14, 5, 6, 16);
    average_pooling_layer<sigmoid_activation> S4(10, 10, 16, 2);
    convolutional_layer<sigmoid_activation> C5(5, 5, 5, 16, 120);
    fully_connected_layer<sigmoid_activation> F6(120, 84);
    fully_connected_layer<sigmoid_activation> F7(84, 10);

    nn.add(&C1);
    nn.add(&S2);
    nn.add(&C3);
    nn.add(&S4);
    nn.add(&C5);
    nn.add(&F6);
    nn.add(&F7);

    vec_t in(1024), train(10);
    uniform_rand(in.begin(), in.end(), -1.0, 1.0);
    uniform_rand(train.begin(), train.end(), 0.2, 0.8);

    for (int i = 0; i < 1000; i++) {
        nn.train(in, train);
    }

    vec_t predict;
    nn.predict(in, &predict);

}