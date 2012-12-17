#include <iostream>
#include "cnn.h"
#include "fully_connected_layer.h"

using namespace nn;

int main(void) {
    cnn nn(0.3, 0.3);
    neural_net<3, 2> layer;
    nn.add(&layer);


    vec_t a(3, 0.0), t(2, 0.0);

    std::vector<vec_t> data, train;

    a[0] = 3.0; a[2] = -1.0;
    t[0] = 0.3; t[1] = 0.7;

    data.push_back(a);
    train.push_back(t);
    nn.check(data, train);
    for (int i = 0; i < 100; i++)
        nn.train(data, train);
    nn.check(data, train);
    vec_t out;
    nn.predict(a, &out);


}