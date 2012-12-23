#include "picotest.h"
#include "convolutional_layer.h"
#include "fully_connected_layer.h"
#include "network.h"

using namespace nn;

TEST(convolutional, fprop) {
    convolutional_layer<sigmoid_activation> l(5, 5, 3, 1, 1);

    vec_t in(25);

    std::fill(l.bias().begin(), l.bias().end(), 0.0);
    uniform_rand(in.begin(), in.end(), -1.0, 1.0);

    const vec_t& out = l.forward_propagation(in);

    for (auto o: out)
        EXPECT_DOUBLE_EQ(o, 0.5);
}

TEST(convolutional, bprop) {
    network<cross_entropy, gradient_descent> nn;
    convolutional_layer<sigmoid_activation> layer(5, 5, 3, 1, 1);

    nn.add(&layer);

    vec_t a(25, 0.0), t(9, 0.0);

    for (int y = 0; y < 5; y++) {
        a[5*y+3] = 1.0;
    }

    for (int y = 0; y < 3; y++) {
        t[3*y+0] = 0.0;
        t[3*y+1] = 0.5;
        t[3*y+2] = 1.0;
    }

    for (int i = 0; i < 100; i++)
        nn.train(a, t);

    vec_t predicted;
    nn.predict(a, &predicted);
}

TEST(fully_connected, bprop) {
    network<cross_entropy, gradient_descent> nn;
    fully_connected_layer<sigmoid_activation> layer(3, 2);
    nn.add(&layer);

    vec_t a(3, 0.0), t(2, 0.0);

    a[0] = 3.0; a[2] = -1.0;
    t[0] = 0.3; t[1] = 0.7;

    for (int i = 0; i < 100; i++)
        nn.train(a, t);

    vec_t predicted;
    nn.predict(a, &predicted);

    EXPECT_DOUBLE_EQ(predicted[0], t[0]);
    EXPECT_DOUBLE_EQ(predicted[1], t[1]);
}

int main(void) {
    RUN_ALL_TESTS();
}