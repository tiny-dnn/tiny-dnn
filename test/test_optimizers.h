#pragma once
#include "gtest/gtest.h"
#include "tiny_dnn/tiny_dnn.h"

namespace tiny_dnn {

TEST(optimizers, adagrad_update) {
    adagrad optimizer;

    vec_t weights = {.2, .4, .006, -.77, -.010};
    vec_t gradients = {1.0, -3.24, -.6, 2.79, 1.82};

    // Defining the expected updates
    vec_t first_update = {0.1900, 0.4100, 0.0160, -0.7800, -0.0200};
    vec_t second_update = {0.1829, 0.4170, 0.0230, -0.7870, -0.0270};

    // Testing
    optimizer.update(gradients, weights, false);
    for (size_t i = 0; i < weights.size(); i++)
        EXPECT_NEAR(first_update[i], weights[i], 1e-3);

    optimizer.update(gradients, weights, false);
    for (size_t i = 0; i < weights.size(); i++)
        EXPECT_NEAR(second_update[i], weights[i], 1e-3);
}

TEST(optimizers, rmsprop_update) {
    RMSprop optimizer;

    vec_t weights = {-.021, 1.03, -.05, -.749, .009};
    vec_t gradients = {1.0, -3.24, -.6, 2.79, 1.82};

    // Defining the expected updates
    vec_t first_update = {-0.0220, 1.0310, -0.0490, -0.7500, 0.0080};
    vec_t second_update = {-0.0227, 1.0317, -0.0482, -0.7507, 0.0072};

    // Testing
    optimizer.update(gradients, weights, false);
    for (size_t i = 0; i < weights.size(); i++)
        EXPECT_NEAR(first_update[i], weights[i], 1e-3);

    optimizer.update(gradients, weights, false);
    for (size_t i = 0; i < weights.size(); i++)
        EXPECT_NEAR(second_update[i], weights[i], 1e-3);
}

TEST(optimizers, adam_update) {
    adam optimizer;

    vec_t weights = {1.0, .081, -.6201, .96, -.007};
    vec_t gradients = {6.45, -3.24, -.6, 2.79, 1.82};

    // Defining the expected updates
    vec_t first_update = {0.9992, 0.0817, -0.6193, 0.9592, -0.0077};
    vec_t second_update = {0.9983, 0.0826, -0.6184, 0.9583, -0.0086};

    // Testing
    optimizer.update(gradients, weights, false);
    for (size_t i = 0; i < weights.size(); i++)
        EXPECT_NEAR(first_update[i], weights[i], 1e-3);

    optimizer.update(gradients, weights, false);
    for (size_t i = 0; i < weights.size(); i++)
        EXPECT_NEAR(second_update[i], weights[i], 1e-3);
}

TEST(optimizers, naive_sgd_update) {
    gradient_descent optimizer;

    vec_t weights = {-.001, -.90, .005, -.74, .003};
    vec_t gradients = {-2.24, -3.24, .6, .39, .82};

    // Defining the expected updates
    vec_t first_update = {0.0214, -0.8676, -0.0010, -0.7439, -0.0052};
    vec_t second_update = {0.0438, -0.8352, -0.0070, -0.7478, -0.0134};

    // Testing
    optimizer.update(gradients, weights, false);
    for (size_t i = 0; i < weights.size(); i++)
        EXPECT_NEAR(first_update[i], weights[i], 1e-3);

    optimizer.update(gradients, weights, false);
    for (size_t i = 0; i < weights.size(); i++)
        EXPECT_NEAR(second_update[i], weights[i], 1e-3);
}

TEST(optimizers, momentum_update) {
    momentum optimizer;

    vec_t weights = {-.001, -.90, .005, -.74, .003};
    vec_t gradients = {-2.24, -3.24, .6, .39, .82};

    // Defining the expected updates
    vec_t first_update = {.0214, -.8676, -.0010, -.7439, -.0052};
    vec_t second_update = {.0639, -.8060, -.0124, -.7513, -.0207};

    // Testing
    optimizer.update(gradients, weights, false);
    for (size_t i = 0; i < weights.size(); i++)
        EXPECT_NEAR(first_update[i], weights[i], 1e-3);
    
    optimizer.update(gradients, weights, false);
    for (size_t i = 0; i < weights.size(); i++)
        EXPECT_NEAR(second_update[i], weights[i], 1e-3);
}

TEST(optimizers, nesterov_momentum_update) {
    nesterov_momentum optimizer;

    vec_t weights = {.1, .3, .005, -.74, -.008};
    vec_t gradients = {1.0, -3.24, -.6, 2.79, 1.82};

    // Defining the expected updates
    vec_t first_update = {.0810, .3615, .0164, -.7930, -.0425};
    vec_t second_update = {.0539, .4493, .0326, -.8686, -.0919};

    // Testing
    optimizer.update(gradients, weights, false);
    for (size_t i = 0; i < weights.size(); i++)
        EXPECT_NEAR(first_update[i], weights[i], 1e-3);
    
    optimizer.update(gradients, weights, false);
    for (size_t i = 0; i < weights.size(); i++)
        EXPECT_NEAR(second_update[i], weights[i], 1e-3);
}

} // namespace tiny_dnn
