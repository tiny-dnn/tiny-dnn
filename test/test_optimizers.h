#pragma once
#include "gtest/gtest.h"
#include "tiny_dnn/tiny_dnn.h"

using namespace tiny_dnn::activation;

namespace tiny_dnn {

TEST(optimizers, nesterov_momentum_update) {
    nesterov_momentum optimizer;

    vec_t weights = {.1, .3, .005, -.74, -.008};
    vec_t gradients = {1.0, -3.24, -.6, 2.79, 1.82};

    // Defining the expected updates
    vec_t first_update = {.0810, .3615, .0164, -.7930, -.0425};
    vec_t second_update = {.0539, .4493, .0326, -.8686, -.0919};

    // Testing
    optimizer.update(gradients, weights, false);
    for (auto i = 0u; i < weights.size(); i++)
        EXPECT_NEAR(first_update[i], weights[i], 1e-3);
    
    optimizer.update(gradients, weights, false);
    for (auto i = 0u; i < weights.size(); i++)
        EXPECT_NEAR(second_update[i], weights[i], 1e-3);
}

} // namespace tiny_dnn
