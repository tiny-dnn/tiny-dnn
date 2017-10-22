/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.
    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include <gtest/gtest.h>

#include "tiny_dnn/tiny_dnn.h"

namespace tiny_dnn {

TEST(optimizers, adagrad_update) {
  adagrad optimizer;

  vec_t weights = {(float_t)0.20, (float_t)0.40, (float_t)0.006, (float_t)-0.77,
                   (float_t)-0.010};
  vec_t gradients = {(float_t)1.00, (float_t)-3.24, (float_t)-0.600,
                     (float_t)2.79, (float_t)1.820};

  // Defining the expected updates

  vec_t first_update = {(float_t)0.1900, (float_t)0.4100, (float_t)0.0160,
                        (float_t)-0.7800, (float_t)-0.0200};
  vec_t second_update = {(float_t)0.1829, (float_t)0.4170, (float_t)0.0230,
                         (float_t)-0.7870, (float_t)-0.0270};

  // Testing

  optimizer.update(gradients, weights, false);

  for (size_t i = 0; i < weights.size(); i++) {
    EXPECT_NEAR(first_update[i], weights[i], 1e-3);
  }

  optimizer.update(gradients, weights, false);

  for (size_t i = 0; i < weights.size(); i++) {
    EXPECT_NEAR(second_update[i], weights[i], 1e-3);
  }
}

TEST(optimizers, rmsprop_update) {
  RMSprop optimizer;

  vec_t weights = {(float_t)-0.021, (float_t)1.03, (float_t)-0.05,
                   (float_t)-.749, (float_t)0.009};
  vec_t gradients = {(float_t)1.000, (float_t)-3.24, (float_t)-0.60,
                     (float_t)2.79, (float_t)1.820};

  // Defining the expected updates

  vec_t first_update = {(float_t)-0.0220, (float_t)1.0310, (float_t)-0.0490,
                        (float_t)-0.7500, (float_t)0.0080};
  vec_t second_update = {(float_t)-0.0227, (float_t)1.0317, (float_t)-0.0482,
                         (float_t)-0.7507, (float_t)0.0072};

  // Testing

  optimizer.update(gradients, weights, false);

  for (size_t i = 0; i < weights.size(); i++) {
    EXPECT_NEAR(first_update[i], weights[i], 1e-3);
  }

  optimizer.update(gradients, weights, false);

  for (size_t i = 0; i < weights.size(); i++) {
    EXPECT_NEAR(second_update[i], weights[i], 1e-3);
  }
}

TEST(optimizers, adam_update) {
  adam optimizer;

  vec_t weights = {(float_t)1.00, (float_t)0.081, (float_t)-0.6201,
                   (float_t)0.96, (float_t)-0.007};
  vec_t gradients = {(float_t)6.45, (float_t)-3.240, (float_t)-0.6000,
                     (float_t)2.79, (float_t)1.820};

  // Defining the expected updates

  vec_t first_update = {(float_t)0.9992, (float_t)0.0817, (float_t)-0.6193,
                        (float_t)0.9592, (float_t)-0.0077};
  vec_t second_update = {(float_t)0.9983, (float_t)0.0826, (float_t)-0.6184,
                         (float_t)0.9583, (float_t)-0.0086};

  // Testing

  optimizer.update(gradients, weights, false);

  for (size_t i = 0; i < weights.size(); i++) {
    EXPECT_NEAR(first_update[i], weights[i], 1e-3);
  }

  optimizer.update(gradients, weights, false);

  for (size_t i = 0; i < weights.size(); i++) {
    EXPECT_NEAR(second_update[i], weights[i], 1e-3);
  }
}

TEST(optimizers, adamax_update) {
  adamax optimizer;

  vec_t weights = {(float_t)1.00, (float_t)0.081, (float_t)-0.6201,
                   (float_t)0.96, (float_t)-0.007};
  vec_t gradients = {(float_t)6.45, (float_t)-3.240, (float_t)-0.6000,
                     (float_t)2.79, (float_t)1.820};

  // Defining the expected updates

  vec_t first_update = {(float_t)0.9980, (float_t)0.0830, (float_t)-0.6181,
                        (float_t)0.9580, (float_t)-0.0090};
  vec_t second_update = {(float_t)0.9960, (float_t)0.0850, (float_t)-0.6161,
                         (float_t)0.9560, (float_t)-0.0109};

  // Testing

  optimizer.update(gradients, weights, false);

  for (size_t i = 0; i < weights.size(); i++) {
    EXPECT_NEAR(first_update[i], weights[i], 1e-3);
  }

  optimizer.update(gradients, weights, false);

  for (size_t i = 0; i < weights.size(); i++) {
    EXPECT_NEAR(second_update[i], weights[i], 1e-3);
  }
}

TEST(optimizers, naive_sgd_update) {
  gradient_descent optimizer;

  vec_t weights = {(float_t)-0.001, (float_t)-0.90, (float_t)0.005,
                   (float_t)-0.74, (float_t)0.003};
  vec_t gradients = {(float_t)-2.240, (float_t)-3.24, (float_t)0.600,
                     (float_t)0.39, (float_t)0.820};

  // Defining the expected updates

  vec_t first_update = {(float_t)0.0214, (float_t)-0.8676, (float_t)-0.0010,
                        (float_t)-0.7439, (float_t)-0.0052};
  vec_t second_update = {(float_t)0.0438, (float_t)-0.8352, (float_t)-0.0070,
                         (float_t)-0.7478, (float_t)-0.0134};

  // Testing

  optimizer.update(gradients, weights, false);

  for (size_t i = 0; i < weights.size(); i++) {
    EXPECT_NEAR(first_update[i], weights[i], 1e-3);
  }

  optimizer.update(gradients, weights, false);

  for (size_t i = 0; i < weights.size(); i++) {
    EXPECT_NEAR(second_update[i], weights[i], 1e-3);
  }
}

TEST(optimizers, momentum_update) {
  momentum optimizer;

  vec_t weights = {(float_t)-0.001, (float_t)-0.90, (float_t)0.005,
                   (float_t)-0.74, (float_t)0.003};
  vec_t gradients = {(float_t)-2.240, (float_t)-3.24, (float_t)0.600,
                     (float_t)0.39, (float_t)0.820};

  // Defining the expected updates

  vec_t first_update = {(float_t)0.0214, (float_t)-0.8676, (float_t)-0.0010,
                        (float_t)-0.7439, (float_t)-0.0052};
  vec_t second_update = {(float_t)0.0639, (float_t)-0.8060, (float_t)-0.0124,
                         (float_t)-0.7513, (float_t)-0.0207};

  // Testing

  optimizer.update(gradients, weights, false);

  for (size_t i = 0; i < weights.size(); i++) {
    EXPECT_NEAR(first_update[i], weights[i], 1e-3);
  }

  optimizer.update(gradients, weights, false);

  for (size_t i = 0; i < weights.size(); i++) {
    EXPECT_NEAR(second_update[i], weights[i], 1e-3);
  }
}

TEST(optimizers, nesterov_momentum_update) {
  nesterov_momentum optimizer;

  vec_t weights = {(float_t)0.1, (float_t)0.30, (float_t)0.005, (float_t)-0.74,
                   (float_t)-0.008};
  vec_t gradients = {(float_t)1.0, (float_t)-3.24, (float_t)-0.600,
                     (float_t)2.79, (float_t)1.820};

  // Defining the expected updates

  vec_t first_update = {(float_t)0.0810, (float_t)0.3615, (float_t)0.0164,
                        (float_t)-0.7930, (float_t)-0.0425};
  vec_t second_update = {(float_t)0.0539, (float_t)0.4493, (float_t)0.0326,
                         (float_t)-0.8686, (float_t)-0.0919};

  // Testing

  optimizer.update(gradients, weights, false);

  for (size_t i = 0; i < weights.size(); i++) {
    EXPECT_NEAR(first_update[i], weights[i], 1e-3);
  }

  optimizer.update(gradients, weights, false);

  for (size_t i = 0; i < weights.size(); i++) {
    EXPECT_NEAR(second_update[i], weights[i], 1e-3);
  }
}

}  // namespace tiny_dnn
