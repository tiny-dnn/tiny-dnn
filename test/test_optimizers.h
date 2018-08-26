/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.
    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

namespace tiny_dnn {

TEST(optimizers, adagrad_update) {
  adagrad optimizer;

  vec_t weights   = {0.20, 0.40, 0.006, -0.77, -0.010};
  vec_t gradients = {1.00, -3.24, -0.600, 2.79, 1.820};

  // Defining the expected updates

  vec_t first_update  = {0.1900, 0.4100, 0.0160, -0.7800, -0.0200};
  vec_t second_update = {0.1829, 0.4170, 0.0230, -0.7870, -0.0270};

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

  vec_t weights   = {-0.021, 1.03, -0.05, -.749, 0.009};
  vec_t gradients = {1.000, -3.24, -0.60, 2.79, 1.820};

  // Defining the expected updates

  vec_t first_update  = {-0.0220, 1.0310, -0.0490, -0.7500, 0.0080};
  vec_t second_update = {-0.0227, 1.0317, -0.0482, -0.7507, 0.0072};

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

  vec_t weights   = {1.00, 0.081, -0.6201, 0.96, -0.007};
  vec_t gradients = {6.45, -3.240, -0.6000, 2.79, 1.820};

  // Defining the expected updates

  vec_t first_update  = {0.9992, 0.0817, -0.6193, 0.9592, -0.0077};
  vec_t second_update = {0.9983, 0.0826, -0.6184, 0.9583, -0.0086};

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

  vec_t weights   = {1.00, 0.081, -0.6201, 0.96, -0.007};
  vec_t gradients = {6.45, -3.240, -0.6000, 2.79, 1.820};

  // Defining the expected updates

  vec_t first_update  = {0.9980, 0.0830, -0.6181, 0.9580, -0.0090};
  vec_t second_update = {0.9960, 0.0850, -0.6161, 0.9560, -0.0109};

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

  vec_t weights   = {-0.001, -0.90, 0.005, -0.74, 0.003};
  vec_t gradients = {-2.240, -3.24, 0.600, 0.39, 0.820};

  // Defining the expected updates

  vec_t first_update  = {0.0214, -0.8676, -0.0010, -0.7439, -0.0052};
  vec_t second_update = {0.0438, -0.8352, -0.0070, -0.7478, -0.0134};

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

  vec_t weights   = {-0.001, -0.90, 0.005, -0.74, 0.003};
  vec_t gradients = {-2.240, -3.24, 0.600, 0.39, 0.820};

  // Defining the expected updates

  vec_t first_update  = {0.0214, -0.8676, -0.0010, -0.7439, -0.0052};
  vec_t second_update = {0.0639, -0.8060, -0.0124, -0.7513, -0.0207};

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

  vec_t weights   = {0.1, 0.30, 0.005, -0.74, -0.008};
  vec_t gradients = {1.0, -3.24, -0.600, 2.79, 1.820};

  // Defining the expected updates

  vec_t first_update  = {0.0810, 0.3615, 0.0164, -0.7930, -0.0425};
  vec_t second_update = {0.0539, 0.4493, 0.0326, -0.8686, -0.0919};

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
