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

  vec_t weights   = {0.20f,  0.40f,  0.006f, -0.77f, -0.010f};
  vec_t gradients = {1.00f, -3.24f, -0.600f,  2.79f, 1.820f};

  // Defining the expected updates

  vec_t first_update  = {0.1900f, 0.4100f, 0.0160f, -0.7800f, -0.0200f};
  vec_t second_update = {0.1829f, 0.4170f, 0.0230f, -0.7870f, -0.0270f};

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

  vec_t weights   = {-0.021f,  1.03f, -0.05f, -0.749f, 0.009f};
  vec_t gradients = { 1.000f, -3.24f, -0.60f,  2.79f, 1.820f};

  // Defining the expected updates

  vec_t first_update  = {-0.0220f, 1.0310f, -0.0490f, -0.7500f, 0.0080f};
  vec_t second_update = {-0.0227f, 1.0317f, -0.0482f, -0.7507f, 0.0072f};

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

  vec_t weights   = {1.00f,  0.081f, -0.6201f, 0.96f, -0.007f};
  vec_t gradients = {6.45f, -3.240f, -0.6000f, 2.79f, 1.820f};

  // Defining the expected updates

  vec_t first_update  = {0.9992f, 0.0817f, -0.6193f, 0.9592f, -0.0077f};
  vec_t second_update = {0.9983f, 0.0826f, -0.6184f, 0.9583f, -0.0086f};

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

  vec_t weights   = {1.00f,  0.081f, -0.6201f, 0.96f, -0.007f};
  vec_t gradients = {6.45f, -3.240f, -0.6000f, 2.79f,  1.820f};

  // Defining the expected updates

  vec_t first_update  = {0.9980f, 0.0830f, -0.6181f, 0.9580f, -0.0090f};
  vec_t second_update = {0.9960f, 0.0850f, -0.6161f, 0.9560f, -0.0109f};

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

  vec_t weights   = {-0.001f, -0.90f, 0.005f, -0.74f, 0.003f};
  vec_t gradients = {-2.240f, -3.24f, 0.600f, 0.39f, 0.820f};

  // Defining the expected updates

  vec_t first_update  = {0.0214f, -0.8676f, -0.0010f, -0.7439f, -0.0052f};
  vec_t second_update = {0.0438f, -0.8352f, -0.0070f, -0.7478f, -0.0134f};

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

  vec_t weights   = {-0.001f, -0.90f, 0.005f, -0.74f, 0.003f};
  vec_t gradients = {-2.240f, -3.24f, 0.600f,  0.39f, 0.820f};

  // Defining the expected updates

  vec_t first_update  = {0.0214f, -0.8676f, -0.0010f, -0.7439f, -0.0052f};
  vec_t second_update = {0.0639f, -0.8060f, -0.0124f, -0.7513f, -0.0207f};

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

  vec_t weights   = {0.1f, 0.30f, 0.005f, -0.74f, -0.008f};
  vec_t gradients = {1.0f, -3.24f, -0.600f, 2.79f, 1.820f};

  // Defining the expected updates

  vec_t first_update  = {0.0810f, 0.3615f, 0.0164f, -0.7930f, -0.0425f};
  vec_t second_update = {0.0539f, 0.4493f, 0.0326f, -0.8686f, -0.0919f};

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
