#pragma once
#include "gtest/gtest.h"
#include "testhelper.h"
#include "tiny_dnn/tiny_dnn.h"

using namespace tiny_dnn::activation;

namespace tiny_dnn {

TEST(test_large_thread_count, test_large_thread_count) {
  network<sequential> net;
  net << fully_connected_layer(1, 2) << tanh();
  adagrad optimizer;

  std::vector<vec_t> data;
  std::vector<label_t> labels;

  const size_t tnum = 300;

  for (size_t i = 0; i < tnum; i++) {
    bool in    = bernoulli(0.5);
    bool label = bernoulli(0.5);

    data.push_back({static_cast<float_t>(in)});
    labels.push_back(label ? 1 : 0);
  }

  const int n_threads = 200;

  // test different batch sizes
  net.train<mse>(optimizer, data, labels, 1, 1, nop, nop, true, n_threads);
  net.train<mse>(optimizer, data, labels, 100, 1, nop, nop, true, n_threads);
  net.train<mse>(optimizer, data, labels, 200, 1, nop, nop, true, n_threads);
  net.train<mse>(optimizer, data, labels, 300, 1, nop, nop, true, n_threads);
}

}  // namespace tiny-dnn
