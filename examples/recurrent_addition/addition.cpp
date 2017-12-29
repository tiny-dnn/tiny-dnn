/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#include <cstdlib>
#include <iostream>
#include <vector>

#include "tiny_dnn/tiny_dnn.h"
#include "tiny_dnn/util/random.h"

/**
 * Example of using recurrent neural networks for adding two
 * numbers. The dataset consists of random pairs of numbers in
 * the range -1,1.
 *
 */
template <typename N>
void construct_net(N &nn, const std::string rnn_type) {
  using fc              = tiny_dnn::fully_connected_layer;
  using tanh            = tiny_dnn::tanh_layer;
  using recurrent       = tiny_dnn::recurrent_layer;
  const int hidden_size = 100;  // recurrent state size
  const int seq_len     = 2;    // amount of numbers to add

  if (rnn_type == "rnn") {
    nn << recurrent(tiny_dnn::rnn(1, hidden_size), seq_len);
  } else if (rnn_type == "gru") {
    nn << recurrent(tiny_dnn::gru(1, hidden_size), seq_len);
  } else if (rnn_type == "lstm") {
    nn << recurrent(tiny_dnn::lstm(1, hidden_size), seq_len);
  }
  nn << tanh() << fc(hidden_size, 1);
}

/**
 * Generates random dataset where y[t] = x[t] + x[t-1]
 * @param n_samples [in] Number of data points.
 * @param seq_len [in] Recurrent sequence length
 * @param batch_size [in] mini-batch size
 * @return n samples
 */
std::vector<tiny_dnn::tensor_t> gen_dataset(const int n_samples,
                                            const int seq_len,
                                            const int batch_size) {
  // define and initialize input and output data.
  tiny_dnn::tensor_t input;
  tiny_dnn::tensor_t output;
  // sample input from uniform distribution.
  for (int i = 0; i < n_samples; i++) {
    input.push_back({tiny_dnn::uniform_rand<float_t>(-1, 1)});
  }
  // Fill output in order n_samples * seq_len * batch-size
  output.resize(input.size());
  int n_seqs = n_samples / (seq_len * batch_size);
  for (int n = 0; n < n_seqs; n++) {
    int seq_start = n * batch_size * seq_len;
    for (int s = 0; s < seq_len; s++) {
      int item0_start = seq_start + (s - 1) * batch_size;  // x[t]
      int item1_start = seq_start + s * batch_size;        // x[t+1]
      for (int b = 0; b < batch_size; b++) {
        // y[0] = x[0]
        // y[t+1] = x[t] + x[t+1]
        output[item1_start + b].push_back(
          (s == 0 ? input[item1_start + b][0]
                  : input[item1_start + b][0] + input[item0_start + b][0]));
      }
    }
  }
  // fill return buffer
  std::vector<tiny_dnn::tensor_t> ret;
  ret.push_back(input);
  ret.push_back(output);
  return ret;
}

void train(const int n_samples,
           int seq_len,
           double learning_rate,
           const int n_train_epochs,
           int n_minibatch,
           const std::string rnn_type) {
  // to use in templates
  using recurrent_layer = tiny_dnn::recurrent_layer;
  // specify loss-function and learning strategy
  tiny_dnn::network<tiny_dnn::sequential> nn;
  tiny_dnn::adam optimizer;

  std::cout << "load models..." << std::endl;
  construct_net(nn, rnn_type);

  std::cout << "load data..." << std::endl;
  std::vector<tiny_dnn::tensor_t> dataset =
    gen_dataset(n_samples, seq_len, n_minibatch);

  int total_length = n_minibatch * seq_len;

  std::cout << "start learning" << std::endl;

  tiny_dnn::progress_display disp(n_samples);
  tiny_dnn::timer t;

  optimizer.alpha = learning_rate;

  int epoch = 1;
  // create callback
  auto on_enumerate_epoch = [&]() {
    std::cout << "Epoch " << epoch << "/" << n_train_epochs << " finished. "
              << t.elapsed() << "s elapsed." << std::endl;
    ++epoch;

    // set rnn to element-wise prediction (without forget)
    nn.at<recurrent_layer>(0).seq_len(1);
    // clear current recurrent state
    nn.at<recurrent_layer>(0).clear_state();

    nn.set_netphase(tiny_dnn::net_phase::test);
    float loss = nn.get_loss<tiny_dnn::mse>(dataset[0], dataset[1]) / n_samples;
    nn.set_netphase(tiny_dnn::net_phase::train);

    std::cout << "mse: " << loss << std::endl;

    // set rnn input sequence to two again to continue training.
    nn.at<recurrent_layer>(0).seq_len(2);

    disp.restart(n_samples);
    t.restart();
  };

  auto on_enumerate_minibatch = [&]() { disp += total_length; };

  // training
  nn.fit<tiny_dnn::mse>(optimizer, dataset[0], dataset[1], total_length,
                        n_train_epochs, on_enumerate_minibatch,
                        on_enumerate_epoch);

  std::cout << "end training." << std::endl;

  // start demo
  nn.at<recurrent_layer>(0).seq_len(1);
  nn.at<recurrent_layer>(0).bptt_max(2);
  std::cout << "Input  numbers between -1 and 1." << std::endl;
  while (true) {
    nn.at<recurrent_layer>(0).clear_state();
    std::cout << "Input number 1: ";
    float num1 = 0;
    std::cin >> num1;
    std::cout << "Input number 2: ";
    float num2 = 0;
    std::cin >> num2;
    tiny_dnn::vec_t out1 = nn.predict(tiny_dnn::vec_t({num1}));
    tiny_dnn::vec_t out2 = nn.predict(tiny_dnn::vec_t({num2}));
    std::cout << "Sum: " << out2[0] << std::endl;
  }
}

static void usage(const char *argv0) {
  std::cout << "Usage: " << argv0 << " --n_samples 1000"
            << " --learning_rate 0.01"
            << " --epochs 20"
            << " --minibatch_size 100"
            << " --rnn_type gru" << std::endl;
}

int main(int argc, char **argv) {
  int n_samples        = 1000;
  double learning_rate = 0.01;
  int epochs           = 20;
  int minibatch_size   = 100;
  std::string rnn_type = "gru";

  if (argc == 2) {
    std::string argname(argv[1]);
    if (argname == "--help" || argname == "-h") {
      usage(argv[0]);
      return 0;
    }
  }
  for (int count = 1; count + 1 < argc; count += 2) {
    std::string argname(argv[count]);
    if (argname == "--n_samples") {
      n_samples = atoi(argv[count + 1]);
    } else if (argname == "--learning_rate") {
      learning_rate = atof(argv[count + 1]);
    } else if (argname == "--epochs") {
      epochs = atoi(argv[count + 1]);
    } else if (argname == "--minibatch_size") {
      minibatch_size = atoi(argv[count + 1]);
    } else if (argname == "--rnn_type") {
      rnn_type = argv[count + 1];
    } else {
      std::cerr << "Invalid parameter specified - \"" << argname << "\""
                << std::endl;
      usage(argv[0]);
      return -1;
    }
  }
  if (learning_rate <= 0) {
    std::cerr
      << "Invalid learning rate. The learning rate must be greater than 0."
      << std::endl;
    return -1;
  }
  if (epochs <= 0) {
    std::cerr << "Invalid number of epochs. The number of epochs must be "
                 "greater than 0."
              << std::endl;
    return -1;
  }
  if (minibatch_size <= 0 || minibatch_size > 50000) {
    std::cerr
      << "Invalid minibatch size. The minibatch size must be greater than 0"
         " and less than dataset size (50000)."
      << std::endl;
    return -1;
  }
  std::cout << "Running with the following parameters:" << std::endl
            << "n_samples: " << n_samples << std::endl
            << "Learning rate: " << learning_rate << std::endl
            << "Minibatch size: " << minibatch_size << std::endl
            << "Number of epochs: " << epochs << std::endl
            << "Rnn type: " << rnn_type << std::endl
            << std::endl;
  try {
    train(n_samples, 2, learning_rate, epochs, minibatch_size, rnn_type);
  } catch (tiny_dnn::nn_error &err) {
    std::cerr << "Exception: " << err.what() << std::endl;
  }
}
