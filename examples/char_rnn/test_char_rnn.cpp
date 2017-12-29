/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "tiny_dnn/tiny_dnn.h"
#include "tiny_dnn/util/random.h"

// Data helpers
std::vector<tiny_dnn::tensor_t> encode(char &input, std::map<char, int> &dict) {
  std::vector<tiny_dnn::tensor_t> ret(1);
  ret[0].resize(1);
  ret[0][0].resize(dict.size(), 0);
  ret[0][0][dict[input]] = 1;
  return ret;
}

/**
 * Load dictionary to decode output predictions.
 * @return dictionary (array of chars)
 */
const std::pair<std::map<char, int>, std::vector<char>> get_encoding() {
  std::ifstream ifs("../examples/char_rnn/data/encoding.raw",
                    std::ifstream::in);
  std::vector<char> dec;
  std::map<char, int> enc;
  char c;
  while (ifs.read(&c, 1)) {
    enc[c] = dec.size();
    dec.push_back(c);
  }
  const std::pair<std::map<char, int>, std::vector<char>> ret(enc, dec);
  return ret;
}

/**
 * Build rnn
 * @tparam N [in] Graph or sequential
 * @param nn [in] Reference to the network to initialize.
 * @param vocab_size [in] Size of the encoding.
 * @param hidden_size [in] Hidden state size.
 * @param n_layers [in] Number of recurrent layers.
 * @param seq_len [in] Number of steps to remember.
 * @param rnn_type [in] type of rnn cell (lstm, gru, rnn).
 * @param backend_type [in] backend type.
 */
template <typename N>
void construct_net(N &nn,
                   const int vocab_size,
                   const int hidden_size,
                   const int n_layers,
                   const int seq_len,
                   const std::string rnn_type,
                   tiny_dnn::core::backend_t backend_type) {
  // define layer aliases
  using activation = tiny_dnn::selu_layer;
  using fc         = tiny_dnn::fully_connected_layer;
  using recurrent  = tiny_dnn::recurrent_layer;

  // clip gradients
  tiny_dnn::recurrent_layer_parameters params;
  params.clip = 0;

  // add recurrent stack
  int input_size = vocab_size;
  nn << fc(vocab_size, input_size, false, backend_type);
  for (int i = 0; i < n_layers; i++) {
    if (rnn_type == "rnn") {
      nn << recurrent(tiny_dnn::rnn(input_size, hidden_size), seq_len, params);
    } else if (rnn_type == "gru") {
      nn << recurrent(tiny_dnn::gru(input_size, hidden_size), seq_len, params);
    } else if (rnn_type == "lstm") {
      nn << recurrent(tiny_dnn::lstm(input_size, hidden_size), seq_len, params);
    }
    input_size = hidden_size;
    nn << activation();  // << dropout(hidden_size, 0.3);
  }
  // predict next char
  nn << fc(hidden_size, vocab_size, false, backend_type);
}

/**
 * Helper function to get the cross-entropy error.
 * @param data [in] vector of predicted probabilities.
 * @param labels [in] vector of labels in one-hot encoding.
 * @return Cross-Entropy loss.
 */
const float cross_entropy(std::vector<tiny_dnn::tensor_t> data,
                          std::vector<tiny_dnn::tensor_t> labels) {
  float loss = 0;
  for (unsigned int i = 0; i < data.size(); i++) {
    auto &data_   = data[i][0];
    auto &labels_ = labels[i][0];
    for (unsigned int j = 0; j < labels_.size(); j++) {
      loss += labels_[j] * std::log(data_[j] + 1e-6) +
              (1 - labels_[j]) * std::log(1 - data_[j] + 1e-6);
    }
  }
  return -loss / data.size();
}

/**
 * Samples next char given the output probability distribution.
 * @param probs [in] Probability distribution.
 * @return sample.
 */
unsigned int select_one(const tiny_dnn::vec_t probs) {
  std::default_random_engine generator;
  std::discrete_distribution<unsigned int> distribution(probs.begin(),
                                                        probs.end());
  return distribution(generator);
}

/**
 * Applies softmax with temperature. The higher the temperature,
 * the uncertain the output.
 * @param data [in] vector of output
 * @param temperature [in] manages output uncertainty.
 */
void softmax(tiny_dnn::vec_t &data, double temperature = 1.0) {
  tiny_dnn::softmax_layer softmax;
  for (auto &d : data) {
    d /= temperature;
  }
  softmax.forward_activation(data, data);
}

// Train/Val
/**
 * The main training loop.
 * @param hidden_size [in] Hidden state size.
 * @param n_layers [in] Number of hidden layers.
 * @param seq_len [in] Max sequence length.
 * @param temperature [in] The softmax temperature
 * @param rnn_type [in] RNN, LSTM, GRU
 * @param backend_type [in] the backend.
 */
void test(int hidden_size,
          int n_layers,
          int seq_len,
          double temperature,
          const std::string rnn_type,
          tiny_dnn::core::backend_t backend_type) {
  using recurrent_layer = tiny_dnn::recurrent_layer;
  // specify loss-function and learning strategy
  tiny_dnn::network<tiny_dnn::sequential> nn;
  nn.weight_init(tiny_dnn::weight_init::xavier());

  std::cout << "Loading data..." << std::endl;
  auto encoding = get_encoding();
  auto enc_dict = encoding.first;
  auto dec_dict = encoding.second;
  std::cout << "load models..." << std::endl;
  construct_net(nn, dec_dict.size(), hidden_size, n_layers, seq_len, rnn_type,
                backend_type);
  // load nets
  std::ifstream ifs("../examples/char_rnn/data/char_rnn_weights");
  ifs >> nn;
  for (auto n : nn) n->set_parallelize(true);

  nn.set_netphase(tiny_dnn::net_phase::test);
  for (unsigned int i = 0; i < nn.layer_size(); i++) {
    try {
      nn.at<recurrent_layer>(i).seq_len(1);
      nn.at<recurrent_layer>(i).bptt_max(1e9);
      nn.at<recurrent_layer>(i).clear_state();
    } catch (tiny_dnn::nn_error &err) {
    }
  }
  // read stdin
  while (true) {
    std::string input;
    std::getline(std::cin, input);
    unsigned int out_ch = 0;
    for (char &c : input) {
      out_ch = select_one(nn.fprop(encode(c, enc_dict))[0][0]);
    }
    int counter = 0;
    // feed rnn output to input to generate text
    while (dec_dict[out_ch] != '\n' && counter++ < 100) {
      auto output = nn.fprop(encode(dec_dict[out_ch], enc_dict))[0][0];
      softmax(output, temperature);
      out_ch = select_one(output);
      std::cout << dec_dict[out_ch];
    }
  }
}

static tiny_dnn::core::backend_t parse_backend_name(const std::string &name) {
  const std::array<const std::string, 5> names = {
    "internal", "nnpack", "libdnn", "avx", "opencl",
  };
  for (unsigned int i = 0; i < names.size(); ++i) {
    if (name.compare(names[i]) == 0) {
      return static_cast<tiny_dnn::core::backend_t>(i);
    }
  }
  return tiny_dnn::core::default_engine();
}

static void usage(const char *argv0) {
  std::cout << "Usage: " << argv0 << " --temperature 1.0"
            << " --hidden_size 128"
            << " --n_layers 1"
            << " --rnn_type gru"
            << " --backend_type internal" << std::endl;
}

int main(int argc, char **argv) {
  double temperature                     = 1.0;
  int hidden_size                        = 128;
  int n_layers                           = 1;
  int seq_len                            = 100;
  std::string rnn_type                   = "gru";
  tiny_dnn::core::backend_t backend_type = tiny_dnn::core::default_engine();

  if (argc == 2) {
    std::string argname(argv[1]);
    if (argname == "--help" || argname == "-h") {
      usage(argv[0]);
      return 0;
    }
  }
  for (int count = 1; count + 1 < argc; count += 2) {
    std::string argname(argv[count]);
    if (argname == "--temperature") {
      temperature = atof(argv[count + 1]);
    } else if (argname == "--hidden_size") {
      hidden_size = atoi(argv[count + 1]);
    } else if (argname == "--n_layers") {
      n_layers = atoi(argv[count + 1]);
    } else if (argname == "--seq_len") {
      seq_len = atoi(argv[count + 1]);
    } else if (argname == "--rnn_type") {
      rnn_type = argv[count + 1];
    } else if (argname == "--backend_type") {
      backend_type = parse_backend_name(argv[count + 1]);
    } else {
      std::cerr << "Invalid parameter specified - \"" << argname << "\""
                << std::endl;
      usage(argv[0]);
      return -1;
    }
  }
  if (temperature <= 0) {
    std::cerr
      << "Invalid learning rate. The learning rate must be greater than 0."
      << std::endl;
    return -1;
  }

  std::cout << "Running with the following parameters:" << std::endl
            << "temperature: " << temperature << std::endl
            << "Hidden size: " << hidden_size << std::endl
            << "NLayers: " << n_layers << std::endl
            << "Sequence length: " << seq_len << std::endl
            << "Rnn type: " << rnn_type << std::endl
            << "Backend type: " << backend_type << std::endl
            << std::endl;
  try {
    test(hidden_size, n_layers, seq_len, temperature, rnn_type, backend_type);
  } catch (tiny_dnn::nn_error &err) {
    std::cerr << "Exception: " << err.what() << std::endl;
  }
}
