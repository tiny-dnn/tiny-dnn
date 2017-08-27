/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "tiny_dnn/tiny_dnn.h"
#include "tiny_dnn/util/random.h"

/**
 * class to interface with python code
 */
class Model {
 public:
  /**
   * Constructor
   * @param weights [in] path to the weights file.
   * @param encoding [in] path to the encoding file.
   * @param rnn_type [in] rnn cell type.
   * @param depth [in] number of recurrent hidden layers.
   * @param hidden_size [in] hidden state size.
   */
  Model(const char *weights,
        const char *encoding,
        const char *rnn_type,
        int depth,
        int hidden_size) {
    std::cout << "Loading network with:" << std::endl;
    std::cout << "Weights path:  " << weights << std::endl;
    std::cout << "Encoding path: " << encoding << std::endl;
    std::cout << "Rnn type:      " << rnn_type << std::endl;
    std::cout << "Depth:         " << depth << std::endl;
    std::cout << "Hidden size:   " << hidden_size << std::endl;
    std::cout << "===============" << std::endl;
    // define layer aliases
    using activation = tiny_dnn::selu_layer;
    using fc         = tiny_dnn::fully_connected_layer;
    using recurrent  = tiny_dnn::recurrent_layer;

    weights_path_  = std::string(weights);
    encoding_path_ = std::string(encoding);
    auto encoding_ = get_encoding();
    enc_dict_      = encoding_.first;
    dec_dict_      = encoding_.second;
    rnn_type_      = std::string(rnn_type);

    tiny_dnn::recurrent_layer_parameters params;
    // add recurrent stack
    int input_size = dec_dict_.size();
    nn << fc(input_size, input_size, false);
    for (int i = 0; i < depth; i++) {
      if (rnn_type_ == "rnn") {
        nn << recurrent(tiny_dnn::rnn(input_size, hidden_size), 1, params);
      } else if (rnn_type_ == "gru") {
        nn << recurrent(tiny_dnn::gru(input_size, hidden_size), 1, params);
      } else if (rnn_type_ == "lstm") {
        nn << recurrent(tiny_dnn::lstm(input_size, hidden_size), 1, params);
      }
      input_size = hidden_size;
      nn << activation();  // << dropout(hidden_size, 0.3);
    }
    nn << fc(hidden_size, dec_dict_.size(), false);
    // load weights
    std::ifstream ifs(weights_path_);
    ifs >> nn;
    for (auto n : nn) n->set_parallelize(true);

    // set test phase
    nn.set_netphase(tiny_dnn::net_phase::test);
    for (unsigned int i = 0; i < nn.layer_size(); i++) {
      try {
        nn.at<recurrent>(i).seq_len(1);
        nn.at<recurrent>(i).bptt_max(1e9);
        nn.at<recurrent>(i).clear_state();
      } catch (tiny_dnn::nn_error &err) {
      }
    }
  }

  /**
   * The main interface function.
   * @param c [in] input char.
   * @param temperature [in] softmax temperature.
   * @return rnn predicted char.
   */
  char forward(char c, double temperature) {
    unsigned int out_ch = 0;
    auto output         = nn.fprop(encode(c, enc_dict_))[0][0];
    softmax(output, temperature);
    out_ch = select_one(output);
    return dec_dict_[out_ch];
  }

 private:
  /**
   * Helper function to encode the input.
   * @param input [in] input char.
   * @param dict [in] encoding dictionary.
   * @return encoded char.
   */
  std::vector<tiny_dnn::tensor_t> encode(char &input,
                                         std::map<char, int> &dict) {
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
    std::ifstream ifs(encoding_path_, std::ifstream::in);
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
   * Softmax function.
   * @param data [in] outputs vector.
   * @param temperature [in] softmax temperature.
   */
  void softmax(tiny_dnn::vec_t &data, double temperature = 1.0) {
    tiny_dnn::softmax_layer softmax;
    for (auto &d : data) {
      d /= temperature;
    }
    softmax.forward_activation(data, data);
  }

  // internal variables.
  std::string encoding_path_;
  std::string rnn_type_;
  std::string weights_path_;

  std::map<char, int> enc_dict_;
  std::vector<char> dec_dict_;

  tiny_dnn::network<tiny_dnn::sequential> nn;
};
