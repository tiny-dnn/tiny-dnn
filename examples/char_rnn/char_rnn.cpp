/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

#include "tiny_dnn/tiny_dnn.h"
#include "tiny_dnn/util/random.h"

// Data helpers

/**
 * Handles loading data from bytestream dataset.
 */
class CharDataset {
 public:
  /**
   * Constructor. Receives the path to the data and labels, batch_size,
   * and dimensionality of its one-hot encoding. Data and labels are
   * assumed to contain the same number of samples with the following
   * dimensonality: n_samples x sequence_length x batch_size x 1
   * @param x [in] Data path
   * @param y [in] Labels path
   * @param batch_size [in] batch size.
   * @param dims [in] one-hot dimensonality.
   */
  CharDataset(const std::string x,
              const std::string y,
              const int batch_size,
              const int dims)
    : batch_size_(batch_size), dims_(dims) {
    // open input streams
    fdata_.open(x);
    flabels_.open(y);
    // resize internal data buffers
    data_buffer_.resize(batch_size);
    labels_buffer_.resize(batch_size);
    for (int i = 0; i < batch_size; i++) {
      data_buffer_[i].resize(1);
      labels_buffer_[i].resize(1);
      data_buffer_[i][0].resize(dims);
      labels_buffer_[i][0].resize(dims);
    }
    // guess dataset size
    fdata_.seekg(0, fdata_.end);
    size_ = fdata_.tellg();
    fdata_.seekg(0, fdata_.beg);
  }

  /**
   * Fill internal buffers with the next batch.
   * @return Whether an epoch has been completed.
   */
  const bool get() {
    int count = 0;
    char x;
    char y;
    if (static_cast<int>(fdata_.tellg()) + batch_size_ > size_) {
      return false;
    }
    while (count < batch_size_ && fdata_.read(&x, 1) && flabels_.read(&y, 1)) {
      tiny_dnn::fill_tensor(data_buffer_[count], 0.0);
      tiny_dnn::fill_tensor(labels_buffer_[count], 0.0);
      data_buffer_[count][0][x]   = 1;
      labels_buffer_[count][0][y] = 1;
      count++;
    }
    const bool ret = fdata_.good() && flabels_.good();
    return ret;
  }

  /**
   * Restarts from the begining.
   */
  inline void rewind() {
    fdata_.seekg(fdata_.beg);
    flabels_.seekg(flabels_.beg);
  }

  /**
   * Getter of the dataset size.
   * @return dataset size.
   */
  const int size() { return size_; }

  /**
   * Get data buffer.
   * @return data_buffer
   */
  std::vector<tiny_dnn::tensor_t> &get_data() { return data_buffer_; }
  /**
   * Get labels buffer.
   * @return labels_buffer
   */
  std::vector<tiny_dnn::tensor_t> &get_labels() { return labels_buffer_; }

 private:
  // state variables
  int batch_size_;
  int dims_;
  int size_;

  // input stream
  std::ifstream fdata_;
  std::ifstream flabels_;

  // buffers
  std::vector<tiny_dnn::tensor_t> data_buffer_;
  std::vector<tiny_dnn::tensor_t> labels_buffer_;
};

/**
 * Load dictionary to decode output predictions.
 * @return dictionary (array of chars)
 */
const std::vector<char> get_dec_dict() {
  std::ifstream ifs("../examples/char_rnn/encoding.raw", std::ifstream::in);
  std::vector<char> ret;
  char c;
  while (ifs.read(&c, 1)) {
    ret.push_back(c);
  }
  return ret;
}

// Model helpers

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
                   float dropout_rate,
                   tiny_dnn::core::backend_t backend_type) {
  // define layer aliases
  using activation = tiny_dnn::selu_layer;
  using dropout    = tiny_dnn::dropout_layer;
  using fc         = tiny_dnn::fully_connected_layer;
  using recurrent  = tiny_dnn::recurrent_layer;
  using softmax    = tiny_dnn::softmax_layer;

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
    nn << activation();
    if (dropout_rate > 0) nn << dropout(hidden_size, dropout_rate);
  }
  // predict next char
  nn << fc(hidden_size, vocab_size, false, backend_type) << softmax(vocab_size);
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

template <typename N>
void set_train(N &nn, const int seq_len) {
  nn.set_netphase(tiny_dnn::net_phase::train);
  for (unsigned int i = 0; i < nn.layer_size(); i++) {
    try {
      nn.template at<tiny_dnn::dropout_layer>(i).set_context(
        tiny_dnn::net_phase::train);
    } catch (tiny_dnn::nn_error &err) {
    }
    try {
      nn.template at<tiny_dnn::recurrent_layer>(i).seq_len(seq_len);
      nn.template at<tiny_dnn::recurrent_layer>(i).bptt_max(seq_len);
      nn.template at<tiny_dnn::recurrent_layer>(i).clear_state();
    } catch (tiny_dnn::nn_error &err) {
    }
  }
}

template <typename N>
void set_test(N &nn) {
  nn.set_netphase(tiny_dnn::net_phase::test);
  for (unsigned int i = 0; i < nn.layer_size(); i++) {
    try {
      nn.template at<tiny_dnn::dropout_layer>(i).set_context(
        tiny_dnn::net_phase::test);
    } catch (tiny_dnn::nn_error &err) {
    }
    try {
      nn.template at<tiny_dnn::recurrent_layer>(i).clear_state();
    } catch (tiny_dnn::nn_error &err) {
    }
  }
}

// Train/Val
/**
 * The main training loop.
 * @param hidden_size [in] Hidden state size.
 * @param n_layers [in] Number of hidden layers.
 * @param seq_len [in] Max sequence length.
 * @param learning_rate [in] The optimizer learning rate.
 * @param n_train_epochs [in] Number of epochs to train.
 * @param n_minibatch [in] Mini-batch size.
 * @param rnn_type [in] RNN, LSTM, GRU
 * @param backend_type [in] the backend.
 */
void train(int hidden_size,
           int n_layers,
           int seq_len,
           double learning_rate,
           const int n_train_epochs,
           int n_minibatch,
           const std::string rnn_type,
           float dropout_rate,
           tiny_dnn::core::backend_t backend_type) {
  // specify loss-function and learning strategy
  tiny_dnn::network<tiny_dnn::sequential> nn;
  tiny_dnn::adam optimizer;
  optimizer.alpha = learning_rate;

  // aux variable
  int total_length = n_minibatch * seq_len;

  std::cout << "Loading data..." << std::endl;
  auto dict = get_dec_dict();
  CharDataset train_dataset("../examples/char_rnn/train.raw",
                            "../examples/char_rnn/train_labels.raw",
                            total_length, dict.size());
  CharDataset val_dataset("../examples/char_rnn/val.raw",
                          "../examples/char_rnn/val_labels.raw", total_length,
                          dict.size());

  std::cout << "load models..." << std::endl;
  construct_net(nn, dict.size(), hidden_size, n_layers, seq_len, rnn_type,
                dropout_rate, backend_type);
  nn.weight_init(tiny_dnn::weight_init::xavier());
  for (auto n : nn) n->set_parallelize(true);
  optimizer.reset();

  std::cout << "start learning" << std::endl;
  tiny_dnn::timer t;
  int epoch       = 0;
  float best_loss = std::numeric_limits<float>::max();
  // aux vector for calling bprop
  std::vector<tiny_dnn::tensor_t> cost;
  while (epoch++ < n_train_epochs) {
    if (epoch % 10 == 0) {
      optimizer.alpha *= 0.97;
    }
    // training phase
    set_train(nn, seq_len);
    // start reading dataset by the first element
    train_dataset.rewind();
    float loss  = 0;
    int counter = 0;
    // train epoch
    while (train_dataset.get()) {
      std::vector<tiny_dnn::tensor_t> &data   = train_dataset.get_data();
      std::vector<tiny_dnn::tensor_t> &labels = train_dataset.get_labels();
      data                                    = nn.fprop(data);
      // show current batch loss
      if (counter++ % 20 == 0) {
        loss = cross_entropy(data, labels);
        std::cout << "Train loss: " << loss << std::endl;
      }
      nn.bprop<tiny_dnn::cross_entropy>(data, labels, cost);
      nn.update_weights(&optimizer);
    }
    // validation phase
    set_test(nn);
    float val_loss = 0;
    val_dataset.rewind();
    counter = 0;
    while (val_dataset.get()) {
      std::vector<tiny_dnn::tensor_t> &data   = val_dataset.get_data();
      std::vector<tiny_dnn::tensor_t> &labels = val_dataset.get_labels();
      data                                    = nn.fprop(data);
      val_loss += cross_entropy(data, labels);
      // Sample predictions
      for (int n = 0; n < seq_len; n++) {
        auto &d           = data[n * n_minibatch][0];
        unsigned int pred = select_one(d);
        std::cout << dict[pred];
      }
      std::cout << std::endl;
      counter += 1;
    }
    std::cout << "Val loss: " << val_loss / counter << std::endl;
    if (val_loss < best_loss) {
      best_loss = val_loss;
      std::cout << "Save checkpoint" << std::endl;
      // save network
      std::ofstream ofs("char_rnn_weights");
      ofs << nn;
    }
  }
  std::cout << "end training." << std::endl;
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
  std::cout << "Usage: " << argv0 << " --learning_rate 0.01"
            << " --epochs 20"
            << " --hidden_size 128"
            << " --minibatch_size 32"
            << " --n_layers 1"
            << " --rnn_type gru"
            << " --backend_type internal" << std::endl;
}

int main(int argc, char **argv) {
  double learning_rate                   = 0.01;
  int epochs                             = 20;
  int hidden_size                        = 128;
  int minibatch_size                     = 32;
  int n_layers                           = 1;
  int seq_len                            = 100;
  std::string rnn_type                   = "gru";
  float dropout_rate                     = 0.0;
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
    if (argname == "--learning_rate") {
      learning_rate = atof(argv[count + 1]);
    } else if (argname == "--epochs") {
      epochs = atoi(argv[count + 1]);
    } else if (argname == "--hidden_size") {
      hidden_size = atoi(argv[count + 1]);
    } else if (argname == "--n_layers") {
      n_layers = atoi(argv[count + 1]);
    } else if (argname == "--minibatch_size") {
      minibatch_size = atoi(argv[count + 1]);
    } else if (argname == "--seq_len") {
      seq_len = atoi(argv[count + 1]);
    } else if (argname == "--rnn_type") {
      rnn_type = argv[count + 1];
    } else if (argname == "--dropout") {
      dropout_rate = atof(argv[count + 1]);
    } else if (argname == "--backend_type") {
      backend_type = parse_backend_name(argv[count + 1]);
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
            << "Learning rate: " << learning_rate << std::endl
            << "Hidden size: " << hidden_size << std::endl
            << "Minibatch size: " << minibatch_size << std::endl
            << "NLayers: " << n_layers << std::endl
            << "Sequence length: " << seq_len << std::endl
            << "Number of epochs: " << epochs << std::endl
            << "Rnn type: " << rnn_type << std::endl
            << "Dropout rate: " << dropout_rate << std::endl
            << "Backend type: " << backend_type << std::endl
            << std::endl;
  try {
    train(hidden_size, n_layers, seq_len, learning_rate, epochs, minibatch_size,
          rnn_type, dropout_rate, backend_type);
  } catch (tiny_dnn::nn_error &err) {
    std::cerr << "Exception: " << err.what() << std::endl;
  }
}
