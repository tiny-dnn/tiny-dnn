/*
Copyright (c) 2016, Taiga Nomi
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
* Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.
* Neither the name of the <organization> nor the
names of its contributors may be used to endorse or promote products
derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
#include <iostream>
#include <vector>
#include "tiny_dnn/tiny_dnn.h"

using namespace tiny_dnn;
using namespace tiny_dnn::activation;
using namespace std;

template <typename N>
void construct_net(N& nn) {
    typedef convolutional_layer<activation::identity> conv;
    typedef max_pooling_layer<relu> pool;

    const serial_size_t n_fmaps = 32;   ///< number of feature maps for upper layer
    const serial_size_t n_fmaps2 = 64;  ///< number of feature maps for lower layer
    const serial_size_t n_fc = 64;      ///< number of hidden units in fully-connected layer

    nn << conv(32, 32, 5, 3, n_fmaps, padding::same)
       << pool(32, 32, n_fmaps, 2)
       << conv(16, 16, 5, n_fmaps, n_fmaps, padding::same)
       << pool(16, 16, n_fmaps, 2)
       << conv(8, 8, 5, n_fmaps, n_fmaps2, padding::same)
       << pool(8, 8, n_fmaps2, 2)
       << fully_connected_layer<activation::identity>(4 * 4 * n_fmaps2, n_fc)
       << fully_connected_layer<softmax>(n_fc, 10);
}

void train_cifar10(string data_dir_path, double learning_rate, ostream& log) {
    // specify loss-function and learning strategy
    network<sequential> nn;
    adam optimizer;

    construct_net(nn);

    log << "learning rate:" << learning_rate << endl;

    cout << "load models..." << endl;

    // load cifar dataset
    vector<label_t> train_labels, test_labels;
    vector<vec_t> train_images, test_images;

    for (int i = 1; i <= 5; i++) {
        parse_cifar10(data_dir_path + "/data_batch_" + to_string(i) + ".bin",
                      &train_images, &train_labels, -1.0, 1.0, 0, 0);
    }

    parse_cifar10(data_dir_path + "/test_batch.bin",
                  &test_images, &test_labels, -1.0, 1.0, 0, 0);

    cout << "start learning" << endl;

    progress_display disp(train_images.size());
    timer t;
    const int n_minibatch = 10;     ///< minibatch size
    const int n_train_epochs = 30;  ///< training duration

    optimizer.alpha *= static_cast<tiny_dnn::float_t>(sqrt(n_minibatch) * learning_rate);

    // create callback
    auto on_enumerate_epoch = [&]() {
        cout << t.elapsed() << "s elapsed." << endl;
        tiny_dnn::result res = nn.test(test_images, test_labels);
        log << res.num_success << "/" << res.num_total << endl;

        disp.restart(train_images.size());
        t.restart();
    };

    auto on_enumerate_minibatch = [&]() {
        disp += n_minibatch;
    };

    // training
    nn.train<cross_entropy>(optimizer, train_images, train_labels,
                            n_minibatch, n_train_epochs, on_enumerate_minibatch,
                            on_enumerate_epoch);

    cout << "end training." << endl;

    // test and show results
    nn.test(test_images, test_labels).print_detail(cout);

    // save networks
    ofstream ofs("cifar-weights");
    ofs << nn;
}

int main(int argc, char **argv) {
    if (argc != 3) {
        cerr << "Usage : " << argv[0]
             << "arg[0]: path_to_data (example:../data)"
             << "arg[1]: learning rate (example:0.01)" << endl;
        return -1;
    }
    train_cifar10(argv[1], stod(argv[2]), cout);
}
