/*
    Copyright (c) 2013, Taiga Nomi
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
#include "tiny_cnn/tiny_cnn.h"

using namespace tiny_cnn;
using namespace tiny_cnn::activation;

///////////////////////////////////////////////////////////////////////////////
// recongnition on MNIST similar to LaNet-5 adding deconvolution

void deLaNet(network<sequential>& nn,
    std::vector<label_t> train_labels,
    std::vector<label_t> test_labels,
    std::vector<vec_t> train_images,
    std::vector<vec_t> test_images) {

    // connection table [Y.Lecun, 1998 Table.1]
    #define O true
    #define X false
        static const bool tbl[] = {
            O, X, X, X, O, O, O, X, X, O, O, O, O, X, O, O,
            O, O, X, X, X, O, O, O, X, X, O, O, O, O, X, O,
            O, O, O, X, X, X, O, O, O, X, X, O, X, O, O, O,
            X, O, O, O, X, X, O, O, O, O, X, X, O, X, O, O,
            X, X, O, O, O, X, X, O, O, O, O, X, O, O, X, O,
            X, X, X, O, O, O, X, X, O, O, O, O, X, O, O, O
        };
    #undef O
    #undef X

    // construct nets
    nn << convolutional_layer<tan_h>(32, 32, 5, 1, 6) // 32x32 in, 5x5 kernel, 1-6 fmaps conv
       << average_pooling_layer<tan_h>(28, 28, 6, 2) // 28x28 in, 6 fmaps, 2x2 subsampling
       << deconvolutional_layer<tan_h>(14, 14, 5, 6, 16,
                                     connection_table(tbl, 6, 16)) // with connection-table
       << average_pooling_layer<tan_h>(18, 18, 16, 2)
       << convolutional_layer<tan_h>(9, 9, 9, 16, 120)
       << fully_connected_layer<tan_h>(120, 10);

    std::cout << "start training" << std::endl;

    progress_display disp((unsigned long)train_images.size());
    timer t;
    int minibatch_size = 10;
    int num_epochs = 30;

    adagrad optimizer;
    optimizer.alpha *= tiny_cnn::float_t(std::sqrt(minibatch_size));

    // create callback
    auto on_enumerate_epoch = [&](){
        std::cout << t.elapsed() << "s elapsed." << std::endl;
        tiny_cnn::result res = nn.test(test_images, test_labels);
        std::cout << res.num_success << "/" << res.num_total << std::endl;

        disp.restart((unsigned long)train_images.size());
        t.restart();
    };

    auto on_enumerate_minibatch = [&](){
        disp += minibatch_size;
    };

    // training
    nn.train<mse>(optimizer, train_images, train_labels, minibatch_size,
                  num_epochs, on_enumerate_minibatch, on_enumerate_epoch);

    std::cout << "end training." << std::endl;

    // test and show results
    nn.test(test_images, test_labels).print_detail(std::cout);

    // save networks
    std::ofstream ofs("deLeNet-weights");
    ofs << nn;
}

///////////////////////////////////////////////////////////////////////////////
// Deconcolutional Auto-encoder
void DeAE(network<sequential>& nn, 
    std::vector<label_t> train_labels,
    std::vector<label_t> test_labels,
    std::vector<vec_t> train_images,
    std::vector<vec_t> test_images) {

    // construct nets
    nn << convolutional_layer<tan_h>(32, 32, 5, 1, 18)  // C1, 1@32x32-in, 6@32x32-out
       << average_pooling_layer<tan_h>(28, 28, 18, 2)   // S2, 6@28x28-in, 6@14x14-out
       << average_unpooling_layer<tan_h>(14, 14, 18, 2) // U3, 6@14x14-in, 6@28x28-out
       << deconvolutional_layer<tan_h>(28, 28, 5, 18, 3); // D4, 6@28x28-in, 16@32x32-out

    // load train-data and make corruption

    std::vector<vec_t> training_images_corrupted(train_images);

    for (auto& d : training_images_corrupted) {
        d = corrupt(move(d), tiny_cnn::float_t(0.1), tiny_cnn::float_t(0.0)); // corrupt 10% data
    }

    gradient_descent optimizer;

    // learning deconcolutional Auto-encoder
    nn.train<mse>(optimizer, training_images_corrupted, train_images);

    std::cout << "end training." << std::endl;

    // save networks
    std::ofstream ofs("DeAE-weights");
    ofs << nn;
}

void train(std::string data_dir_path, std::string experiment) {

    std::cout << "load traing and testing data..." << std::endl;

    // load MNIST dataset
    std::vector<label_t> train_labels, test_labels;
    std::vector<vec_t> train_images, test_images;

    parse_mnist_labels(data_dir_path+"/train-labels.idx1-ubyte",
                       &train_labels);
    parse_mnist_images(data_dir_path+"/train-images.idx3-ubyte",
                       &train_images, -1.0, 1.0, 2, 2);
    parse_mnist_labels(data_dir_path+"/t10k-labels.idx1-ubyte",
                       &test_labels);
    parse_mnist_images(data_dir_path+"/t10k-images.idx3-ubyte",
                       &test_images, -1.0, 1.0, 2, 2);
    // specify loss-function and learning strategy
    network<sequential> nn;

    if (experiment == "deLaNet")
        deLaNet(nn, train_labels, test_labels, train_images, test_images); // recongnition on MNIST similar to LaNet-5 adding deconvolution
    else if (experiment == "DeAE")
        DeAE(nn, train_labels, test_labels, train_images, test_images); // Deconcolution Auto-encoder on MNIST
}

int main(int argc, char **argv) {
    if (argc != 3) {
        std::cerr << "Usage : " << argv[0]
                  << " path_to_data (example:../data) (example:deLaNet or DeAE)" << std::endl;
        return -1;
    }
    train(argv[1], argv[2]);
}
