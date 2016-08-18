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
#include "tiny_dnn/tiny_dnn.h"

using namespace tiny_dnn;
using namespace tiny_dnn::activation;

///////////////////////////////////////////////////////////////////////////////
// recongnition on MNIST similar to LaNet-5 adding deconvolution

void deconv_lanet(network<graph>& nn,
    std::vector<label_t> train_labels,
    std::vector<label_t> test_labels,
    std::vector<vec_t> train_images,
    std::vector<vec_t> test_images)
{
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

    // declare nodes
    input_layer i1(shape3d(32,32,1));
    convolutional_layer<tan_h> c1(32, 32, 5, 1, 6);
    average_pooling_layer<tan_h> p1(28, 28, 6, 2);
    deconvolutional_layer<tan_h> d1(14, 14, 5, 6, 16, connection_table(tbl, 6, 16));
    average_pooling_layer<tan_h> p2(18, 18, 16, 2);
    convolutional_layer<tan_h> c2(9, 9, 9, 16, 120);
    fully_connected_layer<tan_h> f1(120, 10);

    // connect them to graph
    i1 << c1 << p1 << d1 << p2 << c2 << f1;
    construct_graph(nn, { &i1 }, { &f1 });

    std::cout << "start training" << std::endl;

    progress_display disp((unsigned long)train_images.size());
    timer t;
    int minibatch_size = 10;
    int num_epochs = 30;

    adagrad optimizer;
    optimizer.alpha *= static_cast<tiny_dnn::float_t>(std::sqrt(minibatch_size));

    // create callback
    auto on_enumerate_epoch = [&](){
        std::cout << t.elapsed() << "s elapsed." << std::endl;
        tiny_dnn::result res = nn.test(test_images, test_labels);
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
    std::ofstream ofs("deconv_lanet_weights");
    ofs << nn;
}

///////////////////////////////////////////////////////////////////////////////
// Deconcolutional Auto-encoder
void deconv_ae(network<sequential>& nn,
    std::vector<label_t> train_labels,
    std::vector<label_t> test_labels,
    std::vector<vec_t> train_images,
    std::vector<vec_t> test_images) {

    // construct nets
    nn << convolutional_layer<tan_h>(32, 32, 5, 1, 6)
       << average_pooling_layer<tan_h>(28, 28, 6, 2)
       << convolutional_layer<tan_h>(14, 14, 3, 6, 16)
       << deconvolutional_layer<tan_h>(12, 12, 3, 16, 6)
       << average_unpooling_layer<tan_h>(14, 14, 6, 2)
       << deconvolutional_layer<tan_h>(28, 28, 5, 6, 1);

    // load train-data and make corruption

    std::vector<vec_t> training_images_corrupted(train_images);

    for (auto& d : training_images_corrupted) {
        d = corrupt(move(d), tiny_dnn::float_t(0.1), tiny_dnn::float_t(0.0)); // corrupt 10% data
    }

    gradient_descent optimizer;

    // learning deconcolutional Auto-encoder
    nn.train<mse>(optimizer, training_images_corrupted, train_images);

    std::cout << "end training." << std::endl;

    // save networks
    std::ofstream ofs("deconv_ae_weights");
    ofs << nn;
}

///////////////////////////////////////////////////////////////////////////////
// ENet
void enet(network<graph>& nn,
    std::vector<label_t> train_labels,
    std::vector<label_t> test_labels,
    std::vector<vec_t> train_images,
    std::vector<vec_t> test_images) {

    // initial module
    input_layer ii0(shape3d(32,32,1));
    convolutional_layer<tan_h> ic1(32, 32, 3, 1, 8, padding::same, true, 2, 2);
    max_pooling_layer<tan_h> ip1(32, 32, 1, 2);
    convolutional_layer<tan_h> ic2(16, 16, 1, 1, 8, padding::same);
    concat_layer icc1(2, 16*16*8);

    ii0 << ip1 << ic2;
    ii0 << ic1;
    (ic2, ic1) << icc1;

    // bottle neck module 1
    max_pooling_layer<tan_h> b1p1(16, 16, 16, 2);
    convolutional_layer<tan_h> b1c2(8, 8, 1, 16, 32, padding::same);
    convolutional_layer<tan_h> b1c1(16, 16, 1, 16, 32, padding::same);
    convolutional_layer<tan_h> b1c3(16, 16, 2, 32, 32, padding::same, true, 2, 2);
    convolutional_layer<tan_h> b1c4(8, 8, 1, 32, 32, padding::same);
    concat_layer b1cc1(2, 8*8*32);

    icc1 << b1p1 << b1c2;
    icc1 << b1c1 << b1c3 << b1c4;
    (b1c2, b1c4) << b1cc1;

    // bottle neck module 2
    deconvolutional_layer<tan_h> b2d1(8, 8, 1, 64, 16, padding::same, true, 2, 2);
    deconvolutional_layer<tan_h> b2d2(16, 16, 1, 16, 1, padding::same, true, 2, 2);
    fully_connected_layer<tan_h> f1(32*32, 10);
    b1cc1 << b2d1 << b2d2 << f1;

    // construct whole network
    construct_graph(nn, { &ii0 }, { &f1 });

    // load train-data and make corruption

    std::cout << "start training" << std::endl;

    progress_display disp((unsigned long)train_images.size());
    timer t;
    int minibatch_size = 10;
    int num_epochs = 30;

    adagrad optimizer;
    optimizer.alpha *= tiny_dnn::float_t(std::sqrt(minibatch_size));

    // create callback
    auto on_enumerate_epoch = [&](){
        std::cout << t.elapsed() << "s elapsed." << std::endl;
        tiny_dnn::result res = nn.test(test_images, test_labels);
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
    std::ofstream ofs("deconv_lanet_weights");
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
    network<sequential> nn_s;
    network<graph> nn_g;

    if (experiment == "deconv_lanet")
        deconv_lanet(nn_g, train_labels, test_labels, train_images, test_images); // recongnition on MNIST similar to LaNet-5 adding deconvolution
    else if (experiment == "deconv_ae")
        deconv_ae(nn_s, train_labels, test_labels, train_images, test_images); // Deconcolution Auto-encoder on MNIST
    else if (experiment == "enet")
        enet(nn_g, train_labels, test_labels, train_images, test_images); // Bottle neck module based ENet
}

int main(int argc, char **argv) {
    if (argc != 3) {
        std::cerr << "Usage : " << argv[0]
                  << " path_to_data (example:../data) (example:deconv_lanet, deconv_ae or enet)" << std::endl;
        return -1;
    }
    train(argv[1], argv[2]);
}
