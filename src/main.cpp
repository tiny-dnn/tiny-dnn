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
#include <boost/timer.hpp>
#include <boost/progress.hpp>

#include "tiny_cnn.h"
//#define NOMINMAX
//#include "imdebug.h"

void sample1_convnet();
void sample2_mlp();
void sample3_dae();
void sample4_dropout();

using namespace tiny_cnn;
using namespace tiny_cnn::activation;

int main(void) {
    sample1_convnet();
}

///////////////////////////////////////////////////////////////////////////////
// learning convolutional neural networks (LeNet-5 like architecture)
void sample1_convnet(void) {
    // construct LeNet-5 architecture
    typedef network<mse, gradient_descent_levenberg_marquardt> CNN;
    CNN nn;
    convolutional_layer<CNN, tan_h> C1(32, 32, 5, 1, 6);
    average_pooling_layer<CNN, tan_h> S2(28, 28, 6, 2);
    // connection table [Y.Lecun, 1998 Table.1]
#define O true
#define X false
    static const bool connection[] = {
        O, X, X, X, O, O, O, X, X, O, O, O, O, X, O, O,
        O, O, X, X, X, O, O, O, X, X, O, O, O, O, X, O,
        O, O, O, X, X, X, O, O, O, X, X, O, X, O, O, O,
        X, O, O, O, X, X, O, O, O, O, X, X, O, X, O, O,
        X, X, O, O, O, X, X, O, O, O, O, X, O, O, X, O,
        X, X, X, O, O, O, X, X, O, O, O, O, X, O, O, O
    };
#undef O
#undef X
    convolutional_layer<CNN, tan_h> C3(14, 14, 5, 6, 16, connection_table(connection, 6, 16));
    average_pooling_layer<CNN, tan_h> S4(10, 10, 16, 2);
    convolutional_layer<CNN, tan_h> C5(5, 5, 5, 16, 120);
    fully_connected_layer<CNN, tan_h> F6(120, 10);

    assert(C1.param_size() == 156 && C1.connection_size() == 122304);
    assert(S2.param_size() == 12 && S2.connection_size() == 5880);
    assert(C3.param_size() == 1516 && C3.connection_size() == 151600);
    assert(S4.param_size() == 32 && S4.connection_size() == 2000);
    assert(C5.param_size() == 48120 && C5.connection_size() == 48120);

    nn.add(&C1);
    nn.add(&S2);
    nn.add(&C3);
    nn.add(&S4);
    nn.add(&C5);
    nn.add(&F6);

    std::cout << "load models..." << std::endl;

    // load MNIST dataset
    std::vector<label_t> train_labels, test_labels;
    std::vector<vec_t> train_images, test_images;

    parse_mnist_labels("train-labels.idx1-ubyte", &train_labels);
    parse_mnist_images("train-images.idx3-ubyte", &train_images);
    parse_mnist_labels("t10k-labels.idx1-ubyte", &test_labels);
    parse_mnist_images("t10k-images.idx3-ubyte", &test_images);

    std::cout << "start learning" << std::endl;

    boost::progress_display disp(train_images.size());
    boost::timer t;
    int minibatch_size = 10;

    nn.optimizer().alpha *= std::sqrt(minibatch_size);

    // create callback
    auto on_enumerate_epoch = [&](){
        std::cout << t.elapsed() << "s elapsed." << std::endl;

        tiny_cnn::result res = nn.test(test_images, test_labels);

        std::cout << nn.optimizer().alpha << "," << res.num_success << "/" << res.num_total << std::endl;

        nn.optimizer().alpha *= 0.85; // decay learning rate
        nn.optimizer().alpha = std::max(0.00001, nn.optimizer().alpha);

        disp.restart(train_images.size());
        t.restart();
    };

    auto on_enumerate_minibatch = [&](){ 
        disp += minibatch_size; 
    
        // weight visualization in imdebug
        /*static int n = 0;    
        n+=minibatch_size;
        if (n >= 1000) {
            image img;
            C3.weight_to_image(img);
            imdebug("lum b=8 w=%d h=%d %p", img.width(), img.height(), &img.data()[0]);
            n = 0;
        }*/
    };
    
    // training
    nn.train(train_images, train_labels, minibatch_size, 20, on_enumerate_minibatch, on_enumerate_epoch);

    std::cout << "end training." << std::endl;

    // test and show results
    nn.test(test_images, test_labels).print_detail(std::cout);

    // save networks
    std::ofstream ofs("LeNet-weights");
    ofs << C1 << S2 << C3 << S4 << C5 << F6;
}


///////////////////////////////////////////////////////////////////////////////
// learning 3-Layer Networks
void sample2_mlp()
{
    const int num_hidden_units = 500;

#if defined(_MSC_VER) && _MSC_VER < 1800
    // initializer-list is not supported
    int num_units[] = { 28 * 28, num_hidden_units, 10 };
    auto nn = make_mlp<mse, gradient_descent, tan_h>(num_units, num_units + 3);
#else
    auto nn = make_mlp<mse, gradient_descent, tan_h>({ 28 * 28, num_hidden_units, 10 });
#endif

    // load MNIST dataset
    std::vector<label_t> train_labels, test_labels;
    std::vector<vec_t> train_images, test_images;

    parse_mnist_labels("train-labels.idx1-ubyte", &train_labels);
    parse_mnist_images("train-images.idx3-ubyte", &train_images, -1.0, 1.0, 0, 0);
    parse_mnist_labels("t10k-labels.idx1-ubyte", &test_labels);
    parse_mnist_images("t10k-images.idx3-ubyte", &test_images, -1.0, 1.0, 0, 0);

    nn.optimizer().alpha = 0.001;
    
    boost::progress_display disp(train_images.size());
    boost::timer t;

    // create callback
    auto on_enumerate_epoch = [&](){
        std::cout << t.elapsed() << "s elapsed." << std::endl;

        tiny_cnn::result res = nn.test(test_images, test_labels);

        std::cout << nn.optimizer().alpha << "," << res.num_success << "/" << res.num_total << std::endl;

        nn.optimizer().alpha *= 0.85; // decay learning rate
        nn.optimizer().alpha = std::max(0.00001, nn.optimizer().alpha);

        disp.restart(train_images.size());
        t.restart();
    };

    auto on_enumerate_data = [&](){ 
        ++disp; 
    };  

    nn.train(train_images, train_labels, 1, 20, on_enumerate_data, on_enumerate_epoch);
}

///////////////////////////////////////////////////////////////////////////////
// denoising auto-encoder
void sample3_dae()
{
#if defined(_MSC_VER) && _MSC_VER < 1800
    // initializer-list is not supported
    int num_units[] = { 100, 400, 100 };
    auto nn = make_mlp<mse, gradient_descent, tan_h>(num_units, num_units + 3);
#else
    auto nn = make_mlp<mse, gradient_descent, tan_h>({ 100, 400, 100 });
#endif

    std::vector<vec_t> train_data_original;

    // load train-data

    std::vector<vec_t> train_data_corrupted(train_data_original);

    for (auto& d : train_data_corrupted) {
        corrupt(d, 0.1, 0.0, &d); // corrupt 10% data
    }

    // learning 100-400-100 denoising auto-encoder
    nn.train(train_data_corrupted, train_data_original);
}

///////////////////////////////////////////////////////////////////////////////
// dropout-learning

void sample4_dropout()
{
    typedef network<mse, gradient_descent> Network;
    Network nn;
    int input_dim    = 28*28;
    int hidden_units = 800;
    int output_dim   = 10;

    fully_connected_dropout_layer<Network, tan_h> f1(input_dim, hidden_units, dropout::per_data);
    fully_connected_layer<Network, tan_h> f2(hidden_units, output_dim);
    nn.add(&f1); nn.add(&f2);

    nn.optimizer().alpha = 0.003; // TODO: not optimized
    nn.optimizer().lambda = 0.0;

    // load MNIST dataset
    std::vector<label_t> train_labels, test_labels;
    std::vector<vec_t> train_images, test_images;

    parse_mnist_labels("train-labels.idx1-ubyte", &train_labels);
    parse_mnist_images("train-images.idx3-ubyte", &train_images, -1.0, 1.0, 0, 0);
    parse_mnist_labels("t10k-labels.idx1-ubyte", &test_labels);
    parse_mnist_images("t10k-images.idx3-ubyte", &test_images, -1.0, 1.0, 0, 0);

    // load train-data, label_data
    boost::progress_display disp(train_images.size());
    boost::timer t;

    // create callback
    auto on_enumerate_epoch = [&](){
        std::cout << t.elapsed() << "s elapsed." << std::endl;

        f1.set_context(dropout::test_phase);
        tiny_cnn::result res = nn.test(test_images, test_labels);
        f1.set_context(dropout::train_phase);


        std::cout << nn.optimizer().alpha << "," << res.num_success << "/" << res.num_total << std::endl;

        nn.optimizer().alpha *= 0.99; // decay learning rate
        nn.optimizer().alpha = std::max(0.00001, nn.optimizer().alpha);

        disp.restart(train_images.size());
        t.restart();
    };

    auto on_enumerate_data = [&](){
        ++disp;
    };

    nn.train(train_images, train_labels, 1, 100, on_enumerate_data, on_enumerate_epoch);

    // change context to enable all hidden-units
    //f1.set_context(dropout::test_phase);
    //std::cout << res.num_success << "/" << res.num_total << std::endl;
}
