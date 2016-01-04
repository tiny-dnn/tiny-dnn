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
#include <memory>
#define CNN_CAFFEINATED
#include "tiny_cnn/tiny_cnn.h"
//#define NOMINMAX
//#include "imdebug.h"

void sample1_convnet();
void sample2_mlp();
void sample3_dae();
void sample4_dropout();

using namespace tiny_cnn;
using namespace tiny_cnn::activation;
using namespace std;

#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

cv::Mat compute_mean(const string& mean_file, int width, int height)
{
    caffe::BlobProto mean_blob;
    detail::read_proto_from_binary(mean_file, &mean_blob);

    /* The format of the mean_blob file is planar 32-bit float BGR or grayscale. */
    std::vector<cv::Mat> channels;
    float* data = mean_blob.mutable_data()->mutable_data();
    for (int i = 0; i < mean_blob.channels(); ++i) {
        /* Extract an individual channel. */
        cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
        channels.push_back(channel);
        data += mean_blob.height() * mean_blob.width();
    }

    /* Merge the separate channels into a single image. */
    cv::Mat mean;
    cv::merge(channels, mean);

    /* Compute the global mean pixel value and create a mean image
    * filled with this value. */
    cv::Scalar channel_mean = cv::mean(mean);
    return cv::Mat(cv::Size(width, height), mean.type(), channel_mean);
}

cv::ColorConversionCodes get_cvt_codes(int src_channels, int dst_channels)
{
    assert(src_channels != dst_channels);

    if (dst_channels == 3) 
      return src_channels == 1 ? cv::COLOR_GRAY2BGR : cv::COLOR_BGRA2BGR;
    else if (dst_channels == 1) 
      return src_channels == 3 ? cv::COLOR_BGR2GRAY : cv::COLOR_BGRA2GRAY;
    else
      throw std::runtime_error("unsupported color code");
}

void preprocess(const cv::Mat& img, const cv::Mat& mean, int num_channels, cv::Size geometry, std::vector<cv::Mat>* input_channels)
{
    cv::Mat sample;

    // convert color
    if (img.channels() != num_channels)
        cv::cvtColor(img, sample, get_cvt_codes(img.channels(), num_channels));
    else
        sample = img;

    // resize
    cv::Mat sample_resized;
    cv::resize(sample, sample_resized, geometry);

    cv::Mat sample_float;
    sample_resized.convertTo(sample_float, num_channels == 3 ? CV_32FC3 : CV_32FC1);
 
    // subtract mean
    if (mean.size().width > 0) {
        cv::Mat sample_normalized;
        cv::subtract(sample_float, mean, sample_normalized);
        cv::split(sample_normalized, *input_channels);
    }
    else {
        cv::split(sample_float, *input_channels);
    }
}

void test(const string& model_file, const string& trained_file, const string& mean_file, const string& label_file, const string& img_file)
{
    auto net = create_net_from_caffe_prototxt(model_file);
    reload_weight_from_caffe_protobinary(trained_file, net.get());

    int channels = net->in_shape().depth_;
    int width = net->in_shape().width_;
    int height = net->in_shape().height_;

    cv::Mat img = cv::imread(img_file, -1);

    auto mean = compute_mean(mean_file, width, height);

    std::vector<float> inputvec(width*height*channels);
    std::vector<cv::Mat> input_channels;
    for (int i = 0; i < channels; i++) {
        cv::Mat channel(height, width, CV_32FC1, &inputvec[width*height*i]);
        input_channels.push_back(channel);
    }

    preprocess(img, mean, 3, cv::Size(width, height), &input_channels);

    std::vector<double> vec(width*height*channels);
    for (size_t i = 0; i < vec.size(); i++)
        vec[i] = inputvec[i];
    auto result = net->predict(vec);
    //auto net = create_net_from_caffeproto("C:\\Users\\knowme\\Documents\\GitHub\\tiny-cnn\\vc_caffe\\caffe2tinycnn\\caffe2tinycnn\\deploy.prototxt");

}

int main(int argc, char** argv) {
    int arg_channel = 1;
    string model_file = argv[arg_channel++];
    string trained_file = argv[arg_channel++];
    string mean_file = argv[arg_channel++];
    string label_file = argv[arg_channel++];
    string img_file = argv[arg_channel++];

    try {
    test(model_file, trained_file, mean_file, label_file, img_file);
    }
    catch (const nn_error& e)
    {
        std::cout << e.what() << std::endl;
    }

}

///////////////////////////////////////////////////////////////////////////////
// learning convolutional neural networks (LeNet-5 like architecture)
void sample1_convnet(void) {
    // construct LeNet-5 architecture
    network<mse, gradient_descent_levenberg_marquardt> nn;

    // connection table [Y.Lecun, 1998 Table.1]
#define O true
#define X false
    static const bool connection [] = {
        O, X, X, X, O, O, O, X, X, O, O, O, O, X, O, O,
        O, O, X, X, X, O, O, O, X, X, O, O, O, O, X, O,
        O, O, O, X, X, X, O, O, O, X, X, O, X, O, O, O,
        X, O, O, O, X, X, O, O, O, O, X, X, O, X, O, O,
        X, X, O, O, O, X, X, O, O, O, O, X, O, O, X, O,
        X, X, X, O, O, O, X, X, O, O, O, O, X, O, O, O
    };
#undef O
#undef X

    nn << convolutional_layer<tan_h>(32, 32, 5, 1, 6) // 32x32 in, 5x5 kernel, 1-6 fmaps conv
       << average_pooling_layer<tan_h>(28, 28, 6, 2) // 28x28 in, 6 fmaps, 2x2 subsampling
       << convolutional_layer<tan_h>(14, 14, 5, 6, 16,
                                     connection_table(connection, 6, 16)) // with connection-table
       << average_pooling_layer<tan_h>(10, 10, 16, 2)
       << convolutional_layer<tan_h>(5, 5, 5, 16, 120)
       << fully_connected_layer<tan_h>(120, 10);

    std::cout << "load models..." << std::endl;

    // load MNIST dataset
    std::vector<label_t> train_labels, test_labels;
    std::vector<vec_t> train_images, test_images;

    parse_mnist_labels("../../data/train-labels.idx1-ubyte", &train_labels);
    parse_mnist_images("../../data/train-images.idx3-ubyte", &train_images, -1.0, 1.0, 2, 2);
    parse_mnist_labels("../../data/t10k-labels.idx1-ubyte", &test_labels);
    parse_mnist_images("../../data/t10k-images.idx3-ubyte", &test_images, -1.0, 1.0, 2, 2);

    std::cout << "start learning" << std::endl;

    progress_display disp(train_images.size());
    timer t;
    int minibatch_size = 10;

    nn.optimizer().alpha *= std::sqrt(minibatch_size);

    // create callback
    auto on_enumerate_epoch = [&](){
        std::cout << t.elapsed() << "s elapsed." << std::endl;

        tiny_cnn::result res = nn.test(test_images, test_labels);

        std::cout << nn.optimizer().alpha << "," << res.num_success << "/" << res.num_total << std::endl;

        nn.optimizer().alpha *= 0.85; // decay learning rate
        nn.optimizer().alpha = std::max((tiny_cnn::float_t)0.00001, nn.optimizer().alpha);

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
    ofs << nn;
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

    parse_mnist_labels("../../data/train-labels.idx1-ubyte", &train_labels);
    parse_mnist_images("../../data/train-images.idx3-ubyte", &train_images, -1.0, 1.0, 0, 0);
    parse_mnist_labels("../../data/t10k-labels.idx1-ubyte", &test_labels);
    parse_mnist_images("../../data/t10k-images.idx3-ubyte", &test_images, -1.0, 1.0, 0, 0);

    nn.optimizer().alpha = 0.001;
    
    progress_display disp(train_images.size());
    timer t;

    // create callback
    auto on_enumerate_epoch = [&](){
        std::cout << t.elapsed() << "s elapsed." << std::endl;

        tiny_cnn::result res = nn.test(test_images, test_labels);

        std::cout << nn.optimizer().alpha << "," << res.num_success << "/" << res.num_total << std::endl;

        nn.optimizer().alpha *= 0.85; // decay learning rate
        nn.optimizer().alpha = std::max((tiny_cnn::float_t)0.00001, nn.optimizer().alpha);

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
        d = corrupt(move(d), 0.1, 0.0); // corrupt 10% data
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

    fully_connected_layer<tan_h> f1(input_dim, hidden_units);
    dropout_layer dropout(hidden_units, 0.5);
    fully_connected_layer<tan_h> f2(hidden_units, output_dim);
    nn << f1 << dropout << f2;

    nn.optimizer().alpha = 0.003; // TODO: not optimized
    nn.optimizer().lambda = 0.0;

    // load MNIST dataset
    std::vector<label_t> train_labels, test_labels;
    std::vector<vec_t> train_images, test_images;

    parse_mnist_labels("../../data/train-labels.idx1-ubyte", &train_labels);
    parse_mnist_images("../../data/train-images.idx3-ubyte", &train_images, -1.0, 1.0, 0, 0);
    parse_mnist_labels("../../data/t10k-labels.idx1-ubyte", &test_labels);
    parse_mnist_images("../../data/t10k-images.idx3-ubyte", &test_images, -1.0, 1.0, 0, 0);

    // load train-data, label_data
    progress_display disp(train_images.size());
    timer t;

    // create callback
    auto on_enumerate_epoch = [&](){
        std::cout << t.elapsed() << "s elapsed." << std::endl;
  
        dropout.set_context(net_phase::test);
        tiny_cnn::result res = nn.test(test_images, test_labels);
        dropout.set_context(net_phase::train);


        std::cout << nn.optimizer().alpha << "," << res.num_success << "/" << res.num_total << std::endl;

        nn.optimizer().alpha *= 0.99; // decay learning rate
        nn.optimizer().alpha = std::max((tiny_cnn::float_t)0.00001, nn.optimizer().alpha);

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
