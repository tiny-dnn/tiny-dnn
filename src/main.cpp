#include <iostream>
#include <iomanip>
#include <map>

#include "network.h"
#include "fully_connected_layer.h"
#include "convolutional_layer.h"

#include "pooling_layer.h"
#include "mnist_parser.h"

#include "fixed_point.h"

#include <boost/timer.hpp>
#include <boost/progress.hpp>

using namespace nn;

int main(void) {
    typedef network<mse, gradient_descent> CNN;
    CNN nn;
    convolutional_layer<CNN, tanh_activation> C1(32, 32, 5, 1, 6);
    average_pooling_layer<CNN, tanh_activation> S2(28, 28, 6, 2);
    static const bool connection[] = {
        true,  false, false, false, true,  true,  true,  false, false, true,  true,  true,  true,  false, true,  true,
        true,  true,  false, false, false, true,  true,  true,  false, false, true,  true,  true,  true,  false, true,
        true,  true,  true,  false, false, false, true,  true,  true,  false, false, true,  false, true,  true,  true,
        false, true,  true,  true,  false, false, true,  true,  true,  true,  false, false, true,  false, true,  true,
        false, false, true,  true,  true,  false, false, true,  true,  true,  true,  false, true,  true,  false, true,
        false, false, false, true,  true,  true,  false, false, true,  true,  true,  true,  false, true,  true,  true
    };
    convolutional_layer<CNN, tanh_activation> C3(14, 14, 5, 6, 16, connection_table(connection, 6, 16));
    average_pooling_layer<CNN, tanh_activation> S4(10, 10, 16, 2);
    convolutional_layer<CNN, tanh_activation> C5(5, 5, 5, 16, 120);
    fully_connected_layer<CNN, tanh_activation> F6(120, 10);

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
    nn.init_weight();
 
    std::vector<label_t> train_labels, test_labels;
    std::vector<vec_t> train_images, test_images;

    parse_labels("train-labels.idx1-ubyte", &train_labels);
    parse_images("train-images.idx3-ubyte", &train_images);
    parse_labels("t10k-labels.idx1-ubyte", &test_labels);
    parse_images("t10k-images.idx3-ubyte", &test_images);

    boost::progress_display disp(train_images.size());
    boost::timer t;

    auto on_enumerate_epoch = [&](){
        std::cout << t.elapsed() << "s elapsed." << std::endl;

        nn::result res = nn.test(test_images, test_labels);

        std::cout << nn.learner().alpha << "," << res.num_success << "/" << res.num_total << std::endl;

        nn.learner().alpha *= 0.85;
        nn.learner().alpha = std::max(0.00001, nn.learner().alpha);

        disp.restart(train_images.size());
        t.restart();
    };

    auto on_enumerate_data = [&]() { ++disp; };

    nn.train(train_images, train_labels, 20, on_enumerate_data, on_enumerate_epoch);
    
    /*int success = 0;

    std::map<int, std::map<int, int> > confusion_matrix;

    for (size_t i = 0; i < test_labels.size(); i++) {
        vec_t out;
        nn.predict(test_images[i], &out);

        const label_t predicted = max_index(out);
        const label_t actual = test_labels[i];

        confusion_matrix[predicted][actual]++;
        if (predicted == actual) success++;
    }

    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            std::cout << std::setw(5) << confusion_matrix[i][j] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "result:" << success << "/" << test_labels.size();*/
}