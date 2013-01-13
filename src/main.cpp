#include <iostream>
#include <boost/timer.hpp>
#include <boost/progress.hpp>

#include "tiny_cnn.h"
//#define NOMINMAX
//#include "imdebug.h"

using namespace tiny_cnn;

int main(void) {
    // construct LeNet-5 architecture
    typedef network<mse, gradient_descent> CNN;
    CNN nn;
    convolutional_layer<CNN, tanh_activation> C1(32, 32, 5, 1, 6);
    average_pooling_layer<CNN, tanh_activation> S2(28, 28, 6, 2);
    // connection table [Y.Lecun, 1998 Table.1]
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

    // load MNIST dataset
    std::vector<label_t> train_labels, test_labels;
    std::vector<vec_t> train_images, test_images;

    parse_labels("train-labels.idx1-ubyte", &train_labels);
    parse_images("train-images.idx3-ubyte", &train_images);
    parse_labels("t10k-labels.idx1-ubyte", &test_labels);
    parse_images("t10k-images.idx3-ubyte", &test_images);

    boost::progress_display disp(train_images.size());
    boost::timer t;

    // create callback
    auto on_enumerate_epoch = [&](){
        std::cout << t.elapsed() << "s elapsed." << std::endl;

        tiny_cnn::result res = nn.test(test_images, test_labels);

        std::cout << nn.learner().alpha << "," << res.num_success << "/" << res.num_total << std::endl;

        nn.learner().alpha *= 0.85;
        nn.learner().alpha = std::max(0.00001, nn.learner().alpha);

        disp.restart(train_images.size());
        t.restart();
    };

    auto on_enumerate_data = [&](){ 
		static int n = 0;
		++disp; 
	
		n++;
		if (n == 1000) {
			image img;
			C3.weight_to_image(img);
			//imdebug("lum b=8 w=%d h=%d %p", img.width(), img.height(), &img.data()[0]);
			n = 0;
		}
	};
    
    // training
    nn.init_weight();
    nn.train(train_images, train_labels, 20, on_enumerate_data, on_enumerate_epoch);

    // test and show results
    nn.test(test_images, test_labels).print_detail(std::cout);

    // save networks
    std::ofstream ofs("LeNet-weights");
    ofs << C1 << S2 << C3 << S4 << C5 << F6;
}