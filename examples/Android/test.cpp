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
//#define _CRT_SECURE_NO_WARNINGS
//#include "picotest/picotest.h"
//#include "tiny_cnn/tiny_cnn.h"
//
//using namespace tiny_cnn::activation;
//#include "test_network.h"
//#include "test_average_pooling_layer.h"
//#include "test_dropout_layer.h"
//#include "test_max_pooling_layer.h"
//#include "test_fully_connected_layer.h"
//#include "test_convolutional_layer.h"
//#include "test_lrn_layer.h"
//#include "test_target_cost.h"
//#include "test_large_thread_count.h"
//
//
//int main(void) {
//    return RUN_ALL_TESTS();
//}

#include <iostream>
#include <opencv2/opencv.hpp>
/*#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>*/
#include "tiny_cnn/tiny_cnn.h"

using namespace tiny_cnn;
using namespace tiny_cnn::activation;
using namespace std;


typedef void *PredictorHandle;

// rescale output to 0-100
template <typename Activation>
double rescale(double x) {
	Activation a;
	return 100.0 * (x - a.scale().first) / (a.scale().second - a.scale().first);
}

// convert tiny_cnn::image to cv::Mat and resize
//cv::Mat image2mat(image<>& img) {
//
//	cv::Mat ori(img.height(), img.width(), CV_8U, &img.at(0, 0));
//	cv::Mat resized;
//	cv::resize(ori, resized, cv::Size(), 3 , 3 , cv::INTER_AREA);
//	return resized;
//}

//void convert_image(const std::string& imagefilename,
//	double scale,
//	int w,
//	int h,
//	vector<vec_t>& data ) {
//	auto img = cv::imread(imagefilename, cv::IMREAD_GRAYSCALE);
//	if (img.data == nullptr) return; // cannot open, or it's not an image
//
//	cv::Mat_<uint8_t> resized;
//	cv::resize(img, resized , cv::Size(w, h));
//
//	vec_t d;
//	// mnist dataset is "white on black", so negate required
//	std::transform(resized.begin(), resized.end(), std::back_inserter(d),
//		[=](uint8_t c) { return c * scale; });
//	data.push_back(d);
//}

void convert_image(cv::Mat* img,
	double scale ,
	int w,
	int h,
	vec_t& data){

	//auto img = cv::imread(imagefilename, cv::IMREAD_GRAYSCALE);
	if (img->data == nullptr) return; // cannot open, or it's not an image

	cv::Mat_<uint8_t> resized;
	cv::resize(*img, resized, cv::Size(w, h));

	// mnist dataset is "white on black", so negate required
	std::transform(resized.begin(), resized.end(), std::back_inserter(data),
		[=](uint8_t c) { return (255 - c) * (1 - (-1)) / 255.0 + -1; });
}

void convert_image(const std::string& imagefilename ,

	double minv,
	double maxv,
	int w,
	int h,
	vec_t& data) {
	auto img = cv::imread(imagefilename, cv::IMREAD_GRAYSCALE);
	if (img.data == nullptr) return; // cannot open, or it's not an image

	cv::Mat_<uint8_t> resized;
	cv::resize(img, resized, cv::Size(w, h));

	// mnist dataset is "white on black", so negate required
	std::transform(resized.begin(), resized.end(), std::back_inserter(data),
		[=](uint8_t c) { return (255 - c) * (maxv - minv) / 255.0 + minv; });
}


void construct_net(network<sequential>& nn) {

	// connection table [Y.Lecun, 1998 Table.1]
#define O true
#define X false
	static const bool tbl[] = {
		O, X, O, O, O, O, O, O, O, O, O, X, O, O, O, O,
		O, X, X, X, O, O, O, X, X, O, O, O, X, X, O, O,
		O, O, X, X, X, O, O, O, X, X, O, O, X, X, X, O,
		O, O, O, X, X, X, O, O, O, X, X, O, X, X, O, O,
		O, O, O, O, X, X, O, O, O, O, X, X, O, X, X, O,
		O, X, O, O, O, X, X, O, O, O, O, X, O, O, X, O,
	};
#undef O
#undef X

	// construct nets
	nn << convolutional_layer<tan_h>(40, 40, 3, 1, 6)
		<< average_pooling_layer<tan_h>(38, 38, 6, 2)
		<< convolutional_layer<tan_h>(19, 19, 4, 6, 16,
		connection_table(tbl, 6, 16))
		<< average_pooling_layer<tan_h>(16, 16, 16, 2)
		<< convolutional_layer<tan_h>(8, 8, 3, 16, 16)
		<< fully_connected_layer<tan_h>(16 * 6 * 6, 64)
		<< fully_connected_layer<relu>(64, 2);
}


void createPredictor(
	const string& trained_file,
	PredictorHandle& handle
	)
{
	static network<sequential> nn;
	construct_net(nn);
	// load nets
	ifstream ifs(trained_file.c_str());
	ifs >> nn;
	handle = &nn;
}

void recognize(const std::string& dictionary, cv::Mat * img) {

	network<sequential> nn;
	//<mse, adagrad> nn;
	construct_net(nn);
	// load nets
	ifstream ifs(dictionary.c_str());
	ifs >> nn;
	// convert imagefile to vec_t
	vec_t data;
	convert_image(img, 1.0, 40 , 40 , data);
	// recognize
	auto res = nn.predict(data);
	vector<pair<double, int>> scores;

	// sort & print top-3
	for (int i = 0; i < 10; i++)
		scores.emplace_back(rescale<tan_h>(res[i]), i);

	sort(scores.begin(), scores.end(), greater<pair<double, int>>());

	for (int i = 0; i < 10; i++)
		cout << scores[i].second << "," << scores[i].first << endl;
	// visualize outputs of each layer
	//for (size_t i = 0; i < nn.depth(); i++) {
	//	auto out_img = nn[i]->output_to_image();
	//	cv::imshow("layer:" + std::to_string(i), image2mat(out_img));
	//}
	// visualize filter shape of first convolutional layer
	//auto weight = nn.at<convolutional_layer<tan_h>>(0).weight_to_image();
	//cv::imshow("weights:", image2mat(weight));
	cv::waitKey(0);
}



void recognize(const std::string& dictionary, const std::string& filename) {

	network<sequential>  nn;

	construct_net(nn);
	// load nets
	ifstream ifs(dictionary.c_str());
	ifs >> nn;
	PredictorHandle  handle;
	handle = &nn;

	long h = (long)handle;

	PredictorHandle handle2 = (PredictorHandle)h;

	auto xx = (network<sequential> *)(handle2);

	// convert imagefile to vec_t
	vec_t data;
	convert_image(filename, -1.0, 1.0 , 32, 32, data);
	// recognize
	auto res = xx->predict(data);
	vector<pair<double, int>> scores;

	// sort & print top-3
	for (int i = 0; i < 10; i++)
		scores.emplace_back(rescale<tan_h>(res[i]), i);

	sort(scores.begin(), scores.end(), greater<pair<double, int>>());

	for (int i = 0; i < 3; i++)
		cout << scores[i].second << "," << scores[i].first << endl;

	cv::waitKey(0);
}

int getResult(long h ,
	cv::Mat * img)
{
	//("getResult1");
	PredictorHandle handle = (PredictorHandle)h;
	auto nn = (network<sequential> *)(handle);
	//debug("getResult2");
	vec_t data;
	convert_image(img , 1.0 , 40 , 40 , data);
	//convert_image(image_path, -1.0, 1.0, 32, 32, data);
	//debug("getResult3");
	// recognize
	auto res = nn->predict(data);

	//debug("getResult4");
	vector<pair<double, int> > scores;
	// sort & print top-3
	for (int i = 0; i < 2; i++)
		scores.emplace_back(rescale<tan_h>(res[i]), i);
	sort(scores.begin(), scores.end(), greater<pair<double, int>>());
	for (int i = 0; i < 2 ; i++)
	{
		//debug("result = %d , %f", scores[i].second, scores[i].first);
	}
	// visualize outputs of each layer
	//for (size_t i = 0; i < nn.depth(); i++) {
	//	auto out_img = nn[i]->output_to_image();
	//	cv::imshow("layer:" + std::to_string(i), image2mat(out_img));
	//}
	// visualize filter shape of first convolutional layer
	//auto weight = nn.at<convolutional_layer<tan_h>>(0).weight_to_image();
	//cv::imshow("weights:", image2mat(weight));
	return scores[0].second;
}


int main(int argc, char** argv)
{
	//if (argc != 2) {
	//	cout << "please specify image file";
	//	return 0;
	//}
	cout << "hehe" << endl;

	PredictorHandle pre = 0;

	char * image_path = "/home/jaychou/Downloads/gender_test_data/1.png";
	char * train_path = "/home/jaychou/Downloads/gender_test_data/carlogo";

	cv::Mat mat = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
	
	cv::imshow("hehe" , mat);
	cv::waitKey(0);
	//recognize("E:\\tiny-cnn-master\\vc\\vc12\\LeNet-weights", &mat);
	//recognize(train_path , image_path);

	long addr;

	createPredictor(train_path , pre);

	addr = (long)pre;

	cout << getResult(addr, &mat);

	//long addr;
	//cv::Mat* m = (cv::Mat *)(addr);
}
