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
#include <memory>
#define CNN_USE_CAFFE_CONVERTER
#include "tiny_cnn/tiny_cnn.h"

using namespace tiny_cnn;
using namespace tiny_cnn::activation;
using namespace std;

#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

cv::Mat compute_mean(const string& mean_file, int width, int height)
{
    caffe::BlobProto blob;
    detail::read_proto_from_binary(mean_file, &blob);

    vector<cv::Mat> channels;
    auto data = blob.mutable_data()->mutable_data();

    for (int i = 0; i < blob.channels(); i++, data += blob.height() * blob.width())
        channels.emplace_back(blob.height(), blob.width(), CV_32FC1, data);

    cv::Mat mean;
    cv::merge(channels, mean);

    return cv::Mat(cv::Size(width, height), mean.type(), cv::mean(mean));
}

cv::ColorConversionCodes get_cvt_codes(int src_channels, int dst_channels)
{
    assert(src_channels != dst_channels);

    if (dst_channels == 3)
        return src_channels == 1 ? cv::COLOR_GRAY2BGR : cv::COLOR_BGRA2BGR;
    else if (dst_channels == 1)
        return src_channels == 3 ? cv::COLOR_BGR2GRAY : cv::COLOR_BGRA2GRAY;
    else
        throw runtime_error("unsupported color code");
}

void preprocess(const cv::Mat& img,
                const cv::Mat& mean,
                int num_channels,
                cv::Size geometry,
                vector<cv::Mat>* input_channels)
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

vector<string> get_label_list(const string& label_file)
{
    string line;
    ifstream ifs(label_file.c_str());

    if (ifs.fail() || ifs.bad())
        throw runtime_error("failed to open:" + label_file);

    vector<string> lines;
    while (getline(ifs, line))
        lines.push_back(line);

    return lines;
}

void test(const string& model_file,
          const string& trained_file,
          const string& mean_file,
          const string& label_file,
          const string& img_file)
{
    auto labels = get_label_list(label_file);
    auto net = create_net_from_caffe_prototxt(model_file);
    reload_weight_from_caffe_protobinary(trained_file, net.get());

    int channels = net->in_shape().depth_;
    int width = net->in_shape().width_;
    int height = net->in_shape().height_;

    cv::Mat img = cv::imread(img_file, -1);

    auto mean = compute_mean(mean_file, width, height);

    vector<float> inputvec(width*height*channels);
    vector<cv::Mat> input_channels;

    for (int i = 0; i < channels; i++)
        input_channels.emplace_back(height, width, CV_32FC1, &inputvec[width*height*i]);

    preprocess(img, mean, 3, cv::Size(width, height), &input_channels);

    vector<tiny_cnn::float_t> vec(inputvec.begin(), inputvec.end());

    auto result = net->predict(vec);
    vector<tiny_cnn::float_t> sorted(result.begin(), result.end());

    int top_n = 5;
    partial_sort(sorted.begin(), sorted.begin()+top_n, sorted.end(), greater<tiny_cnn::float_t>());

    for (int i = 0; i < top_n; i++) {
        size_t idx = distance(result.begin(), find(result.begin(), result.end(), sorted[i]));
        cout << labels[idx] << "," << sorted[i] << endl;
    }
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
    } catch (const nn_error& e) {
        cout << e.what() << endl;
    }
}
