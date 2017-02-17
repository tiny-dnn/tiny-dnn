#include <iostream>
#include "tiny_dnn/tiny_dnn.h"

using namespace tiny_dnn;
using namespace tiny_dnn::activation;
using namespace std;

// rescale output to 0-100
template <typename Activation>
double rescale(double x) {
  Activation a;
  return 100.0 * (x - a.scale().first) / (a.scale().second - a.scale().first);
}

void convert_image(const std::string &imagefilename,
                   double minv,
                   double maxv,
                   int w,
                   int h,
                   vec_t &data) {
  image<> img(imagefilename, image_type::rgb);
  image<> resized = resize_image(img, w, h);
  data.resize(resized.width() * resized.height() * resized.depth());
  for (size_t c = 0; c < resized.depth(); ++c) {
    for (size_t y = 0; y < resized.height(); ++y) {
      for (size_t x = 0; x < resized.width(); ++x) {
        data[c * resized.width() * resized.height() + y * resized.width() + x] =
          (maxv - minv) * (resized[y * resized.width() + x + c]) / 255.0 + minv;
      }
    }
  }
}

template <typename N>
void construct_net(N &nn) {
  typedef convolutional_layer<activation::identity> conv;
  typedef max_pooling_layer<relu> pool;

  const int n_fmaps  = 32;  ///< number of feature maps for upper layer
  const int n_fmaps2 = 64;  ///< number of feature maps for lower layer
  const int n_fc     = 64;  ///< number of hidden units in fully-connected layer

  nn << conv(32, 32, 5, 3, n_fmaps, padding::same) << pool(32, 32, n_fmaps, 2)
     << conv(16, 16, 5, n_fmaps, n_fmaps, padding::same)
     << pool(16, 16, n_fmaps, 2)
     << conv(8, 8, 5, n_fmaps, n_fmaps2, padding::same)
     << pool(8, 8, n_fmaps2, 2)
     << fully_connected_layer<activation::identity>(4 * 4 * n_fmaps2, n_fc)
     << fully_connected_layer<softmax>(n_fc, 10);
}

void recognize(const std::string &dictionary, const std::string &src_filename) {
  network<sequential> nn;

  construct_net(nn);

  // load nets
  ifstream ifs(dictionary.c_str());
  ifs >> nn;

  // convert imagefile to vec_t
  vec_t data;
  convert_image(src_filename, -1.0, 1.0, 32, 32, data);

  // recognize
  auto res = nn.predict(data);
  vector<pair<double, int>> scores;

  // sort & print top-3
  for (int i = 0; i < 10; i++) scores.emplace_back(rescale<tan_h>(res[i]), i);

  sort(scores.begin(), scores.end(), greater<pair<double, int>>());

  for (int i = 0; i < 3; i++)
    cout << scores[i].second << "," << scores[i].first << endl;
}

int main(int argc, char **argv) {
  if (argc != 2) {
    cout << "please specify image file";
    return 0;
  }
  recognize("cifar-weights", argv[1]);
}
