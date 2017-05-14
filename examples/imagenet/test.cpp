#include <iostream>
#include "tiny_dnn/tiny_dnn.h"

using namespace tiny_dnn;
using namespace tiny_dnn::activation;
using namespace std;

// rescale output to 0-100
template <typename Activation>
double rescale(double x) {
  Activation a(1);
  return 100.0 * (x - a.scale().first) / (a.scale().second - a.scale().first);
}

void convert_image(const std::string &imagefilename, int w, int h,
                   vec_t &data) {
  image<> img(imagefilename, image_type::bgr);
  image<> resized = resize_image(img, w, h);
  data.resize(resized.width() * resized.height() * resized.depth());
  float imageNetMeans[] = {103.939, 116.779, 123.68};  // BGR
  for (size_t c = 0; c < resized.depth(); ++c) {
    for (size_t y = 0; y < resized.height(); ++y) {
      for (size_t x = 0; x < resized.width(); ++x) {
        data[c * resized.width() * resized.height() + y * resized.width() + x] =
            resized[y * resized.width() + x + c] - imageNetMeans[c];
      }
    }
  }
}

void recognize(const std::string &model_name,
               const std::string &class_labels_file_name,
               const std::string &src_filename) {
  network<sequential> nn;

  // Load the model
  nn.load(model_name);

  // convert imagefile to vec_t
  vec_t data;
  convert_image(src_filename, 227, 227, data);

  // recognize
  auto res = nn.predict(data);
  vector<pair<double, int>> scores;

  // Load class labels
  ifstream class_labels_file(class_labels_file_name, ios::in);
  string line;
  vector<string> class_labels;
  while (getline(class_labels_file, line)) {
    class_labels.push_back(line);
  }

  // sort & print top-5
  for (int i = 0; i < 1000; i++)
    scores.emplace_back(rescale<tanh_layer>(res[i]), i);

  sort(scores.begin(), scores.end(), greater<pair<double, int>>());

  for (int i = 0; i < 5; i++) {
    cout << "Predicted class: " << class_labels[scores[i].second] << " ("
         << scores[i].second << ") | Confidence: " << scores[i].first << " %"
         << endl;
  }
}

int main(int argc, char **argv) {
  if (argc != 4) {
    cout << "Usage: " << argv[0]
         << " <Model file> <Classes label file> <Input image file>" << endl;
    return 0;
  }
  recognize(argv[1], argv[2], argv[3]);
}
