/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

#include "tiny_dnn/tiny_dnn.h"
#define INPUT_SIZE 300
#define N_ANCHORS 8732
#define N_CLASSES 21
#define BG_CLASS_ID 0
#define NMS_THRESHOLD 0.5
#define MEAN_B 123
#define MEAN_G 117
#define MEAN_R 104

void convert_image(const std::string& imagefilename,
                   int w,
                   int h,
                   tiny_dnn::vec_t& data) {
  const int MEAN_BGR[] = {MEAN_B, MEAN_G, MEAN_R};

  tiny_dnn::image<> img(imagefilename, tiny_dnn::image_type::bgr);
  tiny_dnn::image<> resized = resize_image(img, w, h);
  data                      = resized.to_vec();

  size_t spatial_size = resized.height() * resized.width();
  for (size_t c = 0; c < resized.depth(); ++c) {
    for (size_t i = 0; i < spatial_size; ++i) {
      data[c * spatial_size + i] -= MEAN_BGR[c];
    }
  }
}

void concat_hwc_features(tiny_dnn::vec_t& collections,
                         size_t n_collection_items,
                         tiny_dnn::vec_t& feature,
                         size_t in_spatial_size,
                         size_t in_channels) {
  tiny_dnn::vec_t t_feature;
  t_feature.resize(feature.size());

  // transpose features to HxWxC
  for (size_t i = 0; i < in_spatial_size; ++i) {
    for (size_t j = 0; j < in_channels; ++j) {
      t_feature[i * in_channels + j] = feature[j * in_spatial_size + i];
    }
  }

  // Append features to vectors
  for (size_t i = 0; i < t_feature.size(); ++i) {
    collections[n_collection_items + i] = t_feature[i];
  }
}

void inline_softmax(tiny_dnn::vec_t& confidences) {
  for (size_t i = 0; i < N_ANCHORS; ++i) {
    float sum = 0;
    for (size_t j = 0; j < N_CLASSES; ++j) {
      sum += exp(confidences[i * N_CLASSES + j]);
    }

    for (size_t j = 0; j < N_CLASSES; ++j) {
      confidences[i * N_CLASSES + j] =
        exp(confidences[i * N_CLASSES + j]) / sum;
    }
  }
}

void save_default_boxes(tiny_dnn::vec_t& default_boxes,
                        size_t box_index,
                        float cx,
                        float cy,
                        float width,
                        float height) {
  default_boxes[box_index * 4]     = cx;
  default_boxes[box_index * 4 + 1] = cy;
  default_boxes[box_index * 4 + 2] = width;
  default_boxes[box_index * 4 + 3] = height;
}

void init_default_boxes(tiny_dnn::vec_t& default_boxes) {
  float steps[] = {8.0 / INPUT_SIZE,  16.0 / INPUT_SIZE,  32.0 / INPUT_SIZE,
                   64.0 / INPUT_SIZE, 100.0 / INPUT_SIZE, 300.0 / INPUT_SIZE};
  float sizes[] = {30.0 / INPUT_SIZE,  60.0 / INPUT_SIZE,  111.0 / INPUT_SIZE,
                   162.0 / INPUT_SIZE, 213.0 / INPUT_SIZE, 264.0 / INPUT_SIZE,
                   315.0 / INPUT_SIZE};
  size_t feature_map_sizes[]                    = {38, 19, 10, 5, 3, 1};
  std::vector<std::vector<float>> aspect_ratios = {{2},    {2, 3}, {2, 3},
                                                   {2, 3}, {2},    {2}};

  const size_t N_FEATURES = 6;
  size_t box_index        = 0;
  for (size_t i = 0; i < N_FEATURES; ++i) {
    size_t fm_size = feature_map_sizes[i];

    for (size_t h = 0; h < fm_size; ++h) {
      for (size_t w = 0; w < fm_size; ++w) {
        float cx = (w + 0.5) * steps[i];
        float cy = (h + 0.5) * steps[i];

        float s = sizes[i];
        save_default_boxes(default_boxes, box_index++, cx, cy, s, s);

        s = sqrt(sizes[i] * sizes[i + 1]);
        save_default_boxes(default_boxes, box_index++, cx, cy, s, s);

        s = sizes[i];
        for (float ar : aspect_ratios[i]) {
          save_default_boxes(default_boxes, box_index++, cx, cy, s * sqrt(ar),
                             s / sqrt(ar));
          save_default_boxes(default_boxes, box_index++, cx, cy, s / sqrt(ar),
                             s * sqrt(ar));
        }
      }
    }
  }
}

void construct_nets(std::vector<tiny_dnn::network<tiny_dnn::sequential>>& nets,
                    const std::string& modelFolder) {
  using conv   = tiny_dnn::convolutional_layer;
  using pool   = tiny_dnn::max_pooling_layer;
  using relu   = tiny_dnn::relu_layer;
  using l2norm = tiny_dnn::l2_normalization_layer;
  using pad    = tiny_dnn::zero_pad_layer;

  tiny_dnn::network<tiny_dnn::sequential> nn1;
  nn1 << conv(300, 300, 3, 3, 64, tiny_dnn::padding::same)             // vgg.0
      << relu() << conv(300, 300, 3, 64, 64, tiny_dnn::padding::same)  // vgg.2
      << relu() << pool(300, 300, 64, 2, 2, false)
      << conv(150, 150, 3, 64, 128, tiny_dnn::padding::same)  // vgg.5
      << relu()
      << conv(150, 150, 3, 128, 128, tiny_dnn::padding::same)  // vgg.7
      << relu() << pool(150, 150, 128, 2, 2, false)
      << conv(75, 75, 3, 128, 256, tiny_dnn::padding::same)            // vgg.10
      << relu() << conv(75, 75, 3, 256, 256, tiny_dnn::padding::same)  // vgg.12
      << relu() << conv(75, 75, 3, 256, 256, tiny_dnn::padding::same)  // vgg.14
      << relu() << pool(75, 75, 256, 2, 2, true)
      << conv(38, 38, 3, 256, 512, tiny_dnn::padding::same)            // vgg.17
      << relu() << conv(38, 38, 3, 512, 512, tiny_dnn::padding::same)  // vgg.19
      << relu() << conv(38, 38, 3, 512, 512, tiny_dnn::padding::same)  // vgg.21
      << relu() << l2norm(38 * 38, 512, 1e-12, 20);
  nets.push_back(nn1);

  tiny_dnn::network<tiny_dnn::sequential> nn2;
  nn2 << pool(38, 38, 512, 2, 2, false)
      << conv(19, 19, 3, 512, 512, tiny_dnn::padding::same)            // vgg.24
      << relu() << conv(19, 19, 3, 512, 512, tiny_dnn::padding::same)  // vgg.26
      << relu() << conv(19, 19, 3, 512, 512, tiny_dnn::padding::same)  // vgg.28
      << relu() << pad(19, 19, 512, 1, 1) << pool(21, 21, 512, 3, 1, true)
      << pad(19, 19, 512, 6, 6)
      << conv(31, 31, 3, 512, 1024, tiny_dnn::padding::valid, true, 1, 1, 6,
              6)  // vgg.31
      << relu()
      << conv(19, 19, 1, 1024, 1024, tiny_dnn::padding::same)  // vgg.33
      << relu();
  nets.push_back(nn2);

  tiny_dnn::network<tiny_dnn::sequential> nn3;
  nn3 << conv(19, 19, 1, 1024, 256, tiny_dnn::padding::same)  // extra.0
      << relu() << pad(19, 19, 256, 1, 1)
      << conv(21, 21, 3, 256, 512, tiny_dnn::padding::valid, true, 2,
              2)  // extra.1
      << relu();
  nets.push_back(nn3);

  tiny_dnn::network<tiny_dnn::sequential> nn4;
  nn4 << conv(10, 10, 1, 512, 128, tiny_dnn::padding::same)  // extra.2
      << relu() << pad(10, 10, 128, 1, 1)
      << conv(12, 12, 3, 128, 256, tiny_dnn::padding::valid, true, 2,
              2)  // extra.3
      << relu();
  nets.push_back(nn4);

  tiny_dnn::network<tiny_dnn::sequential> nn5;
  nn5 << conv(5, 5, 1, 256, 128, tiny_dnn::padding::same)             // extra.4
      << relu() << conv(5, 5, 3, 128, 256, tiny_dnn::padding::valid)  // extra.5
      << relu();
  nets.push_back(nn5);

  tiny_dnn::network<tiny_dnn::sequential> nn6;
  nn6 << conv(3, 3, 1, 256, 128, tiny_dnn::padding::same)             // extra.6
      << relu() << conv(3, 3, 3, 128, 256, tiny_dnn::padding::valid)  // extra.7
      << relu();
  nets.push_back(nn6);

  // Locations
  tiny_dnn::network<tiny_dnn::sequential> nn7;
  nn7 << conv(38, 38, 3, 512, 16, tiny_dnn::padding::same);  // loc.0
  nets.push_back(nn7);

  tiny_dnn::network<tiny_dnn::sequential> nn8;
  nn8 << conv(19, 19, 3, 1024, 24, tiny_dnn::padding::same);  // loc.1
  nets.push_back(nn8);

  tiny_dnn::network<tiny_dnn::sequential> nn9;
  nn9 << conv(10, 10, 3, 512, 24, tiny_dnn::padding::same);  // loc.2
  nets.push_back(nn9);

  tiny_dnn::network<tiny_dnn::sequential> nn10;
  nn10 << conv(5, 5, 3, 256, 24, tiny_dnn::padding::same);  // loc.3
  nets.push_back(nn10);

  tiny_dnn::network<tiny_dnn::sequential> nn11;
  nn11 << conv(3, 3, 3, 256, 16, tiny_dnn::padding::same);  // loc.4
  nets.push_back(nn11);

  tiny_dnn::network<tiny_dnn::sequential> nn12;
  nn12 << conv(1, 1, 3, 256, 16, tiny_dnn::padding::same);  // loc.5
  nets.push_back(nn12);

  // Confidences
  tiny_dnn::network<tiny_dnn::sequential> nn13;
  nn13 << conv(38, 38, 3, 512, 4 * N_CLASSES,
               tiny_dnn::padding::same);  // conf.0
  nets.push_back(nn13);

  tiny_dnn::network<tiny_dnn::sequential> nn14;
  nn14 << conv(19, 19, 3, 1024, 6 * N_CLASSES,
               tiny_dnn::padding::same);  // conf.1
  nets.push_back(nn14);

  tiny_dnn::network<tiny_dnn::sequential> nn15;
  nn15 << conv(10, 10, 3, 512, 6 * N_CLASSES,
               tiny_dnn::padding::same);  // conf.2
  nets.push_back(nn15);

  tiny_dnn::network<tiny_dnn::sequential> nn16;
  nn16 << conv(5, 5, 3, 256, 6 * N_CLASSES, tiny_dnn::padding::same);  // conf.3
  nets.push_back(nn16);

  tiny_dnn::network<tiny_dnn::sequential> nn17;
  nn17 << conv(3, 3, 3, 256, 4 * N_CLASSES, tiny_dnn::padding::same);  // conf.4
  nets.push_back(nn17);

  tiny_dnn::network<tiny_dnn::sequential> nn18;
  nn18 << conv(1, 1, 3, 256, 4 * N_CLASSES, tiny_dnn::padding::same);  // conf.5
  nets.push_back(nn18);

  for (size_t i = 0; i < 18; ++i) {
    std::ostringstream modelPath;
    modelPath << modelFolder << std::setfill('0') << std::setw(2) << i + 1
              << ".weights";
    std::ifstream ifs(modelPath.str());
    if (ifs.fail()) {
      std::cout << "Failed to load weights from " << modelPath.str()
                << std::endl;
    } else {
      std::cout << "Loading weights from " << modelPath.str() << std::endl;
    }
    ifs >> nets[i];
  }
}

void detect(std::vector<tiny_dnn::network<tiny_dnn::sequential>>& nets,
            const std::string& src_filename) {
  // convert imagefile to vec_t
  tiny_dnn::vec_t img;
  convert_image(src_filename, INPUT_SIZE, INPUT_SIZE, img);

  // multi-scale features
  auto feature1 = nets[0].predict(img);
  auto feature2 = nets[1].predict(feature1);
  auto feature3 = nets[2].predict(feature2);
  auto feature4 = nets[3].predict(feature3);
  auto feature5 = nets[4].predict(feature4);
  auto feature6 = nets[5].predict(feature5);

  // locations
  auto loc_feature1 = nets[6].predict(feature1);   // 16x38x38
  auto loc_feature2 = nets[7].predict(feature2);   // 24x19x19
  auto loc_feature3 = nets[8].predict(feature3);   // 24x10x10
  auto loc_feature4 = nets[9].predict(feature4);   // 24x5x5
  auto loc_feature5 = nets[10].predict(feature5);  // 16x3x3
  auto loc_feature6 = nets[11].predict(feature6);  // 16x1x1

  // locations
  auto conf_feature1 = nets[12].predict(feature1);  // (4*n_classes)x38x38
  auto conf_feature2 = nets[13].predict(feature2);  // (6*n_classes)x19x19
  auto conf_feature3 = nets[14].predict(feature3);  // (6*n_classes)x10x10
  auto conf_feature4 = nets[15].predict(feature4);  // (6*n_classes)x5x5
  auto conf_feature5 = nets[16].predict(feature5);  // (4*n_classes)x3x3
  auto conf_feature6 = nets[17].predict(feature6);  // (4*n_classes)x1x1

  tiny_dnn::vec_t locations;
  size_t n_location_items = 0;
  locations.resize(N_ANCHORS * 4);
  concat_hwc_features(locations, n_location_items, loc_feature1, 38 * 38, 16);
  n_location_items += loc_feature1.size();
  concat_hwc_features(locations, n_location_items, loc_feature2, 19 * 19, 24);
  n_location_items += loc_feature2.size();
  concat_hwc_features(locations, n_location_items, loc_feature3, 10 * 10, 24);
  n_location_items += loc_feature3.size();
  concat_hwc_features(locations, n_location_items, loc_feature4, 5 * 5, 24);
  n_location_items += loc_feature4.size();
  concat_hwc_features(locations, n_location_items, loc_feature5, 3 * 3, 16);
  n_location_items += loc_feature5.size();
  concat_hwc_features(locations, n_location_items, loc_feature6, 1 * 1, 16);
  n_location_items += loc_feature6.size();

  tiny_dnn::vec_t default_boxes;
  default_boxes.resize(N_ANCHORS * 4);
  init_default_boxes(default_boxes);

  tiny_dnn::vec_t bounding_boxes;
  bounding_boxes.resize(N_ANCHORS * 4);
  for (size_t i = 0; i < N_ANCHORS; ++i) {
    // regress center x and center y for bounding boxes
    float cx =
      locations[i * 4] * 0.1 * default_boxes[i * 4 + 2] + default_boxes[i * 4];
    float cy = locations[i * 4 + 1] * 0.1 * default_boxes[i * 4 + 3] +
               default_boxes[i * 4 + 1];
    // regress width and height for bounding boxes
    float width  = exp(locations[i * 4 + 2] * 0.2) * default_boxes[i * 4 + 2];
    float height = exp(locations[i * 4 + 3] * 0.2) * default_boxes[i * 4 + 3];

    bounding_boxes[i * 4]     = cx - width / 2;
    bounding_boxes[i * 4 + 1] = cy - height / 2;
    bounding_boxes[i * 4 + 2] = cx + width / 2;
    bounding_boxes[i * 4 + 3] = cy + height / 2;
  }

  tiny_dnn::vec_t confidences;
  size_t n_confidence_items = 0;
  confidences.resize(N_ANCHORS * N_CLASSES);
  concat_hwc_features(confidences, n_confidence_items, conf_feature1, 38 * 38,
                      4 * N_CLASSES);
  n_confidence_items += conf_feature1.size();
  concat_hwc_features(confidences, n_confidence_items, conf_feature2, 19 * 19,
                      6 * N_CLASSES);
  n_confidence_items += conf_feature2.size();
  concat_hwc_features(confidences, n_confidence_items, conf_feature3, 10 * 10,
                      6 * N_CLASSES);
  n_confidence_items += conf_feature3.size();
  concat_hwc_features(confidences, n_confidence_items, conf_feature4, 5 * 5,
                      6 * N_CLASSES);
  n_confidence_items += conf_feature4.size();
  concat_hwc_features(confidences, n_confidence_items, conf_feature5, 3 * 3,
                      4 * N_CLASSES);
  n_confidence_items += conf_feature5.size();
  concat_hwc_features(confidences, n_confidence_items, conf_feature6, 1 * 1,
                      4 * N_CLASSES);
  n_confidence_items += conf_feature6.size();
  // Softmax
  inline_softmax(confidences);

  // Get class labels for bounding boxes
  std::vector<int> bounding_box_indexes;
  std::vector<int> bounding_box_classes;
  std::vector<float> bounding_box_confidences;
  for (size_t i = 0; i < N_ANCHORS; ++i) {
    int max_conf_id = -1;
    float max_conf  = -1;
    for (size_t j = 0; j < N_CLASSES; ++j) {
      float conf = confidences[i * N_CLASSES + j];
      if (conf > max_conf) {
        max_conf_id = j;
        max_conf    = conf;
      }
    }
    if (max_conf_id != BG_CLASS_ID) {
      bounding_box_indexes.push_back(i);
      bounding_box_classes.push_back(max_conf_id);
      bounding_box_confidences.push_back(max_conf);
    }
  }

  // Get valid bounding boxes
  std::vector<tiny_dnn::bounding_box> bounding_box_candidates;
  for (size_t i = 0; i < bounding_box_indexes.size(); ++i) {
    int bbox_index = bounding_box_indexes[i];

    tiny_dnn::bounding_box bbox;
    bbox.x_min = bounding_boxes[bbox_index * 4] * INPUT_SIZE;
    bbox.y_min = bounding_boxes[bbox_index * 4 + 1] * INPUT_SIZE;
    bbox.x_max = bounding_boxes[bbox_index * 4 + 2] * INPUT_SIZE;
    bbox.y_max = bounding_boxes[bbox_index * 4 + 3] * INPUT_SIZE;
    bbox.score = bounding_box_confidences[i];
    bounding_box_candidates.push_back(bbox);
  }

  // Print coordinates of bounding boxes
  bounding_box_indexes = tiny_dnn::nms(bounding_box_candidates, NMS_THRESHOLD);
  if (bounding_box_indexes.size()) {
    std::cout << "Bounding box coordinates:" << std::endl;
  } else {
    std::cout << "No targets detected." << std::endl;
  }
  for (size_t i = 0; i < bounding_box_indexes.size(); ++i) {
    int bbox_index              = bounding_box_indexes[i];
    tiny_dnn::bounding_box bbox = bounding_box_candidates[bbox_index];

    std::cout << "x_min = " << bbox.x_min << ", "
              << "x_max = " << bbox.x_max << ", "
              << "y_min = " << bbox.y_min << ", "
              << "y_max = " << bbox.y_max << ", "
              << "class = " << bounding_box_classes[bbox_index] << ", "
              << "score = " << bounding_box_confidences[bbox_index]
              << std::endl;
  }
}

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cout << "Usage: example_ssd_test model_folder_path img_file_path";
    return -1;
  }
  std::vector<tiny_dnn::network<tiny_dnn::sequential>> nets;

  construct_nets(nets, argv[1]);
  detect(nets, argv[2]);
}
