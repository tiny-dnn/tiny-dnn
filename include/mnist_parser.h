#pragma once
#include "util.h"
#include <fstream>
#include <cstdint>
#include <boost/detail/endian.hpp>

namespace tiny_cnn {

void parse_labels(const std::string& label_file, std::vector<label_t> *labels) {
    std::ifstream ifs(label_file.c_str(), std::ios::in | std::ios::binary);

    if (ifs.bad() || ifs.fail())
        throw nn_error("failed to open file:" + label_file);

    uint32_t magic_number, num_items;

    ifs.read((char*)&magic_number, 4);
    ifs.read((char*)&num_items, 4);
#if defined(BOOST_LITTLE_ENDIAN)
    reverse_endian(&magic_number);
    reverse_endian(&num_items);
#endif

    if (magic_number != 0x00000801 || num_items <= 0) 
        throw nn_error("MNIST label-file format error");

    for (size_t i = 0; i < num_items; i++) {
        uint8_t label;
        ifs.read((char*)&label, 1);
        labels->push_back((label_t)label);
    }
}

void parse_images(const std::string& image_file, std::vector<vec_t> *images, float_t scale_min = -1.0, float_t scale_max = 1.0, int x_padding = 2, int y_padding = 2) {
    std::ifstream ifs(image_file.c_str(), std::ios::in | std::ios::binary);

    if (ifs.bad() || ifs.fail())
        throw nn_error("failed to open file:" + image_file);

    uint32_t magic_number, num_items, num_rows, num_cols;

    ifs.read((char*)&magic_number, 4);
    ifs.read((char*)&num_items, 4);
    ifs.read((char*)&num_rows, 4);
    ifs.read((char*)&num_cols, 4);
#if defined(BOOST_LITTLE_ENDIAN)
    reverse_endian(&magic_number);
    reverse_endian(&num_items);
    reverse_endian(&num_rows);
    reverse_endian(&num_cols);
#endif

    if (magic_number != 0x00000803 || num_items <= 0) 
        throw nn_error("MNIST label-file format error");

    const int width = num_cols + 2 * x_padding;
    const int height = num_rows + 2 * y_padding;

    for (size_t i = 0; i < num_items; i++) {
        vec_t image(width * height, scale_min);
        std::vector<uint8_t> image_vec(num_rows * num_cols);

        ifs.read((char*)&image_vec[0], num_rows * num_cols);

        for (size_t y = 0; y < num_rows; y++)
            for (size_t x = 0; x < num_cols; x++)
            image[width * (y + y_padding) + x + x_padding]
              = (image_vec[y * num_cols + x] / 255.0) * (scale_max - scale_min) + scale_min;

        images->push_back(image);
    }
}


}
