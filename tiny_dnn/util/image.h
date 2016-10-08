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
#pragma once
#include <vector>
#include <fstream>
#include <cstdint>
#include <algorithm>
#include "tiny_dnn/util/util.h"

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_STATIC // We need this define to avoid multiple definition
#include "third_party/stb/stb_image.h"

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#define STB_IMAGE_RESIZE_STATIC
#include "third_party/stb/stb_image_resize.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_WRITE_STATIC
#include "third_party/stb/stb_image_write.h"

namespace tiny_dnn {

inline bool ends_with(std::string const & value, std::string const & ending) {
    if (ending.size() > value.size()) return false;
    return std::equal(ending.rbegin(), ending.rend(), value.rbegin());
}

enum class image_load_type {
    grayscale,    ///< load image and convert automatically to 8-bit grayscale
    keep_original ///< load image and keep original color channels
};

/**
 * Simple image utility class
 */
template<typename T = unsigned char>
class image {
public:
    typedef uint8_t intensity_t;
    typedef typename std::vector<intensity_t>::iterator iterator;
    typedef typename std::vector<intensity_t>::const_iterator const_iterator;

    image() : width_(0), height_(0), depth_(1) {}

    /**
     * create image from raw pointer
     */
    image(const T* data, size_t width, size_t height) : width_(width), height_(height), depth_(1), data_(depth_ * width_ * height_, 0) 
    {
        memcpy(&data_[0],data, depth_ * width * height*sizeof(T));
    }

    /**
     * create WxHxD image filled with 0
     */
    image(const shape3d& size) : width_(size.width_), height_(size.height_), depth_(size.depth_), data_(depth_ * width_ * height_, 0) {}

    /**
     * create WxH image filled with 0
     */
    image(size_t width, size_t height) : width_(width), height_(height), depth_(1), data_(width * height, 0) {}

    image(const image& rhs) : width_(rhs.width_), height_(rhs.height_), depth_(rhs.depth_), data_(rhs.data_) {}

    image(const image&& rhs) : width_(rhs.width_), height_(rhs.height_), depth_(rhs.depth_), data_(std::move(rhs.data_)) {}

    /**
     * create image from file
     * supported file format: JPEG/PNG/TGA/BMP/PSD/GIF/HDR/PIC/PNM
     *                        (see detail at the comments in thrid_party/stb/stb_image.h)
     */
    image(const std::string& filename, image_load_type load_type = image_load_type::grayscale)
    {
        stbi_uc* input_pixels = stbi_load(filename.c_str(), &width_, &height_, &depth_, load_type == image_load_type::grayscale ? 1 : 0);
        if (input_pixels == nullptr) {
            throw nn_error("failed to open image:" + std::string(stbi_failure_reason()));
        }

        data_.resize(width_*height_*depth_);
        std::copy(input_pixels, input_pixels + data_.size(), data_.begin());

        stbi_image_free(input_pixels);
    }

    /**
     * create image from file with specific size
     * supported file format: JPEG/PNG/TGA/BMP/PSD/GIF/HDR/PIC/PNM
     *                        (see detail at the comments in thrid_party/stb/stb_image.h)
     */
    image(const std::string& filename, size_t width, size_t height)
    {
        int srcw, srch, depth;
        stbi_uc* input_pixels = stbi_load(filename.c_str(), &srcw, &srch, &depth, 1);
        if (input_pixels == nullptr) {
            throw nn_error("failed to open image:" + filename);
        }
        depth_ = 1;
        data_.resize(width*height);

        if (!stbir_resize_uint8(input_pixels, srcw, srch, 0, &data_[0], width, height, 0, 1)) {
            throw nn_error("failed to resize image");
        }

        stbi_image_free(input_pixels);
    }

    image& operator = (const image& rhs) {
        width_ = rhs.width_;
        height_ = rhs.height_;
        depth_ = rhs.depth_;
        data_ = rhs.data_;
        return *this;
    }

    image& operator = (const image&& rhs) {
        width_ = rhs.width_;
        height_ = rhs.height_;
        depth_ = rhs.depth_;
        data_ = std::move(rhs.data_);
        return *this;
    }

    void save(const std::string& path) const {
        int ret;
        if (ends_with(path, "png")) {
            ret = stbi_write_png(path.c_str(), width_, height_, depth_, (const void*)&data_[0], 0);
        }
        else {
            ret = stbi_write_bmp(path.c_str(), width_, height_, depth_, (const void*)&data_[0]);
        }
        if (ret == 0) {
            throw nn_error("failed to save image:" + path);
        }
    }

    void write(const std::string& path) const {
        save(path);
    }

    void resize(size_t width, size_t height) 
    {
        data_.resize(width * height);
        width_ = width;
        height_ = height;
        //depth_ = depth;
    }

    void fill(intensity_t value) {
        std::fill(data_.begin(), data_.end(), value);
    }

    intensity_t& at(size_t x, size_t y, size_t z = 0) {
        assert(x < width_);
        assert(y < height_);
        assert(z < depth_);
        return data_[z * width_ * height_ + y * width_ + x];
    }

    const intensity_t& at(size_t x, size_t y, size_t z = 0) const {
        assert(x < width_);
        assert(y < height_);
        assert(z < depth_);
        return data_[z * width_ * height_ + y * width_ + x];
    }

    iterator begin() { return data_.begin(); }
    iterator end() { return data_.end();  }
    const_iterator begin() const { return data_.begin(); }
    const_iterator end() const { return data_.end(); }

    intensity_t& operator[](std::size_t idx)       { return data_[idx]; };
    const intensity_t& operator[](std::size_t idx) const { return data_[idx]; };

    size_t width() const { return width_; }
    size_t height() const { return height_; }
    size_t depth() const {return depth_;}
    const std::vector<intensity_t>& data() const { return data_; }

    vec_t to_vec() const {
        return vec_t(data_.begin(), data_.end());
    }

private:
    size_t width_;
    size_t height_;
    size_t depth_;
    std::vector<intensity_t> data_;
};

/**
 * visualize 1d-vector
 *
 * @example
 *
 * vec:[1,5,3]
 *
 * img:
 *   ----------
 *   -11-55-33-
 *   -11-55-33-
 *   ----------
 **/
template<typename T>
inline image<T> vec2image(const vec_t& vec, cnn_size_t block_size = 2, cnn_size_t max_cols = 20)
{
    if (vec.empty())
        throw nn_error("failed to visialize image: vector is empty");

    image<T> img;
    const cnn_size_t border_width = 1;
    const auto cols = vec.size() >= (cnn_size_t)max_cols ? (cnn_size_t)max_cols : vec.size();
    const auto rows = (vec.size() - 1) / cols + 1;
    const auto pitch = block_size + border_width;
    const auto width = pitch * cols + border_width;
    const auto height = pitch * rows + border_width;
    const typename image<T>::intensity_t bg_color = 255;
    cnn_size_t current_idx = 0;

    img.resize(width, height);
    img.fill(bg_color);

    auto minmax = std::minmax_element(vec.begin(), vec.end());

    for (unsigned int r = 0; r < rows; r++) {
        cnn_size_t topy = pitch * r + border_width;

        for (unsigned int c = 0; c < cols; c++, current_idx++) {
            cnn_size_t leftx = pitch * c + border_width;
            const float_t src = vec[current_idx];
            image<>::intensity_t dst
                = static_cast<typename image<T>::intensity_t>(rescale(src, *minmax.first, *minmax.second, 0, 255));

            for (cnn_size_t y = 0; y < block_size; y++)
              for (cnn_size_t x = 0; x < block_size; x++)
                img.at(x + leftx, y + topy) = dst;

            if (current_idx == vec.size()) return img;
        }
    }
    return img;
}

/**
 * visualize 1d-vector
 *
 * @example
 *
 * vec:[5,2,1,3,6,3,0,9,8,7,4,2] maps:[width=2,height=3,depth=2]
 *
 * img:
 *  -------
 *  -52-09-
 *  -13-87-
 *  -63-42-
 *  -------
 **/
template<typename T>
inline image<T> vec2image(const vec_t& vec, const index3d<cnn_size_t>& maps) {
    if (vec.empty())
        throw nn_error("failed to visualize image: vector is empty");
    if (vec.size() != maps.size())
        throw nn_error("failed to visualize image: vector size invalid");

    const cnn_size_t border_width = 1;
    const auto pitch = maps.width_ + border_width;
    const auto width = maps.depth_ * pitch + border_width;
    const auto height = maps.height_ + 2 * border_width;
    const typename image<T>::intensity_t bg_color = 255;
    image<T> img;

    img.resize(width, height);
    img.fill(bg_color);

    auto minmax = std::minmax_element(vec.begin(), vec.end());

    for (cnn_size_t c = 0; c < maps.depth_; ++c) {
        const auto top = border_width;
        const auto left = c * pitch + border_width;

        for (cnn_size_t y = 0; y < maps.height_; ++y) {
            for (cnn_size_t x = 0; x < maps.width_; ++x) {
                const float_t val = vec[maps.get_index(x, y, c)];

                img.at(left + x, top + y)
                    = static_cast<typename image<T>::intensity_t>(rescale(val, *minmax.first, *minmax.second, 0, 255));
            }
        }
    }
    return img;
}

} // namespace tiny_dnn
