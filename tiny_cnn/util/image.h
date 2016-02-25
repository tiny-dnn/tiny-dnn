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

namespace tiny_cnn {

//TODO finish update this class with 'depth'
template<typename T = unsigned char>
class image {
public:
    typedef T intensity_t;

    image() : width_(0), height_(0), depth_(1) {}

    image(const T* data, size_t width, size_t height) : width_(width), height_(height), depth_(1), data_(depth_ * width_ * height_, 0) 
    {
        memcpy(&data_[0],data, depth_ * width * height*sizeof(T));
    }

    image(index3d<cnn_size_t> rhs) : width_(rhs.width_), height_(rhs.height_), depth_(rhs.depth_), data_(depth_ * width_ * height_, 0) {}

    image(size_t width, size_t height) : width_(width), height_(height), depth_(1), data_(width * height, 0) {}

    image(const image& rhs) : width_(rhs.width_), height_(rhs.height_), depth_(rhs.depth_), data_(rhs.data_) {}

    image(const image&& rhs) : width_(rhs.width_), height_(rhs.height_), depth_(rhs.depth_), data_(std::move(rhs.data_)) {}

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

    void write(const std::string& path) const { // WARNING: This is OS dependent (writes of bytes with reinterpret_cast depend on endianness)
        std::ofstream ofs(path.c_str(), std::ios::binary | std::ios::out);

        if (!is_little_endian())
            throw nn_error("image::write for bit-endian is not supported");

        const uint32_t line_pitch = ((width_ + 3) / 4) * 4;
        const uint32_t header_size = 14 + 12 + 256 * 3;
        const uint32_t data_size = line_pitch * height_;
        
        // file header(14 byte)
        const uint16_t file_type = ('M' << 8) | 'B';
        const uint32_t file_size = header_size + data_size;
        const uint32_t reserved = 0;
        const uint32_t offset_bytes = header_size;

        ofs.write(reinterpret_cast<const char*>(&file_type), 2);
        ofs.write(reinterpret_cast<const char*>(&file_size), 4);
        ofs.write(reinterpret_cast<const char*>(&reserved), 4);
        ofs.write(reinterpret_cast<const char*>(&offset_bytes), 4);

        // info header(12byte)
        const uint32_t info_header_size = 12;
        const int16_t width = static_cast<int16_t>(width_);
        const int16_t height = static_cast<int16_t>(height_);
        const uint16_t planes = 1;
        const uint16_t bit_count = 8;

        ofs.write(reinterpret_cast<const char*>(&info_header_size), 4);
        ofs.write(reinterpret_cast<const char*>(&width), 2);
        ofs.write(reinterpret_cast<const char*>(&height), 2);
        ofs.write(reinterpret_cast<const char*>(&planes), 2);
        ofs.write(reinterpret_cast<const char*>(&bit_count), 2);

        // color palette (256*3byte)
        for (int i = 0; i < 256; i++) {
            const auto v = static_cast<const char>(i);
            ofs.write(&v, 1);//R
            ofs.write(&v, 1);//G
            ofs.write(&v, 1);//B
        }

        // data
        for (size_t i = 0; i < height_; i++) {
            ofs.write(reinterpret_cast<const char*>(&data_[(height_ - 1 - i) * width_]), width_);
            if (line_pitch != width_) {
                uint32_t dummy = 0;
                ofs.write(reinterpret_cast<const char*>(&dummy), line_pitch - width_);
            }
        }
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

    intensity_t& operator[](std::size_t idx)       { return data_[idx]; };
    const intensity_t& operator[](std::size_t idx) const { return data_[idx]; };

    size_t width() const { return width_; }
    size_t height() const { return height_; }
    size_t depth() const {return depth_;}
    const std::vector<intensity_t>& data() const { return data_; }
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

} // namespace tiny_cnn
