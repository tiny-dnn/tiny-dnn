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
#include <array>
#include "tiny_dnn/util/util.h"

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable:4996) // suppress warnings about using fopen
#endif

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_INLINE // We need this define to avoid multiple definition
#include "third_party/stb/stb_image.h"

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#define STB_IMAGE_RESIZE_INLINE
#include "third_party/stb/stb_image_resize.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_WRITE_INLINE
#include "third_party/stb/stb_image_write.h"


namespace tiny_dnn {

namespace detail {

template <typename T>
typename std::enable_if<std::is_unsigned<T>::value, T>::type saturated_sub(T s1, T s2) {
    return s1 > s2 ? static_cast<T>(s1 - s2) : 0;
}

template <typename T>
typename std::enable_if<!std::is_unsigned<T>::value, T>::type saturated_sub(T s1, T s2) {
    return static_cast<T>(s1 - s2);
}

inline bool ends_with(std::string const & value, std::string const & ending) {
    if (ending.size() > value.size()) return false;
    return std::equal(ending.rbegin(), ending.rend(), value.rbegin());
}

inline void resize_image_core(const uint8_t* src, int srcw, int srch, uint8_t* dst, int dstw, int dsth, int channels)
{
    stbir_resize_uint8(src, srcw, srch, 0, dst, dstw, dsth, 0, channels);
}

inline void resize_image_core(const float* src, int srcw, int srch, float* dst, int dstw, int dsth, int channels)
{
    stbir_resize_float(src, srcw, srch, 0, dst, dstw, dsth, 0, channels);
}

} // namespace detail

enum class image_type {
    grayscale,    ///< load image and convert automatically to 8-bit grayscale
    rgb, ///< load image and keep original color channels
    bgr
};

/**
 * Simple image utility class
 */
template<typename T = unsigned char>
class image {
public:
    typedef T intensity_t;
    typedef typename std::vector<intensity_t>::iterator iterator;
    typedef typename std::vector<intensity_t>::const_iterator const_iterator;

    image() : width_(0), height_(0), depth_(1) {}

    /**
     * create image from raw pointer
     */
    image(const T* data, size_t width, size_t height, image_type type)
        : width_(width), height_(height), depth_(type == image_type::grayscale ? 1: 3), type_(type), data_(depth_ * width_ * height_, 0)
    {
        std::copy(data, data + width * height * depth_, &data_[0]);
    }

    /**
     * create WxHxD image filled with 0
     */
    image(const shape3d& size, image_type type)
        : width_(size.width_), height_(size.height_), depth_(size.depth_),
          type_(type),
          data_(depth_ * width_ * height_, 0){
        if (type == image_type::grayscale && size.depth_ != 1) {
            throw nn_error("depth must be 1 in grayscale");
        }
        else if (type != image_type::grayscale && size.depth_ != 3) {
            throw nn_error("depth must be 3 in rgb/bgr");
        }
    }

    template <typename U>
    image(const image<U>& rhs) : width_(rhs.width()), height_(rhs.height()), depth_(rhs.depth()), type_(rhs.type()), data_(rhs.shape().size()) {
        std::transform(rhs.begin(), rhs.end(), data_.begin(), [](T src) { return static_cast<intensity_t>(src); });
    }

    /**
     * create image from file
     * supported file format: JPEG/PNG/TGA/BMP/PSD/GIF/HDR/PIC/PNM
     *                        (see detail at the comments in thrid_party/stb/stb_image.h)
     */
    image(const std::string& filename, image_type type)
    {
        int w, h, d;
        stbi_uc* input_pixels = stbi_load(filename.c_str(), &w, &h, &d, type == image_type::grayscale ? 1 : 3);
        if (input_pixels == nullptr) {
            throw nn_error("failed to open image:" + std::string(stbi_failure_reason()));
        }

        width_  = static_cast<size_t>(w);
        height_ = static_cast<size_t>(h);
        depth_  = type == image_type::grayscale ? 1 : 3;
        type_ = type;

        data_.resize(width_*height_*depth_);

        // reorder to HxWxD -> DxHxW
        from_rgb(input_pixels, input_pixels + data_.size());
   
        stbi_image_free(input_pixels);
    }

    void save(const std::string& path) const {
        int ret;
        std::vector<uint8_t> buf = to_rgb<uint8_t>();

        if (detail::ends_with(path, "png")) {
            ret = stbi_write_png(path.c_str(),
                                 static_cast<int>(width_),
                                 static_cast<int>(height_),
                                 static_cast<int>(depth_),
                                 (const void*)&buf[0], 0);
        }
        else {
            ret = stbi_write_bmp(path.c_str(),
                                 static_cast<int>(width_),
                                 static_cast<int>(height_),
                                 static_cast<int>(depth_),
                                 (const void*)&buf[0]);
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
        data_.resize(width * height * depth_);
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

    bool empty() const { return data_.empty(); }
    iterator begin() { return data_.begin(); }
    iterator end() { return data_.end();  }
    const_iterator begin() const { return data_.begin(); }
    const_iterator end() const { return data_.end(); }

    intensity_t& operator[](std::size_t idx)       { return data_[idx]; };
    const intensity_t& operator[](std::size_t idx) const { return data_[idx]; };

    size_t width() const { return width_; }
    size_t height() const { return height_; }
    size_t depth() const {return depth_;}
    image_type type() const { return type_; }
    shape3d shape() const {
        return shape3d(static_cast<serial_size_t>(width_),
                       static_cast<serial_size_t>(height_),
                       static_cast<serial_size_t>(depth_));
    }
    const std::vector<intensity_t>& data() const { return data_; }
    vec_t to_vec() const { return vec_t(begin(), end()); }

    template <typename U>
    std::vector<U> to_rgb() const {
        if (depth_ == 1) {
            return std::vector<U>(data_.begin(), data_.end());
        }
        else {
            std::vector<U> buf(shape().size());
            auto order = depth_order(type_);
            auto dst = buf.begin();

            for (size_t y = 0; y < height_; y++)
                for (size_t x = 0; x < width_; x++)
                    for (size_t i = 0; i < depth_; i++)
                        *dst++ = static_cast<U>(at(x, y, order[i]));
            return buf;
        }
    }

    template <typename Iter>
    void from_rgb(Iter begin, Iter end) { 
        if (depth_ == 1) {
            std::copy(begin, end, data_.begin());
        }
        else {
            auto order = depth_order(type_);
            assert(static_cast<serial_size_t>(
                std::distance(begin, end)) == data_.size());

            for (size_t y = 0; y < height_; y++)
                for (size_t x = 0; x < width_; x++)
                    for (size_t i = 0; i < depth_; i++)
                        at(x, y, order[i]) = static_cast<intensity_t>(*begin++);
        }
    }

private:
    std::array<size_t, 3> depth_order(image_type img) const {
        if (img == image_type::rgb) {
            return{ {0,1,2} };
        }
        else {
            assert(img == image_type::bgr);
            return{ {2,1,0 } };
        }
    }
    size_t width_;
    size_t height_;
    size_t depth_;
    image_type type_;
    std::vector<intensity_t> data_;
};

template <typename T>
image<float_t> mean_image(const image<T>& src)
{
    image<float_t> mean(shape3d(1, 1, (serial_size_t)src.depth()), src.type());

    for (size_t i = 0; i < src.depth(); i++) {
        float_t sum = 0.0f;
        for (size_t y = 0; y < src.height(); y++) {
            for (size_t x = 0; x < src.width(); x++) {
                sum += src.at(x, y, i);
            }
        }
        mean.at(0, 0, i) = sum / (src.width() * src.height());
    }

    return mean;
}

/**
 * resize image into width x height
 * This function use Mitchell-Netrevalli filter with B=1/3, C=1/3 for downsampling, and
 * and cubic spline algorithm for upsampling.
 */
template <typename T>
inline image<T> resize_image(const image<T>& src, int width, int height)
{
    image<T> resized(shape3d(static_cast<serial_size_t>(width),
                             static_cast<serial_size_t>(height),
                             static_cast<serial_size_t>(src.depth())),
                     src.type());
    std::vector<T> src_rgb = src.template to_rgb<T>();
    std::vector<T> dst_rgb(resized.shape().size());

    detail::resize_image_core(&src_rgb[0],
                              static_cast<int>(src.width()),
                              static_cast<int>(src.height()),
                              &dst_rgb[0],
                              width,
                              height,
                              static_cast<int>(src.depth()));

    resized.from_rgb(dst_rgb.begin(), dst_rgb.end());

    return resized;
}


// dst[x,y,d] = lhs[x,y,d] - rhs[x,y,d]
template <typename T>
image<T> subtract_image(const image<T>& lhs, const image<T>& rhs)
{
    if (lhs.shape() != rhs.shape()) {
        throw nn_error("Shapes of lhs/rhs must be same. lhs:" + to_string(lhs.shape()) + ",rhs:" + to_string(rhs.shape()));
    }

    image<T> dst(lhs.shape(), lhs.type());

    auto dstit = dst.begin();
    auto lhsit = lhs.begin();
    auto rhsit = rhs.begin();

    for (; dstit != dst.end(); ++dstit, ++lhsit, ++rhsit) {
        *dstit = detail::saturated_sub(*lhsit, *rhsit);
    }
    return dst;
}

template <typename T>
image<T> subtract_scalar(const image<T>& lhs, const image<T>& rhs)
{
    if (lhs.depth() != rhs.depth()) {
        throw nn_error("Depth of lhs/rhs must be same. lhs:" + to_string(lhs.depth()) + ",rhs:" + to_string(rhs.depth()));
    }
    if (rhs.width() != 1 || rhs.height() != 1) {
        throw nn_error("rhs must be 1x1xN");
    }

    image<T> dst(lhs.shape(), lhs.type());

    auto dstit = dst.begin();
    auto lhsit = lhs.begin();
    auto rhsit = rhs.begin();

    for (size_t i = 0; i < lhs.depth(); i++, ++rhsit) {
        for (size_t j = 0; j < lhs.width() * lhs.height(); j++, ++dstit, ++lhsit) {
            *dstit = detail::saturated_sub(*lhsit, *rhsit);
        }
    }

    return dst;
}

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
inline image<T> vec2image(const vec_t& vec, serial_size_t block_size = 2, serial_size_t max_cols = 20)
{
    if (vec.empty())
        throw nn_error("failed to visialize image: vector is empty");

    image<T> img;
    const serial_size_t border_width = 1;
    const auto cols = vec.size() >= (serial_size_t)max_cols ? (serial_size_t)max_cols : vec.size();
    const auto rows = (vec.size() - 1) / cols + 1;
    const auto pitch = block_size + border_width;
    const auto width = pitch * cols + border_width;
    const auto height = pitch * rows + border_width;
    const typename image<T>::intensity_t bg_color = 255;
    serial_size_t current_idx = 0;

    img.resize(width, height);
    img.fill(bg_color);

    auto minmax = std::minmax_element(vec.begin(), vec.end());

    for (unsigned int r = 0; r < rows; r++) {
        serial_size_t topy = pitch * r + border_width;

        for (unsigned int c = 0; c < cols; c++, current_idx++) {
            serial_size_t leftx = pitch * c + border_width;
            const float_t src = vec[current_idx];
            image<>::intensity_t dst
                = static_cast<typename image<T>::intensity_t>(rescale(src, *minmax.first, *minmax.second, 0, 255));

            for (serial_size_t y = 0; y < block_size; y++)
              for (serial_size_t x = 0; x < block_size; x++)
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
inline image<T> vec2image(const vec_t& vec, const index3d<serial_size_t>& maps) {
    if (vec.empty())
        throw nn_error("failed to visualize image: vector is empty");
    if (vec.size() != maps.size())
        throw nn_error("failed to visualize image: vector size invalid");

    const serial_size_t border_width = 1;
    const auto pitch = maps.width_ + border_width;
    const auto width = maps.depth_ * pitch + border_width;
    const auto height = maps.height_ + 2 * border_width;
    const typename image<T>::intensity_t bg_color = 255;
    image<T> img;

    img.resize(width, height);
    img.fill(bg_color);

    auto minmax = std::minmax_element(vec.begin(), vec.end());

    for (serial_size_t c = 0; c < maps.depth_; ++c) {
        const auto top = border_width;
        const auto left = c * pitch + border_width;

        for (serial_size_t y = 0; y < maps.height_; ++y) {
            for (serial_size_t x = 0; x < maps.width_; ++x) {
                const float_t val = vec[maps.get_index(x, y, c)];

                img.at(left + x, top + y)
                    = static_cast<typename image<T>::intensity_t>(rescale(val, *minmax.first, *minmax.second, 0, 255));
            }
        }
    }
    return img;
}

} // namespace tiny_dnn

#ifdef _MSC_VER
#pragma warning(pop)
#endif
