/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include <cstdio>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "tiny_dnn/tiny_dnn.h"
#include "tiny_dnn/util/gradient_check.h"

#ifdef _WIN32
#define NOMINMAX
#include <Windows.h>
#endif

#include <sys/types.h>
#include <sys/stat.h>

namespace tiny_dnn {

template <typename Container, typename T>
inline bool is_near_container(const Container &expected,
                              const Container &actual,
                              T abs_error) {
  auto i1 = std::begin(expected);
  auto i2 = std::begin(actual);

  for (; i1 != std::end(expected); ++i1, ++i2) {
    if (std::abs(*i1 - *i2) > abs_error) return false;
  }
  return true;
}

template <typename Container>
inline bool is_different_container(const Container &expected,
                                   const Container &actual) {
  auto i1 = std::begin(expected);
  auto i2 = std::begin(actual);

  for (; i1 != std::end(expected); ++i1, ++i2) {
    if (*i1 != *i2) return true;
  }
  return false;
}

inline bool exists(const std::string &path) {
  if (FILE *file = std::fopen(path.c_str(), "r")) {
    fclose(file);
    return true;
  } else {
    return false;
  }
}

inline std::string unique_path() {
  std::string pattern;
  do {
    pattern = "%%%%-%%%%-%%%%-%%%%";
    for (auto p = pattern.begin(); p != pattern.end(); ++p) {
      // https://linux.die.net/man/3/rand_r
      // RAND_MAX == 32767
      if (*p == '%') *p = (uniform_rand(0, 32767) % 10) + '0';
    }
  } while (exists(pattern));
  // return std::experimental::filesystem::v1::temp_directory_path().string() +
  // pattern;
  char path[256];
#ifdef _WIN32
  ::GetTempPath(256, path);
#else
  // TODO(edgarriba): Almost always, snprintf is better than strcpy
  strcpy(path, "/tmp/");  // NOLINT
#endif
  // TODO(edgarriba): Almost always, snprintf is better than strcpy
  strcat(path, pattern.c_str());  // NOLINT
  struct stat buffer;
  return (stat(path, &buffer) == 0) ? unique_path() : path;
}

vec_t forward_pass(layer &src, const vec_t &vec) {
  src.setup(false);
  (*src.inputs()[0]->get_data())[0] = vec;
  src.forward();
  std::vector<const tensor_t *> out;
  src.output(out);
  return (*out[0])[0];
}

template <typename N>
vec_t forward_pass(network<N> &net, const vec_t &vec) {
  return net.predict(vec);
}

template <typename T>
void network_serialization_test(T &src, T &dst) {
  // EXPECT_FALSE(src.has_same_weights(dst, 1E-5));

  std::string tmp_file_path = unique_path();

  // write and read
  src.save(tmp_file_path);
  dst.load(tmp_file_path);

  std::remove(tmp_file_path.c_str());

  vec_t v(src.in_data_size());
  uniform_rand(v.begin(), v.end(), float_t{-1.0}, float_t{1.0});

  // a bit more than precision limit
  float_t epsilon = std::numeric_limits<float_t>::epsilon() * 2;

  EXPECT_TRUE(src.has_same_weights(dst, epsilon));

  vec_t r1 = forward_pass(src, v);
  vec_t r2 = forward_pass(dst, v);

  EXPECT_TRUE(is_near_container(r1, r2, epsilon * 10));
}

template <typename T>
void serialization_test(T &src, T &dst) {
  // EXPECT_FALSE(src.has_same_weights(dst, 1E-5));

  std::string tmp_file_path = unique_path();

  // write
  {
    std::ofstream ofs(tmp_file_path.c_str());
    src.save(ofs);
  }
  // read
  {
    std::ifstream ifs(tmp_file_path.c_str());
    dst.load(ifs);
  }

  std::remove(tmp_file_path.c_str());

  vec_t v(src.in_data_size());
  uniform_rand(v.begin(), v.end(), float_t{-1.0}, float_t{1.0});

  // a bit more than precision limit
  float_t epsilon = std::numeric_limits<float_t>::epsilon() * 2;
  EXPECT_TRUE(src.has_same_weights(dst, epsilon));

  vec_t r1 = forward_pass(src, v);
  vec_t r2 = forward_pass(dst, v);

  EXPECT_TRUE(is_near_container(r1, r2, epsilon * 10));
}

template <typename T>
void quantized_serialization_test(T &src, T &dst) {
  // EXPECT_FALSE(src.has_same_weights(dst, 1E-5));

  std::string tmp_file_path = unique_path();

  // write
  {
    std::ofstream ofs(tmp_file_path.c_str());
    src.save(ofs);
  }

  // read
  {
    std::ifstream ifs(tmp_file_path.c_str());
    dst.load(ifs);
  }

  std::remove(tmp_file_path.c_str());

  vec_t v(src.in_data_size());
  uniform_rand(v.begin(), v.end(), -1.0, 1.0);

  EXPECT_TRUE(src.has_same_weights(dst, 1E-5));

  vec_t r1 = forward_pass(src, v);
  vec_t r2 = forward_pass(dst, v);

  EXPECT_TRUE(is_near_container(r1, r2, 1E-2));
}

template <typename T>
inline T epsilon() {
  return 0;
}

template <>
inline float epsilon() {
  return 1e-2f;
}

template <>
inline double epsilon() {
  return 1e-4;
}

inline bool resolve_path(const std::string &filename, std::string &path) {
  // TODO(edgarriba): use std::vector instead
  static const char *path_list[] = {"",
                                    "./test/",
                                    "../test/",
                                    "../../test/",
                                    "../../../test/",
                                    "../../tiny-dnn/test/",
                                    "../../../tiny-dnn/test/"};

  for (size_t i = 0; i < sizeof(path_list) / sizeof(path_list[0]); i++) {
    if (exists(path_list[i] + filename)) {
      path = path_list[i] + filename;
      return true;
    }
  }
  return false;
}

std::pair<
  std::vector<tensor_t>,
  std::vector<std::vector<
    label_t>>> inline generate_gradient_check_data(size_t input_dimension,
                                                   size_t sample_count = 5,
                                                   size_t class_count  = 2) {
  const size_t input_channel_count  = 1;
  const size_t output_channel_count = 1;
  std::vector<tensor_t> a(
    sample_count, tensor_t(input_channel_count, vec_t(input_dimension, 0.0)));
  std::vector<std::vector<label_t>> t(
    sample_count, std::vector<label_t>(output_channel_count));

  for (size_t sample = 0; sample < sample_count; ++sample) {
    for (size_t input_channel = 0; input_channel < input_channel_count;
         ++input_channel) {
      vec_t &v = a[sample][input_channel];
      uniform_rand(v.begin(), v.end(), -1, 1);
    }
    for (size_t output_channel = 0; output_channel < output_channel_count;
         ++output_channel) {
      t[sample][output_channel] = sample % class_count;
    }
  }

  return std::make_pair(a, t);
}

std::vector<tensor_t> generate_test_data(const std::vector<size_t> nsamples,
                                         const std::vector<size_t> sizes) {
  assert(nsamples.size() == sizes.size());
  std::vector<tensor_t> ret(nsamples.size());
  for (size_t i = 0; i < nsamples.size(); i++) {
    ret[i].resize(nsamples[i]);
    for (size_t j = 0; j < nsamples[i]; j++) {
      ret[i][j].resize(sizes[i]);
      uniform_rand(ret[i][j].begin(), ret[i][j].end(), -1.0, 1.0);
    }
  }
  return ret;
}

#ifndef CNN_NO_SERIALIZATION
inline std::string layer_to_json(const layer &src) {
  std::ostringstream os;
  {
    cereal::JSONOutputArchive oa(os);
    layer::save_layer(oa, src);
  }
  return os.str();
}

inline std::shared_ptr<layer> json_to_layer(const std::string &src) {
  std::stringstream ss;
  ss << src;
  cereal::JSONInputArchive oa(ss);
  return layer::load_layer(oa);
}
#endif

}  // namespace tiny_dnn
