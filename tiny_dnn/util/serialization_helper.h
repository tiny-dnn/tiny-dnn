/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <typeindex>

#include <cereal/archives/json.hpp>
#include <cereal/types/memory.hpp>

#include "tiny_dnn/layers/layers.h"
#include "tiny_dnn/util/macro.h"
#include "tiny_dnn/util/nn_error.h"
#include "tiny_dnn/util/serialization_functions.h"
#include "tiny_dnn/util/serialization_layer_list.h"

namespace tiny_dnn {

template <typename OutputArchive>
class serialization_helper {
 public:
  void register_saver(
    const std::string &name,
    std::function<void(OutputArchive &, const layer *)> func) {
    savers_[name] = [=](void *ar, const layer *l) {
      return func(*reinterpret_cast<OutputArchive *>(ar), l);
    };
  }

  template <typename T>
  void register_type(const std::string &name) {
    type_names_[typeid(T)] = name;
  }

  void save(const std::string &layer_name, OutputArchive &ar, const layer *l) {
    check_if_enabled();

    if (savers_.find(layer_name) == savers_.end()) {
      throw nn_error("Failed to save layer. Saver for " + layer_name +
                     " is not found.\n"
                     "Please use CNN_REGISTER_LAYER macro to register "
                     "appropriate saver.");
    }

    savers_[layer_name](reinterpret_cast<void *>(&ar), l);
  }

  const std::string &type_name(std::type_index index) const {
    if (type_names_.find(index) == type_names_.end()) {
      throw nn_error("Typename is not registered");
    }
    return type_names_.at(index);
  }

  static serialization_helper &get_instance() {
    static serialization_helper instance;
    return instance;
  }

 private:
  void check_if_enabled() const {
#ifdef CNN_NO_SERIALIZATION
    static_assert(
      sizeof(OutputArchive) == 0,
      "You are using save functions, but serialization function is "
      "disabled "
      "in current configuration.\n\n"
      "You need to undef CNN_NO_SERIALIZATION to enable these functions.\n"
      "If you are using cmake, you can use -DUSE_SERIALIZER=ON "
      "option.\n\n");
#endif
  }

  /** layer-type -> generator  */
  std::map<std::string, std::function<void(void *, const layer *)>> savers_;

  std::map<std::type_index, std::string> type_names_;

  template <typename T>
  static void save_layer_impl(OutputArchive &oa, const layer *layer);

  template <typename T>
  friend void register_layers(T *h);

  template <typename T>
  void register_layer(const char *layer_name) {
    register_type<T>(layer_name);
    register_saver(layer_name, save_layer_impl<T>);
  }

  serialization_helper() { register_layers(this); }
};  // class serialization_helper

template <typename OutputArchive>
template <typename T>
void serialization_helper<OutputArchive>::save_layer_impl(OutputArchive &oa,
                                                          const layer *layer) {
  oa(cereal::make_nvp(
    serialization_helper<OutputArchive>::get_instance().type_name(typeid(T)),
    *dynamic_cast<const T *>(layer)));
}

template <typename OutputArchive>
void layer::save_layer(OutputArchive &oa, const layer &l) {
  const std::string &name =
    serialization_helper<OutputArchive>::get_instance().type_name(typeid(l));
  serialization_helper<OutputArchive>::get_instance().save(name, oa, &l);
}

template <class Archive>
void layer::serialize_prolog(Archive &ar) {
  std::string name =
    serialization_helper<Archive>::get_instance().type_name(typeid(*this));
  ar(cereal::make_nvp("type", name));
}

}  // namespace tiny_dnn
