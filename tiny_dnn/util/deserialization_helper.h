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
    * Neither the name of the tiny-dnn nor the
    names of its contributors may be used to endorse or promote products
    derived from this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
   AND ANY
    EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
   FOR ANY
    DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
    LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
    ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
   THIS
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
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

template <typename InputArchive>
class deserialization_helper {
 public:
  void register_loader(
    const std::string &name,
    std::function<std::shared_ptr<layer>(InputArchive &)> func) {
    loaders_[name] = [=](void *ar) {
      return func(*reinterpret_cast<InputArchive *>(ar));
    };
  }

  template <typename T>
  void register_type(const std::string &name) {
    type_names_[typeid(T)] = name;
  }

  std::shared_ptr<layer> load(const std::string &layer_name, InputArchive &ar) {
    check_if_enabled();

    if (loaders_.find(layer_name) == loaders_.end()) {
      throw nn_error("Failed to load layer. Loader for " + layer_name +
                     " is not found.\n"
                     "Please use CNN_REGISTER_LAYER macro to register "
                     "appropriate loader.");
    }

    return loaders_[layer_name](reinterpret_cast<void *>(&ar));
  }

  const std::string &type_name(std::type_index index) const {
    if (type_names_.find(index) == type_names_.end()) {
      throw nn_error("Typename is not registered");
    }
    return type_names_.at(index);
  }

  static deserialization_helper &get_instance() {
    static deserialization_helper instance;
    return instance;
  }

 private:
  void check_if_enabled() const {
#ifdef CNN_NO_SERIALIZATION
    static_assert(
      sizeof(InputArchive) == 0,
      "You are using load functions, but deserialization function is "
      "disabled in current configuration.\n\n"
      "You need to undef CNN_NO_SERIALIZATION to enable these functions.\n"
      "If you are using cmake, you can use -DUSE_SERIALIZER=ON "
      "option.\n\n");
#endif
  }

  /** layer-type -> generator  */
  std::map<std::string, std::function<std::shared_ptr<layer>(void *)>> loaders_;

  std::map<std::type_index, std::string> type_names_;

  template <typename T>
  static std::shared_ptr<layer> load_layer_impl(InputArchive &ia);

  template <typename T>
  friend void register_layers(T *h);

  template <typename T>
  void register_layer(const char *layer_name) {
    register_loader(layer_name, load_layer_impl<T>);
    register_type<T>(layer_name);
  }

  deserialization_helper() { register_layers(this); }

};  // class deserialization_helper

template <typename InputArchive>
template <typename T>
std::shared_ptr<layer> deserialization_helper<InputArchive>::load_layer_impl(
  InputArchive &ia) {
  using ST = typename std::aligned_storage<sizeof(T), alignof(T)>::type;

  std::unique_ptr<ST> bn(new ST());

  cereal::memory_detail::LoadAndConstructLoadWrapper<InputArchive, T> wrapper(
    reinterpret_cast<T *>(bn.get()));

  wrapper.CEREAL_SERIALIZE_FUNCTION_NAME(ia);

  std::shared_ptr<layer> t;
  t.reset(reinterpret_cast<T *>(bn.get()));
  bn.release();

  return t;
}

template <typename T>
void start_loading_layer(T &ar) {
  CNN_UNREFERENCED_PARAMETER(ar);
}

template <typename T>
void finish_loading_layer(T &ar) {
  CNN_UNREFERENCED_PARAMETER(ar);
}

inline void start_loading_layer(cereal::JSONInputArchive &ia) {
  ia.startNode();
}

inline void finish_loading_layer(cereal::JSONInputArchive &ia) {
  ia.finishNode();
}

/**
* generate layer from cereal's Archive
**/
template <typename InputArchive>
std::shared_ptr<layer> layer::load_layer(InputArchive &ia) {
  start_loading_layer(ia);

  std::string p;
  ia(cereal::make_nvp("type", p));
  auto l = deserialization_helper<InputArchive>::get_instance().load(p, ia);

  finish_loading_layer(ia);

  return l;
}

}  // namespace tiny_dnn
