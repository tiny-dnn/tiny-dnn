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
#include <typeindex>
#include <map>
#include <functional>
#include <memory>
#include <string>
#include <cereal/archives/json.hpp>
#include <cereal/types/memory.hpp>
#include "tiny_dnn/util/nn_error.h"
#include "tiny_dnn/util/macro.h"
#include "tiny_dnn/layers/layers.h"

namespace tiny_dnn {

template <typename OutputArchive>
class serialization_helper {
public:
    void register_saver(const std::string& name, std::function<void(OutputArchive&, const layer*)> func) {
        savers_[name] = [=](void* ar, const layer* l) {
            return func(*reinterpret_cast<OutputArchive*>(ar), l);
        };
    }

    template <typename T>
    void register_type(const std::string& name) {
        type_names_[typeid(T)] = name;
    }

    void save(const std::string& layer_name, OutputArchive & ar, const layer *l) {
        check_if_enabled();

        if (savers_.find(layer_name) == savers_.end()) {
            throw nn_error("Failed to generate layer. Generator for " + layer_name + " is not found.\n"
                "Please use CNN_REGISTER_LAYER_DESERIALIZER macro to register appropriate generator");
        }

        savers_[layer_name](reinterpret_cast<void*>(&ar), l);
    }

    const std::string& type_name(std::type_index index) const {
        if (type_names_.find(index) == type_names_.end()) {
            throw nn_error("Typename is not registered");
        }
        return type_names_.at(index);
    }

    static serialization_helper& get_instance() {
        static serialization_helper instance;
        return instance;
    }

private:
    void check_if_enabled() const {
#ifdef CNN_NO_SERIALIZATION
        static_assert(sizeof(OutputArchive)==0,
                             "You are using save functions, but serialization function is disabled in current configuration.\n\n"
                             "You need to undef CNN_NO_SERIALIZATION to enable these functions.\n"
                             "If you are using cmake, you can use -DUSE_SERIALIZER=ON option.\n\n");
#endif
    }

    /** layer-type -> generator  */
    std::map<std::string, std::function<void(void*, const layer*)>> savers_;

    std::map<std::type_index, std::string> type_names_;
    
    template <typename T>
    static void save_layer_impl(OutputArchive& oa, const layer* layer);

#define CNN_REGISTER_LAYER_BODY(layer_type, layer_name) \
    register_type<layer_type>(layer_name);\
    register_saver(layer_name, save_layer_impl<layer_type>)

#define CNN_REGISTER_LAYER(layer_type, layer_name) CNN_REGISTER_LAYER_BODY(layer_type, #layer_name)

#define CNN_REGISTER_LAYER_WITH_ACTIVATION(layer_type, activation_type, layer_name) \
CNN_REGISTER_LAYER_BODY(layer_type<activation::activation_type>, #layer_name "<" #activation_type ">")

#define CNN_REGISTER_LAYER_WITH_ACTIVATIONS(layer_type, layer_name) \
CNN_REGISTER_LAYER_WITH_ACTIVATION(layer_type, tan_h, layer_name); \
CNN_REGISTER_LAYER_WITH_ACTIVATION(layer_type, softmax, layer_name); \
CNN_REGISTER_LAYER_WITH_ACTIVATION(layer_type, identity, layer_name); \
CNN_REGISTER_LAYER_WITH_ACTIVATION(layer_type, sigmoid, layer_name); \
CNN_REGISTER_LAYER_WITH_ACTIVATION(layer_type, relu, layer_name); \
CNN_REGISTER_LAYER_WITH_ACTIVATION(layer_type, leaky_relu, layer_name); \
CNN_REGISTER_LAYER_WITH_ACTIVATION(layer_type, elu, layer_name); \
CNN_REGISTER_LAYER_WITH_ACTIVATION(layer_type, tan_hp1m2, layer_name)

    serialization_helper() {
#include "serialization_layer_list.h"
    }

#undef CNN_REGISTER_LAYER_BODY
#undef CNN_REGISTER_LAYER
#undef CNN_REGISTER_LAYER_WITH_ACTIVATION
#undef CNN_REGISTER_LAYER_WITH_ACTIVATIONS

}; // class serialization_helper

template <typename OutputArchive>
template <typename T>
void serialization_helper<OutputArchive>::save_layer_impl(OutputArchive& oa, const layer* layer) {
    oa (cereal::make_nvp(serialization_helper<OutputArchive>::get_instance().type_name(typeid(T)),
                         *dynamic_cast<const T*>(layer)));
}

template <typename OutputArchive>
void layer::save_layer(OutputArchive & oa, const layer& l) {
    const std::string& name = serialization_helper<OutputArchive>::get_instance().type_name(typeid(l));
    serialization_helper<OutputArchive>::get_instance().save(name, oa, &l);
}

template <class Archive>
void layer::serialize_prolog(Archive & ar) {
    ar(cereal::make_nvp("type",
        serialization_helper<Archive>::get_instance().type_name(typeid(*this))));
}

} // namespace tiny_dnn

