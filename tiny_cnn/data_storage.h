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

#include "tiny_cnn/util/util.h"

namespace tiny_cnn {

class data_storage {
public:
    data_storage() : task_size_(1) {}

    int allocate(const index3d<cnn_size_t>& shape, bool worker_specific = false) {
        int num_alloc = worker_specific ? task_size_ : 1;
        storage_idx_.push_back({});

        for (int i = 0; i < num_alloc; i++) {
            storage_.push_back(vec_t(shape.size()));
            storage_idx_.back().push_back((int)storage_.size() - 1);
        }

        worker_specific_.push_back(worker_specific ? 1 : 0);
        return (int)storage_idx_.size() - 1;
    }

    std::vector<const vec_t*> get(const std::vector<int>& ids, cnn_size_t worker_index = 0) const {
        std::vector<const vec_t*> vec;
        for (cnn_size_t i = 0; i < ids.size(); i++)
            vec.push_back(get(ids[i], worker_index));
        return vec;
    }

    std::vector<vec_t*> get(const std::vector<int>& ids, cnn_size_t worker_index = 0) {
        std::vector<vec_t*> vec;
        for (cnn_size_t i = 0; i < ids.size(); i++)
            vec.push_back(get(ids[i], worker_index));
        return vec;
    }

    vec_t* get(int id, cnn_size_t worker_index = 0) {
        if (id == -1)
            throw nn_error("invalid index");
        return worker_specific_[id] ? &storage_.at(storage_idx_[id][worker_index]) : &storage_.at(storage_idx_[id][0]);
    }

    const vec_t* get(int id, cnn_size_t worker_index = 0) const {
        if (id == -1)
            throw nn_error("invalid index");
        return worker_specific_[id] ? &storage_.at(storage_idx_[id][worker_index]) : &storage_.at(storage_idx_[id][0]);
    }

    template <typename Pred>
    void foreach(int id, Pred p) {
        for (auto idx : storage_idx_[id]) {
            vec_t* v = &storage_[idx];
            p(v);
        }
    }

    void clear(int id, cnn_size_t worker_size) {
        for (cnn_size_t i = 0; i < worker_size; i++) {
            vec_t& v = *get(id, i);
            std::fill(v.begin(), v.end(), float_t(0));
        }
    }

    void merge(int id, cnn_size_t worker_size, vec_t *dst) {
        vec_t& first = *get(id);
        *dst = first;

        for (cnn_size_t i = 1; i < worker_size; i++) {
            vec_t& src = *get(id, i);
            vectorize::reduce<float_t>(&src[0], dst->size(), &(*dst)[0]);
        }
    }

    int allocate(bool worker_specific = false) {
        storage_.push_back(vec_t());
        worker_specific_.push_back(worker_specific ? 1 : 0);
        return (int)storage_.size() - 1;
    }

    void set_worker_size(cnn_size_t size) {
        task_size_ = size;
        // replicate worker-specific vector
        for (cnn_size_t i = 0; i < storage_idx_.size(); i++) {
            if (!worker_specific_[i]) {
                continue;
            }
            if (storage_idx_[i].size() < size) {
                for (cnn_size_t j = storage_idx_[i].size(); j < size; j++) {
                    storage_.push_back(storage_[storage_idx_[i][0]]);
                    storage_idx_[i].push_back(storage_.size() - 1);
                }
            }
        }
    }
private:
    std::vector<vec_t> storage_;
    std::vector<std::vector<int>> storage_idx_;// [id,worker_idx] -> storage
    std::vector<uint8_t> worker_specific_;
    int task_size_;
};



} // namespace tiny_cnn
