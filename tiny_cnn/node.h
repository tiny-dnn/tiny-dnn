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
#include <sstream>
#include <iomanip>
#include <memory>
#include <numeric>
#include <vector>
#include <set>
#include <queue>
#include <unordered_set>

#include "tiny_cnn/util/util.h"
#include "tiny_cnn/util/product.h"
#include "tiny_cnn/util/image.h"
#include "tiny_cnn/util/weight_init.h"
#include "tiny_cnn/optimizers/optimizer.h"

#include "tiny_cnn/activations/activation_function.h"

namespace tiny_cnn {

class node;
class layer;
class edge;

typedef node* nodeptr_t;
typedef std::shared_ptr<edge> edgeptr_t;

typedef layer* layerptr_t;

/**
 * base class of all kind of tinny-cnn data
 **/
class node : public std::enable_shared_from_this<node> {
public:
    node(cnn_size_t in_size, cnn_size_t out_size)
        : prev_(in_size), next_(out_size) {}
    virtual ~node() {}

    const std::vector<edgeptr_t>& prev() const { return prev_; }
    const std::vector<edgeptr_t>& next() const { return next_; }

    cnn_size_t prev_port(const edge& e) const {
        auto it = std::find_if(prev_.begin(), prev_.end(),
                               [&](edgeptr_t ep) { return ep.get() == &e; });
        return (cnn_size_t)std::distance(prev_.begin(), it);
    }

    cnn_size_t next_port(const edge& e) const {
        auto it = std::find_if(next_.begin(), next_.end(),
                               [&](edgeptr_t ep) { return ep.get() == &e; });
        return (cnn_size_t)std::distance(next_.begin(), it);
    }

    std::vector<node*> prev_nodes() const; // @todo refactor and remove this method
    std::vector<node*> next_nodes() const; // @todo refactor and remove this method
 protected:
    node() = delete;

    friend void connect(layerptr_t head, layerptr_t tail,
                        cnn_size_t head_index, cnn_size_t tail_index);

    mutable std::vector<edgeptr_t> prev_;
    mutable std::vector<edgeptr_t> next_;
};

/**
 * class containing input/output data
 **/
class edge {
 public:
    edge(node* prev, const shape3d& shape, vector_type vtype)
        : worker_specific_data_(!is_trainable_weight(vtype)),
          shape_(shape),
          vtype_(vtype),
          data_(1, vec_t(shape.size())),
          prev_(prev) {
      grad_.resize(1, vec_t(shape.size()));
    }

	template <typename Float, typename Vec>
	static void merge_grads_impl(int begin, int end, cnn_size_t worker_size, std::vector<Vec>& grad, Vec& dst, Float scale) {
		size_t sz = end - begin;
		switch (worker_size) {
		case 1:
			{
				auto& g = grad[0];
				for (int i=begin; i<end; ++i) {
					dst[i] = g[i] * scale;
				}
			}
			break;
		case 2:
			{
				auto& g0 = grad[0];
				auto& g1 = grad[1];
				for (int i = begin; i<end; ++i) {
					dst[i] = (g0[i] + g1[i]) * scale;
				}
			}
			break;
		default:
			std::copy(&grad[0][begin], &grad[0][end], &dst[begin]);
			for (cnn_size_t i = 1; i < worker_size; i++) {
				vectorize::reduce<float_t>(&grad[i][begin], sz, &dst[begin]);
			}
			for (int i = begin; i<end; ++i) {
				dst[i] *= scale;
			}
			break;
		}
	}

#ifdef CNN_USE_AVX
	static void merge_grads_impl(int begin, int end, cnn_size_t worker_size, std::vector<fvec_t>& grad, fvec_t& dst, float scale) {
		size_t sz = end - begin;
		__m256 vscale = _mm256_set1_ps(scale);
		switch (worker_size) {
		case 1:
		{
			auto& g = grad[0];
			for (int i = begin; i<end; ++i) {
				dst[i] = g[i] * scale;
			}
		}
		break;
		case 2:
		{
			auto& g0 = grad[0];
			auto& g1 = grad[1];
			const float* pg0 = &g0[begin];
			const float* pg1 = &g1[begin];
			float* pdst = &dst[begin];
			size_t nblocks = sz >> 4;
			for (size_t i=0; i<nblocks; ++i) {
				__m256 vg00 = _mm256_loadu_ps(pg0 + 0);
				__m256 vg01 = _mm256_loadu_ps(pg0 + 8);
				__m256 vg10 = _mm256_loadu_ps(pg1 + 0);
				__m256 vg11 = _mm256_loadu_ps(pg1 + 8);
				__m256 vres0 = _mm256_add_ps(vg00, vg10);
				__m256 vres1 = _mm256_add_ps(vg01, vg11);
				vres0 = _mm256_mul_ps(vres0, vscale);
				vres1 = _mm256_mul_ps(vres1, vscale);
				_mm256_storeu_ps(pdst + 0, vres0);
				_mm256_storeu_ps(pdst + 8, vres1);
				pg0 += 16;
				pg1 += 16;
				pdst += 16;
			}
			for (int i=begin+(sz << 4); i<end; ++i) {
				dst[i] = (g0[i] + g1[i]) * scale;
			}
		}
		break;
		case 4:
		{
			auto& g0 = grad[0];
			auto& g1 = grad[1];
			auto& g2 = grad[2];
			auto& g3 = grad[3];
			if (begin & 7) {
				int head_size = 8 - (begin & 7);
				int head_end = std::min(end, begin + head_size);
				for (int i = begin; i<head_end; ++i) {
					dst[i] = (g0[i] + g1[i] + g2[i] + g3[i]) * scale;
				}
				if (end == head_end) {
					return;
				}
				begin += head_size;
				sz = end - begin;
			}
			const float* pg0 = &g0[begin];
			const float* pg1 = &g1[begin];
			const float* pg2 = &g2[begin];
			const float* pg3 = &g3[begin];
			float* pdst = &dst[begin];
			size_t nblocks = sz >> 4;
			for (size_t i = 0; i<nblocks; ++i) {
				__m256 vg00 = _mm256_load_ps(pg0 + 0);
				__m256 vg01 = _mm256_load_ps(pg0 + 8);
				__m256 vg10 = _mm256_load_ps(pg1 + 0);
				__m256 vg11 = _mm256_load_ps(pg1 + 8);
				__m256 vg20 = _mm256_load_ps(pg2 + 0);
				__m256 vg21 = _mm256_load_ps(pg2 + 8);
				__m256 vg30 = _mm256_load_ps(pg3 + 0);
				__m256 vg31 = _mm256_load_ps(pg3 + 8);
				vg00 = _mm256_add_ps(vg00, vg10);
				vg01 = _mm256_add_ps(vg01, vg11);
				vg20 = _mm256_add_ps(vg20, vg30);
				vg21 = _mm256_add_ps(vg21, vg31);
				__m256 vres0 = _mm256_add_ps(vg00, vg20);
				vres0 = _mm256_mul_ps(vres0, vscale);
				_mm256_store_ps(pdst + 0, vres0);
				__m256 vres1 = _mm256_add_ps(vg01, vg21);
				vres1 = _mm256_mul_ps(vres1, vscale);
				_mm256_store_ps(pdst + 8, vres1);
				pg0 += 16;
				pg1 += 16;
				pg2 += 16;
				pg3 += 16;
				pdst += 16;
			}
			for (int i = begin + (nblocks << 4); i<end; ++i) {
				dst[i] = (g0[i] + g1[i] + g2[i] + g3[i]) * scale;
			}
		}
		break;
		default:
		{
			float* pbegin = &grad[0][begin];
			std::copy(pbegin, (pbegin + sz), &dst[begin]);
			for (cnn_size_t i = 1; i < worker_size; i++) {
				vectorize::reduce<float>(&grad[i][begin], sz, &dst[begin]);
			}
			for (int i = begin; i<end; ++i) {
				dst[i] *= scale;
			}
		}
		break;
		}
	}

	static void merge_grads_impl(int begin, int end, cnn_size_t worker_size, std::vector<dvec_t>& grad, dvec_t& dst, double scale) {
		size_t sz = end - begin;
		__m256d vscale = _mm256_set1_pd(scale);
		switch (worker_size) {
		case 1:
		{
			auto& g = grad[0];
			for (int i = begin; i<end; ++i) {
				dst[i] = g[i] * scale;
			}
		}
		break;
		case 2:
		{
			auto& g0 = grad[0];
			auto& g1 = grad[1];
			const double* pg0 = &g0[begin];
			const double* pg1 = &g1[begin];
			double* pdst = &dst[begin];
			size_t nblocks = sz >> 3;
			for (size_t i=0; i<nblocks; ++i) {
				__m256d vg00 = _mm256_loadu_pd(pg0 + 0);
				__m256d vg01 = _mm256_loadu_pd(pg0 + 4);
				__m256d vg10 = _mm256_loadu_pd(pg1 + 0);
				__m256d vg11 = _mm256_loadu_pd(pg1 + 4);
				__m256d vres0 = _mm256_add_pd(vg00, vg10);
				__m256d vres1 = _mm256_add_pd(vg01, vg11);
				vres0 = _mm256_mul_pd(vres0, vscale);
				vres1 = _mm256_mul_pd(vres1, vscale);
				_mm256_storeu_pd(pdst + 0, vres0);
				_mm256_storeu_pd(pdst + 4, vres1);
				pg0 += 8;
				pg1 += 8;
				pdst += 8;
			}
			for (int i=begin+(sz << 3); i<end; ++i) {
				dst[i] = (g0[i] + g1[i]) * scale;
			}
		}
		break;
		case 4:
		{
			auto& g0 = grad[0];
			auto& g1 = grad[1];
			auto& g2 = grad[2];
			auto& g3 = grad[3];
			if (begin & 3) {
				int head_size = 4 - (begin & 3);
				int head_end = std::min(end, begin + head_size);
				for (int i = begin; i<head_end; ++i) {
					dst[i] = (g0[i] + g1[i] + g2[i] + g3[i]) * scale;
				}
				if (end == head_end) {
					return;
				}
				begin += head_size;
				sz = end - begin;
			}
			const double* pg0 = &g0[begin];
			const double* pg1 = &g1[begin];
			const double* pg2 = &g2[begin];
			const double* pg3 = &g3[begin];
			double* pdst = &dst[begin];
			size_t nblocks = sz >> 3;
			for (size_t i = 0; i<nblocks; ++i) {
				__m256d vg00 = _mm256_load_pd(pg0 + 0);
				__m256d vg01 = _mm256_load_pd(pg0 + 4);
				__m256d vg10 = _mm256_load_pd(pg1 + 0);
				__m256d vg11 = _mm256_load_pd(pg1 + 4);
				__m256d vg20 = _mm256_load_pd(pg2 + 0);
				__m256d vg21 = _mm256_load_pd(pg2 + 4);
				__m256d vg30 = _mm256_load_pd(pg3 + 0);
				__m256d vg31 = _mm256_load_pd(pg3 + 4);
				vg00 = _mm256_add_pd(vg00, vg10);
				vg01 = _mm256_add_pd(vg01, vg11);
				vg20 = _mm256_add_pd(vg20, vg30);
				vg21 = _mm256_add_pd(vg21, vg31);
				__m256d vres0 = _mm256_add_pd(vg00, vg20);
				vres0 = _mm256_mul_pd(vres0, vscale);
				_mm256_store_pd(pdst + 0, vres0);
				__m256d vres1 = _mm256_add_pd(vg01, vg21);
				vres1 = _mm256_mul_pd(vres1, vscale);
				_mm256_store_pd(pdst + 4, vres1);
				pg0 += 8;
				pg1 += 8;
				pg2 += 8;
				pg3 += 8;
				pdst += 8;
			}
			for (int i = begin + (nblocks << 3); i<end; ++i) {
				dst[i] = (g0[i] + g1[i] + g2[i] + g3[i]) * scale;
			}
		}
		break;
		default:
		{
			double* pbegin = &grad[0][begin];
			std::copy(pbegin, (pbegin + sz), &dst[begin]);
			for (cnn_size_t i = 1; i < worker_size; i++) {
				vectorize::reduce<double>(&grad[i][begin], sz, &dst[begin]);
			}
			for (int i = begin; i<end; ++i) {
				dst[i] *= scale;
			}
		}
		break;
		}
	}
#endif // #ifdef CNN_USE_AVX

	static inline void merge_grads(int begin, int end, cnn_size_t worker_size, std::vector<vec_t>& grad, vec_t& dst, float_t scale) {
		merge_grads_impl(begin, end, worker_size, grad, dst, scale);
	}

    void merge_grads(cnn_size_t worker_size, vec_t& dst, float_t scale) {
		size_t sz = grad_[0].size();
		if (sz > dst.size()) {
			dst.resize(sz);
		}
		size_t sz_per_thread = sz / std::thread::hardware_concurrency();
		if (sz_per_thread <= 64) {
			merge_grads(0, sz, worker_size, grad_, dst, scale);
		}else {
			for_(true, 0, sz,
				[&](const blocked_range& r) {
					merge_grads(r.begin(), r.end(), worker_size, grad_, dst, scale);
				}
			);
		}
    }

    void clear_grads(cnn_size_t worker_size) {
        for (cnn_size_t i = 0; i < worker_size; i++)
            clear_grad_onwork(i);
    }

    void clear_grad_onwork(cnn_size_t index) {
        std::fill(grad_[index].begin(), grad_[index].end(), (float_t)0);
    }

    void set_worker_size(cnn_size_t size) {
        if (worker_specific_data_) data_.resize(size, data_[0]);
        grad_.resize(size, grad_[0]);
    }

    vec_t* get_data(cnn_size_t worker_index = 0) {
        return worker_specific_data_ ? &data_[worker_index] : &data_[0];
    }

    const vec_t* get_data(cnn_size_t worker_index = 0) const {
        return worker_specific_data_ ? &data_[worker_index] : &data_[0];
    }

    vec_t* get_gradient(cnn_size_t worker_index = 0) {
        return &grad_[worker_index];
    }

    const vec_t* get_gradient(cnn_size_t worker_index = 0) const {
        return &grad_[worker_index];
    }

    const std::vector<node*>& next() const { return next_; }
    node* prev() { return prev_; }
    const node* prev() const { return prev_; }

    const shape3d& shape() const { return shape_; }
    vector_type vtype() const { return vtype_; }
    void add_next_node(node* next) { next_.push_back(next); }

 private:
    bool worker_specific_data_;
    shape3d shape_;
    vector_type vtype_;
    std::vector<vec_t> data_;
    std::vector<vec_t> grad_;
    node* prev_;               // previous node, "producer" of this tensor
    std::vector<node*> next_;  // next nodes, "consumers" of this tensor
};

inline std::vector<node*> node::prev_nodes() const {
    std::set<node*> sets;
    for (auto& e : prev_) {
        if (e && e->prev()) sets.insert(e->prev());
    }
    return std::vector<node*>(sets.begin(), sets.end());
}

inline std::vector<node*> node::next_nodes() const {
    std::set<node*> sets;
    for (auto& e : next_) {
        if (e) {
            auto n = e->next();
            sets.insert(n.begin(), n.end());
        }
    }
    return std::vector<node*>(sets.begin(), sets.end());
}

template <typename T>
struct node_tuple {
    node_tuple(T l1, T l2) {
        nodes_.push_back(l1); nodes_.push_back(l2);
    }
    std::vector<T> nodes_;
};

template <typename T>
node_tuple<T*> operator , (T& l1, T& l2) {
    return node_tuple<T*>(&l1, &l2);
}

template <typename T>
node_tuple<std::shared_ptr<T>> operator , (std::shared_ptr<T> l1, std::shared_ptr<T> l2) {
    return node_tuple<std::shared_ptr<T>>(l1, l2);
}

template <typename T>
node_tuple<std::shared_ptr<T>> operator , (node_tuple<std::shared_ptr<T>> lhs, std::shared_ptr<T>& rhs) {
    lhs.nodes_.push_back(rhs);
    return lhs;
}

template <typename T>
node_tuple<T*> operator , (node_tuple<T*> lhs, T& rhs) {
    lhs.nodes_.push_back(&rhs);
    return lhs;
}

template <typename T, typename U>
inline std::shared_ptr<U>& operator << (std::shared_ptr<T>& lhs,
                                        std::shared_ptr<U>& rhs) {
    connect(lhs.get(), rhs.get());
    return rhs;
}

template <typename T, typename U>
inline U& operator << (const node_tuple<T>& lhs, U& rhs) {
    for (size_t i = 0; i < lhs.nodes_.size(); i++) {
        connect(&*lhs.nodes_[i], &*rhs, 0, i);
    }
    return rhs;
}

template <typename T, typename U>
inline node_tuple<T>& operator << (U& lhs, const node_tuple<T>& rhs) {
    for (size_t i = 0; i < rhs.nodes_.size(); i++) {
        connect(&*lhs, &*rhs.nodes_[i], i, 0);
    }
    return rhs;
}

template <typename T, typename U>
inline U& operator << (const node_tuple<T*>& lhs, U& rhs) {
    for (size_t i = 0; i < lhs.nodes_.size(); i++) {
        connect(lhs.nodes_[i], &rhs, 0, i);
    }
    return rhs;
}

template <typename T, typename U>
inline node_tuple<T*>& operator << (U& lhs, const node_tuple<T*>& rhs) {
    for (size_t i = 0; i < rhs.nodes_.size(); i++) {
        connect(&lhs, rhs.nodes_[i], i, 0);
    }
    return rhs;
}


}   // namespace tiny_cnn
