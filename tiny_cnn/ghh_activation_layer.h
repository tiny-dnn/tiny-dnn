/*
    Copyright (c) 2013, Taiga Nomi, Kwang Moo Yi, Yannick Verdie
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

/*
    This  is an  implementation of  the  GHH activation  function  proposed in  "Learning to  Assign
    Orientations to Feature Points", Kwang Moo Yi,  Yannick Verdie, Pascal Fua, and Vincent Lepetit,
    2015. For details, please see arXiv:1511.04273. Also, when using this activation layer, please
    cite the paper.

 */

#pragma once
#include "layer.h"
#include "product.h"
#include "dropout.h"

namespace tiny_cnn {

template<typename Activation, typename Filter = filter_none>
class ghh_activation_layer : public layer<Activation> {
public:
    typedef layer<Activation> Base;
    CNN_USE_LAYER_MEMBERS;

    ghh_activation_layer(layer_size_t out_dim, size_t num_in_sum, size_t num_in_max)
        : Base(size_t(out_dim) * num_in_sum * num_in_max, out_dim, 0, 0), filter_(out_dim) {
		m_num_in_sum = num_in_sum;
		m_num_in_max = num_in_max;
	}

	// Stupid dummies to be compatible
    size_t connection_size() const override {
        return size_t(in_size_) * out_size_ + out_size_;
    }

	// Stupid dummies to be compatible
    size_t fan_in_size() const override {
        return in_size_;
    }

	// Stupid dummies to be compatible
    size_t fan_out_size() const override {
        return out_size_;
    }

    const vec_t& forward_propagation(const vec_t& in, size_t index) override {
        vec_t &a = a_[index];
        vec_t &out = output_[index];

		int step_size = out_size_; // we move channel-wise to be compatible (might break cache!)
		
		// simple implementation with for loops for now
        for_i(parallelize_, out_size_, [&](int i) {
		    int idx = i; // set base for i
			float_t sum_out = static_cast<float_t>(0.0);
			float_t sum_sign = static_cast<float_t>(1.0);
			for(int n=0; n < m_num_in_sum; ++n){
				assert(idx < in.size());
				float_t max_out = in[idx]; // start with first as max
				idx += step_size;		   // move index to next
				for(int m=1; m < m_num_in_max; ++m){
					assert(idx < in.size());
					max_out = std::max(max_out,in[idx]); // do the max operation for others
					idx += step_size;					 // move index to next
				}
				sum_out += sum_sign * max_out; // add with sign
				sum_sign = -sum_sign; // flip sign
			}
			a[i] = sum_out;	// final activated output
		});

        for_i(parallelize_, out_size_, [&](int i) {
            out[i] = h_.f(a, i);
        });
		
        auto& this_out = filter_.filter_fprop(out, index);

        return next_ ? next_->forward_propagation(this_out, index) : this_out;
    }

	// We use the fully connected implementation for now (TODO: Fixme)
    const vec_t& back_propagation(const vec_t& current_delta, size_t index) override {
        const vec_t& curr_delta = filter_.filter_bprop(current_delta, index);
        const vec_t& prev_out = prev_->output(index);
        const vec_t& in = prev_->output(index); // note that this works only with identity!
        const activation::function& prev_h = prev_->activation_function();
        vec_t& prev_delta = prev_delta_[index];
        vec_t& dW = dW_[index];
        vec_t& db = db_[index];

		if (!W_.empty())
			throw nn_error("Weights not empty for GHH!");

		int step_size = out_size_; // we move channel-wise to be compatible (might break cache!)
		
		// Compute the coefficients to be multiplied to delta (basically picking the things for max)
		vec_t bp_coeff(this->in_size_,0);
        for_i(parallelize_, out_size_, [&](int i) {
		    int idx = i; // set base for i
			float_t sum_sign = static_cast<float_t>(1.0);
			for(int n=0; n < m_num_in_sum; ++n){
				// start with first as max
				float_t max_out = in[idx]; 
				assert(idx < in.size());
				int max_idx = idx;
				idx += step_size;
				// find max_idx for assigning bp_coeff
				for(int m=1; m < m_num_in_max; ++m){
					assert(idx < in.size());
					if (max_out < in[idx]){
						max_out = in[idx];
						max_idx = idx;
					}
					idx += step_size;
				}
				bp_coeff[max_idx] = sum_sign; // save the coefficient to be multiplied
				sum_sign = -sum_sign; // flip sign
			}
		});

        for (int c = 0; c < this->in_size_; c++) {
            // propagate delta to previous layer

			// Use bp_coeff to see if we propagate the curr_delta
			int r = c % this->out_size_;
			prev_delta[c] = curr_delta[r] * bp_coeff[c]; // TODO: this is wrong
			
            // // prev_delta[c] += current_delta[r] * W_[c * out_size_ + r]
            // prev_delta[c] = vectorize::dot(&curr_delta[0], &W_[c*out_size_], out_size_);

			prev_delta[c] *= prev_h.df(prev_out[c]);
        }

        // for (int c = 0; c < this->in_size_; c++) {
        //     // propagate delta to previous layer
        //     // prev_delta[c] += current_delta[r] * W_[c * out_size_ + r]
        //     prev_delta[c] = vectorize::dot(&curr_delta[0], &W_[c*out_size_], out_size_);
        //     prev_delta[c] *= prev_h.df(prev_out[c]);
        // }

		
		// No need to prepare weight updates for this layer
        // for_(parallelize_, 0, out_size_, [&](const blocked_range& r) {
        //     // accumulate weight-step using delta
        //     // dW[c * out_size + i] += current_delta[i] * prev_out[c]
        //     for (int c = 0; c < in_size_; c++)
        //         vectorize::muladd(&curr_delta[0], prev_out[c], r.end() - r.begin(), &dW[c*out_size_ + r.begin()]);

        //     for (int i = r.begin(); i < r.end(); i++) 
        //         db[i] += curr_delta[i];
        // });

        return prev_->back_propagation(prev_delta_[index], index);
    }

	// We use the fully connected implementation for now (TODO: Fixme)
    const vec_t& back_propagation_2nd(const vec_t& current_delta2) override {
        const vec_t& prev_out = prev_->output(0);
        const vec_t& in = prev_->output(0); // note that this works only with identity!
        const activation::function& prev_h = prev_->activation_function();

		printf("I should not have been called\n");
		
		// No need to prepare weight updates for this layer
        // for (int c = 0; c < in_size_; c++) 
        //     for (int r = 0; r < out_size_; r++)
        //         Whessian_[c*out_size_ + r] += current_delta2[r] * sqr(prev_out[c]);

        // for (int r = 0; r < out_size_; r++)
        //     bhessian_[r] += current_delta2[r];


		int step_size = out_size_; // we move channel-wise to be compatible (might break cache!)

        // Compute the coefficients to be multiplied to delta (basically picking the things for max)
		vec_t bp_coeff(this->in_size_,0);
        for_i(parallelize_, out_size_, [&](int i) {
		    int idx = i; // set base for i
			float_t sum_sign = static_cast<float_t>(1.0);
			for(int n=0; n < m_num_in_sum; ++n){
				// start with first as max
				float_t max_out = in[idx]; 
				assert(idx < in.size());
				int max_idx = idx;
				idx += step_size;
				// find max_idx for assigning bp_coeff
				for(int m=1; m < m_num_in_max; ++m){
					assert(idx < in.size());
					if (max_out < in[idx]){
						max_out = in[idx];
						max_idx = idx;
					}
					idx += step_size;
				}
				bp_coeff[max_idx] = sum_sign; // save the coefficient to be multiplied
				sum_sign = -sum_sign; // flip sign
			}
		});
		
        for (int c = 0; c < in_size_; c++) { 
            prev_delta2_[c] = 0.0;

            for (int r = 0; r < out_size_; r++) 
                prev_delta2_[c] += current_delta2[r];

            prev_delta2_[c] *= sqr(bp_coeff[c]);
            prev_delta2_[c] *= sqr(prev_h.df(prev_out[c]));
        }

        return prev_->back_propagation_2nd(prev_delta2_);
    }


	// // We override the weight initialization for this guy
    // void init_weight() override {

	// 	printf("OVERRIDED INIT for GHH\n");

	// 	W_.clear();
	// 	b_.clear();
		
    //     // weight_init_->fill(&W_, fan_in_size(), fan_out_size());
    //     // bias_init_->fill(&b_, fan_in_size(), fan_out_size());

    //     // std::fill(Whessian_.begin(), Whessian_.end(), 0.0);
    //     // std::fill(bhessian_.begin(), bhessian_.end(), 0.0);
    //     // clear_diff(CNN_TASK_SIZE);
    // }
	
	// // Optimizer should do nothing for this guy. So we override
    // template <typename Optimizer>
    // void update_weight(Optimizer *o, int worker_size, size_t batch_size) override {

	// 	printf("OVERRIDED Optimizer for GHH\n");
		
    //     if (W_.empty()) return;

    //     // merge(worker_size, batch_size);

    //     // o->update(dW_[0], Whessian_, W_);
    //     // o->update(db_[0], bhessian_, b_);

    //     // clear_diff(worker_size);
    //     // post_update();
    // }
	

    std::string layer_type() const override { return "ghh-activation"; }

protected:
    Filter filter_;

	int m_num_in_sum;
	int m_num_in_max;
};

} // namespace tiny_cnn
