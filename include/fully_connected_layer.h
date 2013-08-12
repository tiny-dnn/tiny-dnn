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
#include "layer.h"
#include "updater.h"

namespace tiny_cnn {

// normal 
template<typename N, typename Activation>
class fully_connected_layer : public layer<N, Activation> {
public:
    typedef layer<N, Activation> Base;
    typedef typename Base::Updater Updater;

    fully_connected_layer(int in_dim, int out_dim) : layer<N, Activation>(in_dim, out_dim, in_dim * out_dim, out_dim) {}

    int connection_size() const {
        return this->in_size_ * this->out_size_ + this->out_size_;
    }

    int fan_in_size() const {
        return this->in_size_;
    }

    const vec_t& forward_propagation(const vec_t& in) {

        for (int r = 0; r < this->out_size_; r++) {
            float_t z = 0.0;
            for (int c = 0; c < this->in_size_; c++) 
                z += this->W_[r*this->in_size_+c] * in[c];
            z += this->b_[r];
            this->output_[r] = this->a_.f(z);
        }

        return this->next_ ? this->next_->forward_propagation(this->output_) : this->output_;
    }

    const vec_t& back_propagation(const vec_t& current_delta, Updater *l) {
        const vec_t& prev_out = this->prev_->output();
        const activation& prev_h = this->prev_->activation_function();

        for (int c = 0; c < this->in_size_; c++) { 
            this->prev_delta_[c] = 0.0;

            for (int r = 0; r < this->out_size_; r++) 
                this->prev_delta_[c] += current_delta[r] * this->W_[r*this->in_size_+c];

            this->prev_delta_[c] *= prev_h.df(prev_out[c]);
        }

        if (l) {
            parallel_for(0,this->out_size_, [&](const blocked_range& r) {
                for (int i = r.begin(); i < r.end(); i++) 
                    for (int c = 0; c < this->in_size_; c++) 
                        l->update(current_delta[i] * prev_out[c], this->Whessian_[i*this->in_size_+c], &this->W_[i*this->in_size_+c]); 

                for (int i = r.begin(); i < r.end(); i++) 
                    l->update(current_delta[i], this->bhessian_[i], &this->b_[i]);   
            });
        }

        return this->prev_->back_propagation(this->prev_delta_, l);
    }


    const vec_t& back_propagation_2nd(const vec_t& current_delta2) {
        const vec_t& prev_out = this->prev_->output();
        const activation& prev_h = this->prev_->activation_function();

        for (int r = 0; r < this->out_size_; r++)
            for (int c = 0; c < this->in_size_; c++) 
                this->Whessian_[r*this->in_size_+c] += current_delta2[r] * prev_out[c] * prev_out[c];

        for (int r = 0; r < this->out_size_; r++)
            this->bhessian_[r] += current_delta2[r];

        for (int c = 0; c < this->in_size_; c++) { 
            this->prev_delta2_[c] = 0.0;

            for (int r = 0; r < this->out_size_; r++) 
                this->prev_delta2_[c] += current_delta2[r] * this->W_[r*this->in_size_+c] * this->W_[r*this->in_size_+c];

            this->prev_delta2_[c] *= prev_h.df(prev_out[c]) * prev_h.df(prev_out[c]);
        }

        return this->prev_->back_propagation_2nd(this->prev_delta2_);
    }
};

}
