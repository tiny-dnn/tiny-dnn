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
#include <stdexcept>
#include <algorithm>
#include <iterator>
#include <map>
#include <set>

#include "util.h"
#include "activation.h"
#include "updater.h"
#include "layer.h"

namespace tiny_cnn {

struct result {
    result() : num_success(0), num_total(0) {}

    double accuracy() const {
        return num_success * 100.0 / num_total;
    }

    template <typename Char, typename CharTraits>
    void print_summary(std::basic_ostream<Char, CharTraits>& os) const {
        os << "accuracy:" << accuracy() << "% (" << num_success << "/" << num_total << std::endl;
    }

    template <typename Char, typename CharTraits>
    void print_detail(std::basic_ostream<Char, CharTraits>& os) {
        print_summary(os);
        auto all_labels = labels();

        os << std::setw(5) << "*" << " ";
        for (auto c : all_labels) 
            os << std::setw(5) << c << " ";
        os << std::endl;

        for (auto r : all_labels) {
            os << std::setw(5) << r << " ";           
            for (auto c : all_labels) 
                os << std::setw(5) << confusion_matrix[r][c] << " ";
            os << std::endl;
        }
    }

    std::set<label_t> labels() const {
        std::set<label_t> all_labels;
        for (auto r : confusion_matrix) {
            all_labels.insert(r.first);
            for (auto c : r.second)
                all_labels.insert(c.first);
        }
        return all_labels;
    }

    int num_success;
    int num_total;
    std::map<label_t, std::map<label_t, int> > confusion_matrix;
};

template<typename L, typename U>
class network {
public:
    typedef L LossFunction;
    typedef U Updater;
    typedef std::function<void()> Listener;

    network(const std::string& name = "") {}

    void init_weight() { 
        layers_.reset(); 
    }

    template<typename T>
    void add(layer_base<T> *layer) { layers_.add(layer); }

    // input data dimension of whole networks
    int in_dim() const { return layers_.head()->in_size(); }

    // output data dimension of whole networks
    int out_dim() const { return layers_.tail()->out_size(); }

    std::string name() const { return name_; }

    LossFunction& loss_function() { return E_; }

    Updater& learner() { return updater_; }

    void predict(const vec_t& in, vec_t *out) {
        *out = forward_propagation(in);
    }

    // classification
    template <typename OnDataEnumerate, typename OnEpochEnumerate>
    void train(const std::vector<vec_t>& in, const std::vector<label_t>& t, int epoch, OnDataEnumerate on_data_enumerate, OnEpochEnumerate on_epoch_enumerate) {
        for (int iter = 0; iter < epoch; iter++) {
            calc_hessian(in);
            for (size_t i = 0; i < in.size(); i++) {
                train_once(in[i], t[i]);
                on_data_enumerate();
            }
            on_epoch_enumerate();
        }
    }

    void train(const std::vector<vec_t>& in, const std::vector<label_t>& t, int epoch = 1) {
        train(in, t, epoch, nop, nop);
    }

    result test(const std::vector<vec_t>& in, const std::vector<label_t>& t) {
        result test_result;

        for (size_t i = 0; i < in.size(); i++) {
            vec_t out;
            predict(in[i], &out);

            const label_t predicted = max_index(out);
            const label_t actual = t[i];

            if (predicted == actual) test_result.num_success++;
            test_result.num_total++;
            test_result.confusion_matrix[predicted][actual]++;
        }
        return test_result;
    }

    // regression
    template <typename OnDataEnumerate, typename OnEpochEnumerate>
    void train(const std::vector<vec_t>& in, const std::vector<vec_t>& t, int epoch,  OnDataEnumerate on_data_enumerate, OnEpochEnumerate on_epoch_enumerate) {
        for (int iter = 0; iter < epoch; iter++) {
            calc_hessian(in);
            for (size_t i = 0; i < in.size(); i++) {
                train_once(in[i], t[i]);
                on_data_enumerate();
            }
            on_epoch_enumerate();
        }
    }

    void train(const std::vector<vec_t>& in, const std::vector<vec_t>& t, int epoch = 1) {
        train(in, t, epoch, nop, nop);
    }

private:

    void train_once(const vec_t& in, const label_t& t) {
        const vec_t& out = forward_propagation(in);
        vec_t tvec(out.size(), target_value_min());

        if (static_cast<size_t>(t) >= out.size())
            throw nn_error("training label must be less than output neurons");

        tvec[t] = target_value_max();
        back_propagation(out, tvec);
    }

    void train_once(const vec_t& in, const vec_t& t) {
        const vec_t& out = forward_propagation(in);
        back_propagation(out, t);
    }   

    void calc_hessian(const std::vector<vec_t>& in, int size_initialize_hessian = 500) {
        int size = std::min((int)in.size(), size_initialize_hessian);

        for (int i = 0; i < size; i++) {
            const vec_t& out = forward_propagation(in[i]);
            back_propagation_2nd(out);
        }
        layers_.divide_hessian(size);
    }

    template<typename T, typename Loss>
    bool is_canonical_link(const T& h, const Loss& E) {
        if (typeid(h) == typeid(sigmoid_activation) && typeid(E) == typeid(cross_entropy)) return true;
        if (typeid(h) == typeid(tanh_activation) && typeid(E) == typeid(cross_entropy)) return true;
        if (typeid(h) == typeid(identity_activation) && typeid(E) == typeid(mse)) return true;
        return false;
    }

    const vec_t& forward_propagation(const vec_t& in) {
        if (in.size() != (size_t)in_dim())
            throw nn_error("input dimension mismatch");
        return layers_.head()->forward_propagation(in);
    }

    void back_propagation_2nd(const vec_t& out) {
        vec_t delta(out_dim());
        const activation& h = layers_.tail()->activation_function();

        if (is_canonical_link(h, E_)) {
            for (int i = 0; i < out_dim(); i++)
                delta[i] = target_value_max() * h.df(out[i]);  
        } else {
            for (int i = 0; i < out_dim(); i++)
                delta[i] = target_value_max() * h.df(out[i]) * h.df(out[i]); // FIXME
        }

        layers_.tail()->back_propagation_2nd(delta);
    }

    void back_propagation(const vec_t& out, const vec_t& t) {
        vec_t delta(out_dim());
        const activation& h = layers_.tail()->activation_function();

        if (is_canonical_link(h, E_)) {
            for (int i = 0; i < out_dim(); i++)
                delta[i] = out[i] - t[i];  
        } else {
            for (int i = 0; i < out_dim(); i++)
                delta[i] = E_.df(out[i], t[i]) * h.df(out[i]);
        }

        layers_.tail()->back_propagation(delta, &updater_);
    }

    float_t target_value_min() const { return layers_.tail()->activation_function().scale().first; }
    float_t target_value_max() const { return layers_.tail()->activation_function().scale().second; }

    std::string name_;
    LossFunction E_;
    Updater updater_;
    layers<network<L, U> > layers_;
};

}
