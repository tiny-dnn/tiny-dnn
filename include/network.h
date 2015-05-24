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
#include <iomanip>
#include <map>
#include <set>

#include "util.h"
#include "activation_function.h"
#include "loss_function.h"
#include "optimizer.h"
#include "fully_connected_layer.h"
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

enum grad_check_mode {
    GRAD_CHECK_ALL, ///< check all elements of weights
    GRAD_CHECK_RANDOM ///< check 10 randomly selected weights
};

template<typename LossFunction, typename Optimizer>
class network {
public:
    explicit network(const std::string& name = "") : name_(name) {}

    // getter
    layer_size_t in_dim() const         { return layers_.head()->in_size(); }
    layer_size_t out_dim() const        { return layers_.tail()->out_size(); }
    std::string  name() const           { return name_; }
    Optimizer&   optimizer()            { return optimizer_; }

    void         init_weight()          { layers_.reset(); }
    void         add(layer_base *layer) { layers_.add(layer); }
    vec_t        predict(const vec_t& in) { return fprop(in); }

    /**
     * training conv-net
     *
     * @param in                 array of input data
     * @param t                  array of training signals(label or vector)
     * @param epoch              number of training epochs
     * @param on_batch_enumerate callback for each mini-batch enumerate
     * @param on_epoch_enumerate callback for each epoch 
     */
    template <typename OnBatchEnumerate, typename OnEpochEnumerate, typename T>
    void train(const std::vector<vec_t>& in,
               const std::vector<T>&     t,
               size_t                    batch_size,
               int                       epoch,
               OnBatchEnumerate          on_batch_enumerate,
               OnEpochEnumerate          on_epoch_enumerate)
    {
        init_weight();
        layers_.set_parallelize(batch_size < CNN_TASK_SIZE);
        optimizer_.reset();

        for (int iter = 0; iter < epoch; iter++) {
            if (optimizer_.requires_hessian())
                calc_hessian(in);
            for (size_t i = 0; i < in.size(); i+=batch_size) {
                train_once(&in[i], &t[i], std::min(batch_size, in.size() - i));
                on_batch_enumerate();
            }
            on_epoch_enumerate();
        }
    }

    /**
     * training conv-net without callback
     **/
    template<typename T>
    void train(const std::vector<vec_t>& in, const std::vector<T>& t, size_t batch_size = 1, int epoch = 1) {
        train(in, t, batch_size, epoch, nop, nop);
    }

    /**
     * test and generate confusion-matrix
     **/
    result test(const std::vector<vec_t>& in, const std::vector<label_t>& t) {
        result test_result;

        for (size_t i = 0; i < in.size(); i++) {
            const label_t predicted = max_index(predict(in[i]));
            const label_t actual = t[i];

            if (predicted == actual) test_result.num_success++;
            test_result.num_total++;
            test_result.confusion_matrix[predicted][actual]++;
        }
        return test_result;
    }

    void save(std::ostream& os) const {
        auto l = layers_.head();
        while (l) { l->save(os); l = l->next(); }
    }

    void load(std::istream& is) {
        auto l = layers_.head();
        while (l) { l->load(is); l = l->next(); }
    }

    bool gradient_check(const vec_t* in, const label_t* t, int data_size, float_t eps, grad_check_mode mode) {
        assert(!layers_.empty());
        std::vector<vec_t> v;
        label2vector(t, data_size, &v);

        auto current = layers_.head();

        while ((current = current->next()) != 0) { // ignore first input layer
            vec_t& w = current->weight();
            vec_t& b = current->bias();
            vec_t& dw = current->weight_diff(0);
            vec_t& db = current->bias_diff(0);

            if (w.empty()) continue;
            
            switch (mode) {
            case GRAD_CHECK_ALL:
                for (int i = 0; i < (int)w.size(); i++)
                    if (calc_delta(in, &v[0], data_size, w, dw, i) > eps) return false;
                for (int i = 0; i < (int)b.size(); i++)
                    if (calc_delta(in, &v[0], data_size, b, db, i) > eps) return false;
                break;
            case GRAD_CHECK_RANDOM:
                for (int i = 0; i < 10; i++)
                    if (calc_delta(in, &v[0], data_size, w, dw, uniform_idx(w)) > eps) return false;
                for (int i = 0; i < 10; i++)
                    if (calc_delta(in, &v[0], data_size, b, db, uniform_idx(b)) > eps) return false;
                break;
            default:
                throw nn_error("unknown grad-check type");
            }
        }
        return true;
    }

private:

    void label2vector(const label_t* t, int num, std::vector<vec_t> *vec) const {
        layer_size_t outdim = out_dim();

        assert(num > 0);
        assert(outdim > 0);

        vec->reserve(num);
        for (int i = 0; i < num; i++) {
            assert(t[i] < outdim);
            vec->emplace_back(outdim, target_value_min());
            vec->back()[t[i]] = target_value_max();
        }
    }

    void train_once(const vec_t* in, const label_t* t, int size) {
        std::vector<vec_t> v;
        label2vector(t, size, &v);
        train_once(in, &v[0], size);
    }

    void train_once(const vec_t* in, const vec_t* t, int size) {
        if (size == 1) {
            bprop(fprop(in[0]), t[0]);
            layers_.update_weights(&optimizer_, 1, 1);
        } else {
            task_group g;
            int num_tasks = size < CNN_TASK_SIZE ? 1 : CNN_TASK_SIZE;
            int data_per_thread = size / num_tasks;
            int remaining = size;

            for (int i = 0; i < num_tasks; i++) {
                int num = i == num_tasks - 1 ? remaining : data_per_thread;
                
                g.run([=]{                    
                    for (int j = 0; j < num; j++) bprop(fprop(in[j], i), t[j], i);
                });

                remaining -= num;
                in += num;
                t += num;
            }

            assert(remaining == 0);
            g.wait();
            layers_.update_weights(&optimizer_, num_tasks, size);
        }
    }   

    void calc_hessian(const std::vector<vec_t>& in, int size_initialize_hessian = 500) {
        int size = std::min((int)in.size(), size_initialize_hessian);

        for (int i = 0; i < size; i++)
            bprop_2nd(fprop(in[i]));

        layers_.divide_hessian(size);
    }

    template<typename Activation, typename Loss>
    bool is_canonical_link(const Activation& h, const Loss& E) {
        if (typeid(h) == typeid(activation::sigmoid) && typeid(E) == typeid(cross_entropy)) return true;
        if (typeid(h) == typeid(activation::tan_h) && typeid(E) == typeid(cross_entropy)) return true;
        if (typeid(h) == typeid(activation::identity) && typeid(E) == typeid(mse)) return true;
        return false;
    }

    const vec_t& fprop(const vec_t& in, int idx = 0) {
        if (in.size() != (size_t)in_dim())
            throw nn_error("input dimension mismatch");
        return layers_.head()->forward_propagation(in, idx);
    }

    float_t get_loss(const vec_t& out, const vec_t& t) {
        float_t e = 0.0;
        assert(out.size() == t.size());
        for_i(out.size(), [&](int i){ e += E_.f(out[i], t[i]); });
        return e;
    }

    void bprop_2nd(const vec_t& out) {
        vec_t delta(out_dim());
        const activation::function& h = layers_.tail()->activation_function();

        if (is_canonical_link(h, E_)) {
            for_i(out_dim(), [&](int i){ delta[i] = target_value_max() * h.df(out[i]);});
        } else {
            for_i(out_dim(), [&](int i){ delta[i] = target_value_max() * h.df(out[i]) * h.df(out[i]);}); // FIXME
        }

        layers_.tail()->back_propagation_2nd(delta);
    }

    void bprop(const vec_t& out, const vec_t& t, int idx = 0) {
        vec_t delta(out_dim());
        const activation::function& h = layers_.tail()->activation_function();

        if (is_canonical_link(h, E_))
            for_i(out_dim(), [&](int i){ delta[i] = out[i] - t[i]; });
        else
            for_i(out_dim(), [&](int i){ delta[i] = E_.df(out[i], t[i]) * h.df(out[i]);});

        layers_.tail()->back_propagation(delta, idx);
    }

    float_t calc_delta(const vec_t* in, const vec_t* v, int data_size, vec_t& w, vec_t& dw, int check_index) {
        static const float_t delta = 1e-10;

        std::fill(dw.begin(), dw.end(), 0.0);

        // calculate dw/dE by numeric
        float_t prev_w = w[check_index];

        w[check_index] = prev_w + delta;
        float_t f_p = 0.0;
        for_i(data_size, [&](int i){ f_p += get_loss(fprop(in[i]), v[i]); });

        float_t f_m = 0.0;
        w[check_index] = prev_w - delta;
        for_i(data_size, [&](int i){ f_m += get_loss(fprop(in[i]), v[i]); });

        float_t delta_by_numerical = (f_p - f_m) / (2.0 * delta);
        w[check_index] = prev_w;

        // calculate dw/dE by bprop
        for_i(data_size, [&](int i){ bprop(fprop(in[i]), v[i]); });

        float_t delta_by_bprop = dw[check_index];

        return std::abs(delta_by_bprop - delta_by_numerical);
    }

    float_t target_value_min() const { return layers_.tail()->activation_function().scale().first; }
    float_t target_value_max() const { return layers_.tail()->activation_function().scale().second; }

    std::string name_;
    LossFunction E_;
    Optimizer optimizer_;
    layers layers_;
};

/**
* create multi-layer perceptron
*/
template<typename loss_func, typename algorithm, typename activation, typename Iter>
network<loss_func, algorithm> make_mlp(Iter first, Iter last) {
    typedef network<loss_func, algorithm> net_t;
    net_t n;

    Iter next = first + 1;
    for (; next != last; ++first, ++next)
        n.add(new fully_connected_layer<activation>(*first, *next));
    return n;
}

/**
 * create multi-layer perceptron
 */
template<typename loss_func, typename algorithm, typename activation>
network<loss_func, algorithm> make_mlp(const std::vector<int>& units) {
    return make_mlp<loss_func, algorithm, activation>(units.begin(), units.end());
}

template <typename L, typename O, typename Layer>
network<L, O>& operator << (network<L, O>& n, const Layer&& l) {
    n.add(new Layer(l));
    return n;
}

template <typename L, typename O, typename Layer>
network<L, O>& operator << (network<L, O>& n, Layer& l) {
    n.add(&l);
    return n;
}

template <typename L, typename O, typename Char, typename CharTraits>
std::basic_ostream<Char, CharTraits>& operator << (std::basic_ostream<Char, CharTraits>& os, const network<L, O>& n) {
    n.save(os);
    return os;
}

template <typename L, typename O, typename Char, typename CharTraits>
std::basic_istream<Char, CharTraits>& operator >> (std::basic_istream<Char, CharTraits>& os, const network<L, O>& n) {
    n.load(os);
    return os;
}

} // namespace tiny_cnn
