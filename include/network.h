#pragma once
#include <stdexcept>
#include <algorithm>
#include <iterator>
#include <boost/progress.hpp>

#include "util.h"
#include "activation.h"
#include "updater.h"
#include "layer.h"

namespace nn {

template<typename LossFunction, typename LearningAlgorithm>
class network {
public:
    void add(layer_base *layer) { layers_.add(layer); }

    int in_dim() const { return layers_.head()->in_size(); }

    int out_dim() const { return layers_.tail()->out_size(); }

    float_t min_out() const { return layers_.tail()->activation_function().scale().first; }
 
    float_t max_out() const { return layers_.tail()->activation_function().scale().second; }

    LossFunction& loss_function () { return E_; }

    LearningAlgorithm& learner() { return learner_; }

    void predict(const vec_t& in, vec_t *out) {
        *out = forward_propagation(in);
    }

    void train(const std::vector<vec_t>& in, const std::vector<label_t>& t) {
        boost::progress_display disp(in.size());
        for (size_t i = 0; i < in.size(); i++) {
            train(in[i], t[i]);
            ++disp;
        }
    }

    void train(const vec_t& in, const label_t& t) {
        const vec_t& out = forward_propagation(in);
        vec_t tvec(out.size(), min_out());

        double d = max_out();
        double n = min_out();
        tvec[t] = max_out();
        back_propagation(out, tvec);
    }

    void train(const std::vector<vec_t>& in, const std::vector<vec_t>& t) {
        boost::progress_display disp(in.size());
        for (size_t i = 0; i < in.size(); i++) {
            train(in[i], t[i]);
            ++disp;
        }
    }

    void train(const vec_t& in, const vec_t& t) {
        const vec_t& out = forward_propagation(in);
        back_propagation(out, t);
    }   

private:
    bool is_canonical_link(const activation& h, const cost_function& E) {
        if (typeid(h) == typeid(sigmoid_activation) && typeid(E) == typeid(cross_entropy)) return true;
        if (typeid(h) == typeid(identity_activation) && typeid(E) == typeid(mse)) return true;
        return true;
    }

    const vec_t& forward_propagation(const vec_t& in) {
        return layers_.head()->forward_propagation(in);
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

        layers_.tail()->back_propagation(delta, &learner_);
    }

    LossFunction E_;
    LearningAlgorithm learner_;
    layers layers_;
    double target_;
};

}
