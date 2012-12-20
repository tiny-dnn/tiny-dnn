#pragma once
#include <stdexcept>
#include <algorithm>
#include <iterator>

#include "util.h"
#include "activation.h"
#include "learner.h"
#include "layer.h"

namespace nn {

class cnn {
public:
    cnn(double alpha, double lambda) : lambda_(lambda) {
        learner_ = new gradient_descent(alpha);
    }

    cnn(learner *l, double lambda) : lambda_(lambda), learner_(l) {}

    ~cnn(){
        delete learner_;
    }

    void add(layer *layer) {
        layers_.add(layer);
        layer->set_learner(learner_);
    }

    int in_dim() const { return layers_.head()->in_dim(); }
    int out_dim() const { return layers_.tail()->out_dim(); }

    void predict(const vec_t& in, vec_t *out) {
        *out = forward_propagation(in);
    }

    void train(const std::vector<vec_t>& in, const std::vector<vec_t>& t) {
        for (size_t i = 0; i < in.size(); i++)
            train(in[i], t[i]);
    }

    void train(const vec_t& in, const vec_t& t) {
        const vec_t& out = forward_propagation(in);
        back_propagation(out, t);
    }

private:

    const vec_t& forward_propagation(const vec_t& in) {
        return layers_.head()->forward_propagation(in);
    }

    void back_propagation(const vec_t& out, const vec_t& t) {
        vec_t delta(out_dim());
        const activation& h = layers_.tail()->activation_function();

        for (int i = 0; i < out_dim(); i++)
            delta[i] = (out[i] - t[i]) * h.df(out[i]);  

        layers_.tail()->back_propagation(delta, true);
    }

    const double lambda_; // weight decay
    layers layers_;
    learner *learner_;
};

}
