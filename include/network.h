#pragma once
#include <stdexcept>
#include <algorithm>
#include <iterator>

#include "util.h"
#include "activation.h"
#include "updater.h"
#include "layer.h"

namespace nn {

template<typename L, typename U>
class network {
public:
    typedef L LossFunction;
    typedef U Updater;

    void init_weight(const std::vector<vec_t>& in, int size_initialize_hessian = 500) { 
        layers_.reset(); 
        init_hessian(in, size_initialize_hessian);
    }

    template<typename T>
    void add(layer_base<T> *layer) { layers_.add(layer); }

    int in_dim() const { return layers_.head()->in_size(); }

    int out_dim() const { return layers_.tail()->out_size(); }

    float_t min_out() const { return layers_.tail()->activation_function().scale().first; }
 
    float_t max_out() const { return layers_.tail()->activation_function().scale().second; }

    LossFunction& loss_function () { return E_; }

    Updater& learner() { return updater_; }

    void predict(const vec_t& in, vec_t *out) {
        *out = forward_propagation(in);
    }

    void train(const std::vector<vec_t>& in, const std::vector<label_t>& t) {
        for (size_t i = 0; i < in.size(); i++) 
            train(in[i], t[i]);
    }

    void train(const vec_t& in, const label_t& t) {
        const vec_t& out = forward_propagation(in);
        vec_t tvec(out.size(), min_out());

        if (static_cast<size_t>(t) >= out.size())
            throw nn_error("training label must be less than output neurons");

        tvec[t] = max_out();
        back_propagation(out, tvec);
    }

    void train(const std::vector<vec_t>& in, const std::vector<vec_t>& t) {
        for (size_t i = 0; i < in.size(); i++) {
            train(in[i], t[i]);
        }
    }

    void train(const vec_t& in, const vec_t& t) {
        const vec_t& out = forward_propagation(in);
        back_propagation(out, t);
    }   

private:
    void init_hessian(const std::vector<vec_t>& in, int size_initialize_hessian) {
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
        return layers_.head()->forward_propagation(in);
    }

    void back_propagation_2nd(const vec_t& out) {
        vec_t delta(out_dim());
        const activation& h = layers_.tail()->activation_function();

        if (is_canonical_link(h, E_)) {
            for (int i = 0; i < out_dim(); i++)
                delta[i] = max_out() * h.df(out[i]);  
        } else {
            for (int i = 0; i < out_dim(); i++)
                delta[i] = max_out() * h.df(out[i]) * h.df(out[i]);  
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

    LossFunction E_;
    Updater updater_;
    layers<network<L, U> > layers_;
};

}
