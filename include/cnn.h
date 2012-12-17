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
        //layer->unroll(&params_, &diffs_);
    }

    int in_dim() const { return layers_.head()->in_dim(); }
    int out_dim() const { return layers_.tail()->out_dim(); }

    void predict(const vec_t& in, vec_t *out) {
        *out = *layers_.head()->forward_propagation(in);
    }

    void train(const std::vector<vec_t>& in, const std::vector<vec_t>& training) {
        calc_diff(in, training);
        // update by delta and learning algorithm
        learner_->update(layers_.all_param(), layers_.all_diff());
    }

    bool check(const std::vector<vec_t>& in, const std::vector<vec_t>& training) {
        const int dim = layers_.all_param().size();
        vec_t diff1(dim), diff2(dim);

        calc_diff(in, training);
        for (int i = 0; i < dim; i++)
            diff1[i] = *layers_.all_diff()[i];

        calc_diff_numeric(in, training);
        for (int i = 0; i < dim; i++)
            diff2[i] = *layers_.all_diff()[i];

        float_t diff = sum_square(diff1 - diff2) / dim;
        return diff < 1E-5;
    }

    float_t loss_function(const std::vector<vec_t>& in, const std::vector<vec_t>& training) {
        const int m = in.size();
        float_t loss_score = 0.0;
        float_t norm_score = 0.0;

        for (int i = 0; i < m; i++) {
            layers_.head()->forward_propagation(in[i]);
            loss_score += sum_square(layers_.tail()->output() - training[i]);
        }      
        loss_score /= (2 * m);

        norm_score = lambda_ * sum_square(layers_.weight()) / 2.0; // bias�͊܂߂Ȃ�
        return loss_score + norm_score;
    }

private:
    void calc_diff(const std::vector<vec_t>& in, const std::vector<vec_t>& training) {
        const int m = in.size();
        layers_.reset_diff();

        for (int i = 0; i < m; i++) {
            layers_.head()->forward_propagation(in[i]);
            layers_.tail()->back_propagation(in[i], training[i]);
        } 

        pvec_t& w  = layers_.weight(); 
        pvec_t& dw = layers_.weight_diff();
        for (size_t i = 0; i < w.size(); i++) 
            *dw[i] = *dw[i] / m + lambda_ * *w[i];   

        pvec_t& b  = layers_.bias(); 
        pvec_t& db = layers_.bias_diff();
        for (size_t i = 0; i < b.size(); i++) 
            *db[i] = *db[i] / m;   
    }

    void calc_diff_numeric(const std::vector<vec_t>& in, const std::vector<vec_t>& training) {
        static const float_t EPSILON = 1e-4;
        const int m = in.size();
        layers_.reset_diff();

        const int dim = layers_.all_param().size();

        for (int i = 0; i < dim; i++) {
            const float_t v = *layers_.all_param()[i];

            *layers_.all_param()[i] = v + EPSILON;
            const float_t Jp = loss_function(in, training);

            *layers_.all_param()[i] = v - EPSILON;
            const float_t Jm = loss_function(in, training);

            const float_t diff = (Jp - Jm) / (2.0 * EPSILON);
            *layers_.all_diff()[i] = diff;
        }
    }

    const double lambda_; // weight decay
    layers layers_;
    learner *learner_;
};

}
