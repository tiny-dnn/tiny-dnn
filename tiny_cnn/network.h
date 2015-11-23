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
#include <iostream>
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
#include "layer.h"
#include "layers.h"
#include "fully_connected_layer.h"

namespace tiny_cnn {

struct result {
    result() : num_success(0), num_total(0) {}

    double accuracy() const {
        return num_success * 100.0 / num_total;
    }

    template <typename Char, typename CharTraits>
    void print_summary(std::basic_ostream<Char, CharTraits>& os) const {
        os << "accuracy:" << accuracy() << "% (" << num_success << "/" << num_total << ")" << std::endl;
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
class network 
{
public:
    typedef LossFunction E;

    explicit network(const std::string& name = "") : name_(name) {}

    // getter
    layer_size_t in_dim() const         { return layers_.head()->in_size(); }
    layer_size_t out_dim() const        { return layers_.tail()->out_size(); }
    std::string  name() const           { return name_; }
    Optimizer&   optimizer()            { return optimizer_; }

    void         init_weight()          { layers_.init_weight(); }
    void         add(std::shared_ptr<layer_base> layer) { layers_.add(layer); }
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
    bool train(const std::vector<vec_t>& in,
               const std::vector<T>&     t,
               size_t                    batch_size,
               int                       epoch,
               OnBatchEnumerate          on_batch_enumerate,
               OnEpochEnumerate          on_epoch_enumerate,

               const bool                _init_weight = true,
               const int                 nbThreads = CNN_TASK_SIZE
               )
    {
        check_training_data(in, t);
        if (_init_weight)
            init_weight();
        layers_.set_parallelize(batch_size < CNN_TASK_SIZE);
        optimizer_.reset();

        for (int iter = 0; iter < epoch; iter++) {
            if (optimizer_.requires_hessian())
                calc_hessian(in);
            for (size_t i = 0; i < in.size(); i+=batch_size) {
                train_once(&in[i], &t[i], std::min(batch_size, in.size() - i), nbThreads);
                on_batch_enumerate();

                if (i % 100 == 0 && layers_.is_exploded()) {
                    std::cout << "[Warning]Detected infinite value in weight. stop learning." << std::endl;
                    return false;
                }
            }
            on_epoch_enumerate();
        }
        return true;
    }

    /**
     * training conv-net without callback
     **/
    template<typename T>
    bool train(const std::vector<vec_t>& in, const std::vector<T>& t, size_t batch_size = 1, int epoch = 1) {
        return train(in, t, batch_size, epoch, nop, nop);
    }

    /**
     * test and generate confusion-matrix for classification task
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


    std::vector<vec_t> test(const std::vector<vec_t>& in)
     {
            std::vector<vec_t> test_result(in.size());

            for_i(in.size(), [&](int i)
            {
                test_result[i] = predict(in[i]);
            });
            return test_result;
    }

    std::vector<float_t> scoreRegressor(const std::vector<vec_t>& in)
    {
            std::vector<float_t> test_result(in.size());

            for_i(in.size(), [&](int i)
            {
                test_result[i] = predict(in[i])[0];
            });
            return test_result;
    }


#ifdef CODE_READY
/**
 * @brief [TODO--------- NOT FINISHED]
 * @details [long description]
 * 
 * @param data [description]
 * @return [description]
 */
    image<float_t> fast_scoreRegressor(const float_t* data, const int rows, const int cols)
     {
        return test_image(image<float_t>(data, cols, rows));
     }

    image<float_t> fast_scoreRegressor(const image<float_t>& data)
    {
        int cols, rows, depth;
        image<float_t> res(data.width(),data.height());
        auto current = layers_.head();

        //int count = 0;
        image<float_t> in;
        image<float_t> out(data);
        //image<float_t> output = current->forward_propagation(in,0);

        while ((current = current->next()) != 0) 
        { 
            in = out;//transfer out of previous layer to in of current

            //format out image
            index3d<layer_size_t> shape =  current->out_shape();
            out = image<float_>(shape);

            cols = in.width(); rows = in.height();depth = in.depth();
            //std::cout<<++count<<std::endl;
            if (!current->layer_type().compare("conv"))
            {

                //convolve all the image (instead of the patch, so faster)
                vec_t& kernel = current->weight();//w
                vec_t& b = current->bias();
                
                //depth = out.depth();
                //convolution
                //for_i((cols-kernel_size)*(rows-kernel_size), [&](int count)
                for (int count = 0; count < (cols-kernel_size)*(rows-kernel_size); ++count)
                {
                    int mm, nn, ii, jj, m, n;
                    const int j = count / (cols-kernel_size) + kernel_size/2;
                    const int i = count % (cols-kernel_size) + kernel_size/2;

                    for_i(out.depth(), [&](int k)
                    {
                        //convolve
                        float sum = 0;
                        for (m = 0; m < kernel_size; ++m)
                        {
                            mm = kernel_size - 1 - m;
                            for (n = 0; n < kernel_size; ++n) {
                                nn = kernel_size - 1 - n;

                                ii = i + (m - kernel_size / 2);
                                jj = j + (n - kernel_size / 2);

                                //if (ii >= 0 && ii < rows && jj >= 0 && jj < cols) 
                                {
                                    float val = (in[ii * cols +jj]);
                                    //for (k=0;k<depth_;k++)
                                    sum += val *  kernel[shape.getIndex(nn,mm,k)];//+mm * kernel_size + nn];
                                }
                            }
                        }

                        //add the bias
                         for (m = 0; m < kernel_size; ++m)
                            for (n = 0; n < kernel_size; ++n)
                                out[k * (rows*cols) + i * cols + j] = sum + b[k * (kernel_size*kernel_size) + m * kernel_size + n];
                    }
                    );
                }
                //);

            }

            if (!current->layer_type().compare("max-pool"))
            {
                size_t size = current->pool_size();

                for_i(depth, [&](int k)
                {
                    for (int count = 0; count < out.width()*out.height(); ++count)
                    {
                        int mm, nn, ii, jj, m, n;
                        const int j = count / out.width();
                        const int i = count % out.width();
                        out[k * out.width()*out.height() + i  * out.width() + j] = in[k * width * height + i * size * width + j * size];    
                }
                );

            }


         }

        return res;

    }
#endif
    
    /**
     * calculate loss value (the smaller, the better) for regression task
     **/
    float_t get_loss(const std::vector<vec_t>& in, const std::vector<vec_t>& t) {
        float_t sum_loss = (float_t)0.0;

        for (size_t i = 0; i < in.size(); i++) {
            const vec_t predicted = predict(in[i]);
            sum_loss += get_loss(predict(in[i]), t[i]);
        }
        return sum_loss;
    }

    void save(std::ostream& os) const {
        os.precision(std::numeric_limits<tiny_cnn::float_t>::digits10);

        auto l = layers_.head();
        while (l) { l->save(os); l = l->next(); }
    }

    void load(std::istream& is) {
        is.precision(std::numeric_limits<tiny_cnn::float_t>::digits10);

        auto l = layers_.head();
        while (l) { l->load(is); l = l->next(); }
    }

    /**
     * checking gradients calculated by bprop
     * detail information:
     * http://ufldl.stanford.edu/wiki/index.php/Gradient_checking_and_advanced_optimization
     **/
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

    template <typename L, typename O>
    bool has_same_weights(const network<L, O>& others, float_t eps) const {
        auto h1 = layers_.head();
        auto h2 = others.layers_.head();

        while (h1 && h2) {
            if (!h1->has_same_weights(*h2, eps))
                return false;
            h1 = h1->next();
            h2 = h2->next();
        }
        return true;
    }

    template <typename T>
    const T& at(size_t index) const {
        return layers_.at<T>(index);
    }

    const layer_base* operator [] (size_t index) const {
        return layers_[index];
    }

    layer_base* operator [] (size_t index) {
        return layers_[index];
    }

    size_t depth() const {
        return layers_.depth();
    }

    template <typename WeightInit>
    network& weight_init(const WeightInit& f) {
        auto ptr = std::make_shared<WeightInit>(f);
        for (size_t i = 0; i < depth(); i++)
          layers_[i]->weight_init(ptr);
        return *this;
    }

    template <typename BiasInit>
    network& bias_init(const BiasInit& f) { 
        auto ptr = std::make_shared<BiasInit>(f);
        for (size_t i = 0; i < depth(); i++)
            layers_[i]->bias_init(ptr);
        return *this;
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

    void train_once(const vec_t* in, const label_t* t, int size, const int nbThreads = CNN_TASK_SIZE) {
        std::vector<vec_t> v;
        label2vector(t, size, &v);
        train_once(in, &v[0], size, nbThreads );
    }

    void train_once(const vec_t* in, const vec_t* t, int size, const int nbThreads = CNN_TASK_SIZE) {
        if (size == 1) {
            bprop(fprop(in[0]), t[0]);
            layers_.update_weights(&optimizer_, 1, 1);
        } else {
            train_onebatch(in, t, size, nbThreads);
        }
    }   

    void train_onebatch(const vec_t* in, const vec_t* t, int batch_size, const int num_tasks = CNN_TASK_SIZE) {
        task_group g;
        //int num_tasks = batch_size < CNN_TASK_SIZE ? 1 : CNN_TASK_SIZE;
        int data_per_thread = batch_size / num_tasks;
        int remaining = batch_size;

        // divide batch data and invoke [num_tasks] tasks
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
        // merge all dW and update W by optimizer
        layers_.update_weights(&optimizer_, num_tasks, batch_size);
    }

    void calc_hessian(const std::vector<vec_t>& in, int size_initialize_hessian = 500) {
        int size = std::min((int)in.size(), size_initialize_hessian);

        for (int i = 0; i < size; i++)
            bprop_2nd(fprop(in[i]));

        layers_.divide_hessian(size);
    }

    template<typename Activation>
    bool is_canonical_link(const Activation& h) {
        if (typeid(h) == typeid(activation::sigmoid) && typeid(E) == typeid(cross_entropy)) return true;
        if (typeid(h) == typeid(activation::tan_h) && typeid(E) == typeid(cross_entropy)) return true;
        if (typeid(h) == typeid(activation::identity) && typeid(E) == typeid(mse)) return true;
        if (typeid(h) == typeid(activation::softmax) && typeid(E) == typeid(cross_entropy_multiclass)) return true;
        return false;
    }

    const vec_t& fprop(const vec_t& in, int idx = 0) {
        if (in.size() != (size_t)in_dim())
            data_mismatch(*layers_[0], in);
        return layers_.head()->forward_propagation(in, idx);
    }

    float_t get_loss(const vec_t& out, const vec_t& t) {
        float_t e = 0.0;
        assert(out.size() == t.size());
        for_i(out.size(), [&](int i){ e += E::f(out[i], t[i]); });
        return e;
    }

    void bprop_2nd(const vec_t& out) {
        vec_t delta(out_dim());
        const activation::function& h = layers_.tail()->activation_function();

        if (is_canonical_link(h)) {
            for_i(out_dim(), [&](int i){ delta[i] = target_value_max() * h.df(out[i]);});
        } else {
            for_i(out_dim(), [&](int i){ delta[i] = target_value_max() * h.df(out[i]) * h.df(out[i]);}); // FIXME
        }

        layers_.tail()->back_propagation_2nd(delta);
    }

    void bprop(const vec_t& out, const vec_t& t, int idx = 0) {
        vec_t delta(out_dim());
        const activation::function& h = layers_.tail()->activation_function();

        if (is_canonical_link(h)) {
            for_i(out_dim(), [&](int i){ delta[i] = out[i] - t[i]; });
        } else {
            vec_t dE_dy = gradient<E>(out, t);

            // delta = dE/da = (dE/dy) * (dy/da)
            for (size_t i = 0; i < out_dim(); i++) {
                vec_t dy_da = h.df(out, i);
                delta[i] = vectorize::dot(&dE_dy[0], &dy_da[0], out_dim());
            }
        }

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

    void check_t(size_t i, label_t t, layer_size_t dim_out) {
        if (t >= dim_out) {
            std::ostringstream os;
            os << format_str("t[%u]=%u, dim(network output)=%u", i, t, dim_out) << std::endl;
            os << "in classification task, dim(network output) must be greater than max class id." << std::endl;
            if (dim_out == 1)
                os << std::endl << "(for regression, use vector<vec_t> instead of vector<label_t> for training signal)" << std::endl;

            throw nn_error("output dimension mismatch!\n " + os.str());
        }
    }

    void check_t(size_t i, const vec_t& t, layer_size_t dim_out) {
        if (t.size() != dim_out)
            throw nn_error(format_str("output dimension mismatch!\n dim(target[%u])=%u, dim(network output size=%u", i, t.size(), dim_out));
    }

    template <typename T>
    void check_training_data(const std::vector<vec_t>& in, const std::vector<T>& t) {
        layer_size_t dim_in = in_dim();
        layer_size_t dim_out = out_dim();

        if (in.size() != t.size())
            throw nn_error("number of training data must be equal to label data");

        size_t num = in.size();

        for (size_t i = 0; i < num; i++) {
            if (in[i].size() != dim_in)
                throw nn_error(format_str("input dimension mismatch!\n dim(data[%u])=%d, dim(network input)=%u", i, in[i].size(), dim_in));

            check_t(i, t[i], dim_out);
        }
    }

    float_t target_value_min() const { return layers_.tail()->activation_function().scale().first; }
    float_t target_value_max() const { return layers_.tail()->activation_function().scale().second; }

    std::string name_;
    Optimizer optimizer_;
    layers layers_;
};

/**
 * @brief [cut an image in samples to be tested (slow)]
 * @details [long description]
 * 
 * @param data [pointer to the data]
 * @param rows [self explained]
 * @param cols [self explained]
 * @param sizepatch [size of the patch, such as the total number of pixel in the patch is sizepatch*sizepatch ]
 * @return [vector of vec_c (sample) to be passed to test function]
 */
std::vector<vec_t> image2vec(const float_t* data, const unsigned int  rows, const unsigned int cols, const unsigned int sizepatch, const unsigned int step=1)
{
    assert(step>0);
    std::vector<vec_t> res((cols-sizepatch)*(rows-sizepatch)/(step*step),vec_t(sizepatch*sizepatch));
        for_i((cols-sizepatch)*(rows-sizepatch)/(step*step), [&](int count)
        {
            const int j = step*(count / ((cols-sizepatch)/step));
            const int i = step*(count % ((cols-sizepatch)/step));

            //vec_t sample(sizepatch*sizepatch);

            if (i+sizepatch < cols && j+sizepatch < rows)
            for (unsigned int k=0;k<sizepatch*sizepatch;k++)
            //for_i(sizepatch*sizepatch, [&](int k)
            {
                unsigned int y = k / sizepatch + j;
                unsigned int x = k % sizepatch + i;
                res[count][k] = data[x+y*cols];
            }
            //);
            //res[count] = (sample);
        });


    return res;
}

// void vec2image(const std::vector<vec_t> &in, float_t* out)
// {
//     //printf("%d\n",in.size());
//     for (int i=0;i<in.size();i++)
//     //for_i(in.size(), [&](int i)
//     {
//         out[i] = in[i][0];
//     }
//     //);
// }

/**
* create multi-layer perceptron
*/
template<typename loss_func, typename algorithm, typename activation, typename Iter>
network<loss_func, algorithm> make_mlp(Iter first, Iter last) {
    typedef network<loss_func, algorithm> net_t;
    net_t n;

    Iter next = first + 1;
    for (; next != last; ++first, ++next)
        n << fully_connected_layer<activation>(*first, *next);
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
    n.add(std::make_shared<Layer>(l));
    return n;
}

template <typename L, typename O, typename Layer>
network<L, O>& operator << (network<L, O>& n, Layer& l) {
    n.add(std::make_shared<Layer>(l));
    return n;
}

template <typename L, typename O, typename Char, typename CharTraits>
std::basic_ostream<Char, CharTraits>& operator << (std::basic_ostream<Char, CharTraits>& os, const network<L, O>& n) {
    n.save(os);
    return os;
}

template <typename L, typename O, typename Char, typename CharTraits>
std::basic_istream<Char, CharTraits>& operator >> (std::basic_istream<Char, CharTraits>& os, network<L, O>& n) {
    n.load(os);
    return os;
}

} // namespace tiny_cnn
