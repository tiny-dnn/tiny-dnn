/*
    COPYRIGHT

    All contributions by Taiga Nomi
    Copyright (c) 2013, Taiga Nomi
    All rights reserved.

    All other contributions:
    Copyright (c) 2013-2016, the respective contributors.
    All rights reserved.

    Each contributor holds copyright over their respective contributions.
    The project versioning (Git) records all such contribution source information.

    LICENSE

    The BSD 3-Clause License


    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice, this
      list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright notice,
      this list of conditions and the following disclaimer in the documentation
      and/or other materials provided with the distribution.

    * Neither the name of tiny-cnn nor the names of its
      contributors may be used to endorse or promote products derived from
      this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
    FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
    DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
    CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
    OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
#pragma once
#include <sstream>
#include <iomanip>
#include <memory>
#include <numeric>
#include <algorithm>
#include <vector>
#include <string>
#include <utility>
#include <queue>

#include "tiny_cnn/node.h"

#include "tiny_cnn/core/backend.h"
#include "tiny_cnn/core/params/conv_params.h"

#include "tiny_cnn/core/framework/device.fwd.h"

#include "tiny_cnn/util/util.h"
#include "tiny_cnn/util/product.h"
#include "tiny_cnn/util/image.h"
#include "tiny_cnn/util/weight_init.h"

#include "tiny_cnn/optimizers/optimizer.h"
#include "tiny_cnn/activations/activation_function.h"

#ifdef CNN_USE_LIBDNN
#include "libdnn.hpp"
#endif

#ifdef USE_OPENCL
#include "third_party/CLCudaAPI/clpp11.h"
#endif

namespace tiny_cnn {

/**
 * base class of all kind of NN layers
 *
 * sub-class should override these methods:
 * - forward_propagation ... body of forward-pass calculation
 * - back_propagation    ... body of backward-pass calculation
 * - in_shape            ... specify input data shapes
 * - out_shape           ... specify output data shapes
 * - layer_type          ... name of layer
 **/
class layer : public node {
 public:
    friend void connection_mismatch(const layer& from,
                                    const layer& to);

    virtual ~layer() = default;

    /**
     * construct N-input, M-output layer
     * @param in_type[N] type of input vector (data, weight, bias...)
     * @param out_type[M] type of output vector
     **/
    layer(const std::vector<vector_type>& in_type,
          const std::vector<vector_type>& out_type);

    layer(const layer&) = default;
    layer &operator =(const layer&) = default;

#ifdef CNN_USE_DEFAULT_MOVE_CONSTRUCTORS
    layer(layer&&) = default;
    layer &operator = (layer&&) = default;
#endif

    void set_parallelize(bool parallelize);
    void set_backend(std::shared_ptr<core::backend> backend);

    // Creates a new program based on the kernel string. Note that the kernel string is moved-out when
	// constructing the program to save copying: it should no longer be used in the remainder of this
	// function.
#if defined(USE_OPENCL) || defined(USE_CUDA)
    void tune_kernel(const std::string& program_string,
                     std::vector<std::string>& compiler_options,
                     const CLCudaAPI::Context& context,
                     const CLCudaAPI::Device& device);
#endif  // USE_OPENCL OR USE_CUDA

#ifdef CNN_USE_LIBDNN
    void tune_kernel(const CLCudaAPI::Context& context,
                     const CLCudaAPI::Device& device,
                     const CLCudaAPI::Queue& queue,
                     const int id,
                     const int id_list,
                     const core::conv_params& params);
#endif

    /////////////////////////////////////////////////////////////////////////
    // getter

    bool parallelize() const; 
    bool initialized() const;

    core::backend_t backend_type() const;

    std::shared_ptr<core::backend> backend();

    device* get_device() { return device_; }

    void set_device(device* device) {
        device_ = device;
    }

    core::params params() const; 

    ///< number of incoming edges in this layer
    cnn_size_t in_channels() const;

    ///< number of outgoing edges in this layer
    cnn_size_t out_channels() const;

    cnn_size_t in_data_size() const; 
    cnn_size_t out_data_size() const; 

    std::vector<shape3d> in_data_shape();
    std::vector<shape3d> out_data_shape();

    ///! @deprecated use in_data_size() instead
    cnn_size_t in_size() const;

    ///! @deprecated use out_data_size() instead
    cnn_size_t out_size() const;
    
    std::vector<const vec_t*> weights() const;
    std::vector<vec_t*> weights();

    std::vector<tensor_t*> weight_grads();

    std::vector<edgeptr_t> inputs();

    std::vector<edgeptr_t> outputs();
    std::vector<edgeptr_t> outputs() const;

    void set_out_grads(const std::vector<tensor_t>& grad);
    void set_in_data(const std::vector<tensor_t>& data);

    std::vector<tensor_t> output() const;

    std::vector<vector_type> in_types() const;
    std::vector<vector_type> out_types() const;

    /**
     * return output value range
     * used only for calculating target value from label-id in final(output) layer
     * override properly if the layer is intended to be used as output layer
     **/
    virtual std::pair<float_t, float_t> out_value_range() const;

    /**
     * array of input shapes (width x height x depth)
     **/
    virtual std::vector<shape3d> in_shape() const = 0;

    /**
     * array of output shapes (width x height x depth)
     **/
    virtual std::vector<shape3d> out_shape() const = 0;

    /**
     * name of layer, should be unique for each concrete class
     **/
    virtual std::string layer_type() const = 0;

    /**
     * number of incoming connections for each output unit
     * used only for weight/bias initialization methods which require fan-in size (e.g. xavier)
     * override if the layer has trainable weights, and scale of initialization is important
     **/
    virtual size_t fan_in_size() const;

    /**
     * number of outgoing connections for each input unit
     * used only for weight/bias initialization methods which require fan-out size (e.g. xavier)
     * override if the layer has trainable weights, and scale of initialization is important
     **/
    virtual size_t fan_out_size() const;

    /////////////////////////////////////////////////////////////////////////
    // setter
    template <typename WeightInit>
    layer& weight_init(const WeightInit& f);

    template <typename BiasInit>
    layer& bias_init(const BiasInit& f);

    template <typename WeightInit>
    layer& weight_init(std::shared_ptr<WeightInit> f);

    template <typename BiasInit>
    layer& bias_init(std::shared_ptr<BiasInit> f);

    /////////////////////////////////////////////////////////////////////////
    // save/load

    virtual void save(std::ostream& os) const;

    virtual void load(std::istream& is);

    virtual void load(const std::vector<float_t>& src, int& idx);

    /////////////////////////////////////////////////////////////////////////
    // visualize

    ///< visualize latest output of this layer
    ///< default implementation interpret output as 1d-vector,
    ///< so "visual" layer(like convolutional layer) should override this for better visualization.
    virtual image<> output_to_image(size_t channel = 0) const;

    /////////////////////////////////////////////////////////////////////////
    // fprop/bprop

    /**
     * @param in_data      input vectors of this layer (data, weight, bias)
     * @param out_data     output vectors
     **/
    virtual void forward_propagation(const std::vector<tensor_t*>& in_data,
                                     std::vector<tensor_t*>& out_data) = 0;

    /**
     * return delta of previous layer (delta=\frac{dE}{da}, a=wx in fully-connected layer)
     * @param in_data      input vectors (same vectors as forward_propagation)
     * @param out_data     output vectors (same vectors as forward_propagation)
     * @param out_grad     gradient of output vectors (i-th vector correspond with out_data[i])
     * @param in_grad      gradient of input vectors (i-th vector correspond with in_data[i])
     **/
    virtual void back_propagation(const std::vector<tensor_t*>& in_data,
                                  const std::vector<tensor_t*>& out_data,
                                  std::vector<tensor_t*>&       out_grad,
                                  std::vector<tensor_t*>&       in_grad) = 0;

    /**
     * return delta2 of previous layer (delta2=\frac{d^2E}{da^2}, diagonal of hessian matrix)
     * it is never called if optimizer is hessian-free
     **/
    //virtual void back_propagation_2nd(const std::vector<vec_t>& delta_in) = 0;

    // called afrer updating weight
    virtual void post_update();

    /**
    * notify changing context (train <=> test)
    **/
    virtual void set_context(net_phase ctx);

    std::vector<tensor_t> forward(const std::vector<tensor_t>& input);

    std::vector<tensor_t> backward(const std::vector<tensor_t>& out_grads);

    void forward();
    void backward();

    // allocate & reset weight
    void setup(bool reset_weight);

    void init_weight();
    void clear_grads();

    void update_weight(optimizer *o, cnn_size_t batch_size);
    bool has_same_weights(const layer& rhs, float_t eps) const;

    virtual void set_sample_count(cnn_size_t sample_count);

 protected:
    bool initialized_;
    bool parallelize_;
    cnn_size_t in_channels_;   // number of input vectors
    cnn_size_t out_channels_;  // number of output vectors
    std::vector<vector_type> in_type_;
    std::vector<vector_type> out_type_;

    std::shared_ptr<core::backend> backend_;

    // TODO(edgar): check if remove or not
    device* device_;
    core::params params_;

 private:
    std::shared_ptr<weight_init::function> weight_init_;
    std::shared_ptr<weight_init::function> bias_init_;

    void alloc_input(cnn_size_t i) const;
    void alloc_output(cnn_size_t i) const;
    
    edgeptr_t ith_in_node(cnn_size_t i);
    edgeptr_t ith_out_node(cnn_size_t i);

    vec_t* get_weight_data(cnn_size_t i);
    const vec_t* get_weight_data(cnn_size_t i) const;
};

}  // namespace tiny_cnn
