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

    * Neither the name of tiny-dnn nor the names of its
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

namespace tiny_dnn {

/* Class modelling a Tensor
 */
class Tensor {
 public:
    /* Default constructor for the Tensor class
     * @param batch The size of the batch
     * @param width The width of the tensor
     * @param heigth The height of the tensor
     * @param depth The number of channels
     */
    explicit Tensor(const cnn_size_t batch,
                    const cnn_size_t width,
                    const cnn_size_t height,
                    const cnn_size_t depth) : shape_(4) {
        init_data (batch, width, height, depth);
        init_shape(batch, width, height, depth);
    }
    
    // Returns the tensor shape
    std::vector<cnn_size_t> shape() const { return shape_; }
    
    // Returns the value of a requested position in the tensor
    float_t at(const cnn_size_t batch,  const cnn_size_t width,
               const cnn_size_t height, const cnn_size_t depth) const {
        return *access_data(batch, width, height, depth);
    }

    // Returns the pointer to a requested position in the tensor
    float_t* ptr(const cnn_size_t batch,  const cnn_size_t width,
                 const cnn_size_t height, const cnn_size_t depth) const {
        return access_data(batch, width, height, depth);
    }

 private:
    // Initializes the data buffer with zeroes
    void init_data(const cnn_size_t batch,  const cnn_size_t width,
                   const cnn_size_t height, const cnn_size_t depth) {
        if (batch <= 0 || width <= 0 || height <= 0 || depth <= 0) {
            nn_error("Tensor size must be a positive number");
        }
        data_cpu_ = std::make_shared<std::vector<float_t> >(
            batch * width * height * depth, float_t(0));
    }

    // Initializes the shape vector
    void init_shape(const cnn_size_t batch,  const cnn_size_t width,
                    const cnn_size_t height, const cnn_size_t depth) {
       shape_[0] = batch;  shape_[1] = width;
       shape_[2] = height; shape_[3] = depth;
    }

    // Method to access to the tensor data.
    // It checks if the requested position is feasible or not.
    float_t* access_data(const cnn_size_t batch,
                         const cnn_size_t width,
                         const cnn_size_t height,
                         const cnn_size_t depth) const {
        if ((batch  < 0 || batch  > shape_[0]) ||
            (width  < 0 || width  > shape_[1]) ||
            (height < 0 || height > shape_[2]) ||
            (depth  < 0 || depth  > shape_[3])) {
            nn_error("Access tensor out of dimension");
        }
        /*return &data_cpu_->at(batch  * shape_[1] * shape_[2] * shape_[3] + 
                                depth  * shape_[1] * shape_[2] +
                                height * shape_[1] + width);*/
        return &data_cpu_->at(shape_[1] * shape_[2] * (
                      batch * shape_[3] + depth) + height + width);
    }


 private:
    /* Vector with the size of the tensor
     * shape_[0]: batch
     * shape_[1]: width
     * shape_[2]: height
     * shape_[3]: depth
     */
    std::vector<cnn_size_t> shape_;

    /* Pointer to the Tensor data in CPU mode */
    std::shared_ptr<std::vector<float_t> > data_cpu_;
};

// Overloaded method to print the Tensor class to the standard output
inline std::ostream& operator<< (std::ostream &os, const Tensor& tensor) {
    std::vector<cnn_size_t> shape = tensor.shape();
    for (cnn_size_t i = 0; i < shape[0]; ++i) {
        os << "-- Batch: " << i << "\n";
        for (cnn_size_t j = 0; j < shape[3]; ++j) {
            os << "-- Channel: " << j << "\n";
            os << "-- Data:\n";
            for (cnn_size_t k = 0; k < shape[1]; ++k) {
                for (cnn_size_t l = 0; l < shape[2]; ++l) {
                    os << tensor.at(i,k,l,j) << " ";
                }
                os << ";\n";
            }
        }
    }
    os << "----------------\n" 
       << "--> Tensor size: [ " 
       << shape[0] << " x " << shape[1] << " x "
       << shape[2] << " x " << shape[3] << " ]\n";
    return os;
}

}  // namespace tiny_dnn
