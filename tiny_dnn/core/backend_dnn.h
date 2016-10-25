// Copyright (c) 2016, Taiga Nomi, Edgar Riba. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#pragma once

#include "tiny_dnn/core/backend.h"

namespace tiny_dnn {
namespace core {

class dnn_backend : public backend {
 public:
    // context holds solution-dependent parameters
    // context should be able to hold any types of structures (like boost::any)
    dnn_backend() {}

    // core math functions

    void conv2d(const std::vector<tensor_t*>& in_data,
                std::vector<tensor_t*>&       out_data) override {
        throw nn_error("not implemented yet.");
    }

    void conv2d_q(const std::vector<tensor_t*>& in_data,
                  std::vector<tensor_t*>&       out_data) override {
        throw nn_error("not implemented yet.");
    }

    void conv2d_eq(const std::vector<tensor_t*>& in_data,
                   std::vector<tensor_t*>&       out_data) override {
        throw nn_error("not implemented yet.");
    }

    void conv2d(const std::vector<tensor_t*>& in_data,
                const std::vector<tensor_t*>& out_data,
                std::vector<tensor_t*>&       out_grad,
                std::vector<tensor_t*>&       in_grad) override {
        throw nn_error("not implemented yet.");
    }

    void conv2d_q(const std::vector<tensor_t*>& in_data,
                  const std::vector<tensor_t*>& out_data,
                  std::vector<tensor_t*>&       out_grad,
                  std::vector<tensor_t*>&       in_grad) override {
        throw nn_error("not implemented yet.");
    }

    void deconv2d(const std::vector<tensor_t*>& in_data,
                  std::vector<tensor_t*>&       out_data) override {
        throw nn_error("not implemented yet.");
    }

    void deconv2d_q(const std::vector<tensor_t*>& in_data,
                    std::vector<tensor_t*>&       out_data) override {
        throw nn_error("not implemented yet.");
    }

    void deconv2d_eq(const std::vector<tensor_t*>& in_data,
                     std::vector<tensor_t*>&       out_data) override {
        throw nn_error("not implemented yet.");
    }

    void deconv2d(const std::vector<tensor_t*>& in_data,
                  const std::vector<tensor_t*>& out_data,
                  std::vector<tensor_t*>&       out_grad,
                  std::vector<tensor_t*>&       in_grad) override {
        throw nn_error("not implemented yet.");
    }

    void deconv2d_q(const std::vector<tensor_t*>& in_data,
                    const std::vector<tensor_t*>& out_data,
                    std::vector<tensor_t*>&       out_grad,
                    std::vector<tensor_t*>&       in_grad) override {
        throw nn_error("not implemented yet.");
    }

    void maxpool(const std::vector<tensor_t*>& in_data,
                 std::vector<tensor_t*>&       out_data) override {
        throw nn_error("not implemented yet.");
    }

    void maxpool(const std::vector<tensor_t*>& in_data,
                 const std::vector<tensor_t*>& out_data,
                 std::vector<tensor_t*>&       out_grad,
                 std::vector<tensor_t*>&       in_grad) override {
        throw nn_error("not implemented yet.");
    }

    void fully(const std::vector<tensor_t*>& in_data,
               std::vector<tensor_t*>&       out_data) override {
        throw nn_error("not implemented yet.");
    }

    void fully_q(const std::vector<tensor_t*>& in_data,
                 std::vector<tensor_t*>&       out_data) override {
        throw nn_error("not implemented yet.");
    }

    void fully_eq(const std::vector<tensor_t*>& in_data,
                  std::vector<tensor_t*>&       out_data) override {
        throw nn_error("not implemented yet.");
    }

    void fully(const std::vector<tensor_t*>& in_data,
               const std::vector<tensor_t*>& out_data,
               std::vector<tensor_t*>&       out_grad,
               std::vector<tensor_t*>&       in_grad) override {
        throw nn_error("not implemented yet.");
    }

    void fully_q(const std::vector<tensor_t*>& in_data,
                 const std::vector<tensor_t*>& out_data,
                 std::vector<tensor_t*>&       out_grad,
                 std::vector<tensor_t*>&       in_grad) override {
        throw nn_error("not implemented yet.");
    }

    backend_t type() const override { return backend_t::libdnn; }
};

}  // namespace core
}  // namespace tiny_dnn
