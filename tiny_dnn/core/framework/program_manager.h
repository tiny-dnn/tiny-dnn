/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include <unordered_map>

#include "tiny_dnn/layers/layer.h"

#include "tiny_dnn/core/framework/device.fwd.h"
#include "tiny_dnn/core/framework/program.h"

#if defined(USE_OPENCL) || defined(USE_CUDA)
#ifdef USE_OPENCL
#include "third_party/CLCudaAPI/clpp11.h"
#else
#include "third_party/CLCudaAPI/cupp11.h"
#endif
#endif

namespace tiny_dnn {

/* The class models a general manager to handle compiled OpenCL programs.
 * Since we need to retain compiled programs per layer type, it's
 * implemented as Singleton.
 */
class ProgramManager {
 public:
  /* This function is called to create an instance of the class.
   * Calling the constructor publicly is not allowed.
   * The constructor is private and is only called by this Instance function.
   */
  static ProgramManager &getInstance() {
    static ProgramManager instance;
    return instance;
  }

  /* Registers and compiles a kernel source code.
   *
   * Creates a new program based on the kernel string.
   * Note that the kernel string is moved-out when constructing the
   * program to save copying: it should no longer be used in the
   * remainder of this function.
   */
  void registerOp(const Device &device, layer &layer) {
#if defined(USE_OPENCL) || defined(USE_CUDA)
    // Register device to layer
    layer.setDevice(device);
    layer.createOp();

/*
        // retrieve incoming device an layer
        CLCudaAPI::Device  device_  = device.device();
        CLCudaAPI::Context context_ = device.context();

        // check if previous program was build with this
        // Devce and Layer.
        Program key_program(&device, &layer);

        auto iter = programs_.find(key_program);
        if (iter != programs_.end()) {
            nn_warn("Program already registered.");
            return;
        }

        // Define op kernel string and instantiate program
        // TODO(edgar): load from `cl_kernels` dir.
        // std::ifstream cl_file("opencl_hello_world.cl");
        std::ifstream cl_file(layer.kernel_file());
        std::string program_tail{std::istreambuf_iterator<char>(cl_file),
                                 std::istreambuf_iterator<char>()};
        // fixed kernel params
        std::string program_head =
            std::string("#define Dtype float\n") +
            std::string("#define Dtype4 float4\n") +
            std::string("#define int_tp int\n") +
            std::string("#define CONCAT(A,B) A##_##B\n") +
            std::string("#define TEMPLATE(name,type) CONCAT(name,type)\n");

        // per layer params
        program_head += layer.kernel_header();

        std::cout << layer.kernel_header() << std::endl;

        std::string program_string = std::string{program_head} +
   std::string{program_tail};
        auto program = CLCudaAPI::Program(context_, std::move(program_string));
*/
/*
 * Builds this program and checks for any compilation errors.
 * If there are any, they are printed and execution is halted.
 */
/*        nn_info("Compiling the kernel ...");
        auto compiler_options = std::vector<std::string>{};
        auto build_status = program.Build(device_, compiler_options);

        if (build_status != CLCudaAPI::BuildStatus::kSuccess) {
            auto message = program.GetBuildInfo(device_);
            //throw nn_error("Compiler error(s)/warning(s) found: " +
            //                to_string(message.c_str()));
            nn_warn("Compiler error(s)/warning(s) found: " +
                    to_string(message.c_str()));
            return;
        }
        nn_info("Compiling the kernel ... OK");

        // Kernel compilation succeed: Register program.
        programs_.insert({ key_program, program });
*/
#else  // USE_OPENCL OR USE_CUDA
    CNN_UNREFERENCED_PARAMETER(device);
    CNN_UNREFERENCED_PARAMETER(layer);
#endif
  }

  // Returns the number of registered programs
  size_t num_programs() const {
#if defined(USE_OPENCL) || defined(USE_CUDA)
    return programs_.size();
#else
    return size_t(0);
#endif
  }

// Returns a CLCudaProgram given a key Program
// based on internal device and op.
#if defined(USE_OPENCL) || defined(USE_CUDA)
  CLCudaAPI::Program program(const Program &program) {
    auto p = programs_.find(program);
    if (p == programs_.end()) {
      throw nn_error("Cannot retrieve program.");
    }
    return p->second;
  }
#endif

  // Removes the current programs from the general state
  void reset() {
#if defined(USE_OPENCL) || defined(USE_CUDA)
    programs_.clear();
#endif
  }

 protected:
  ProgramManager()                       = default;
  ProgramManager(const ProgramManager &) = delete;
  ProgramManager &operator=(const ProgramManager &) = delete;

#if defined(USE_OPENCL) || defined(USE_CUDA)
  /* Container holding compiled kernels */
  std::unordered_map<Program, CLCudaAPI::Program, ProgramHash> programs_;
#endif
};

}  // namespace tiny_dnn
