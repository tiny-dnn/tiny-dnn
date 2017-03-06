
// =================================================================================================
// This file is part of the CLCudaAPI project. The project is licensed under Apache Version 2.0. The
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements a bunch of C++11 classes that act as wrappers around OpenCL objects and API
// calls. The main benefits are increased abstraction, automatic memory management, and portability.
// Portability here means that a similar header exists for CUDA with the same classes and
// interfaces. In other words, moving from the OpenCL API to the CUDA API becomes a one-line change.
//
// This is version 8.0 of CLCudaAPI.
//
// =================================================================================================
//
// Copyright 2015 SURFsara
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// 
//  http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// =================================================================================================

#ifndef CLCUDAAPI_CLPP11_H_
#define CLCUDAAPI_CLPP11_H_

// C++
#include <algorithm> // std::copy
#include <string>    // std::string
#include <vector>    // std::vector
#include <memory>    // std::shared_ptr
#include <stdexcept> // std::runtime_error
#include <numeric>   // std::accumulate

// OpenCL
#if defined(__APPLE__) || defined(__MACOSX)
  #include <OpenCL/opencl.h>
#else
  #include <CL/opencl.h>
#endif

namespace CLCudaAPI {
// =================================================================================================

// Error occurred in the C++11 OpenCL header (this file)
inline void Error(const std::string &message) {
  throw std::runtime_error("Internal OpenCL error: "+message);
}

// Error occurred in OpenCL
inline void CheckError(const cl_int status) {
  if (status != CL_SUCCESS) {
    throw std::runtime_error("Internal OpenCL error: "+std::to_string(status));
  }
}

// =================================================================================================

// C++11 version of 'cl_event'
class Event {
 public:

  // Constructor based on the regular OpenCL data-type: memory management is handled elsewhere
  explicit Event(const cl_event event):
      event_(new cl_event) {
    *event_ = event;
  }

  // Regular constructor with memory management
  explicit Event():
      event_(new cl_event, [](cl_event* e) {
        if (*e) { CheckError(clReleaseEvent(*e)); }
        delete e;
      }) {
    *event_ = nullptr;
  }

  // Waits for completion of this event
  void WaitForCompletion() const {
    CheckError(clWaitForEvents(1, &(*event_)));
  }

  // Retrieves the elapsed time of the last recorded event. Note that no error checking is done on
  // the 'clGetEventProfilingInfo' function, since there is a bug in Apple's OpenCL implementation:
  // http://stackoverflow.com/questions/26145603/clgeteventprofilinginfo-bug-in-macosx
  float GetElapsedTime() const {
    WaitForCompletion();
    const auto bytes = sizeof(cl_ulong);
    auto time_start = cl_ulong{0};
    clGetEventProfilingInfo(*event_, CL_PROFILING_COMMAND_START, bytes, &time_start, nullptr);
    auto time_end = cl_ulong{0};
    clGetEventProfilingInfo(*event_, CL_PROFILING_COMMAND_END, bytes, &time_end, nullptr);
    return static_cast<float>(time_end - time_start) * 1.0e-6f;
  }

  // Accessor to the private data-member
  cl_event& operator()() { return *event_; }
  const cl_event& operator()() const { return *event_; }
  cl_event* pointer() { return &(*event_); }
  const cl_event* pointer() const { return &(*event_); }
 private:
  std::shared_ptr<cl_event> event_;
};

// Pointer to an OpenCL event
using EventPointer = cl_event*;

// =================================================================================================

// C++11 version of 'cl_platform_id'
class Platform {
 public:

  // Constructor based on the regular OpenCL data-type
  explicit Platform(const cl_platform_id platform): platform_(platform) { }

  // Initializes the platform
  explicit Platform(const size_t platform_id) {
    auto num_platforms = cl_uint{0};
    CheckError(clGetPlatformIDs(0, nullptr, &num_platforms));
    if (num_platforms == 0) { Error("no platforms found"); }
    auto platforms = std::vector<cl_platform_id>(num_platforms);
    CheckError(clGetPlatformIDs(num_platforms, platforms.data(), nullptr));
    if (platform_id >= num_platforms) { Error("invalid platform ID "+std::to_string(platform_id)); }
    platform_ = platforms[platform_id];
  }

  // Returns the number of devices on this platform
  size_t NumDevices() const {
    auto result = cl_uint{0};
    CheckError(clGetDeviceIDs(platform_, CL_DEVICE_TYPE_ALL, 0, nullptr, &result));
    return static_cast<size_t>(result);
  }

  // Accessor to the private data-member
  const cl_platform_id& operator()() const { return platform_; }
 private:
  cl_platform_id platform_;
};

// Retrieves a vector with all platforms
inline std::vector<Platform> GetAllPlatforms() {
  auto num_platforms = cl_uint{0};
  CheckError(clGetPlatformIDs(0, nullptr, &num_platforms));
  auto all_platforms = std::vector<Platform>();
  for (size_t platform_id = 0; platform_id < static_cast<size_t>(num_platforms); ++platform_id) {
    all_platforms.push_back(Platform(platform_id));
  }
  return all_platforms;
}

// =================================================================================================

// C++11 version of 'cl_device_id'
class Device {
 public:

  // Constructor based on the regular OpenCL data-type
  explicit Device(const cl_device_id device): device_(device) { }

  // Initialize the device. Note that this constructor can throw exceptions!
  explicit Device(const Platform &platform, const size_t device_id) {
    auto num_devices = platform.NumDevices();
    if (num_devices == 0) { Error("no devices found"); }
    auto devices = std::vector<cl_device_id>(num_devices);
    CheckError(clGetDeviceIDs(platform(), CL_DEVICE_TYPE_ALL, static_cast<cl_uint>(num_devices),
                              devices.data(), nullptr));
    if (device_id >= num_devices) { Error("invalid device ID "+std::to_string(device_id)); }
    device_ = devices[device_id];
  }

  // Methods to retrieve device information
  std::string Version() const { return GetInfoString(CL_DEVICE_VERSION); }
  size_t VersionNumber() const
  {
    std::string version_string = Version().substr(7);
    // Space separates the end of the OpenCL version number from the beginning of the
    // vendor-specific information.
    size_t next_whitespace = version_string.find(' ');
    size_t version = (size_t) (100.0 * std::stod(version_string.substr(0, next_whitespace)));
    return version;
  }
  std::string Vendor() const { return GetInfoString(CL_DEVICE_VENDOR); }
  std::string Name() const { return GetInfoString(CL_DEVICE_NAME); }
  std::string Type() const {
    auto type = GetInfo<cl_device_type>(CL_DEVICE_TYPE);
    switch(type) {
      case CL_DEVICE_TYPE_CPU: return "CPU";
      case CL_DEVICE_TYPE_GPU: return "GPU";
      case CL_DEVICE_TYPE_ACCELERATOR: return "accelerator";
      default: return "default";
    }
  }
  size_t MaxWorkGroupSize() const { return GetInfo<size_t>(CL_DEVICE_MAX_WORK_GROUP_SIZE); }
  size_t MaxWorkItemDimensions() const {
    return static_cast<size_t>(GetInfo<cl_uint>(CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS));
  }
  std::vector<size_t> MaxWorkItemSizes() const {
    return GetInfoVector<size_t>(CL_DEVICE_MAX_WORK_ITEM_SIZES);
  }
  unsigned long LocalMemSize() const {
    return static_cast<unsigned long>(GetInfo<cl_ulong>(CL_DEVICE_LOCAL_MEM_SIZE));
  }
  std::string Capabilities() const { return GetInfoString(CL_DEVICE_EXTENSIONS); }
  size_t CoreClock() const {
    return static_cast<size_t>(GetInfo<cl_uint>(CL_DEVICE_MAX_CLOCK_FREQUENCY));
  }
  size_t ComputeUnits() const {
    return static_cast<size_t>(GetInfo<cl_uint>(CL_DEVICE_MAX_COMPUTE_UNITS));
  }
  unsigned long MemorySize() const {
    return static_cast<unsigned long>(GetInfo<cl_ulong>(CL_DEVICE_GLOBAL_MEM_SIZE));
  }
  unsigned long MaxAllocSize() const {
    return static_cast<unsigned long>(GetInfo<cl_ulong>(CL_DEVICE_MAX_MEM_ALLOC_SIZE));
  }
  size_t MemoryClock() const { return 0; } // Not exposed in OpenCL
  size_t MemoryBusWidth() const { return 0; } // Not exposed in OpenCL

  // Configuration-validity checks
  bool IsLocalMemoryValid(const cl_ulong local_mem_usage) const {
    return (local_mem_usage <= LocalMemSize());
  }
  bool IsThreadConfigValid(const std::vector<size_t> &local) const {
    auto local_size = size_t{1};
    for (const auto &item: local) { local_size *= item; }
    for (auto i=size_t{0}; i<local.size(); ++i) {
      if (local[i] > MaxWorkItemSizes()[i]) { return false; }
    }
    if (local_size > MaxWorkGroupSize()) { return false; }
    if (local.size() > MaxWorkItemDimensions()) { return false; }
    return true;
  }

  // Query for a specific type of device or brand
  bool IsCPU() const { return Type() == "CPU"; }
  bool IsGPU() const { return Type() == "GPU"; }
  bool IsAMD() const { return Vendor() == "AMD" || Vendor() == "Advanced Micro Devices, Inc." ||
                              Vendor() == "AuthenticAMD";; }
  bool IsNVIDIA() const { return Vendor() == "NVIDIA" || Vendor() == "NVIDIA Corporation"; }
  bool IsIntel() const { return Vendor() == "INTEL" || Vendor() == "Intel" ||
                                Vendor() == "GenuineIntel"; }
  bool IsARM() const { return Vendor() == "ARM"; }

  // Accessor to the private data-member
  const cl_device_id& operator()() const { return device_; }
 private:
  cl_device_id device_;

  // Private helper functions
  template <typename T>
  T GetInfo(const cl_device_info info) const {
    auto bytes = size_t{0};
    CheckError(clGetDeviceInfo(device_, info, 0, nullptr, &bytes));
    auto result = T(0);
    CheckError(clGetDeviceInfo(device_, info, bytes, &result, nullptr));
    return result;
  }
  template <typename T>
  std::vector<T> GetInfoVector(const cl_device_info info) const {
    auto bytes = size_t{0};
    CheckError(clGetDeviceInfo(device_, info, 0, nullptr, &bytes));
    auto result = std::vector<T>(bytes/sizeof(T));
    CheckError(clGetDeviceInfo(device_, info, bytes, result.data(), nullptr));
    return result;
  }
  std::string GetInfoString(const cl_device_info info) const {
    auto bytes = size_t{0};
    CheckError(clGetDeviceInfo(device_, info, 0, nullptr, &bytes));
    auto result = std::string{};
    result.resize(bytes);
    CheckError(clGetDeviceInfo(device_, info, bytes, &result[0], nullptr));
    return std::string{result.c_str()}; // Removes any trailing '\0'-characters
  }
};

// =================================================================================================

// C++11 version of 'cl_context'
class Context {
 public:

  // Constructor based on the regular OpenCL data-type: memory management is handled elsewhere
  explicit Context(const cl_context context):
      context_(new cl_context) {
    *context_ = context;
  }

  // Regular constructor with memory management
  explicit Context(const Device &device):
      context_(new cl_context, [](cl_context* c) { CheckError(clReleaseContext(*c)); delete c; }) {
    auto status = CL_SUCCESS;
    const cl_device_id dev = device();
    *context_ = clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &status);
    CheckError(status);
  }

  // Accessor to the private data-member
  const cl_context& operator()() const { return *context_; }
  cl_context* pointer() const { return &(*context_); }
 private:
  std::shared_ptr<cl_context> context_;
};

// Pointer to an OpenCL context
using ContextPointer = cl_context*;

// =================================================================================================

// Enumeration of build statuses of the run-time compilation process
enum class BuildStatus { kSuccess, kError, kInvalid };

// C++11 version of 'cl_program'. Additionally holds the program's source code.
class Program {
 public:
  // Note that there is no constructor based on the regular OpenCL data-type because of extra state

  // Source-based constructor with memory management
  explicit Program(const Context &context, std::string source):
      program_(new cl_program, [](cl_program* p) { CheckError(clReleaseProgram(*p)); delete p; }),
      length_(source.length()),
      source_(std::move(source)),
      source_ptr_(&source_[0]) {
    auto status = CL_SUCCESS;
    *program_ = clCreateProgramWithSource(context(), 1, &source_ptr_, &length_, &status);
    CheckError(status);
  }

  // Binary-based constructor with memory management
  explicit Program(const Device &device, const Context &context, const std::string& binary):
      program_(new cl_program, [](cl_program* p) { CheckError(clReleaseProgram(*p)); delete p; }),
      length_(binary.length()),
      source_(binary),
      source_ptr_(&source_[0]) {
    auto status1 = CL_SUCCESS;
    auto status2 = CL_SUCCESS;
    const cl_device_id dev = device();
    *program_ = clCreateProgramWithBinary(context(), 1, &dev, &length_,
                                          reinterpret_cast<const unsigned char**>(&source_ptr_),
                                          &status1, &status2);
    CheckError(status1);
    CheckError(status2);
  }

  // Compiles the device program and returns whether or not there where any warnings/errors
  BuildStatus Build(const Device &device, std::vector<std::string> &options) {
    auto options_string = std::accumulate(options.begin(), options.end(), std::string{" "});
    const cl_device_id dev = device();
    auto status = clBuildProgram(*program_, 1, &dev, options_string.c_str(), nullptr, nullptr);
    if (status == CL_BUILD_PROGRAM_FAILURE) {
      return BuildStatus::kError;
    }
    else if (status == CL_INVALID_BINARY) {
      return BuildStatus::kInvalid;
    }
    else {
      CheckError(status);
      return BuildStatus::kSuccess;
    }
  }

  // Retrieves the warning/error message from the compiler (if any)
  std::string GetBuildInfo(const Device &device) const {
    auto bytes = size_t{0};
    auto query = cl_program_build_info{CL_PROGRAM_BUILD_LOG};
    CheckError(clGetProgramBuildInfo(*program_, device(), query, 0, nullptr, &bytes));
    auto result = std::string{};
    result.resize(bytes);
    CheckError(clGetProgramBuildInfo(*program_, device(), query, bytes, &result[0], nullptr));
    return result;
  }

  // Retrieves a binary or an intermediate representation of the compiled program
  std::string GetIR() const {
    auto bytes = size_t{0};
    CheckError(clGetProgramInfo(*program_, CL_PROGRAM_BINARY_SIZES, sizeof(size_t), &bytes, nullptr));
    auto result = std::string{};
    result.resize(bytes);
    auto result_ptr = result.data();
    CheckError(clGetProgramInfo(*program_, CL_PROGRAM_BINARIES, sizeof(char*), &result_ptr, nullptr));
    return result;
  }

  // Accessor to the private data-member
  const cl_program& operator()() const { return *program_; }
 private:
  std::shared_ptr<cl_program> program_;
  size_t length_;
  std::string source_; // Note: the source can also be a binary or IR
  const char* source_ptr_;
};

// =================================================================================================

// C++11 version of 'cl_command_queue'
class Queue {
 public:

  // Constructor based on the regular OpenCL data-type: memory management is handled elsewhere
  explicit Queue(const cl_command_queue queue):
      queue_(new cl_command_queue) {
    *queue_ = queue;
  }

  // Regular constructor with memory management
  explicit Queue(const Context &context, const Device &device):
      queue_(new cl_command_queue, [](cl_command_queue* s) { CheckError(clReleaseCommandQueue(*s));
                                                             delete s; }) {
    auto status = CL_SUCCESS;
    #ifdef CL_VERSION_2_0
      size_t ocl_version = device.VersionNumber();
      if (ocl_version >= 200)
      {
        cl_queue_properties properties[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
        *queue_ = clCreateCommandQueueWithProperties(context(), device(), properties, &status);
      }
      else
      {
        *queue_ = clCreateCommandQueue(context(), device(), CL_QUEUE_PROFILING_ENABLE, &status);
      }
    #else
      *queue_ = clCreateCommandQueue(context(), device(), CL_QUEUE_PROFILING_ENABLE, &status);
    #endif
    CheckError(status);
  }

  // Synchronizes the queue
  void Finish(Event &) const {
    Finish();
  }
  void Finish() const {
    CheckError(clFinish(*queue_));
  }

  // Retrieves the corresponding context or device
  Context GetContext() const {
    auto bytes = size_t{0};
    CheckError(clGetCommandQueueInfo(*queue_, CL_QUEUE_CONTEXT, 0, nullptr, &bytes));
    cl_context result;
    CheckError(clGetCommandQueueInfo(*queue_, CL_QUEUE_CONTEXT, bytes, &result, nullptr));
    return Context(result);
  }
  Device GetDevice() const {
    auto bytes = size_t{0};
    CheckError(clGetCommandQueueInfo(*queue_, CL_QUEUE_DEVICE, 0, nullptr, &bytes));
    cl_device_id result;
    CheckError(clGetCommandQueueInfo(*queue_, CL_QUEUE_DEVICE, bytes, &result, nullptr));
    return Device(result);
  }

  // Accessor to the private data-member
  const cl_command_queue& operator()() const { return *queue_; }
 private:
  std::shared_ptr<cl_command_queue> queue_;
};

// =================================================================================================

// C++11 version of host memory
template <typename T>
class BufferHost {
 public:

  // Regular constructor with memory management
  explicit BufferHost(const Context &, const size_t size):
      buffer_(new std::vector<T>(size)) {
  }

  // Retrieves the actual allocated size in bytes
  size_t GetSize() const {
    return buffer_->size()*sizeof(T);
  }

  // Compatibility with std::vector
  size_t size() const { return buffer_->size(); }
  T* begin() { return &(*buffer_)[0]; }
  T* end() { return &(*buffer_)[buffer_->size()-1]; }
  T& operator[](const size_t i) { return (*buffer_)[i]; }
  T* data() { return buffer_->data(); }
  const T* data() const { return buffer_->data(); }

 private:
  std::shared_ptr<std::vector<T>> buffer_;
};

// =================================================================================================

// Enumeration of buffer access types
enum class BufferAccess { kReadOnly, kWriteOnly, kReadWrite, kNotOwned };

// C++11 version of 'cl_mem'
template <typename T>
class Buffer {
 public:

  // Constructor based on the regular OpenCL data-type: memory management is handled elsewhere
  explicit Buffer(const cl_mem buffer):
      buffer_(new cl_mem),
      access_(BufferAccess::kNotOwned) {
    *buffer_ = buffer;
  }

  // Regular constructor with memory management. If this class does not own the buffer object, then
  // the memory will not be freed automatically afterwards.
  explicit Buffer(const Context &context, const BufferAccess access, const size_t size):
      buffer_(new cl_mem, [access](cl_mem* m) {
        if (access != BufferAccess::kNotOwned) { CheckError(clReleaseMemObject(*m)); }
        delete m;
      }),
      access_(access) {
    auto flags = cl_mem_flags{CL_MEM_READ_WRITE};
    if (access_ == BufferAccess::kReadOnly) { flags = CL_MEM_READ_ONLY; }
    if (access_ == BufferAccess::kWriteOnly) { flags = CL_MEM_WRITE_ONLY; }
    auto status = CL_SUCCESS;
    *buffer_ = clCreateBuffer(context(), flags, size*sizeof(T), nullptr, &status);
    CheckError(status);
  }

  // As above, but now with read/write access as a default
  explicit Buffer(const Context &context, const size_t size):
    Buffer<T>(context, BufferAccess::kReadWrite, size) {
  }

  // Constructs a new buffer based on an existing host-container
  template <typename Iterator>
  explicit Buffer(const Context &context, const Queue &queue, Iterator start, Iterator end):
    Buffer(context, BufferAccess::kReadWrite, static_cast<size_t>(end - start)) {
    auto size = static_cast<size_t>(end - start);
    auto pointer = &*start;
    CheckError(clEnqueueWriteBuffer(queue(), *buffer_, CL_FALSE, 0, size*sizeof(T), pointer, 0,
                                    nullptr, nullptr));
    queue.Finish();
  }

  // Copies from device to host: reading the device buffer a-synchronously
  void ReadAsync(const Queue &queue, const size_t size, T* host, const size_t offset = 0) const {
    if (access_ == BufferAccess::kWriteOnly) { Error("reading from a write-only buffer"); }
    CheckError(clEnqueueReadBuffer(queue(), *buffer_, CL_FALSE, offset*sizeof(T), size*sizeof(T),
                                   host, 0, nullptr, nullptr));
  }
  void ReadAsync(const Queue &queue, const size_t size, std::vector<T> &host,
                 const size_t offset = 0) const {
    if (host.size() < size) { Error("target host buffer is too small"); }
    ReadAsync(queue, size, host.data(), offset);
  }
  void ReadAsync(const Queue &queue, const size_t size, BufferHost<T> &host,
                 const size_t offset = 0) const {
    if (host.size() < size) { Error("target host buffer is too small"); }
    ReadAsync(queue, size, host.data(), offset);
  }

  // Copies from device to host: reading the device buffer
  void Read(const Queue &queue, const size_t size, T* host, const size_t offset = 0) const {
    ReadAsync(queue, size, host, offset);
    queue.Finish();
  }
  void Read(const Queue &queue, const size_t size, std::vector<T> &host,
            const size_t offset = 0) const {
    Read(queue, size, host.data(), offset);
  }
  void Read(const Queue &queue, const size_t size, BufferHost<T> &host,
            const size_t offset = 0) const {
    Read(queue, size, host.data(), offset);
  }

  // Copies from host to device: writing the device buffer a-synchronously
  void WriteAsync(const Queue &queue, const size_t size, const T* host, const size_t offset = 0) {
    if (access_ == BufferAccess::kReadOnly) { Error("writing to a read-only buffer"); }
    if (GetSize() < (offset+size)*sizeof(T)) { Error("target device buffer is too small"); }
    CheckError(clEnqueueWriteBuffer(queue(), *buffer_, CL_FALSE, offset*sizeof(T), size*sizeof(T),
                                    host, 0, nullptr, nullptr));
  }
  void WriteAsync(const Queue &queue, const size_t size, const std::vector<T> &host,
                  const size_t offset = 0) {
    WriteAsync(queue, size, host.data(), offset);
  }
  void WriteAsync(const Queue &queue, const size_t size, const BufferHost<T> &host,
                  const size_t offset = 0) {
    WriteAsync(queue, size, host.data(), offset);
  }

  // Copies from host to device: writing the device buffer
  void Write(const Queue &queue, const size_t size, const T* host, const size_t offset = 0) {
    WriteAsync(queue, size, host, offset);
    queue.Finish();
  }
  void Write(const Queue &queue, const size_t size, const std::vector<T> &host,
             const size_t offset = 0) {
    Write(queue, size, host.data(), offset);
  }
  void Write(const Queue &queue, const size_t size, const BufferHost<T> &host,
             const size_t offset = 0) {
    Write(queue, size, host.data(), offset);
  }

  // Copies the contents of this buffer into another device buffer
  void CopyToAsync(const Queue &queue, const size_t size, const Buffer<T> &destination) const {
    CheckError(clEnqueueCopyBuffer(queue(), *buffer_, destination(), 0, 0, size*sizeof(T), 0,
                                   nullptr, nullptr));
  }
  void CopyTo(const Queue &queue, const size_t size, const Buffer<T> &destination) const {
    CopyToAsync(queue, size, destination);
    queue.Finish();
  }

  // Retrieves the actual allocated size in bytes
  size_t GetSize() const {
    const auto bytes = sizeof(size_t);
    auto result = size_t{0};
    CheckError(clGetMemObjectInfo(*buffer_, CL_MEM_SIZE, bytes, &result, nullptr));
    return result;
  }

  // Accessor to the private data-member
  const cl_mem& operator()() const { return *buffer_; }
 private:
  std::shared_ptr<cl_mem> buffer_;
  const BufferAccess access_;
};

// =================================================================================================

// C++11 version of 'cl_kernel'
class Kernel {
 public:

  // Constructor based on the regular OpenCL data-type: memory management is handled elsewhere
  explicit Kernel(const cl_kernel kernel):
      kernel_(new cl_kernel) {
    *kernel_ = kernel;
  }

  // Regular constructor with memory management
  explicit Kernel(const Program &program, const std::string &name):
      kernel_(new cl_kernel, [](cl_kernel* k) { CheckError(clReleaseKernel(*k)); delete k; }) {
    auto status = CL_SUCCESS;
    *kernel_ = clCreateKernel(program(), name.c_str(), &status);
    CheckError(status);
  }

  // Sets a kernel argument at the indicated position
  template <typename T>
  void SetArgument(const size_t index, const T &value) {
    CheckError(clSetKernelArg(*kernel_, static_cast<cl_uint>(index), sizeof(T), &value));
  }
  template <typename T>
  void SetArgument(const size_t index, Buffer<T> &value) {
    SetArgument(index, value());
  }

  // Sets all arguments in one go using parameter packs. Note that this overwrites previously set
  // arguments using 'SetArgument' or 'SetArguments'.
  template <typename... Args>
  void SetArguments(Args&... args) {
    SetArgumentsRecursive(0, args...);
  }

  // Retrieves the amount of local memory used per work-group for this kernel
  unsigned long LocalMemUsage(const Device &device) const {
    const auto bytes = sizeof(cl_ulong);
    auto query = cl_kernel_work_group_info{CL_KERNEL_LOCAL_MEM_SIZE};
    auto result = cl_ulong{0};
    CheckError(clGetKernelWorkGroupInfo(*kernel_, device(), query, bytes, &result, nullptr));
    return static_cast<unsigned long>(result);
  }

  // Retrieves the name of the kernel
  std::string GetFunctionName() const {
    auto bytes = size_t{0};
    CheckError(clGetKernelInfo(*kernel_, CL_KERNEL_FUNCTION_NAME, 0, nullptr, &bytes));
    auto result = std::string{};
    result.resize(bytes);
    CheckError(clGetKernelInfo(*kernel_, CL_KERNEL_FUNCTION_NAME, bytes, &result[0], nullptr));
    return std::string{result.c_str()}; // Removes any trailing '\0'-characters
  }

  // Launches a kernel onto the specified queue
  void Launch(const Queue &queue, const std::vector<size_t> &global,
              const std::vector<size_t> &local, EventPointer event) {
    CheckError(clEnqueueNDRangeKernel(queue(), *kernel_, static_cast<cl_uint>(global.size()),
                                      nullptr, global.data(), local.data(),
                                      0, nullptr, event));
  }

  // As above, but with an event waiting list
  void Launch(const Queue &queue, const std::vector<size_t> &global,
              const std::vector<size_t> &local, EventPointer event,
              const std::vector<Event> &waitForEvents) {

    // Builds a plain version of the events waiting list
    auto waitForEventsPlain = std::vector<cl_event>();
    for (auto &waitEvent : waitForEvents) {
      if (waitEvent()) { waitForEventsPlain.push_back(waitEvent()); }
    }

    // Launches the kernel while waiting for other events
    CheckError(clEnqueueNDRangeKernel(queue(), *kernel_, static_cast<cl_uint>(global.size()),
                                      nullptr, global.data(), !local.empty() ? local.data() : nullptr,
                                      static_cast<cl_uint>(waitForEventsPlain.size()),
                                      !waitForEventsPlain.empty() ? waitForEventsPlain.data() : nullptr,
                                      event));
  }

  // Accessor to the private data-member
  const cl_kernel& operator()() const { return *kernel_; }
 private:
  std::shared_ptr<cl_kernel> kernel_;

  // Internal implementation for the recursive SetArguments function.
  template <typename T>
  void SetArgumentsRecursive(const size_t index, T &first) {
    SetArgument(index, first);
  }
  template <typename T, typename... Args>
  void SetArgumentsRecursive(const size_t index, T &first, Args&... args) {
    SetArgument(index, first);
    SetArgumentsRecursive(index+1, args...);
  }
};

// =================================================================================================
} // namespace CLCudaAPI

// CLCUDAAPI_CLPP11_H_
#endif
