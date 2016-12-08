
// =================================================================================================
// This file is part of the CLCudaAPI project. The project is licensed under Apache Version 2.0. The
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements a bunch of C++11 classes that act as wrappers around CUDA objects and API
// calls. The main benefits are increased abstraction, automatic memory management, and portability.
// Portability here means that a similar header exists for OpenCL with the same classes and
// interfaces. In other words, moving from the CUDA API to the OpenCL API becomes a one-line change.
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

#ifndef CLCUDAAPI_CUPP11_H_
#define CLCUDAAPI_CUPP11_H_

// C++
#include <algorithm> // std::copy
#include <string>    // std::string
#include <vector>    // std::vector
#include <memory>    // std::shared_ptr
#include <stdexcept> // std::runtime_error

// CUDA
#include <cuda.h>    // CUDA driver API
#include <nvrtc.h>   // NVIDIA runtime compilation API

namespace CLCudaAPI {
// =================================================================================================

// Max-length of strings
constexpr auto kStringLength = 256;

// =================================================================================================

// Error occurred in the C++11 CUDA header (this file)
inline void Error(const std::string &message) {
  throw std::runtime_error("Internal CUDA error: "+message);
}

// Error occurred in the CUDA driver API
inline void CheckError(const CUresult status) {
  if (status != CUDA_SUCCESS) {
    const char* status_code;
    cuGetErrorName(status, &status_code);
    const char* status_string;
    cuGetErrorString(status, &status_string);
    throw std::runtime_error("Internal CUDA error "+std::string{status_code}+
                             " : "+std::string{status_string});
  }
}

// Error occurred in the NVIDIA runtime compilation API
inline void CheckError(const nvrtcResult status) {
  if (status != NVRTC_SUCCESS) {
    const char* status_string = nvrtcGetErrorString(status);
    throw std::runtime_error("Internal CUDA error: "+std::string{status_string});
  }
}

// =================================================================================================

// C++11 version of two 'CUevent' pointers
class Event {
 public:
  // Note that there is no constructor based on the regular CUDA data-type because of extra state

  // Regular constructor with memory management
  explicit Event():
      start_(new CUevent, [](CUevent* e) { CheckError(cuEventDestroy(*e)); delete e; }),
      end_(new CUevent, [](CUevent* e) { CheckError(cuEventDestroy(*e)); delete e; }) {
    CheckError(cuEventCreate(start_.get(), CU_EVENT_DEFAULT));
    CheckError(cuEventCreate(end_.get(), CU_EVENT_DEFAULT));
  }

  // Waits for completion of this event (not implemented for CUDA)
  void WaitForCompletion() const { }

  // Retrieves the elapsed time of the last recorded event
  float GetElapsedTime() const {
    auto result = 0.0f;
    cuEventElapsedTime(&result, *start_, *end_);
    return result;
  }

  // Accessors to the private data-members
  const CUevent& start() const { return *start_; }
  const CUevent& end() const { return *end_; }
  Event* pointer() { return this; }
 private:
  std::shared_ptr<CUevent> start_;
  std::shared_ptr<CUevent> end_;
};

// Pointer to a CUDA event
using EventPointer = Event*;

// =================================================================================================

// The CUDA platform: initializes the CUDA driver API
class Platform {
 public:

  // Initializes the platform. Note that the platform ID variable is not actually used for CUDA.
  explicit Platform(const size_t platform_id):
    platform_id_(platform_id) {
    CheckError(cuInit(0));
  }

  // Returns the number of devices on this platform
  size_t NumDevices() const {
    auto result = 0;
    CheckError(cuDeviceGetCount(&result));
    return static_cast<size_t>(result);
  }

 private:
  size_t platform_id_;
};

// Retrieves a vector with all platforms. Note that there is just one platform in CUDA.
inline std::vector<Platform> GetAllPlatforms() {
  auto all_platforms = std::vector<Platform>{ Platform(size_t{0}) };
  return all_platforms;
}

// =================================================================================================

// C++11 version of 'CUdevice'
class Device {
 public:

  // Constructor based on the regular CUDA data-type
  explicit Device(const CUdevice device): device_(device) { }

  // Initialization
  explicit Device(const Platform &platform, const size_t device_id) {
    auto num_devices = platform.NumDevices();
    if (num_devices == 0) { Error("no devices found"); }
    CheckError(cuDeviceGet(&device_, device_id % num_devices));
  }

  // Methods to retrieve device information
  std::string Version() const {
    auto result = 0;
    CheckError(cuDriverGetVersion(&result));
    return "CUDA driver "+std::to_string(result);
  }
  size_t VersionNumber() const {
    auto result = 0;
    CheckError(cuDriverGetVersion(&result));
    return static_cast<size_t>(result);
  }
  std::string Vendor() const { return "NVIDIA Corporation"; }
  std::string Name() const {
    auto result = std::string{};
    result.resize(kStringLength);
    CheckError(cuDeviceGetName(&result[0], result.size(), device_));
    return result;
  }
  std::string Type() const { return "GPU"; }
  size_t MaxWorkGroupSize() const {return GetInfo(CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK); }
  size_t MaxWorkItemDimensions() const { return size_t{3}; }
  std::vector<size_t> MaxWorkItemSizes() const {
    return std::vector<size_t>{GetInfo(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X),
                               GetInfo(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y),
                               GetInfo(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z)};
  }
  unsigned long LocalMemSize() const {
    return static_cast<unsigned long>(GetInfo(CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK));
  }
  std::string Capabilities() const {
    auto major = GetInfo(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR);
    auto minor = GetInfo(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR);
    return "SM "+std::to_string(major)+"."+std::to_string(minor);
  }
  size_t CoreClock() const { return 1e-3*GetInfo(CU_DEVICE_ATTRIBUTE_CLOCK_RATE); }
  size_t ComputeUnits() const { return GetInfo(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT); }
  unsigned long MemorySize() const {
    auto result = size_t{0};
    CheckError(cuDeviceTotalMem(&result, device_));
    return static_cast<unsigned long>(result);
  }
  unsigned long MaxAllocSize() const { return MemorySize(); }
  size_t MemoryClock() const { return 1e-3*GetInfo(CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE); }
  size_t MemoryBusWidth() const { return GetInfo(CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH); }

  // Configuration-validity checks
  bool IsLocalMemoryValid(const size_t local_mem_usage) const {
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
  bool IsCPU() const { return false; }
  bool IsGPU() const { return true; }
  bool IsAMD() const { return false; }
  bool IsNVIDIA() const { return true; }
  bool IsIntel() const { return false; }
  bool IsARM() const { return false; }

  // Accessor to the private data-member
  const CUdevice& operator()() const { return device_; }
 private:
  CUdevice device_;

  // Private helper function
  size_t GetInfo(const CUdevice_attribute info) const {
    auto result = 0;
    CheckError(cuDeviceGetAttribute(&result, info, device_));
    return static_cast<size_t>(result);
  }
};

// =================================================================================================

// C++11 version of 'CUcontext'
class Context {
 public:

  // Constructor based on the regular CUDA data-type: memory management is handled elsewhere
  explicit Context(const CUcontext context):
      context_(new CUcontext) {
    *context_ = context;
  }

  // Regular constructor with memory management
  explicit Context(const Device &device):
      context_(new CUcontext, [](CUcontext* c) { CheckError(cuCtxDestroy(*c)); delete c; }) {
    CheckError(cuCtxCreate(context_.get(), 0, device()));
  }

  // Accessor to the private data-member
  const CUcontext& operator()() const { return *context_; }
  CUcontext* pointer() const { return &(*context_); }
 private:
  std::shared_ptr<CUcontext> context_;
};

// Pointer to an OpenCL context
using ContextPointer = CUcontext*;

// =================================================================================================

// Enumeration of build statuses of the run-time compilation process
enum class BuildStatus { kSuccess, kError, kInvalid };

// C++11 version of 'nvrtcProgram'. Additionally holds the program's source code.
class Program {
 public:
  // Note that there is no constructor based on the regular CUDA data-type because of extra state

  // Source-based constructor with memory management
  explicit Program(const Context &, std::string source):
      program_(new nvrtcProgram, [](nvrtcProgram* p) { CheckError(nvrtcDestroyProgram(p));
                                                       delete p; }),
      source_(std::move(source)),
      source_ptr_(&source_[0]),
      from_binary_(false) {
    CheckError(nvrtcCreateProgram(program_.get(), source_ptr_, nullptr, 0, nullptr, nullptr));
  }

  // PTX-based constructor
  explicit Program(const Device &device, const Context &context, const std::string& binary):
      program_(nullptr), // not used
      source_(binary),
      source_ptr_(&source_[0]), // not used
      from_binary_(true) {
  }

  // Compiles the device program and returns whether or not there where any warnings/errors
  BuildStatus Build(const Device &, std::vector<std::string> &options) {
    if (from_binary_) { return BuildStatus::kSuccess; }
    auto raw_options = std::vector<const char*>();
    for (const auto &option: options) {
      raw_options.push_back(option.c_str());
    }
    auto status = nvrtcCompileProgram(*program_, raw_options.size(), raw_options.data());
    if (status == NVRTC_ERROR_COMPILATION) {
      return BuildStatus::kError;
    }
    else if (status == NVRTC_ERROR_INVALID_PROGRAM) {
      return BuildStatus::kInvalid;
    }
    else {
      CheckError(status);
      return BuildStatus::kSuccess;
    }
  }

  // Retrieves the warning/error message from the compiler (if any)
  std::string GetBuildInfo(const Device &) const {
    if (from_binary_) { return std::string{}; }
    auto bytes = size_t{0};
    CheckError(nvrtcGetProgramLogSize(*program_, &bytes));
    auto result = std::string{};
    result.resize(bytes);
    CheckError(nvrtcGetProgramLog(*program_, &result[0]));
    return result;
  }

  // Retrieves an intermediate representation of the compiled program (i.e. PTX)
  std::string GetIR() const {
    if (from_binary_) { return source_; } // holds the PTX
    auto bytes = size_t{0};
    CheckError(nvrtcGetPTXSize(*program_, &bytes));
    auto result = std::string{};
    result.resize(bytes);
    CheckError(nvrtcGetPTX(*program_, &result[0]));
    return result;
  }

  // Accessor to the private data-member
  const nvrtcProgram& operator()() const { return *program_; }
 private:
  std::shared_ptr<nvrtcProgram> program_;
  std::string source_;
  const char* source_ptr_;
  const bool from_binary_;
};

// =================================================================================================

// C++11 version of 'CUstream'
class Queue {
 public:
  // Note that there is no constructor based on the regular CUDA data-type because of extra state

  // Regular constructor with memory management
  explicit Queue(const Context &context, const Device &device):
      queue_(new CUstream, [](CUstream* s) { CheckError(cuStreamDestroy(*s)); delete s; }),
      context_(context),
      device_(device) {
    CheckError(cuStreamCreate(queue_.get(), CU_STREAM_NON_BLOCKING));
  }

  // Synchronizes the queue and optionaly also an event
  void Finish(Event &event) const {
    CheckError(cuEventSynchronize(event.end()));
    Finish();
  }
  void Finish() const {
    CheckError(cuStreamSynchronize(*queue_));
  }

  // Retrieves the corresponding context or device
  Context GetContext() const { return context_; }
  Device GetDevice() const { return device_; }

  // Accessor to the private data-member
  const CUstream& operator()() const { return *queue_; }
 private:
  std::shared_ptr<CUstream> queue_;
  const Context context_;
  const Device device_;
};

// =================================================================================================

// C++11 version of page-locked host memory
template <typename T>
class BufferHost {
 public:

  // Regular constructor with memory management
  explicit BufferHost(const Context &, const size_t size):
      buffer_(new void*, [](void** m) { CheckError(cuMemFreeHost(*m)); delete m; }),
      size_(size) {
    CheckError(cuMemAllocHost(buffer_.get(), size*sizeof(T)));
  }

  // Retrieves the actual allocated size in bytes
  size_t GetSize() const {
    return size_*sizeof(T);
  }

  // Compatibility with std::vector
  size_t size() const { return size_; }
  T* begin() { return &static_cast<T*>(*buffer_)[0]; }
  T* end() { return &static_cast<T*>(*buffer_)[size_-1]; }
  T& operator[](const size_t i) { return static_cast<T*>(*buffer_)[i]; }
  T* data() { return static_cast<T*>(*buffer_); }
  const T* data() const { return static_cast<T*>(*buffer_); }

 private:
  std::shared_ptr<void*> buffer_;
  const size_t size_;
};

// =================================================================================================

// Enumeration of buffer access types
enum class BufferAccess { kReadOnly, kWriteOnly, kReadWrite, kNotOwned };

// C++11 version of 'CUdeviceptr'
template <typename T>
class Buffer {
 public:

  // Constructor based on the regular CUDA data-type: memory management is handled elsewhere
  explicit Buffer(const CUdeviceptr buffer):
      buffer_(new CUdeviceptr),
      access_(BufferAccess::kNotOwned) {
    *buffer_ = buffer;
  }

  // Regular constructor with memory management. If this class does not own the buffer object, then
  // the memory will not be freed automatically afterwards.
  explicit Buffer(const Context &, const BufferAccess access, const size_t size):
      buffer_(new CUdeviceptr, [access](CUdeviceptr* m) {
        if (access != BufferAccess::kNotOwned) { CheckError(cuMemFree(*m)); }
        delete m;
      }),
      access_(access) {
    CheckError(cuMemAlloc(buffer_.get(), size*sizeof(T)));
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
    CheckError(cuMemcpyHtoDAsync(*buffer_, pointer, size*sizeof(T), queue()));
    queue.Finish();
  }

  // Copies from device to host: reading the device buffer a-synchronously
  void ReadAsync(const Queue &queue, const size_t size, T* host, const size_t offset = 0) const {
    if (access_ == BufferAccess::kWriteOnly) { Error("reading from a write-only buffer"); }
    CheckError(cuMemcpyDtoHAsync(host, *buffer_ + offset*sizeof(T), size*sizeof(T), queue()));
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
    CheckError(cuMemcpyHtoDAsync(*buffer_ + offset*sizeof(T), host, size*sizeof(T), queue()));
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
    CheckError(cuMemcpyDtoDAsync(destination(), *buffer_, size*sizeof(T), queue()));
  }
  void CopyTo(const Queue &queue, const size_t size, const Buffer<T> &destination) const {
    CopyToAsync(queue, size, destination);
    queue.Finish();
  }

  // Retrieves the actual allocated size in bytes
  size_t GetSize() const {
    auto result = size_t{0};
    CheckError(cuMemGetAddressRange(nullptr, &result, *buffer_));
    return result;
  }

  // Accessors to the private data-members
  CUdeviceptr operator()() const { return *buffer_; }
  CUdeviceptr& operator()() { return *buffer_; }
 private:
  std::shared_ptr<CUdeviceptr> buffer_;
  const BufferAccess access_;
};

// =================================================================================================

// C++11 version of 'CUfunction'
class Kernel {
 public:

  // Constructor based on the regular CUDA data-type: memory management is handled elsewhere
  explicit Kernel(const CUmodule module, const CUfunction kernel):
    module_(module),
    kernel_(kernel) {
  }

  // Regular constructor with memory management
  explicit Kernel(const Program &program, const std::string &name) {
    CheckError(cuModuleLoadDataEx(&module_, program.GetIR().data(), 0, nullptr, nullptr));
    CheckError(cuModuleGetFunction(&kernel_, module_, name.c_str()));
  }

  // Sets a kernel argument at the indicated position. This stores both the value of the argument
  // (as raw bytes) and the index indicating where this value can be found.
  template <typename T>
  void SetArgument(const size_t index, const T &value) {
    if (index >= arguments_indices_.size()) { arguments_indices_.resize(index+1); }
    arguments_indices_[index] = arguments_data_.size();
    for (auto j=size_t(0); j<sizeof(T); ++j) {
      arguments_data_.push_back(reinterpret_cast<const char*>(&value)[j]);
    }
  }
  template <typename T>
  void SetArgument(const size_t index, Buffer<T> &value) {
    SetArgument(index, value());
  }

  // Sets all arguments in one go using parameter packs. Note that this resets all previously set
  // arguments using 'SetArgument' or 'SetArguments'.
  template <typename... Args>
  void SetArguments(Args&... args) {
    arguments_indices_.clear();
    arguments_data_.clear();
    SetArgumentsRecursive(0, args...);
  }

  // Retrieves the amount of local memory used per work-group for this kernel. Note that this the
  // shared memory in CUDA terminology.
  unsigned long LocalMemUsage(const Device &) const {
    auto result = 0;
    CheckError(cuFuncGetAttribute(&result, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, kernel_));
    return static_cast<unsigned long>(result);
  }

  // Retrieves the name of the kernel
  std::string GetFunctionName() const {
    return std::string{"unknown"}; // Not implemented for the CUDA backend
  }

  // Launches a kernel onto the specified queue
  void Launch(const Queue &queue, const std::vector<size_t> &global,
              const std::vector<size_t> &local, EventPointer event) {

    // Creates the grid (number of threadblocks) and sets the block sizes (threads per block)
    auto grid = std::vector<size_t>{1, 1, 1};
    auto block = std::vector<size_t>{1, 1, 1};
    if (global.size() != local.size()) { Error("invalid thread/workgroup dimensions"); }
    for (auto i=size_t{0}; i<local.size(); ++i) { grid[i] = global[i]/local[i]; }
    for (auto i=size_t{0}; i<local.size(); ++i) { block[i] = local[i]; }

    // Creates the array of pointers from the arrays of indices & data
    std::vector<void*> pointers;
    for (auto &index: arguments_indices_) {
      pointers.push_back(&arguments_data_[index]);
    }

    // Launches the kernel, its execution time is recorded by events
    CheckError(cuEventRecord(event->start(), queue()));
    CheckError(cuLaunchKernel(kernel_, grid[0], grid[1], grid[2], block[0], block[1], block[2],
                              0, queue(), pointers.data(), nullptr));
    CheckError(cuEventRecord(event->end(), queue()));
  }

  // As above, but with an event waiting list
  // TODO: Implement this function
  void Launch(const Queue &queue, const std::vector<size_t> &global,
              const std::vector<size_t> &local, EventPointer event,
              std::vector<Event>& waitForEvents) {
    if (local.size() == 0) {
      Error("launching with a default workgroup size is not implemented for the CUDA back-end");
    }
    else if (waitForEvents.size() != 0) {
      Error("launching with an event waiting list is not implemented for the CUDA back-end");
    }
    else {
     return Launch(queue, global, local, event);
    }
  }

  // Accessors to the private data-members
  const CUfunction& operator()() const { return kernel_; }
  CUfunction operator()() { return kernel_; }
 private:
  CUmodule module_;
  CUfunction kernel_;
  std::vector<size_t> arguments_indices_; // Indices of the arguments
  std::vector<char> arguments_data_; // The arguments data as raw bytes

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

// CLCUDAAPI_CUPP11_H_
#endif
