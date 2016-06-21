#Command line options
#-DUSE_TBB=ON/OFF  (default off)
#-DUSE_OMP=ON/OFF  (default off)
#-DUSE_SSE=ON/OFF  (default on)
#-DUSE_AVX=ON/OFF  (default on)

# ----------------------------------------------------------------------------
#   Basic Configuration
# ----------------------------------------------------------------------------
cmake_minimum_required(VERSION 2.6)

project(tiny_cnn C CXX)
cmake_policy(SET CMP0015 NEW)

set(PROJECT_VERSION "0.1.0")

string(REGEX MATCHALL "[0-9]" PROJECT_VERSION_PARTS "${PROJECT_VERSION}")
list(GET PROJECT_VERSION_PARTS 0 PROJECT_VERSION_MAJOR)
list(GET PROJECT_VERSION_PARTS 1 PROJECT_VERSION_MINOR)
list(GET PROJECT_VERSION_PARTS 2 PROJECT_VERSION_PATCH)
set(PROJECT_SOVERSION "${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}")

#------------------------------------------------------
# Build type
#------------------------------------------------------

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()

#------------------------------------------------------
# Lib Names and Dirs
#------------------------------------------------------

if(WIN32)
    # Postfix of DLLs:
    set(PROJECT_DLLVERSION "${PROJECT_VERSION_MAJOR}${PROJECT_VERSION_MINOR}${PROJECT_VERSION_PATCH}")
    set(RUNTIME_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin CACHE PATH "Directory for dlls and binaries")
    set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin CACHE PATH "Directory for binaries")
    set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin CACHE PATH "Directory for dlls")
else()
    # Postfix of so's:
    set(PROJECT_DLLVERSION)
    set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_INSTALL_PREFIX}/lib/cmake/ /usr/lib/cmake)
endif()
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}")

option(USE_SSE    "Build tiny-cnn with SSE library support" ON)
option(USE_AVX    "Build tiny-cnn with AVX library support" ON)
option(USE_TBB    "Build tiny-cnn with TBB library support" OFF)
option(USE_OMP    "Build tiny-cnn with OMP library support" OFF)
option(USE_NNPACK "Build tiny-cnn with NNPACK library support" OFF)

option(BUILD_TESTS "Set to ON to build tests" OFF)
option(BUILD_EXAMPLES "Set to ON to build examples" ON)
option(BUILD_DOCS "Set to ON to build documentation" OFF)

# ----------------------------------------------------------------------------
#   Find Dependencies
# ----------------------------------------------------------------------------

# Using cmake scripts and modules
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)

if(BUILD_EXAMPLES)
    find_package(OpenCV)
    if(OpenCV_FOUND)
      add_definitions(-DUSE_OPENCV)
      include_directories(${OpenCV_INCLUDE_DIRS})
      set(REQUIRED_LIBRARIES ${REQUIRED_LIBRARIES} ${OpenCV_LIBS})
    endif()
endif()

if(USE_TBB)
    find_package(TBB REQUIRED)
    add_definitions(-DCNN_USE_TBB)
    include_directories(${TBB_INCLUDE_DIRS})
    link_directories(${TBB_LIBRARY_DIRS})
    set(REQUIRED_LIBRARIES ${REQUIRED_LIBRARIES} ${TBB_LIBRARIES})
elseif(USE_OMP)
    find_package(OpenMP REQUIRED)
    add_definitions(-DCNN_USE_OMP)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

if(USE_NNPACK)
   find_package(NNPACK REQUIRED)
   add_definitions(-DCNN_USE_NNPACK)
   include_directories(SYSTEM ${NNPACK_INCLUDE_DIR})
   include_directories(SYSTEM ${NNPACK_INCLUDE_DIR}/../third-party/pthreadpool/include)
   set(REQUIRED_LIBRARIES ${REQUIRED_LIBRARIES} ${NNPACK_LIB})
endif()

# ----------------------------------------------------------------------------
# COMPILER OPTIONS
# ----------------------------------------------------------------------------

if(CMAKE_COMPILER_IS_GNUCXX OR MINGW OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    if(USE_SSE)
        add_definitions(-DCNN_USE_SSE)
        set(EXTRA_C_FLAGS  " ${EXTRA_C_FLAGS} -msse3")
    endif()
    if(USE_AVX)
        add_definitions(-DCNN_USE_AVX)
        set(EXTRA_C_FLAGS  "${EXTRA_C_FLAGS} -mavx -mavx2 -march=core-avx2")
    endif()
    if((NOT USE_TBB) AND (NOT USE_OMP))
        set(EXTRA_C_FLAGS " ${EXTRA_C_FLAGS} -pthread")
    endif()
    set(EXTRA_C_FLAGS " ${EXTRA_C_FLAGS} -std=c++11 -Wall -Wpedantic")
    set(EXTRA_C_FLAGS_RELEASE " -O3")
    set(EXTRA_C_FLAGS_DEBUG " -g3")
else() # MSVC
    if(USE_SSE)
        add_definitions(-DCNN_USE_SSE)
        set(EXTRA_C_FLAGS  " ${EXTRA_C_FLAGS} /arch:SSE2 ")
    endif()
    if(USE_AVX)
        add_definitions(-DCNN_USE_AVX)
        set(EXTRA_C_FLAGS  "${EXTRA_C_FLAGS}  /arch:AVX ")
    endif()
    set(EXTRA_C_FLAGS_RELEASE " /Ox /bigobj ")
    add_definitions(-D _CRT_SECURE_NO_WARNINGS)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${EXTRA_C_FLAGS}")
set(CMAKE_CXX_FLAGS_RELEASE "${EXTRA_C_FLAGS_RELEASE}")
set(CMAKE_CXX_FLAGS_DEBUG "${EXTRA_C_FLAGS_DEBUG}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${EXTRA_EXE_LINKER_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} ${EXTRA_EXE_LINKER_FLAGS_RELEASE}")
set(CMAKE_EXE_LINKER_FLAGS_DEBUG "${CMAKE_EXE_LINKER_FLAGS_DEBUG} ${EXTRA_EXE_LINKER_FLAGS_DEBUG}")


#------------------------------------------------
# TARGETS
#------------------------------------------------


include_directories(${PROJECT_SOURCE_DIR})
link_libraries(${REQUIRED_LIBRARIES})

set(tiny_cnn_hrds
    tiny_cnn/tiny_cnn.h
    tiny_cnn/config.h
    tiny_cnn/network.h
    tiny_cnn/node.h
    tiny_cnn/nodes.h
    tiny_cnn/io/cifar10_parser.h
    tiny_cnn/io/display.h
    tiny_cnn/io/mnist_parser.h
    tiny_cnn/util/image.h
    tiny_cnn/util/product.h
    tiny_cnn/util/util.h
    tiny_cnn/util/deform.h
    tiny_cnn/util/weight_init.h
    tiny_cnn/util/target_cost.h
    tiny_cnn/util/aligned_allocator.h
    tiny_cnn/layers/layer.h
    tiny_cnn/layers/layers.h
    tiny_cnn/layers/convolutional_layer.h
    tiny_cnn/layers/deconvolutional_layer.h
    tiny_cnn/layers/average_pooling_layer.h
    tiny_cnn/layers/average_unpooling_layer.h
    tiny_cnn/layers/fully_connected_layer.h
    tiny_cnn/layers/input_layer.h
    tiny_cnn/layers/max_pooling_layer.h
    tiny_cnn/layers/max_unpooling_layer.h
    tiny_cnn/layers/partial_connected_layer.h
    tiny_cnn/layers/arithmetic_layer.h
    tiny_cnn/layers/concat_layer.h
    tiny_cnn/layers/feedforward_layer.h
    tiny_cnn/layers/dropout_layer.h
    tiny_cnn/layers/lrn_layer.h
    tiny_cnn/layers/linear_layer.h
    tiny_cnn/optimizers/optimizer.h
    tiny_cnn/activations/activation_function.h
    tiny_cnn/lossfunctions/loss_function.h)

set(tiny_cnn_test_headers
    test/test_core.h
    test/test_average_pooling_layer.h
    test/test_convolutional_layer.h
    test/test_deconvolutional_layer.h
    test/test_fully_connected_layer.h
    test/test_lrn_layer.h
    test/test_max_pooling_layer.h
    test/test_dropout_layer.h
    test/test_network.h
    test/test_target_cost.h
    test/test_large_thread_count.h
    test/testhelper.h
    test/picotest/picotest.h)

if(BUILD_EXAMPLES)
    add_executable(example_mnist_train examples/mnist/train.cpp ${tiny_cnn_hrds})
    target_link_libraries(example_mnist_train ${REQUIRED_LIBRARIES})

    add_executable(example_deconv_train examples/deconv/train.cpp ${tiny_cnn_hrds})
    target_link_libraries(example_deconv_train ${REQUIRED_LIBRARIES})

    if(USE_OPENCV)
        add_executable(example_mnist_test examples/mnist/test.cpp ${tiny_cnn_hdrs})
        add_executable(example_deconv_visual examples/deconv/visual.cpp ${tiny_cnn_hdrs})
        target_link_libraries(example_mnist_test ${REQUIRED_LIBRARIES})
        target_link_libraries(example_deconv_visual ${REQUIRED_LIBRARIES})
    else()
        #file(GLOB stb_image_hdrs examples/mnist/stb_image/*.h)
        #add_executable(example_mnist_test examples/mnist/test_with_stb_image.cpp
        #               ${tiny_cnn_hdrs} ${stb_image_hdrs})
        #target_link_libraries(example_mnist_test ${REQUIRED_LIBRARIES})
    endif()

    add_executable(example_cifar_train examples/cifar10/train.cpp ${tiny_cnn_hrds})
    target_link_libraries(example_cifar_train ${REQUIRED_LIBRARIES})

    find_package(Protobuf)
    if(PROTOBUF_FOUND)
        set(proto_file "${CMAKE_CURRENT_SOURCE_DIR}/tiny_cnn/io/caffe/caffe.pb.cc")
        if(EXISTS ${proto_file})
            add_executable(example_caffe_converter
                           examples/caffe_converter/caffe_converter.cpp
                           ${tiny_cnn_hrds} ${proto_file})
            target_link_libraries(example_caffe_converter
                                  ${PROTOBUF_LIBRARIES} ${REQUIRED_LIBRARIES})
# As of Ubuntu 14.04 protoc is no longer a part of libprotobuf-dev package
# and should be installed separately as in: sudo apt-get install protobuf-compiler
        elseif(EXISTS ${PROTOBUF_PROTOC_EXECUTABLE})
            message(STATUS "Found PROTOBUF Compiler: ${PROTOBUF_PROTOC_EXECUTABLE}")
            if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tiny_cnn/io/caffe/caffe.proto)
                PROTOBUF_GENERATE_CPP(PROTO_SRCS PROTO_HDRS ${CMAKE_CURRENT_SOURCE_DIR}/tiny_cnn/io/caffe/caffe.proto)
                if(EXISTS ${PROTO_HDRS})
                    file(COPY ${PROTO_SRCS} DESTINATION "${CMAKE_CURRENT_SOURCE_DIR}/tiny_cnn/io/caffe")
                    file(COPY ${PROTO_HDRS} DESTINATION "${CMAKE_CURRENT_SOURCE_DIR}/tiny_cnn/io/caffe")
                endif()
                add_executable(example_caffe_converter
                               examples/caffe_converter/caffe_converter.cpp
                               ${tiny_cnn_hrds} ${PROTO_SRCS} ${PROTO_HDRS})
                target_link_libraries(example_caffe_converter
                                      ${PROTOBUF_LIBRARIES} ${REQUIRED_LIBRARIES})
            else()
                message(STATUS "Cannot generate C++ proto files, please provide Protobuf file.")
            endif()
        else()
            message(STATUS "Proto is not linked correctly, please make sure file exists.")
        endif()
    else()
        message(STATUS "Cannot generate Caffe Importer. Please install Protobuf.")
    endif()
endif()

if(BUILD_TESTS)
    add_executable(tiny_cnn_test test/test.cpp ${tiny_cnn_hrds} ${tiny_cnn_test_headers})
    target_link_libraries(tiny_cnn_test ${REQUIRED_LIBRARIES})
endif()

#------------------------------------------------
# DOCUMENTATION
#------------------------------------------------

# ========= Doxygen docs =========

find_package(Doxygen)

if(BUILD_DOCS)
    if(NOT DOXYGEN_FOUND)
        message(FATAL_ERROR "Doxygen is needed to build the documentation.")
    endif()

    set(doxyfile_in ${CMAKE_CURRENT_SOURCE_DIR}/doc/Doxyfile.in)
    set(doxyfile ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)
    set(doxy_main_page doc/readme.md)

    configure_file(${doxyfile_in} ${doxyfile} @ONLY)

    add_custom_target(doc
        COMMAND ${DOXYGEN_EXECUTABLE} ${doxyfile}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/doc
        COMMENT "Generating API documentation with Doxygen"
        VERBATIM)

    install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/html DESTINATION share/doc)
endif()


# ----------------------------------------------------------------------------
# display status message for important variables
# ----------------------------------------------------------------------------
message( STATUS )
message( STATUS "-------------------------------------------------------------------------------" )
message( STATUS "General configuration for ${PROJECT_NAME} ${PROJECT_VERSION}")
message( STATUS "-------------------------------------------------------------------------------" )


message( STATUS "BUILD_EXAMPLES=${BUILD_EXAMPLES} BUILD_TESTS=${BUILD_TESTS} USE_TBB=${USE_TBB} USE_OMP=${USE_OMP} USE_SSE=${USE_SSE} USE_AVX=${USE_AVX}")


message( STATUS "-------------------------------------------------------------------------------" )

message( STATUS )
message("    Built as dynamic libs?:" ${BUILD_SHARED_LIBS})
message("    Compiler:"                   "${CMAKE_COMPILER}"   "${CMAKE_CXX_COMPILER}")
message( STATUS "C++ flags (Release):       ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE}")
message( STATUS "C++ flags (Debug):         ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG}")
message( STATUS "CMAKE_CXX_FLAGS:         ${CMAKE_CXX_FLAGS}")
message( STATUS "CMAKE_BINARY_DIR:         ${CMAKE_BINARY_DIR}")
message( STATUS "OpenCV libs:              ${OpenCV_LIBS}")

message( STATUS )
message( STATUS "CMAKE_SYSTEM_PROCESSOR = ${CMAKE_SYSTEM_PROCESSOR}" )
message( STATUS "REQUIRED_LIBRARIES = ${REQUIRED_LIBRARIES}" )
message( STATUS "CMAKE_INSTALL_PREFIX = ${CMAKE_INSTALL_PREFIX}" )
message( STATUS "CMAKE_BUILD_TYPE = ${CMAKE_BUILD_TYPE}" )
message( STATUS "CMAKE_MODULE_PATH = ${CMAKE_MODULE_PATH}" )
message( STATUS "CMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}")
message( STATUS )
message( STATUS )
message( STATUS "Change a value with: cmake -D<Variable>=<Value>" )
message( STATUS )
