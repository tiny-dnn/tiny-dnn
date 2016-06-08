#Command line options
#-DUSE_TBB=ON/OFF     (default off)
#-DUSE_OMP=ON/OFF     (default off)
#-DUSE_SSE=ON/OFF     (default on)
#-DUSE_AVX=ON/OFF     (default on)
#-DUSE_OPENCV=ON/OFF  (default on)

# ----------------------------------------------------------------------------
#   Basic Configuration
# ----------------------------------------------------------------------------
cmake_minimum_required(VERSION 2.6)

project(tiny_cnn C CXX)

set(PROJECT_VERSION "0.0.1")
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

option(USE_TBB        "Set to ON to use TBB"             OFF)
option(USE_OMP        "Set to ON to use OMP"             OFF)
option(USE_SSE        "Set to ON to use sse"             ON)
option(USE_AVX        "Set to ON to use avx"             ON)
option(USE_OPENCV     "Set to ON to use OpenCV"          ON)
option(BUILD_TESTS    "Set to ON to build tests"         OFF)
option(BUILD_EXAMPLES "Set to ON to build examples"      ON)
option(BUILD_DOCS     "Set to ON to build documentation" OFF)

# ----------------------------------------------------------------------------
#   Find Dependencies
# ----------------------------------------------------------------------------

if(BUILD_EXAMPLES)
    find_package(OpenCV)
    if(${OpenCV_FOUND})
        message(STATUS "OpenCV version: ${OpenCV_VERSION}")
        message(STATUS "OpenCV headers: ${OpenCV_INCLUDE_DIRS}")

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
    find_package(OpenMP  REQUIRED)
    add_definitions(-DCNN_USE_OMP)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# ----------------------------------------------------------------------------
# COMPILER optionS
# ----------------------------------------------------------------------------

if(CMAKE_COMPILER_IS_GNUCXX OR MINGW OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    if(USE_SSE)
        add_definitions(-DCNN_USE_SSE)
        set(EXTRA_C_FLAGS " ${EXTRA_C_FLAGS} -msse3 ")
    endif()
    if(USE_AVX)
        add_definitions(-DCNN_USE_AVX)
        set(EXTRA_C_FLAGS " ${EXTRA_C_FLAGS} -mavx ")
    endif()
    if((NOT USE_TBB) AND (NOT USE_OMP))
        set(EXTRA_C_FLAGS " ${EXTRA_C_FLAGS} -pthread ")
    endif()
    set(EXTRA_C_FLAGS " ${EXTRA_C_FLAGS} -std=c++11 -Wall -Wpedantic")
    set(EXTRA_C_FLAGS_RELEASE " -O3 ")
    set(EXTRA_C_FLAGS_DEBUG " -g3 ")
else() # MSVC
    if(USE_SSE )
        add_definitions(-DCNN_USE_SSE)
        set(EXTRA_C_FLAGS " ${EXTRA_C_FLAGS} /arch:SSE2 ")
    endif()
    if(USE_AVX  )
        add_definitions(-DCNN_USE_AVX)
        set(EXTRA_C_FLAGS "${EXTRA_C_FLAGS}  /arch:AVX ")
    endif()
    set(EXTRA_C_FLAGS_RELEASE " /Ox /bigobj ")
    add_definitions(-D _CRT_SECURE_NO_WARNINGS)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${EXTRA_C_FLAGS}")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${EXTRA_C_FLAGS_RELEASE}")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${EXTRA_C_FLAGS_DEBUG}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${EXTRA_EXE_LINKER_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} ${EXTRA_EXE_LINKER_FLAGS_RELEASE}")
set(CMAKE_EXE_LINKER_FLAGS_DEBUG "${CMAKE_EXE_LINKER_FLAGS_DEBUG} ${EXTRA_EXE_LINKER_FLAGS_DEBUG}")

#------------------------------------------------
# TARGETS
#------------------------------------------------

include_directories(${PROJECT_SOURCE_DIR})
link_libraries(${REQUIRED_LIBRARIES})

set(tiny_cnn_hrds
    tiny_cnn/node.h
    tiny_cnn/nodes.h
    tiny_cnn/network.h
    tiny_cnn/config.h
    tiny_cnn/tiny_cnn.h
    tiny_cnn/layers/layer.h
    tiny_cnn/layers/layers.h
    tiny_cnn/layers/convolutional_layer.h
    tiny_cnn/layers/average_pooling_layer.h
    tiny_cnn/layers/fully_connected_layer.h
    tiny_cnn/layers/input_layer.h
    tiny_cnn/layers/max_pooling_layer.h
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

set(tiny_cnn_test_headers test/testhelper.h test/picotest/picotest.h)

if(BUILD_EXAMPLES)
    add_executable(example_mnist_train examples/mnist/train.cpp ${tiny_cnn_hrds})
    add_executable(example_deconv_train examples/deconv/train.cpp ${tiny_cnn_hrds})
    add_executable(example_cifar_train examples/cifar10/train.cpp ${tiny_cnn_hrds})

    if(OpenCV_FOUND)
        add_executable(example_mnist_test examples/mnist/test.cpp ${tiny_cnn_hdrs})
        target_link_libraries(example_mnist_test ${REQUIRED_LIBRARIES})
    else()
        #file(GLOB stb_image_hdrs examples/mnist/stb_image/*.h)
        #add_executable(example_mnist_test examples/mnist/test_with_stb_image.cpp
        #               ${tiny_cnn_hdrs} ${stb_image_hdrs})
    endif()

    set(example_targets ${example_targets} example_mnist_train
                                           example_deconv_train
                                           example_cifar_train
                                           example_mnist_test)
    find_package(Protobuf)
    if(PROTOBUF_FOUND AND OpenCV_FOUND)
        set(proto_file "${CMAKE_CURRENT_SOURCE_DIR}/tiny_cnn/io/caffe/caffe.pb.cc")
        if (EXISTS ${proto_file})
            add_executable(example_caffe_converter
                           examples/caffe_converter/caffe_converter.cpp
                           ${tiny_cnn_hrds} ${proto_file})
            target_link_libraries(example_caffe_converter
                                  ${PROTOBUF_LIBRARIES} ${REQUIRED_LIBRARIES})

            set(example_targets ${example_targets} example_caffe_converter)
        else()
            message(STATUS "Cannot generate Caffe Importer. Please generate Protobuf file and install OpenCV.")
        endif()
    else()
        message(STATUS "Cannot generate Caffe Importer. Please install Protobuf.")
    endif()

    set_target_properties(${example_targets}
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples"
    )
endif()

if(BUILD_TESTS)
    file(GLOB TEST_SRCS test/src/*.cpp)
    foreach(file ${TEST_SRCS})
        get_filename_component(test_fname ${file} NAME_WE)
        set(test_bin tiny_cnn_${test_fname})

        add_executable(${test_bin} test/src/${test_fname}.cpp
                       ${tiny_cnn_hrds} ${tiny_cnn_test_headers})
        target_link_libraries(${test_bin} ${REQUIRED_LIBRARIES})

        set_target_properties(${test_bin}
            PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/test"
        )
    endforeach()
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

message( STATUS "BUILD_EXAMPLES = ${BUILD_EXAMPLES}" )
message( STATUS "BUILD_TESTS    = ${BUILD_TESTS}" )

message( STATUS "optional Dependencies: USE_TBB    = ${USE_TBB}")
message(     "                          USE_OMP    = ${USE_OMP}")
message(     "                          USE_SSE    = ${USE_SSE}")
message(     "                          USE_AVX    = ${USE_AVX}")
message(     "                          USE_NNPACK = ${USE_NNPACK}")

message( STATUS "-------------------------------------------------------------------------------" )

message( STATUS )
message("    Built as dynamic libs?:" ${BUILD_SHARED_LIBS})
message("    Compiler:"                   "${CMAKE_COMPILER}"   "${CMAKE_CXX_COMPILER}")
message( STATUS "C++ flags (Release):      ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE}")
message( STATUS "C++ flags (Debug):        ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG}")
message( STATUS "CMAKE_CXX_FLAGS:          ${CMAKE_CXX_FLAGS}")
message( STATUS "CMAKE_BINARY_DIR:         ${CMAKE_BINARY_DIR}")

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
