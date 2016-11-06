add_executable(benchmarks_all benchmarks/main.cpp)
target_link_libraries(benchmarks_all
    ${project_library_target_name} ${REQUIRED_LIBRARIES})

if(USE_SERIALIZER)

add_executable(example_mnist_train mnist/train.cpp)
target_link_libraries(example_mnist_train
    ${project_library_target_name} ${REQUIRED_LIBRARIES})

add_executable(example_mnist_test mnist/test.cpp)
target_link_libraries(example_mnist_test
    ${project_library_target_name} ${REQUIRED_LIBRARIES})

add_executable(example_deconv_train deconv/train.cpp)
target_link_libraries(example_deconv_train
    ${project_library_target_name} ${REQUIRED_LIBRARIES})

endif()

add_executable(example_deconv_visual deconv/visual.cpp)
target_link_libraries(example_deconv_visual
    ${project_library_target_name} ${REQUIRED_LIBRARIES})

add_executable(example_cifar_train cifar10/train.cpp)
target_link_libraries(example_cifar_train
    ${project_library_target_name} ${REQUIRED_LIBRARIES})

find_package(Protobuf QUIET)
if(PROTOBUF_FOUND)
    set(proto_file "${CMAKE_SOURCE_DIR}/tiny_dnn/io/caffe/caffe.pb.cc")
    if(EXISTS ${proto_file})
        message(STATUS "Found proto-file: ${proto_file}")
        include_directories(${PROTOBUF_INCLUDE_DIRS})
        add_executable(example_caffe_converter
                       caffe_converter/caffe_converter.cpp ${proto_file})
        target_link_libraries(example_caffe_converter
            ${project_library_target_name}
            ${PROTOBUF_LIBRARIES} ${REQUIRED_LIBRARIES})
# As of Ubuntu 14.04 protoc is no longer a part of libprotobuf-dev package
# and should be installed separately as in: sudo apt-get install protobuf-compiler
    elseif(EXISTS ${PROTOBUF_PROTOC_EXECUTABLE})
        message(STATUS "Found PROTOBUF Compiler: ${PROTOBUF_PROTOC_EXECUTABLE}")
        if(EXISTS ${CMAKE_SOURCE_DIR}/tiny_dnn/io/caffe/caffe.proto)
            # Note that this line doesn't invoke protoc at configure time
            PROTOBUF_GENERATE_CPP(PROTO_SRCS PROTO_HDRS
                ${CMAKE_SOURCE_DIR}/tiny_dnn/io/caffe/caffe.proto)
            add_executable(example_caffe_converter
                           caffe_converter/caffe_converter.cpp
                           ${PROTO_SRCS} ${PROTO_HDRS})
            add_custom_target(generated_proto DEPENDS ${PROTO_SRCS} ${PROTO_HDRS})
            add_dependencies(example_caffe_converter generated_proto)

            # We need to invoke the copy after protoc compile finished
            add_custom_command(TARGET generated_proto PRE_BUILD
                               COMMAND ${CMAKE_COMMAND} -E copy
                               ${PROTO_SRCS} ${CMAKE_SOURCE_DIR}/tiny_dnn/io/caffe)
            add_custom_command(TARGET generated_proto PRE_BUILD
                               COMMAND ${CMAKE_COMMAND} -E copy
                               ${PROTO_HDRS} ${CMAKE_SOURCE_DIR}/tiny_dnn/io/caffe)

            target_link_libraries(example_caffe_converter
                ${project_library_target_name}
                ${PROTOBUF_LIBRARIES} ${REQUIRED_LIBRARIES})
        else()
            message(STATUS "Cannot generate C++ proto files, please provide Protobuf file.")
        endif()
    else()
        message(STATUS "Proto is not linked correctly, please make sure file exists.")
    endif()
else(PROTOBUF_FOUND)
    message(STATUS "Cannot generate Caffe Importer. Please install Protobuf.")
endif(PROTOBUF_FOUND)
