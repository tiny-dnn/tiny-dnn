

if(MSVC)
    file(GLOB_RECURSE tiny_dnn_headers "${CMAKE_SOURCE_DIR}/tiny_dnn/*.h")
    file(RELATIVE_PATH tiny_dnn_relpath ${CMAKE_CURRENT_SOURCE_DIR} "${CMAKE_SOURCE_DIR}/tiny_dnn")
    string(REPLACE "/" "\\" tiny_dnn_relpath "${tiny_dnn_relpath}")
    foreach(_source IN ITEMS ${tiny_dnn_headers})
        file(RELATIVE_PATH _source_rel ${CMAKE_CURRENT_SOURCE_DIR} "${_source}")
        get_filename_component(_source_path "${_source_rel}" PATH)
        string(REPLACE "/" "\\" _source_path_msvc "${_source_path}")
        string(REPLACE ${tiny_dnn_relpath} "tiny_dnn" _source_path_msvc ${_source_path_msvc})
        source_group("${_source_path_msvc}" FILES "${_source}")
    endforeach()
endif()

add_executable(benchmarks_all benchmarks/main.cpp ${tiny_dnn_headers})
target_link_libraries(benchmarks_all
    ${project_library_target_name} ${REQUIRED_LIBRARIES})

if(USE_SERIALIZER)

add_executable(example_mnist_train mnist/train.cpp ${tiny_dnn_headers})
target_link_libraries(example_mnist_train
    ${project_library_target_name} ${REQUIRED_LIBRARIES})

add_executable(example_mnist_test mnist/test.cpp ${tiny_dnn_headers})
target_link_libraries(example_mnist_test
    ${project_library_target_name} ${REQUIRED_LIBRARIES})

add_executable(example_mnist_quantized_train mnist/quantized.cpp ${tiny_dnn_headers})
target_link_libraries(example_mnist_quantized_train
    ${project_library_target_name} ${REQUIRED_LIBRARIES})

add_executable(example_deconv_train deconv/train.cpp ${tiny_dnn_headers})
target_link_libraries(example_deconv_train
    ${project_library_target_name} ${REQUIRED_LIBRARIES})

endif()

add_executable(example_deconv_visual deconv/visual.cpp ${tiny_dnn_headers})
target_link_libraries(example_deconv_visual
    ${project_library_target_name} ${REQUIRED_LIBRARIES})

add_executable(example_cifar_train cifar10/train.cpp ${tiny_dnn_headers})
target_link_libraries(example_cifar_train
    ${project_library_target_name} ${REQUIRED_LIBRARIES})

if(PROTO_CPP_AVAILABLE)
    include_directories(${PROTOBUF_INCLUDE_DIRS})
    set_source_files_properties(${proto_file} PROPERTIES GENERATED TRUE)
    add_executable(example_caffe_converter
                   caffe_converter/caffe_converter.cpp ${proto_file} ${tiny_dnn_headers})
    target_link_libraries(example_caffe_converter
            ${project_library_target_name}
            ${PROTOBUF_LIBRARIES} ${REQUIRED_LIBRARIES})
endif()

if(PROTO_CPP_GENERATE)
    add_dependencies(example_caffe_converter generated_proto)
endif()

find_package(Protobuf QUIET)
if(PROTOBUF_FOUND)
    set(proto_file "${CMAKE_CURRENT_SOURCE_DIR}/../tensorflow/core/framework/graph.pb.cc")
    if(EXISTS ${proto_file})
        file(GLOB proto_src "${CMAKE_CURRENT_SOURCE_DIR}/../tensorflow/core/framework/*.pb.cc")
        add_executable(example_tensorflow_converter
                       ${CMAKE_CURRENT_SOURCE_DIR}/../examples/tensorflow_converter/tensorflow_parser.cpp ${proto_src})
        #              ${CMAKE_CURRENT_SOURCE_DIR}/../examples/tensorflow_converter/example_tensorflow_parser.cpp ${proto_file} ${proto_src})
        target_link_libraries(example_tensorflow_converter
            ${project_library_target_name}
            ${PROTOBUF_LIBRARIES} ${REQUIRED_LIBRARIES})
    elseif(EXISTS ${PROTOBUF_PROTOC_EXECUTABLE})
        message(STATUS "Found PROTOBUF Compiler: ${PROTOBUF_PROTOC_EXECUTABLE}")
        if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../tiny_cnn/io/tensorflow/graph.proto)
            PROTOBUF_GENERATE_CPP(PROTO_SRCS PROTO_HDRS
                ${CMAKE_CURRENT_SOURCE_DIR}/../tiny_cnn/io/tensorflow/graph.proto)
            file(GLOB proto_src "${CMAKE_CURRENT_SOURCE_DIR}/../tensorflow/core/framework/*.pb.cc")
            if(EXISTS ${PROTO_HDRS})
                file(COPY ${PROTO_SRCS} DESTINATION "${CMAKE_CURRENT_SOURCE_DIR}/../tiny_cnn/io/tensorflow")
                file(COPY ${PROTO_HDRS} DESTINATION "${CMAKE_CURRENT_SOURCE_DIR}/../tiny_cnn/io/tensorflow")
            endif()
            add_executable(example_tensorflow_converter
                           ${CMAKE_CURRENT_SOURCE_DIR}/../examples/tensorflow_converter/tensorflow_parser.cpp
                           ${proto_src})
            target_link_libraries(example_tensorflow_converter
                ${project_library_target_name}
                ${PROTOBUF_LIBRARIES} ${REQUIRED_LIBRARIES})
        else()
            message(STATUS "Cannot generate C++ proto files, please provide Protobuf file.")
        endif()
    else()
        message(STATUS "Proto is not linked correctly, please make sure file exists.")
    endif()
else()
    message(STATUS "Cannot generate TensorFlow Importer. Please install Protobuf.")
endif()
