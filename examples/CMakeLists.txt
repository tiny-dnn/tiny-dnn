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

add_executable(example_mnist_quantized_train mnist/quantized.cpp)
target_link_libraries(example_mnist_quantized_train
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

if(PROTO_CPP_AVAILABLE)
    include_directories(${PROTOBUF_INCLUDE_DIRS})
    set_source_files_properties(${proto_file} PROPERTIES GENERATED TRUE)
    add_executable(example_caffe_converter
                   caffe_converter/caffe_converter.cpp ${proto_file})
    target_link_libraries(example_caffe_converter
            ${project_library_target_name}
            ${PROTOBUF_LIBRARIES} ${REQUIRED_LIBRARIES})
endif()

if(PROTO_CPP_GENERATE)
    add_dependencies(example_caffe_converter generated_proto)
endif()
