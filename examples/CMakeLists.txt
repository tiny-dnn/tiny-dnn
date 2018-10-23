

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
cotire(benchmarks_all)

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

    add_executable(example_ssd_test ssd_detection/test.cpp ${tiny_dnn_headers})
    target_link_libraries(example_ssd_test
            ${project_library_target_name} ${REQUIRED_LIBRARIES})

    cotire(example_mnist_train example_mnist_test example_mnist_quantized_train example_deconv_train example_ssd_test)
endif()

add_executable(example_deconv_visual deconv/visual.cpp ${tiny_dnn_headers})
target_link_libraries(example_deconv_visual
    ${project_library_target_name} ${REQUIRED_LIBRARIES})

add_executable(example_cifar_train cifar10/train.cpp ${tiny_dnn_headers})
target_link_libraries(example_cifar_train
    ${project_library_target_name} ${REQUIRED_LIBRARIES})

add_executable(example_cifar_test cifar10/test.cpp ${tiny_dnn_headers})
target_link_libraries(example_cifar_test
    ${project_library_target_name} ${REQUIRED_LIBRARIES})

add_executable(example_recurrent_addition recurrent_addition/addition.cpp ${tiny_dnn_headers})
target_link_libraries(example_recurrent_addition
        ${project_library_target_name} ${REQUIRED_LIBRARIES})

add_executable(example_train_char_rnn char_rnn/char_rnn.cpp ${tiny_dnn_headers})
target_link_libraries(example_train_char_rnn
        ${project_library_target_name} ${REQUIRED_LIBRARIES})

add_executable(example_test_char_rnn char_rnn/test_char_rnn.cpp ${tiny_dnn_headers})
target_link_libraries(example_test_char_rnn
        ${project_library_target_name} ${REQUIRED_LIBRARIES})

add_executable(example_sinus_fit sinus_fit/sinus_fit.cpp ${tiny_dnn_headers})
target_link_libraries(example_sinus_fit
        ${project_library_target_name} ${REQUIRED_LIBRARIES})

cotire(example_deconv_visual example_cifar_train example_cifar_test example_recurrent_addition example_train_char_rnn example_test_char_rnn example_sinus_fit)

if(PROTO_CPP_AVAILABLE)
    include_directories(${PROTOBUF_INCLUDE_DIRS})
    set_source_files_properties(${proto_file} PROPERTIES GENERATED TRUE)
    add_executable(example_caffe_converter
                   caffe_converter/caffe_converter.cpp ${proto_file} ${tiny_dnn_headers})
    target_link_libraries(example_caffe_converter
            ${project_library_target_name}
            ${PROTOBUF_LIBRARIES} ${REQUIRED_LIBRARIES})
    cotire(example_caffe_converter)
endif()

if(PROTO_CPP_GENERATE)
    add_dependencies(example_caffe_converter generated_proto)
endif()

if(USE_CBLAS)
    add_executable(example_cblas_mlp backends/cblas/test_mlp.cpp ${tiny_dnn_headers})
    target_link_libraries(example_cblas_mlp
            ${project_library_target_name} ${REQUIRED_LIBRARIES})
endif()
