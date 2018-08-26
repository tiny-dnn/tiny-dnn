include(CTest)
enable_testing()

add_executable(print_device print_device.cpp)
target_link_libraries(print_device
    ${project_library_target_name} ${REQUIRED_LIBRARIES})

add_executable(tiny_dnn_test test.cpp test_main.cpp)

set_target_properties(tiny_dnn_test PROPERTIES LINKER_LANGUAGE CXX)

target_link_libraries(tiny_dnn_test
    ${project_library_target_name} ${REQUIRED_LIBRARIES})


add_test(all_tests tiny_dnn_test)
# workaround for https://gitlab.kitware.com/cmake/cmake/issues/8774
add_custom_target(run_tests COMMAND ${CMAKE_CTEST_COMMAND}
    DEPENDS tiny_dnn_test)

if(COVERALLS)
    message(STATUS "Code coverage: Enabled")
    set_target_properties(tiny_dnn_test
        PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} -g -fprofile-arcs -ftest-coverage"
        LINK_FLAGS "-lgcov --coverage")

else(COVERALLS)
    message(STATUS "Code coverage: Disabled")
endif(COVERALLS)

if(PROTO_CPP_AVAILABLE)
    set_source_files_properties(${proto_file} PROPERTIES GENERATED TRUE)
    target_sources(tiny_dnn_test PUBLIC ${proto_file})
    set_source_files_properties(test.cpp PROPERTIES COMPILE_DEFINITIONS
                                CNN_USE_CAFFE_CONVERTER)
    target_link_libraries(tiny_dnn_test
            ${project_library_target_name}
            ${PROTOBUF_LIBRARIES} ${REQUIRED_LIBRARIES})   
endif()

if(PROTO_CPP_GENERATE)
    add_dependencies(tiny_dnn_test generated_proto)
endif()

### Clang-tidy check
include(../cmake/clang-cxx-dev-tools.cmake)

cotire(print_device tiny_dnn_test)
