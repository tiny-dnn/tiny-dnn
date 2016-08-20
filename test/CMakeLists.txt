enable_testing()

# find test headers
file(GLOB_RECURSE tiny_dnn_test_hdrs *.h)

add_executable(tiny_dnn_test test.cpp ${tiny_dnn_test_headers})
target_link_libraries(tiny_dnn_test 
    ${project_library_target_name} ${REQUIRED_LIBRARIES})
