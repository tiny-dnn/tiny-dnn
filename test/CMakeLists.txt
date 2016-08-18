enable_testing()

# find test headers
file(GLOB_RECURSE tiny_dnn_test_hdrs *.h)

add_executable(tiny_cnn_device device.cpp ${tiny_cnn_test_headers})
target_link_libraries(tiny_cnn_device
    ${project_library_target_name} ${REQUIRED_LIBRARIES})

add_executable(tiny_cnn_test test.cpp ${tiny_cnn_test_headers})
target_link_libraries(tiny_cnn_test 
    ${project_library_target_name} ${REQUIRED_LIBRARIES})
