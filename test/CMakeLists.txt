enable_testing()

add_executable(print_device print_device.cpp)
target_link_libraries(print_device
    ${project_library_target_name} ${REQUIRED_LIBRARIES})

add_executable(tiny_dnn_test test.cpp test_no_duplicate_symbols.cpp)
target_link_libraries(tiny_dnn_test 
    ${project_library_target_name} ${REQUIRED_LIBRARIES})


if(COVERALLS)
	message(STATUS "Code coverage: Enabled")
    set_target_properties(tiny_dnn_test
        PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} -g -fprofile-arcs -ftest-coverage"
                   LINK_FLAGS "-lgcov --coverage")

else(COVERALLS)
	message(STATUS "Code coverage: Disabled")
endif(COVERALLS)
