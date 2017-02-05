#tests

include_directories("${THREAD_POOL_CPP_INC_DIR}")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}")

add_executable(fixed_function_test fixed_function.t.cpp)

target_link_libraries(fixed_function_test)
add_custom_command(
    TARGET fixed_function_test
    POST_BUILD
    COMMAND ./fixed_function_test
)

add_executable(thread_pool_test thread_pool.t.cpp thread_pool2.t.cpp)
target_link_libraries(thread_pool_test pthread)
add_custom_command(
    TARGET thread_pool_test
    POST_BUILD
    COMMAND ./thread_pool_test
)
