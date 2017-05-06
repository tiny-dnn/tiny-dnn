#benchmark

include_directories("${THREAD_POOL_CPP_INC_DIR}")

find_package(Boost REQUIRED COMPONENTS system)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${Boost_INCLUDE_DIR})
add_executable(benchmark benchmark.cpp)
target_link_libraries(benchmark ${Boost_LIBRARIES} pthread)

