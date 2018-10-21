include(../cmake/DownloadProject/DownloadProject.cmake)
download_project(
        PROJ benchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG v1.1.0
        UPDATE_DISCONNECTED 1
)

unset(CMAKE_CXX_STANDARD)
unset(CMAKE_CXX_STANDARD_REQUIRED)

add_subdirectory(${benchmark_SOURCE_DIR} ${benchmark_BINARY_DIR})
include_directories("${benchmark_SOURCE_DIR}/include")

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(tiny_dnn_benchmarks benchmarks.cpp)
target_link_libraries(tiny_dnn_benchmarks
        ${project_library_target_name} ${REQUIRED_LIBRARIES} benchmark)

#### Clang-tidy check
#include(../cmake/clang-cxx-dev-tools.cmake)
