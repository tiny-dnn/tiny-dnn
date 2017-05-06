cmake_minimum_required(VERSION 3.0)

list(APPEND CMAKE_MODULE_PATH
       "${CMAKE_CURRENT_SOURCE_DIR}/cmake"
       "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")

project(thread-pool-cpp)
add_definitions(-std=c++14 -Wall -Werror -O3)


# Get all include files
set(THREAD_POOL_CPP_INC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include/")
include_directories("${THREAD_POOL_CPP_INC_DIR}")
file(GLOB_RECURSE INSTALL_FILES_LIST "${THREAD_POOL_CPP_INC_DIR}/*")

add_subdirectory(tests)
add_subdirectory(benchmark)

# Install as header-only library
set_source_files_properties(${INSTALL_FILES_LIST} PROPERTIES HEADER_FILE_ONLY 1)
add_library(HEADER_ONLY_TARGET STATIC ${INSTALL_FILES_LIST})
set_target_properties(HEADER_ONLY_TARGET PROPERTIES LINKER_LANGUAGE CXX)
install(DIRECTORY ${THREAD_POOL_CPP_INC_DIR} DESTINATION "include")
