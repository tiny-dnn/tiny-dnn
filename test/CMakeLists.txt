include(CTest)
enable_testing()

if(USE_SYS_GTEST)
  find_path(GTEST_HDR "gtest/gtest.h")
  if (GTEST_HDR)
    message(STATUS "Found gtest/gtest.h in ${GTEST_HDR}")
  else()
    message(STATUS "Could not find gtest.h!")
  endif()
  find_path(GMOCK_HDR "gmock/gmock.h")
  if (GMOCK_HDR)
    message(STATUS "Found gmock/mock.h in ${GMOCK_HDR}")
  else()
    message(STATUS "Could not find gmock.h!")
  endif()
  find_library(GTEST_LIB "libgtest.so")
  if (GTEST_LIB)
    message(STATUS "Found libgtest: ${GTEST_LIB}")
  else()
    message(STATUS "Could not find libgtest!")
  endif()
  find_library(GMOCK_LIB "libgmock.so")
  if (GMOCK_LIB)
    message(STATUS "Found libgmock: ${GMOCK_LIB}")
  else()
    message(STATUS "Could not find libgmock!")
  endif()
endif()

if(USE_SYS_GTEST AND GTEST_HDR AND GMOCK_HDR AND GTEST_LIB AND GMOCK_LIB )
  message(STATUS "Using system-wide installation of gtest and gmock")
else()
  message(STATUS "Using local copy of gtest and gmock")
  include(../cmake/DownloadProject/DownloadProject.cmake)
  #set(gtest_disable_pthreads on) #TODO(randl): Windows?
  download_project(
      PROJ googletest
      GIT_REPOSITORY https://github.com/google/googletest.git
      GIT_TAG master
      UPDATE_DISCONNECTED 1
  )
  add_subdirectory(${googletest_SOURCE_DIR} ${googletest_BINARY_DIR})
endif()

# in ../googletest-src/googletest/CMakeLists.txt, BUILD_SHARED_LIBS is set to OFF
if(MSVC)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MTd")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MT")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} /MT")
    set(CMAKE_CXX_FLAGS_MINSIZEREL "${CMAKE_CXX_FLAGS_MINSIZEREL} /MT")
endif()

add_executable(print_device print_device.cpp)
target_link_libraries(print_device
    ${project_library_target_name} ${REQUIRED_LIBRARIES})

add_executable(tiny_dnn_test test.cpp test_no_duplicate_symbols.cpp)
target_link_libraries(tiny_dnn_test
    ${project_library_target_name} ${REQUIRED_LIBRARIES} gtest gmock)

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
