# Additional target to perform clang-format/clang-tidy run
# Requires clang-format and clang-tidy

file(GLOB_RECURSE ALL_CXX_SOURCE_FILES
  ${CMAKE_SOURCE_DIR}/tiny_dnn/*.h
  ${CMAKE_SOURCE_DIR}/benchmarks/*.h
  ${CMAKE_SOURCE_DIR}/test/*.h
  ${CMAKE_SOURCE_DIR}/examples/*.h
  ${CMAKE_SOURCE_DIR}/examples/*.cpp
  )

set(clang-format "clang-format-4.0")

# Adding clang-format target if executable is found
find_program(CLANG_FORMAT ${clang-format})
if(CLANG_FORMAT)
  add_custom_target(
    clang-format
    COMMAND /usr/bin/${clang-format}
    -i
    ${ALL_CXX_SOURCE_FILES}
    )

  add_custom_target(
    clang-format-check
    COMMAND /bin/bash ${CMAKE_SOURCE_DIR}/.travis/clang-format.sh
  )
else()
  message(STATUS "${clang-format} was not found")
endif()

# Adding clang-tidy target if executable is found
find_program(CLANG_TIDY "clang-tidy")
if(CLANG_TIDY)
  add_custom_target(
    clang-tidy
    COMMAND /usr/bin/clang-tidy
    ${ALL_CXX_SOURCE_FILES}
    -config=''
    -checks=*
    --
    -std=c++11
    ${INCLUDE_DIRECTORIES}
    )
endif()

add_custom_target(
   test_lints
   COMMAND /bin/bash ${CMAKE_SOURCE_DIR}/scripts/run_lints.sh
   ${ALL_CXX_SOURCE_FILES}
   )
