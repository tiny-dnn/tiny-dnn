# Additional target to perform clang-format/clang-tidy run
# Requires clang-format and clang-tidy

file(GLOB_RECURSE ALL_CXX_SOURCE_FILES ../tiny_dnn/*.h)
#message(${ALL_CXX_SOURCE_FILES})

if (FALSE)
# Adding clang-format target if executable is found
find_program(CLANG_FORMAT "clang-format")
if(CLANG_FORMAT)
  add_custom_target(
    clang-format
    COMMAND /usr/bin/clang-format
    -i
    -style=file
    ${ALL_CXX_SOURCE_FILES}
    )
endif()
endif()

# Adding clang-tidy target if executable is found
find_program(CLANG_TIDY "clang-tidy")
if(CLANG_TIDY)
  add_custom_target(
    clang-tidy
    COMMAND /usr/bin/clang-tidy
    ${ALL_CXX_SOURCE_FILES}
    -config=''
    --
    -std=c++11
    #${INCLUDE_DIRECTORIES}
    -checks=-*,clang-analyzer-*,-clang-analyzer-cplusplus*
    )
endif()
