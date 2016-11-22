# Try to prepare caffe.pb.cc and caffe.pb.h
#
# - If we can't find Protobuf, do nothing.
# - If we have Protobuf and already have caffe.pb.cc/h under io/caffe,
#   we just set PROTO_CPP_AVAILABLE variable.
# - If we have Protobuf and Protoc but haven't caffe.pb.cc/h,
#   we'll add additional target to generate these files (generated_proto),
#   and set PROTO_CPP_AVAILABLE AND PROTO_CPP_GENERATE variables.
#
#   Since protoc will be invoked at the compile time (not the configuration time),
#   we must add add_dependencies(xxx generated_proto) line to ensure
#   our target needs generated proto files to build.

find_package(Protobuf QUIET)

if(PROTOBUF_FOUND)
    set(proto_file "${CMAKE_SOURCE_DIR}/tiny_dnn/io/caffe/caffe.pb.cc")
    if(EXISTS ${proto_file})
        message(STATUS "Found proto-file: ${proto_file}")
        set (PROTO_CPP_AVAILABLE "YES")
# As of Ubuntu 14.04 protoc is no longer a part of libprotobuf-dev package
# and should be installed separately as in: sudo apt-get install protobuf-compiler
    elseif(EXISTS ${PROTOBUF_PROTOC_EXECUTABLE})
        message(STATUS "Found PROTOBUF Compiler: ${PROTOBUF_PROTOC_EXECUTABLE}")
        if(EXISTS ${CMAKE_SOURCE_DIR}/tiny_dnn/io/caffe/caffe.proto)
            # Note that this line doesn't invoke protoc at configure time
            PROTOBUF_GENERATE_CPP(PROTO_SRCS PROTO_HDRS
                ${CMAKE_SOURCE_DIR}/tiny_dnn/io/caffe/caffe.proto)

            add_custom_target(generated_proto DEPENDS ${PROTO_SRCS} ${PROTO_HDRS})

            # We need to invoke the copy after protoc compile finished
            add_custom_command(TARGET generated_proto PRE_BUILD
                               COMMAND ${CMAKE_COMMAND} -E copy
                               ${PROTO_SRCS} ${CMAKE_SOURCE_DIR}/tiny_dnn/io/caffe)
            add_custom_command(TARGET generated_proto PRE_BUILD
                               COMMAND ${CMAKE_COMMAND} -E copy
                               ${PROTO_HDRS} ${CMAKE_SOURCE_DIR}/tiny_dnn/io/caffe)

            set (PROTO_CPP_AVAILABLE "YES")                
            set (PROTO_CPP_GENERATE "YES")
        else()
            message(STATUS "Cannot generate C++ proto files, please provide Protobuf file.")
        endif()
    else()
        message(STATUS "Proto is not linked correctly, please make sure file exists.")
    endif()
else(PROTOBUF_FOUND)
    message(STATUS "Cannot generate Caffe Importer. Please install Protobuf.")
endif(PROTOBUF_FOUND)
