cmake_minimum_required( VERSION 3.2 )

project( caffe_converter )

find_package( Protobuf REQUIRED )

# generate CPP file for connexion between protobuf and tiny_dnn 
PROTOBUF_GENERATE_CPP( PROTO_SRC PROTO_HDR ${CMAKE_CURRENT_SOURCE_DIR}/../../tiny_dnn/io/caffe/caffe.proto )

# set the executable (main cpp + protobuf generated caffe.pb.h/caffe.pb.cc )
add_executable( caffe_converter caffe_converter.cpp ${PROTO_SRC} ${PROTO_HDR} )

# Set the include dir (binary for caffe.pb.h, protobuf and location of tiny-dnn source)
target_include_directories( caffe_converter PRIVATE ${CMAKE_CURRENT_BINARY_DIR}
                                               ${PROTOBUF_INCLUDE_DIR} 
                                               ${CMAKE_CURRENT_SOURCE_DIR}/../../ )

# Link with protobuf 
target_link_libraries( caffe_converter ${PROTOBUF_LIBRARY} )

# Set C++11 mode 
set_target_properties( caffe_converter PROPERTIES CXX_STANDARD 11 )

cotire(caffe_converter)
