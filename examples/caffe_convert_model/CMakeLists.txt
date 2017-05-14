cmake_minimum_required( VERSION 3.2 )

project( caffe_convert_model )

find_package( Protobuf REQUIRED )

# generate CPP file for connexion between protobuf and tiny_dnn 
PROTOBUF_GENERATE_CPP( PROTO_SRC PROTO_HDR ${CMAKE_CURRENT_SOURCE_DIR}/../../tiny_dnn/io/caffe/caffe.proto )

# set the executable (main cpp + protobuf generated caffe.pb.h/caffe.pb.cc )
add_executable( caffe_convert_model caffe_convert_model.cpp ${PROTO_SRC} ${PROTO_HDR} )

# Set the include dir (binary for caffe.pb.h, protobuf and location of tiny-dnn source)
target_include_directories( caffe_convert_model PRIVATE ${CMAKE_CURRENT_BINARY_DIR}
                                               ${PROTOBUF_INCLUDE_DIR} 
                                               ${CMAKE_CURRENT_SOURCE_DIR}/../../ )

# Link with protobuf 
target_link_libraries( caffe_convert_model ${PROTOBUF_LIBRARY} )

# Set C++11 mode 
set_target_properties( caffe_convert_model PROPERTIES CXX_STANDARD 11 )

cotire(caffe_convert_model)