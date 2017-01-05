cmake_minimum_required( VERSION 3.2 )

project( tensorflow_converter )

find_package( Protobuf 3.0.0 REQUIRED )

# list the needed tensorflow protobuf files
set(PROTO_FILES_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../tiny_dnn/io/tensorflow)

set(PROTO_FILES_NAMES versions
   		      op_def
 	              function
 	              types
 		      tensor
 		      graph
		      attr_value
		      tensor_shape
		      tensor_description
		      allocation_description)

if(FALSE)
set(PROTO_FILES_NAMES cost_graph
		      device_attributes
		      kernel_def
		      log_memory
		      step_stats
		      summary
		      tensor_slice
		      variable)
endif()

foreach(name ${PROTO_FILES_NAMES})
    list(APPEND PROTO_FILES "${PROTO_FILES_PATH}/${name}.proto")
endforeach()

# generate CPP file for connexion between protobuf and tiny_dnn 
PROTOBUF_GENERATE_CPP( PROTO_SRC PROTO_HDR ${PROTO_FILES} )

# set the executable (main cpp + protobuf generated files )
add_executable( tensorflow_converter tensorflow_parser.cpp ${PROTO_SRC} ${PROTO_HDR} )

# Set the include dir protobuf files and location of tiny-dnn source
target_include_directories( tensorflow_converter PRIVATE ${CMAKE_CURRENT_BINARY_DIR}
                                               ${PROTOBUF_INCLUDE_DIR} 
                                               ${CMAKE_CURRENT_SOURCE_DIR}/../../ )

# Link with protobuf 
target_link_libraries( tensorflow_converter ${PROTOBUF_LIBRARY} )

# Set C++11 mode 
set_target_properties( tensorflow_converter PROPERTIES CXX_STANDARD 11 )
