#Command line options
#-DUSE_TBB=ON/OFF  (default off)
#-DUSE_OMP=ON/OFF  (default off)
#-DUSE_SSE=ON/OFF  (default on)
#-DUSE_AVX=ON/OFF  (default on)

# ----------------------------------------------------------------------------
#   Basic Configuration
# ----------------------------------------------------------------------------
CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

PROJECT(tiny_cnn)
set(PROJECT_VERSION "0.0.1")
string(REGEX MATCHALL "[0-9]" PROJECT_VERSION_PARTS "${PROJECT_VERSION}")
list(GET PROJECT_VERSION_PARTS 0 PROJECT_VERSION_MAJOR)
list(GET PROJECT_VERSION_PARTS 1 PROJECT_VERSION_MINOR)
list(GET PROJECT_VERSION_PARTS 2 PROJECT_VERSION_PATCH)
set(PROJECT_SOVERSION "${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}")

#------------------------------------------------------
# Build type
#------------------------------------------------------

IF(NOT CMAKE_BUILD_TYPE )
   SET( CMAKE_BUILD_TYPE "Release" )
ENDIF()

#------------------------------------------------------
# Lib Names and Dirs
#------------------------------------------------------

if(WIN32)
    # Postfix of DLLs:
    SET(PROJECT_DLLVERSION "${PROJECT_VERSION_MAJOR}${PROJECT_VERSION_MINOR}${PROJECT_VERSION_PATCH}")
    SET(RUNTIME_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin CACHE PATH "Directory for dlls and binaries")
    SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin CACHE PATH "Directory for binaries")
    SET(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin CACHE PATH "Directory for dlls")
else()
    # Postfix of so's:
    set(PROJECT_DLLVERSION)
    SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_INSTALL_PREFIX}/lib/cmake/ /usr/lib/cmake)
endif()
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}")

OPTION(USE_TBB 	"Set to ON to use boost" OFF)
OPTION(USE_OMP 	"Set to ON to use boost" OFF)
OPTION(USE_SSE 	"Set to ON to use sse" ON)
OPTION(USE_AVX 	"Set to ON to use avx" ON)

# ----------------------------------------------------------------------------
#   Find Dependencies
# ----------------------------------------------------------------------------

######BOOST (NOT REQUIRED)
set(Boost_USE_STATIC_LIBS        OFF)
find_package( Boost   COMPONENTS   timer   ) #thread iostreams
if(Boost_FOUND)
	include_directories(${Boost_INCLUDE_DIRS})
	link_directories(${Boost_LIBRARY_DIRS})
	SET(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH} ${Boost_LIBRARY_DIRS})
        SET(REQUIRED_LIBRARIES  ${REQUIRED_} ${Boost_TIMER_LIBRARY}  )
ENDIF()


IF(USE_TBB)
    find_package(TBB REQUIRED)
    add_definitions(-DCNN_USE_TBB)
    include_directories(${TBB_INCLUDE_DIRS})
    link_directories(${TBB_LIBRARY_DIRS})
    SET(REQUIRED_LIBRARIES  ${REQUIRED_LIBRARIES} ${TBB_LIBRARIES}  )
ELSEIF(USE_OMP)
find_package(OpenMP  REQUIRED)
        set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        add_definitions( -DCNN_USE_OMP)
ENDIF()
# ----------------------------------------------------------------------------
# COMPILER OPTIONS
# ----------------------------------------------------------------------------

IF(CMAKE_COMPILER_IS_GNUCXX OR MINGW)
    IF(USE_SSE)
        add_definitions(-DCNN_USE_SSE)
        SET(EXTRA_C_FLAGS  " ${EXTRA_C_FLAGS} -msse3 ")
    ENDIF()
    IF(USE_AVX)
        add_definitions(-DCNN_USE_AVX)
        SET(EXTRA_C_FLAGS  "${EXTRA_C_FLAGS} -mavx ")
    ENDIF()
    SET(EXTRA_C_FLAGS " ${EXTRA_C_FLAGS} -std=c++11 ")
    SET(EXTRA_C_FLAGS_RELEASE " -O3 ")
    SET(EXTRA_C_FLAGS_DEBUG " -g3 ")
ELSE()#MSVC
    IF(USE_SSE )
        add_definitions(-DCNN_USE_SSE)
        SET(EXTRA_C_FLAGS  " ${EXTRA_C_FLAGS} /arch:SSE2 ")
    ENDIF()
    IF(USE_AVX  )
        add_definitions(-DCNN_USE_AVX)
        SET(EXTRA_C_FLAGS  "${EXTRA_C_FLAGS}  /arch:AVX ")
    ENDIF()
    SET(EXTRA_C_FLAGS_RELEASE " /Ox ")


ENDIF()

 

    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${EXTRA_C_FLAGS}")
    SET(CMAKE_CXX_FLAGS_RELEASE "${EXTRA_C_FLAGS_RELEASE}")
    SET(CMAKE_CXX_FLAGS_DEBUG "${EXTRA_C_FLAGS_DEBUG}")
    SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${EXTRA_EXE_LINKER_FLAGS}")
    SET(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} ${EXTRA_EXE_LINKER_FLAGS_RELEASE}")
    SET(CMAKE_EXE_LINKER_FLAGS_DEBUG "${CMAKE_EXE_LINKER_FLAGS_DEBUG} ${EXTRA_EXE_LINKER_FLAGS_DEBUG}")

 
#------------------------------------------------
# TARGETS
#------------------------------------------------
INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/tiny_cnn )
LINK_LIBRARIES(${REQUIRED_LIBRARIES} )

SET( tiny_cnn_hrds tiny_cnn/activation_function.h    tiny_cnn/cifar10_parser.h  tiny_cnn/convolutional_layer.h  tiny_cnn/display.h  tiny_cnn/fully_connected_dropout_layer.h  tiny_cnn/image.h        tiny_cnn/layer.h   tiny_cnn/loss_function.h      tiny_cnn/mnist_parser.h  tiny_cnn/optimizer.h                tiny_cnn/product.h   tiny_cnn/util.h
tiny_cnn/average_pooling_layer.h  tiny_cnn/config.h          tiny_cnn/deform.h               tiny_cnn/dropout.h  tiny_cnn/fully_connected_layer.h          tiny_cnn/input_layer.h  tiny_cnn/layers.h  tiny_cnn/max_pooling_layer.h  tiny_cnn/network.h       tiny_cnn/partial_connected_layer.h  tiny_cnn/tiny_cnn.h  tiny_cnn/weight_init.h)

IF(Boost_FOUND)
    ADD_EXECUTABLE(train example/train.cpp ${tiny_cnn_hrds})
ENDIF()
ADD_EXECUTABLE(test example/test.cpp  ${tiny_cnn_hrds})




# ----------------------------------------------------------------------------
# display status message for important variables
# ----------------------------------------------------------------------------
message( STATUS )
MESSAGE( STATUS "-------------------------------------------------------------------------------" )
message( STATUS "General configuration for ${PROJECT_NAME} ${PROJECT_VERSION}")
MESSAGE( STATUS "-------------------------------------------------------------------------------" )


message( STATUS "Boost_FOUND=${Boost_FOUND}   USE_TBB=${USE_TBB} USE_OMP=${USE_OMP} USE_SSE=${USE_SSE} USE_AVX=${USE_AVX}")


MESSAGE( STATUS "-------------------------------------------------------------------------------" )

message( STATUS )
message("    Built as dynamic libs?:" ${BUILD_SHARED_LIBS})
message("    Compiler:"                   "${CMAKE_COMPILER}"   "${CMAKE_CXX_COMPILER}")
message( STATUS "C++ flags (Release):       ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE}")
message( STATUS "C++ flags (Debug):         ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG}")
message( STATUS "CMAKE_CXX_FLAGS:         ${CMAKE_CXX_FLAGS}")
message( STATUS "CMAKE_BINARY_DIR:         ${CMAKE_BINARY_DIR}")

MESSAGE( STATUS )
MESSAGE( STATUS "CMAKE_SYSTEM_PROCESSOR = ${CMAKE_SYSTEM_PROCESSOR}" )
MESSAGE( STATUS "REQUIRED_LIBRARIES = ${REQUIRED_LIBRARIES}" )
MESSAGE( STATUS "CMAKE_INSTALL_PREFIX = ${CMAKE_INSTALL_PREFIX}" )
MESSAGE( STATUS "CMAKE_BUILD_TYPE = ${CMAKE_BUILD_TYPE}" )
MESSAGE( STATUS "CMAKE_MODULE_PATH = ${CMAKE_MODULE_PATH}" )
MESSAGE( STATUS "CMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}")
MESSAGE( STATUS )
MESSAGE( STATUS )
MESSAGE( STATUS "Change a value with: cmake -D<Variable>=<Value>" )
MESSAGE( STATUS )
