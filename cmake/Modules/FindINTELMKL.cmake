SET(INTEL_MKL_INCLUDE_SEARCH_PATHS
    /usr/include
    /usr/local/include
    /opt/intel/compilers_and_libraries/linux/mkl
    /opt/intel/compilers_and_libraries/linux/mkl/include
    ${MKLROOT}
    ${MKLROOT}/include
)

SET(INTEL_MKL_LIB_SEARCH_PATHS
    /lib/
    /lib64/
    /usr/lib
    /usr/lib64
    /usr/local/lib
    /usr/local/lib64
    /opt/intel/compilers_and_libraries/linux/mkl
    /opt/intel/compilers_and_libraries/linux/mkl/lib
    /opt/intel/compilers_and_libraries/linux/mkl/lib/intel
    /opt/intel/compilers_and_libraries/linux/mkl/lib/intel64
    ${MKLROOT}
    ${MKLROOT}/lib
 )

FIND_PATH(INTEL_MKL_INCLUDE_DIR NAMES mkl.h PATHS ${INTEL_MKL_INCLUDE_SEARCH_PATHS})
FIND_LIBRARY(INTEL_MKL_LIB_1 NAMES libmkl_intel_ilp64.a PATHS ${INTEL_MKL_LIB_SEARCH_PATHS})
FIND_LIBRARY(INTEL_MKL_LIB_2 NAMES libmkl_sequential.a PATHS ${INTEL_MKL_LIB_SEARCH_PATHS})
FIND_LIBRARY(INTEL_MKL_LIB_3 NAMES libmkl_core.a PATHS ${INTEL_MKL_LIB_SEARCH_PATHS})

SET(INTELMKL_FOUND ON)

#    Check include files
IF(NOT INTEL_MKL_INCLUDE_DIR)
    SET(INTELMKL_FOUND OFF)
    MESSAGE(STATUS "Could not find Intel MKL include. Turning INTEL_MKL_FOUND off")
ENDIF()

#    Check libraries
IF(NOT INTEL_MKL_LIB_1)
    SET(INTELMKL_FOUND OFF)
    MESSAGE(STATUS "Could not find Intel MKL lib. Turning INTEL_MKL_FOUND off")
ENDIF()
IF(NOT INTEL_MKL_LIB_2)
    SET(INTELMKL_FOUND OFF)
    MESSAGE(STATUS "Could not find Intel MKL lib. Turning INTEL_MKL_FOUND off")
ENDIF()
IF(NOT INTEL_MKL_LIB_3)
    SET(INTELMKL_FOUND OFF)
    MESSAGE(STATUS "Could not find Intel MKL lib. Turning INTEL_MKL_FOUND off")
ENDIF()

IF(INTELMKL_FOUND)
    add_definitions(-DUSE_INTEL_MKL)
    IF(NOT INTEL_MKL_FIND_QUIETLY)
        MESSAGE(STATUS "Found Intel MKL libraries: ${INTEL_MKL_LIB_1} ${INTEL_MKL_LIB_2} ${INTEL_MKL_LIB_3}")
        MESSAGE(STATUS "Found Intel MKL include: ${INTEL_MKL_INCLUDE_DIR}")
    ENDIF(NOT INTEL_MKL_FIND_QUIETLY)
ELSE(INTELMKL_FOUND)
    IF(INTEL_MKL_FIND_REQUIRED)
        MESSAGE(FATAL_ERROR "Could not find Intel MKL")
    ENDIF(INTEL_MKL_FIND_REQUIRED)
ENDIF(INTELMKL_FOUND)

MARK_AS_ADVANCED(
    INTEL_MKL_INCLUDE_DIR
    INTEL_MKL_LIB_1
    INTEL_MKL_LIB_2
    INTELMKL_FOUND
)
