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
FIND_LIBRARY(INTEL_MKL_LIB NAMES libmkl_intel_lp64.a PATHS ${INTEL_MKL_LIB_SEARCH_PATHS})

SET(INTEL_MKL_FOUND ON)

#    Check include files
IF(NOT INTEL_MKL_INCLUDE_DIR)
    SET(INTEL_MKL_FOUND OFF)
    MESSAGE(STATUS "Could not find Intel MKL include. Turning INTEL_MKL_FOUND off")
ENDIF()

#    Check libraries
IF(NOT INTEL_MKL_LIB)
    SET(INTEL_MKL_FOUND OFF)
    MESSAGE(STATUS "Could not find Intel MKL lib. Turning INTEL_MKL_FOUND off")
ENDIF()

IF(INTEL_MKL_FOUND)
    add_definitions(-DUSE_INTEL_MKL)
    IF(NOT INTEL_MKL_FIND_QUIETLY)
        MESSAGE(STATUS "Found Intel MKL libraries: ${INTEL_MKL_LIB}")
        MESSAGE(STATUS "Found Intel MKL include: ${INTEL_MKL_INCLUDE_DIR}")
    ENDIF(NOT INTEL_MKL_FIND_QUIETLY)
ELSE(INTEL_MKL_FOUND)
    IF(INTEL_MKL_FIND_REQUIRED)
        MESSAGE(FATAL_ERROR "Could not find Intel MKL")
    ENDIF(INTEL_MKL_FIND_REQUIRED)
ENDIF(INTEL_MKL_FOUND)

MARK_AS_ADVANCED(
    INTEL_MKL_INCLUDE_DIR
    INTEL_MKL_LIB
    INTEL_MKL
)
