################################################################################################
# TinyDNN status report function.
# Automatically align right column and selects text based on condition.

function(tinydnn_status text)
  set(status_cond)
  set(status_then)
  set(status_else)

  set(status_current_name "cond")
  foreach(arg ${ARGN})
    if(arg STREQUAL "THEN")
      set(status_current_name "then")
    elseif(arg STREQUAL "ELSE")
      set(status_current_name "else")
    else()
      list(APPEND status_${status_current_name} ${arg})
    endif()
  endforeach()

  if(DEFINED status_cond)
    set(status_placeholder_length 23)
    string(RANDOM LENGTH ${status_placeholder_length} ALPHABET " " status_placeholder)
    string(LENGTH "${text}" status_text_length)
    if(status_text_length LESS status_placeholder_length)
      string(SUBSTRING "${text}${status_placeholder}" 0 ${status_placeholder_length} status_text)
    elseif(DEFINED status_then OR DEFINED status_else)
      message(STATUS "${text}")
      set(status_text "${status_placeholder}")
    else()
      set(status_text "${text}")
    endif()

    if(DEFINED status_then OR DEFINED status_else)
      if(${status_cond})
        string(REPLACE ";" " " status_then "${status_then}")
        string(REGEX REPLACE "^[ \t]+" "" status_then "${status_then}")
        message(STATUS "${status_text} ${status_then}")
      else()
        string(REPLACE ";" " " status_else "${status_else}")
        string(REGEX REPLACE "^[ \t]+" "" status_else "${status_else}")
        message(STATUS "${status_text} ${status_else}")
      endif()
    else()
      string(REPLACE ";" " " status_cond "${status_cond}")
      string(REGEX REPLACE "^[ \t]+" "" status_cond "${status_cond}")
      message(STATUS "${status_text} ${status_cond}")
    endif()
  else()
    message(STATUS "${text}")
  endif()
endfunction()

################################################################################################
# Function merging lists of compiler flags to single string.
# Usage:
#   tinydnn_merge_flag_lists(out_variable <list1> [<list2>] [<list3>] ...)
function(tinydnn_merge_flag_lists out_var)
  set(__result "")
  foreach(__list ${ARGN})
    foreach(__flag ${${__list}})
      string(STRIP ${__flag} __flag)
      set(__result "${__result} ${__flag}")
    endforeach()
  endforeach()
  string(STRIP ${__result} __result)
  set(${out_var} ${__result} PARENT_SCOPE)
endfunction()

####
# Prints accumulated tiny-dnn configuration summary
# Usage:
#   tinydnn_print_configuration_summary()

function(tinydnn_print_configuration_summary)

    tinydnn_merge_flag_lists(__flags_rel CMAKE_CXX_FLAGS_RELEASE CMAKE_CXX_FLAGS)
    tinydnn_merge_flag_lists(__flags_deb CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS)

    tinydnn_status("")
    tinydnn_status("******************* tiny-dnn Configuration Summary *******************")
    tinydnn_status("General:")
    tinydnn_status("  Version           :   ${PROJECT_VERSION}")
    tinydnn_status("  System            :   ${CMAKE_SYSTEM_NAME}")
    tinydnn_status("  C++ compiler      :   ${CMAKE_CXX_COMPILER}")
    tinydnn_status("  Release CXX flags :   ${__flags_rel}")
    tinydnn_status("  Debug CXX flags   :   ${__flags_deb}")
    tinydnn_status("  Build type        :   ${CMAKE_BUILD_TYPE}")
    tinydnn_status("")
    tinydnn_status("  BUILD_EXAMPLES    :   ${BUILD_EXAMPLES}")
    tinydnn_status("  BUILD_TESTS       :   ${BUILD_TESTS}")
    tinydnn_status("  BUILD_DOCS        :   ${BUILD_DOCS}")
    tinydnn_status("")
    tinydnn_status("Dependencies:")
    tinydnn_status("  SSE               : " USE_SSE AND COMPILER_HAS_SSE_FLAG THEN "Yes" ELSE "No")
    tinydnn_status("  AVX               : " USE_AVX AND COMPILER_HAS_AVX_FLAG THEN "Yes" ELSE "No")
    tinydnn_status("  AVX2              : " USE_AVX2 AND COMPILER_HAS_AVX2_FLAG THEN "Yes" ELSE "No")
    tinydnn_status("  Pthread           : " USE_PTHREAD THEN "Yes" ELSE "No")
    tinydnn_status("  TBB               : " USE_TBB AND TBB_FOUND THEN "Yes (ver. ${TBB_INTERFACE_VERSION})" ELSE "No")
    tinydnn_status("  OMP               : " USE_OMP AND OMP_FOUND THEN "Yes" ELSE "No")
    tinydnn_status("  NNPACK            : " USE_NNPACK AND NNPACK_FOUND THEN "Yes" ELSE "No")
    tinydnn_status("  OpenCL            : " USE_OPENCL AND OpenCL_FOUND THEN "Yes (ver. ${OpenCL_VERSION_STRING})" ELSE "No")
    tinydnn_status("  LibDNN            : " USE_LIBDNN AND GreenteaLibDNN_FOUND THEN "Yes (ver. ${GreenteaLibDNN_VERSION})" ELSE "No")
    tinydnn_status("")
    tinydnn_status("Utilities:")
    tinydnn_status("  OpenCV            : " USE_OPENCV AND OpenCV_FOUND THEN "Yes (ver. ${OpenCV_VERSION})" ELSE "No")
    tinydnn_status("  Serializer        : " USE_SERIALIZER THEN "Yes" ELSE "No")
    tinydnn_status("  Image API         : " USE_IMAGE_API THEN "Yes" ELSE "No")
    tinydnn_status("  Gemmlowp          : " USE_GEMMLOWP THEN "Yes" ELSE "No")
    tinydnn_status("  THREAD_POOL_KIND  : " ${THREAD_POOL_KIND})
    tinydnn_status("")
    tinydnn_status("Install:")
    tinydnn_status("  Install path      :   ${CMAKE_INSTALL_PREFIX}")
    tinydnn_status("")
endfunction()
