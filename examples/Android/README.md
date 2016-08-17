
#Dependency

The only dependency is Opencv-android library

[Opencv-android-sdk-3.0](http://opencv.org/downloads.html)

Then change the path of your opencv.mk in jni/Android.mk 

#Compile in linux 

1.Setup NDK and configure the environment variables 

2.Some Changes in the Android.mk file
  ```
LOCAL_PATH := $(call my-dir)  
 
include $(CLEAR_VARS)    
OpenCV_INSTALL_MODULES:=on
OPENCV_CAMERA_MODULES:=off
OPENCV_LIB_TYPE:= SHARE
LOCAL_CPP_EXTENSION := .cc

#change to your path
#include /home/jaychou/Downloads/OpenCV-android-sdk/sdk/native/jni/OpenCV.mk


LOCAL_MODULE  := caffe_p
LOCAL_SRC_FILES  := cnnInterface.cpp , common.cpp
LOCAL_LDLIBS += -lm -llog 


LOCAL_C_INCLUDES += $(LOCAL_PATH)/../../../

#change to your path
#LOCAL_C_INCLUDES += /home/jaychou/Downloads/OpenCV-android-sdk/sdk/native/jni/include

include $(BUILD_SHARED_LIBRARY)
  
  ```
  
3.Change the tinycnn.h 

  notes the header file
  ```
#ifdef CNN_USE_CAFFE_CONVERTER
// experimental / require google protobuf
//#include "tiny_cnn/io/caffe/layer_factory.h"
#endif

  ```
  
4.Enter the jni directory and running NDK-build.



