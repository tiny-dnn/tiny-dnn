#include <jni.h>
#include <string.h>
#include <string>
#include <android/log.h>
#include <stdarg.h>
#include <stdio.h>
#include <vector>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <list>
#include <sstream>


//using namespace std;

#ifndef COMMON_H
#define COMMON_H

#define DEBUG_TAG "AndroidLibSvmNDK" 
#define DEBUG_MACRO(x) __android_log_print(ANDROID_LOG_DEBUG, DEBUG_TAG, "NDK: %s", x);
const int debug_message_max=1024;

void debug(const char *s,...);

void cmdToArgv(std::string, std::vector<char*> &);

#endif
