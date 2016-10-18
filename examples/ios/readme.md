# Guild for iOS demo #
## Part 1: Compiling necessary libs ##
**Check on compiling tools**

Here you can have a check on the local iOS SDK version using ```xcodebuild``` which is integrated with Mac OS
on terminal or iterm2.
```bash
xcodebuild -showsdks
```
For example, if the system is macOS 10.12 (16A323) the result might be like this:

```
iOS SDKs:
	iOS 10.0                      	-sdk iphoneos10.0

iOS Simulator SDKs:
	Simulator - iOS 10.0          	-sdk iphonesimulator10.0

macOS SDKs:
	macOS 10.12                   	-sdk macosx10.12

tvOS SDKs:
	tvOS 10.0                     	-sdk appletvos10.0

tvOS Simulator SDKs:
	Simulator - tvOS 10.0         	-sdk appletvsimulator10.0

watchOS SDKs:
	watchOS 3.0                   	-sdk watchos3.0

watchOS Simulator SDKs:
	Simulator - watchOS 3.0       	-sdk watchsimulator3.0
```
Here we can start the Xcode project building:

**Create a hunter folder**

tiny-dnn using [hunter](https://github.com/ruslo/hunter) which is a CMake-driven cross-platform package manager for C++. Linux, Mac, Windows, iOS, Android, Raspberry Pi as the tool for generating 3rd party libraries like OpenCV and Protobuf.
```bash
git clone https://github.com/ruslo/hunter.git
```
And you also need [Gate to Hunter packages](https://github.com/hunter-packages/gate) and place it under the hunter root folder:
```bash
cd hunter
git clone https://github.com/hunter-packages/gate.git
```
**Clone polly for iOS and android Developer**

The next step is apply [polly](https://github.com/ruslo/polly) as CMake toolchain files and scripts:
```bash
git clone https://github.com/ruslo/polly.git
```
**Building 3rd party libraries**

1. First, we can start with [OpenCV](https://github.com/opencv/opencv). Here we can set environment variables.

```bash
export PROJECT_DIR=examples/OpenCV
export TOOLCHAIN=ios-nocodesign-10-0-wo-armv7s
export CXX=g++
export CC=gcc
export PATH="`pwd`/polly/bin:${PATH}"
export PATH="`pwd`/_ci/cmake/bin:${PATH}"
export XCODE_XCCONFIG_FILE="`pwd`/polly/scripts/NoCodeSign.xcconfig"
export HUNTER_ROOT="`pwd`/lib_ios_opencv/Hunter"
export TESTING_URL="`pwd`/lib_ios_opencv/hunter.tar.gz"
```
Then we can start the library compiling using cmake:
```bash
cmake -H`pwd`/${PROJECT_DIR} -B`pwd`/_testing/_builds/${TOOLCHAIN} -GXcode -DCMAKE_TOOLCHAIN_FILE=${HUNTER_ROOT}/polly/${TOOLCHAIN}.cmake -DCMAKE_VERBOSE_MAKEFILE=ON -DPOLLY_STATUS_DEBUG=ON -DHUNTER_STATUS_DEBUG=ON -DHUNTER_ROOT=${HUNTER_ROOT} -DTESTING_URL=${DTESTING_URL} -DTESTING_SHA1=c5eced7e58ece8610d3340401b104900cf8a2183 -DHUNTER_RUN_INSTALL=ON
```
Mind that ```DTESTING_SHA1``` might differs on your compiling environment, so you need to repalce it if there are such errors:
```
CMake Error at Build/Hunter-prefix/src/Hunter-stamp/verify-Hunter.cmake:40 (message):
  error: SHA1 hash of

    /Users/yidawang/Documents/buildboat/hunter/_testing/hunter.tar.gz

  does not match expected value

    expected: '6854b8abf9e391d3b40d23bbed298e3a36d9d050'
      actual: 'c5eced7e58ece8610d3340401b104900cf8a2183'
```

2. Then continue with [protobuf](https://github.com/google/protobuf), the project directory should be changed:

```bash
export PROJECT_DIR=examples/Protobuf
export HUNTER_ROOT="`pwd`/lib_ios_protobuf/Hunter"
export TESTING_URL="`pwd`/lib_ios_protobuf/hunter.tar.gz"
```
Start compiling protobuf similarly to the OpenCV lib
```bash
cmake -H`pwd`/${PROJECT_DIR} -B`pwd`/_testing/_builds/${TOOLCHAIN} -GXcode -DCMAKE_TOOLCHAIN_FILE=${HUNTER_ROOT}/polly/${TOOLCHAIN}.cmake -DCMAKE_VERBOSE_MAKEFILE=ON -DPOLLY_STATUS_DEBUG=ON -DHUNTER_STATUS_DEBUG=ON -DHUNTER_ROOT=${HUNTER_ROOT} -DTESTING_URL=${DTESTING_URL} -DTESTING_SHA1=b8b523e22b115f52b452b63fd9b6fe7c2e133aa6 -DHUNTER_RUN_INSTALL=ON
```
