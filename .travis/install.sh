#!/bin/bash
set -e

if [ "$TRAVIS_OS_NAME" == "osx" ]; then
  brew update
  [ -z $( brew tap | grep 'homebrew/science' ) ] && brew tap homebrew/science
  if [ -f "${HOME}/homebrew-cache/homebrew-cache.tar.gz"]; then
    tar -xvzf "${HOME}/homebrew-cache/homebrew-cache.tar.gz" --directory /usr/local/Cellar
    brew link --force tbb boost cmake opencv3
  else
    [ -z "$( brew ls --versions tbb )" ] && brew install --c++11 tbb
    [ -z "$( brew ls --versions boost )" ] && brew install --c++11 boost
    [ -z "$( brew ls --versions cmake )" ] && brew install cmake
    [ -z "$( brew ls --versions opencv3 )" ] && brew install -v --c++11 --with-tbb opencv3 && brew link --force opencv3 # verbose flag prevents the build from going stale
    mkdir "${HOME}/homebrew-cache"
    tar -czvf "${HOME}/homebrew-cache/homebrew-cache.tar.gz" --directory /usr/local/Cellar tbb boost cmake opencv3
  fi
elif [ "$TRAVIS_OS_NAME" == "linux" ]; then
  version_major=3
  version_minor=1
  version_patch=0  
  OPENCV_DIR="${HOME}/opencv_${version_major}_${version_minor}_${version_patch}"
  if [ ! -d  "${OPENCV_DIR}/lib" ]; then
    git clone https://github.com/Itseez/opencv.git
    cd opencv
    git checkout ${version_major}.${version_minor}.${version_patch}
    mkdir build
    cd build
    cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$OPENCV_DIR" ..
    make -j2 && make -j2 install
  else
    echo "Cached OpenCV ${1}.${2}.${3} found";
  fi
  
  export LD_LIBRARY_PATH=${OPENCV_DIR}/lib:$LD_LIBRARY_PATH

  if [ "$CXX" == "g++" ]; then 
    export CXX="g++-4.8" CC="gcc-4.8"
  fi
fi
