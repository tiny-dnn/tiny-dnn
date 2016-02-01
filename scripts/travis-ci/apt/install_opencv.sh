#!/bin/bash
set -e

# Check if valid OpenCV version was supplied
re='^[0-9]+$'
if [ $# -ne 3 ] || ! [[ $1 =~ $re ]] || ! [[ $2 =~ $re ]] || ! [[ $3 =~ $re ]]; then
  echo "Usage: $0 <version-major> <version-minor> <version-patch>" && exit 1
fi

# check to see if protobuf folder is empty
OPENCV_DIR="$HOME/opencv_${1}_${2}_${3}/lib"
if [ ! -d  "$OPENCV_DIR" ]; then
  git clone https://github.com/Itseez/opencv.git
  cd opencv
  git checkout ${1}.${2}.${3}
  mkdir build
  cd build
  cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$OPENCV_DIR" ..
  make && make install
else
  echo 'Cached OpenCV ${1}.${2}.${3} found.';
fi

ls "$OPENCV_DIR"

export LD_LIBRARY_PATH=$HOME/opencv_${1}_${2}_${3}/lib:$LD_LIBRARY_PATH
