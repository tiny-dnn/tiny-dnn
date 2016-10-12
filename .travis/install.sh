#!/bin/bash
set -e

if [ "$TRAVIS_OS_NAME" == "osx" ]; then
  brew update >/dev/null
  [ -z $( brew tap | grep 'homebrew/science' ) ] && brew tap homebrew/science
  # TODO download homebrew-cache.tar.gz from s3 or similar to speed up the build
  # Travis OSX boxes don't provide caching unfortunately
  if [ -f "${HOME}/homebrew-cache/homebrew-cache.tar.gz" ]; then
    tar -xvzf "${HOME}/homebrew-cache/homebrew-cache.tar.gz" --directory /usr/local/Cellar
    brew link --force tbb cmake
    brew install pkg-config
  else
    [ -z "$( brew ls --versions tbb )" ] && brew install --c++11 tbb
    [ -z "$( brew ls --versions cmake )" ] && brew install cmake
    mkdir "${HOME}/homebrew-cache"
    tar -czvf "${HOME}/homebrew-cache/homebrew-cache.tar.gz" --directory /usr/local/Cellar tbb cmake
  fi
fi
