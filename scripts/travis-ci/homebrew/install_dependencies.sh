#!/bin/bash
set -e

brew update
[ -z $( brew tap | grep 'homebrew/science' ) ] && brew tap homebrew/science
[ -z "$( brew ls --versions tbb )" ] && brew install --c++11 tbb
[ -z "$( brew ls --versions boost )" ] && brew install --c++11 boost
[ -z "$( brew ls --versions cmake )" ] && brew install cmake
[ -z "$( brew ls --versions opencv3 )" ] && brew install -v --c++11 --with-tbb opencv3 && brew link --force opencv3 # verbose flag prevents the build from going stale

