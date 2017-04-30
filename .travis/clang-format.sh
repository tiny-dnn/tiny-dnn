#!/bin/bash

# set color test
COLOR_OFF="\033[0m"
RED="\033[0;31m"
GREEN="\033[0;32m"

# based on https://github.com/lemire/SIMDCompressionAndIntersection/blob/master/tools/clang-format.sh
STYLE=$(which clang-format-4.0)
if [ $? -ne 0 ]; then
    echo "$RED clang-format not installed. Unable to check source file format policy.$COLOR_OFF" >&2
    exit 1
fi

RE=0
BASE=$(git rev-parse --show-toplevel)

ALLFILES=$(git ls-tree --full-tree --name-only -r HEAD tiny_dnn test examples| grep -e ".*\.\(c\|h\|cc\|cpp\|hh\)\$")
for FILE in $ALLFILES; do
    $STYLE $BASE/$FILE | cmp -s $BASE/$FILE -
    if [ $? -ne 0 ]; then
        echo -e "$RED$FILE does not respect the coding style.$COLOR_OFF" >&2
        RE=1
    fi
done

if [ $RE -eq 0 ]; then
    echo -e "$GREEN All files match with clang-format.$COLOR_OFF" >&2
else
    echo -e "$RED Failed matching files with clang-format.$COLOR_OFF" >&2
fi

exit $RE
