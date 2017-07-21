#!/bin/bash

if [ -z "$1" ]; then
    echo "Error, no files provided."
fi

all_cpp_files=$@

clang_format_bin=$(which clang-format-4.0)
if [ $? -ne 0 ]; then
    echo "clang-format-4.0 not installed. Please, install it and try again."
    exit 1
fi

cpplint_bin=$(which cpplint)
if [ $? -ne 0 ]; then
    echo "cpplint not installed. Please, install it and try again."
    exit 1
fi

for cpp_file in $all_cpp_files; do
    
    $clang_format_bin -style=file -output-replacements-xml $cpp_file | grep "<replacement " > /dev/null
    if [ $? -ne 1 ]; then
        echo "Error, clang-format not clean: $cpp_file"
        exit 1
    fi

    # enforce google style guide.  cpplint.py is chatty so pipe non-errors to /dev/null.
    python $cpplint_bin --filter=-runtime/references $cpp_file > /dev/null
    if [ $? -ne 0 ]; then
        echo "Error, cpplint.py failed google style: $cpp_file"
        exit 1
    fi

done
