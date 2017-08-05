#!/bin/bash

script_link="${BASH_SOURCE[0]}"
while [ -h "$script_link" ]; do # resolve $script_link until the file is no longer a symlink
  lib_dir="$( cd -P "$( dirname "$script_link" )" && pwd )"
  script_link="$(readlink "$script_link")"
  [[ $script_link != /* ]] && script_link="$lib_dir/$script_link" # if $script_link was a relative symlink, we need to resolve it relative to the path where the symlink file was located
done
lib_dir="$( cd -P "$( dirname "$script_link" )" && pwd )"

docker run --rm -it -v $lib_dir:/opt/tiny-dnn tinydnn/tinydnn
