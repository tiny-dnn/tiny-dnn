#!/bin/bash

tag_prefix="tinydnn/tinydnn"

docker_images=('dev-ubuntu17.04' 'dev-full-ubuntu17.04')

if [ $# -eq 0 ] ; then
    docker_tag=$tag_prefix/${docker_images[0]}
    docker_file="Dockerfile"
    echo "** No arguments supplied, building default image. **"
else
    match=0
    for img in "${docker_images[@]}"; do
        echo $img
        if [[ $img = "$1" ]]; then
            match=1
            break
        fi
    done
    if [[ $match = 0 ]]; then
        echo "** Not supported image $1, aborting building docker image. **"
        exit 1
    else
        docker_tag=$tag_prefix/$1
        docker_file="Dockerfile.full"
    fi
fi

echo "** running command: docker build -t ${docker_tag} -f ${docker_file}"

docker build -t ${docker_tag} -f ${docker_file} .
