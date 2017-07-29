#!/bin/bash

docker build -t tinydnn/tinydnn:dev-full-ubuntu17.04 -f Dockerfile.full .
docker tag tinydnn/tinydnn:dev-full-ubuntu17.04 tinydnn/tinydnn:latest
