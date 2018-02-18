#!/bin/bash

python third_party/cpplint.py \
      --filter=-runtime/references \
      tiny_dnn/*/*/*.h examples/*/*.cpp examples/*/*/*.cpp test/*.h



