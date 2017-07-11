#!/bin/bash

python third_party/cpplint.py \
      --extensions=cpp,h \
      --filter=-runtime/references \
      tiny_dnn/*/*/* examples/*/* test/*/*/*/*



