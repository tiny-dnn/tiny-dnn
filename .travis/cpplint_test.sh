#!/bin/bash

python third_party/cpplint.py \
      --extensions=cpp,h \
      --filter=-build/header_guard,-runtime/references \
      tiny_dnn/*/* examples/*/* test/*/*/*/*



