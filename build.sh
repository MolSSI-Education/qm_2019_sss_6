#!/bin/bash

g++ -fPIC --shared -std=c++11 -Wall fock_fast.cpp export.cpp \
    -undefined dynamic_lookup \
    -I${CONDA_PREFIX}/include \
    -I${CONDA_PREFIX}/include/python3.7m \
    -I${CONDA_PREFIX}/include/eigen3 \
    -o fock_fast.so
