#!/bin/sh
ARCH=gfx942
#ARCH=gfx90a

rm -rf build_casual_conv1d_test && mkdir build_casual_conv1d_test && cd build_casual_conv1d_test

/opt/rocm/bin/hipcc -x hip ../matrix_core.cc -std=c++17 -I/workspace/aiter/csrc/include --offload-arch=$ARCH  -O3 -Wall -save-temps -o matrix_core_casual_conv1d.exe
./matrix_core_casual_conv1d.exe