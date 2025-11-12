#!/bin/sh
ARCH=gfx942
#ARCH=gfx90a
LIBTORCH_PATH=/root/libtorch
rm -rf build_casual_conv1d_test && mkdir build_casual_conv1d_test && cd build_casual_conv1d_test

# /opt/rocm/bin/hipcc -x hip ../matrix_core.cc -std=c++17 -I/workspace/aiter/csrc/include --offload-arch=$ARCH  -O3 -Wall -save-temps -o matrix_core_casual_conv1d.exe
# /opt/rocm/bin/hipcc -std=c++17 ../matrix_core.cc -o matrix_core_casual_conv1d_debug.exe \
#     -I/workspace/aiter/csrc/include \
#     -I/root/libtorch/include \
#     -I/root/libtorch/include/torch/csrc/api/include \
#     -L/root/libtorch/lib \
#     -Wl,-rpath=/root/libtorch/lib \
#     -ltorch -lc10 -ltorch_cpu

/opt/rocm/bin/hipcc -x hip -std=c++17 ../matrix_core.cc -o matrix_core_casual_conv1d_ref.exe \
    --offload-arch=$ARCH \
    -I/workspace/aiter/csrc/include \
    -I${LIBTORCH_PATH}/include \
    -I${LIBTORCH_PATH}/include/torch/csrc/api/include \
    -L${LIBTORCH_PATH}/lib \
    -Wl,-rpath=/root/libtorch/lib \
    -ltorch -lc10 -ltorch_cpu
# ./matrix_core_casual_conv1d.exe
# ./build_casual_conv1d_test/matrix_core_casual_conv1d.exe