# HIP_VISIBLE_DEVICES=7 rocprofv3 --hip-runtime-trace --kernel-trace --output-format csv pftrace -d trace -o matrix_out --stats -- ./build/matrix_core.exe
# HIP_VISIBLE_DEVICES=7 rocprofv3 --hip-runtime-trace --kernel-trace --output-format csv pftrace -d trace_casual_conv1d -o matrix_casual_conv1d_out_m_1024_n_32_k_128 --stats -- ./build_casual_conv1d_test/matrix_core_casual_conv1d.exe
HIP_VISIBLE_DEVICES=7 rocprofv3 --hip-runtime-trace --kernel-trace --output-format csv pftrace -d trace_casual_conv1d -o casual_conv1d_nch_1_64_2048_k_4 --stats -- ./conv1d_libtorch_ref.exe
# HIP_VISIBLE_DEVICES=7 rocprofv3 -i input_att.yaml -- ./build/matrix_core.exe
# HIP_VISIBLE_DEVICES=7 rocprofv3 -i input_att.yaml -- ./build_casual_conv1d_test/matrix_core_casual_conv1d.exe