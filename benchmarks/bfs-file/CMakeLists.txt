cmake_minimum_required (VERSION 3.1)
project (libnvm-benchmarks)

set (THREADS_PREFER_PTHREAD_FLAG ON)
find_package (Threads REQUIRED)

find_package (CUDA 8.0 REQUIRED)



make_benchmark (bfs-file-benchmark-module bfs-file-bench "main.cu")

make_benchmark_choice (bfs-file-benchmark bfs-file-benchmark-smartio bfs-file-benchmark-module)

target_compile_definitions(bfs-file-benchmark-module PRIVATE NFS_BACKEND)
