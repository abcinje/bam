#include <cuda.h>
#include <nvm_ctrl.h>
#include <nvm_types.h>
#include <nvm_queue.h>
#include <nvm_util.h>
#include <nvm_admin.h>
#include <nvm_error.h>
#include <nvm_cmd.h>
#include <string>
#include <stdexcept>
#include <vector>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <map>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <ctrl.h>
#include <buffer.h>
#include "settings.h"
#include <event.h>
#include <queue.h>
#include <nvm_parallel_queue.h>
#include <nvm_io.h>
#include <page_cache.h>
#include <util.h>
#include <iostream>
#include <fstream>
#include <byteswap.h>

using error = std::runtime_error;
using std::string;

#define FILENAME "foo"

const char* const ctrls_paths[] = {"/dev/libnvm0", "/dev/libnvm1", "/dev/libnvm2", "/dev/libnvm3", "/dev/libnvm4", "/dev/libnvm5", "/dev/libnvm6", "/dev/libnvm7", "/dev/libnvm8", "/dev/libnvm9", "/dev/libnvm10", "/dev/libnvm11", "/dev/libnvm12", "/dev/libnvm13", "/dev/libnvm14", "/dev/libnvm15", "/dev/libnvm16", "/dev/libnvm17", "/dev/libnvm18", "/dev/libnvm19", "/dev/libnvm20", "/dev/libnvm21", "/dev/libnvm22", "/dev/libnvm23", "/dev/libnvm24","/dev/libnvm25", "/dev/libnvm26", "/dev/libnvm27", "/dev/libnvm28", "/dev/libnvm29", "/dev/libnvm30", "/dev/libnvm31"};

#define SIZE (8*4096)

__global__ __launch_bounds__(64, 32)
void write_file(Controller **ctrls, page_cache_d_t *pc, uint32_t n_threads, uint32_t io_size)
{
    uint32_t result, result_count;

    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t laneid = lane_id();

    uint32_t ctrl = 0;
    uint32_t queue;

    if (laneid == 0)
        queue = ctrls[ctrl]->queue_counter.fetch_add(1, simt::memory_order_relaxed) % ctrls[ctrl]->n_qps;
    queue = __shfl_sync(0xFFFFFFFF, queue, 0);

    if (tid < n_threads) {
        uint32_t offset = tid * io_size;
        uint32_t count = io_size;

        nfs_write(ctrls[ctrl]->d_qps + queue, pc, tid, offset, count, &result, &result_count);
    }

    if (result != 0 || result_count != io_size)
        printf("write: %u %u\n", result, result_count);
}

__global__ __launch_bounds__(64, 32)
void read_file(Controller **ctrls, page_cache_d_t *pc, uint32_t n_threads, uint32_t io_size)
{
    uint32_t result, result_count;

    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t laneid = lane_id();

    uint32_t ctrl = 0;
    uint32_t queue;

    if (laneid == 0)
        queue = ctrls[ctrl]->queue_counter.fetch_add(1, simt::memory_order_relaxed) % ctrls[ctrl]->n_qps;
    queue = __shfl_sync(0xFFFFFFFF, queue, 0);

    if (tid < n_threads) {
        uint32_t offset = tid * io_size;
        uint32_t count = io_size;

        nfs_read(ctrls[ctrl]->d_qps + queue, pc, tid, offset, count, &result, &result_count);
    }

    if (result != 0 || result_count != io_size)
        printf("read: %u %u\n", result, result_count);
}

int main(int argc, char** argv)
{
    Settings settings;
    try {
        settings.parseArguments(argc, argv);
    } catch (const string& e) {
        fprintf(stderr, "%s\n", e.c_str());
        fprintf(stderr, "%s\n", Settings::usageString(argv[0]).c_str());
        return 1;
    }

    cudaDeviceProp properties;
    if (cudaGetDeviceProperties(&properties, settings.cudaDevice) != cudaSuccess) {
        fprintf(stderr, "Failed to get CUDA device properties\n");
        return 1;
    }

    try {
        // Init
        cuda_err_chk(cudaSetDevice(settings.cudaDevice));
        std::vector<Controller*> ctrls(settings.n_ctrls);
        for (size_t i = 0 ; i < settings.n_ctrls; i++)
            ctrls[i] = new Controller(ctrls_paths[i], settings.nvmNamespace, settings.cudaDevice, settings.queueDepth, settings.numQueues);

        // PCI Bus ID
        char st[15];
        cuda_err_chk(cudaDeviceGetPCIBusId(st, 15, settings.cudaDevice));
        std::cout << st << std::endl;

        // Parameters
        uint64_t b_size = settings.blkSize;//64;
        uint64_t g_size = (settings.numThreads + b_size - 1)/b_size;//80*16;
        uint64_t n_threads = b_size * g_size;
        uint64_t page_size = settings.pageSize;
        uint64_t n_pages = settings.numPages;
        uint64_t n_blocks = settings.numBlks;
        uint64_t access_type = settings.accessType;
        uint64_t num_reqs = settings.numReqs;
        if (n_pages < n_threads) {
            std::cerr << "Please provide enough pages. Number of pages must be greater than or equal to the number of threads!\n";
            exit(1);
        }
        if (access_type != READ && access_type != WRITE) {
            std::cerr << "Invalid access type\n";
            exit(1);
        }
        if (num_reqs != 1) {
            std::cerr << "Number of requests must be 1\n";
            exit(1);
        }

        // Create page cache
        page_cache_t h_pc(page_size, n_pages, settings.cudaDevice, ctrls[0][0], (uint64_t) 64, ctrls);
        page_cache_d_t* d_pc = (page_cache_d_t*) (h_pc.d_pc_ptr);
        std::cout << "Created page cache" << std::endl;

        // Status
        uint32_t result, *__result;
        cuda_err_chk(cudaMalloc(&__result, sizeof(uint32_t)));

        // Mount
        nfs_mount<<<1, 1>>>(ctrls[0]->d_qps, __result);
        cuda_err_chk(cudaDeviceSynchronize());
        cuda_err_chk(cudaMemcpy(&result, __result, sizeof(uint32_t), cudaMemcpyDeviceToHost));
        if (result) {
            std::cerr << "Failed to mount" << std::endl;
            exit(1);
        }

        // Filename
        char *__filename;
        uint32_t filename_len = strlen(FILENAME);
        if (filename_len > 16) {
            std::cerr << "Filename too long" << std::endl;
            exit(1);
        }
        cuda_err_chk(cudaMalloc(&__filename, filename_len));
        cuda_err_chk(cudaMemcpy(__filename, FILENAME, filename_len, cudaMemcpyHostToDevice));

        // Lookup
        nfs_lookup<<<1, 1>>>(ctrls[0]->d_qps, __result, __filename, filename_len);
        cuda_err_chk(cudaDeviceSynchronize());
        cuda_err_chk(cudaMemcpy(&result, __result, sizeof(uint32_t), cudaMemcpyDeviceToHost));
        if (result == 0) {
            std::cout << "File found: " << FILENAME << std::endl;
        } else if (result == ENOENT) {
            std::cout << "File not found: " << FILENAME << std::endl;
        } else {
            std::cerr << "Failed to lookup: errno " << result << std::endl;
            exit(1);
        }

        // Create if not found
        if (result == ENOENT) {
            nfs_create<<<1, 1>>>(ctrls[0]->d_qps, __result, __filename, filename_len, 0664);
            cuda_err_chk(cudaDeviceSynchronize());
            cuda_err_chk(cudaMemcpy(&result, __result, sizeof(uint32_t), cudaMemcpyDeviceToHost));
            if (result == 0) {
                std::cout << "File created: " << FILENAME << std::endl;
            } else if (result == EEXIST) {
                std::cout << "File already exists: " << FILENAME << std::endl;
            } else {
                std::cerr << "Failed to create: errno " << result << std::endl;
                exit(1);
            }
        }

        if (access_type == READ) {
            write_file<<<g_size, b_size>>>(h_pc.pdt.d_ctrls, d_pc, n_threads, page_size);
            cuda_err_chk(cudaDeviceSynchronize());
        }

        Event before;

        // Launch kernel
        if (access_type == READ)
            read_file<<<g_size, b_size>>>(h_pc.pdt.d_ctrls, d_pc, n_threads, page_size);
        else
            write_file<<<g_size, b_size>>>(h_pc.pdt.d_ctrls, d_pc, n_threads, page_size);

        Event after;

        cuda_err_chk(cudaDeviceSynchronize());

        // Performance
        double elapsed = after - before;
        uint64_t ios = g_size * b_size * num_reqs;
        uint64_t data = ios * page_size;
        double iops = ((double)ios) / (elapsed/1000000);
        double bandwidth = (((double)data) / (elapsed / 1000000)) / (1024ULL * 1024ULL * 1024ULL);
        std::cout << std::dec << "Elapsed Time: " << elapsed << "\tNumber of Ops: "<< ios << "\tData Size (bytes): " << data << std::endl;
        std::cout << std::dec << "Ops/sec: " << iops << "\tEffective Bandwidth(GB/S): " << bandwidth << std::endl;
        //std::cout << std::dec << ctrls[0]->ns.lba_data_size << std::endl;

        cuda_err_chk(cudaFree(__filename));
        cuda_err_chk(cudaFree(__result));

        for (size_t i = 0 ; i < settings.n_ctrls; i++)
            delete ctrls[i];

        std::cout << "Done." << std::endl;
    } catch (const error& e) {
        fprintf(stderr, "Unexpected error: %s\n", e.what());
        return 1;
    }

    return 0;
}
