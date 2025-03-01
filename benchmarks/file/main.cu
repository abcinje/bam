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

const char* const ctrls_paths[] = {"/dev/libnvm0", "/dev/libnvm1", "/dev/libnvm2", "/dev/libnvm3", "/dev/libnvm4", "/dev/libnvm5", "/dev/libnvm6", "/dev/libnvm7", "/dev/libnvm8", "/dev/libnvm9", "/dev/libnvm10", "/dev/libnvm11", "/dev/libnvm12", "/dev/libnvm13", "/dev/libnvm14", "/dev/libnvm15", "/dev/libnvm16", "/dev/libnvm17", "/dev/libnvm18", "/dev/libnvm19", "/dev/libnvm20", "/dev/libnvm21", "/dev/libnvm22", "/dev/libnvm23", "/dev/libnvm24","/dev/libnvm25", "/dev/libnvm26", "/dev/libnvm27", "/dev/libnvm28", "/dev/libnvm29", "/dev/libnvm30", "/dev/libnvm31"};

#define SIZE (8*4096)

template<size_t n>
__global__ __launch_bounds__(64,32)
void random_access_kernel(Controller** ctrls, page_cache_d_t* pc,  uint32_t req_size, uint32_t n_reqs, uint32_t num_ctrls, uint64_t* assignment, uint64_t reqs_per_thread, uint32_t access_type)
{
    //printf("in threads\n");
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t laneid = lane_id();
    uint32_t bid = blockIdx.x;
    uint32_t smid = get_smid();

    uint32_t ctrl;
    uint32_t queue;

    if (laneid == 0) {
        ctrl = smid % (pc->n_ctrls);
        //ctrl = pc->ctrl_counter->fetch_add(1, simt::memory_order_relaxed) % (pc->n_ctrls);
        queue = ctrls[ctrl]->queue_counter.fetch_add(1, simt::memory_order_relaxed) %  (ctrls[ctrl]->n_qps);
        //queue = smid % (ctrls[ctrl]->n_qps);
    }
    ctrl = __shfl_sync(0xFFFFFFFF, ctrl, 0);
    queue = __shfl_sync(0xFFFFFFFF, queue, 0);

    if (tid < n_reqs) {
        uint64_t start_block = (assignment[tid]*req_size) >> ctrls[ctrl]->d_qps[queue].block_size_log;
        //uint64_t start_block = (tid*req_size) >> ctrls[ctrl]->d_qps[queue].block_size_log;
        //start_block = tid;
        uint64_t n_blocks = req_size >> ctrls[ctrl]->d_qps[queue].block_size_log; /// ctrls[ctrl].ns.lba_data_size;;
        //printf("tid: %llu\tstart_block: %llu\tn_blocks: %llu\n", (unsigned long long) tid, (unsigned long long) start_block, (unsigned long long) n_blocks);

        uint16_t cids[n];
        uint16_t sq_poss[n];
        #pragma unroll
        for (size_t i = 0; i < n; i++)
            access_data_async(pc, (ctrls[ctrl]->d_qps)+(queue),start_block, n_blocks, tid, NVM_IO_READ, cids+i, sq_poss+i);
        #pragma unroll
        for (size_t i = 0; i < n; i++)
            poll_async((ctrls[ctrl]->d_qps)+(queue), cids[i], sq_poss[i]);
    }
}

template<size_t n>
__global__ __launch_bounds__(64,32)
void sequential_access_kernel(Controller** ctrls, page_cache_d_t* pc,  uint32_t req_size, uint32_t n_reqs, uint32_t num_ctrls, uint64_t* assignment, uint64_t reqs_per_thread, uint32_t access_type)
{
    //printf("in threads\n");
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t laneid = lane_id();
    uint32_t bid = blockIdx.x;
    uint32_t smid = get_smid();

    uint32_t ctrl;
    uint32_t queue;

    if (laneid == 0) {
        ctrl = smid % (pc->n_ctrls);
        //ctrl = pc->ctrl_counter->fetch_add(1, simt::memory_order_relaxed) % (pc->n_ctrls);
        queue = ctrls[ctrl]->queue_counter.fetch_add(1, simt::memory_order_relaxed) %  (ctrls[ctrl]->n_qps);
        //queue = smid % (ctrls[ctrl]->n_qps);
    }
    ctrl =  __shfl_sync(0xFFFFFFFF, ctrl, 0);
    queue =  __shfl_sync(0xFFFFFFFF, queue, 0);

    if (tid < n_reqs) {
        uint64_t start_block = (tid*req_size) >> ctrls[ctrl]->d_qps[queue].block_size_log;
        //uint64_t start_block = (tid*req_size) >> ctrls[ctrl]->d_qps[queue].block_size_log;
        //start_block = tid;
        uint64_t n_blocks = req_size >> ctrls[ctrl]->d_qps[queue].block_size_log; /// ctrls[ctrl].ns.lba_data_size;;
        //printf("tid: %llu\tstart_block: %llu\tn_blocks: %llu\n", (unsigned long long) tid, (unsigned long long) start_block, (unsigned long long) n_blocks);

        uint16_t cids[n];
        uint16_t sq_poss[n];
        #pragma unroll
        for (size_t i = 0; i < n; i++)
            access_data_async(pc, (ctrls[ctrl]->d_qps)+(queue),start_block, n_blocks, tid, NVM_IO_READ, cids+i, sq_poss+i);
        #pragma unroll
        for (size_t i = 0; i < n; i++)
            poll_async((ctrls[ctrl]->d_qps)+(queue), cids[i], sq_poss[i]);
    }
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
        if (n_pages < n_threads) {
            std::cerr << "Please provide enough pages. Number of pages must be greater than or equal to the number of threads!\n";
            exit(1);
        }

        // Create page cache
        page_cache_t h_pc(page_size, n_pages, settings.cudaDevice, ctrls[0][0], (uint64_t) 64, ctrls);
        page_cache_d_t* d_pc = (page_cache_d_t*) (h_pc.d_pc_ptr);
        std::cout << "Created page cache" << std::endl;

        // Mount
        unsigned long *root;
        cuda_err_chk(cudaMalloc(&root, sizeof(unsigned long)));
        cuda_err_chk(cudaMemset(root, 0, sizeof(unsigned long)));
        nfs_mount<<<1, 1>>>(ctrls[0]->d_qps, root);

        // Assignment for random access
        uint64_t* assignment;
        uint64_t* d_assignment;
        if (settings.random) {
            assignment = (uint64_t*)malloc(n_threads*sizeof(uint64_t));
            for (size_t i = 0; i < n_threads; i++)
                assignment[i] = rand() % n_blocks;
            cuda_err_chk(cudaMalloc(&d_assignment, n_threads*sizeof(uint64_t)));
            cuda_err_chk(cudaMemcpy(d_assignment, assignment,  n_threads*sizeof(uint64_t), cudaMemcpyHostToDevice));
        }

#if 0
        Event before;

        // Launch kernel
        if (settings.random) {
            switch (settings.numReqs) {
            case 1:
                random_access_kernel<1><<<g_size, b_size>>>(h_pc.pdt.d_ctrls, d_pc, page_size, n_threads, settings.n_ctrls, d_assignment, settings.numReqs, settings.accessType);
                break;
            case 2:
                random_access_kernel<2><<<g_size, b_size>>>(h_pc.pdt.d_ctrls, d_pc, page_size, n_threads, settings.n_ctrls, d_assignment, settings.numReqs, settings.accessType);
                break;
            case 3:
                random_access_kernel<3><<<g_size, b_size>>>(h_pc.pdt.d_ctrls, d_pc, page_size, n_threads, settings.n_ctrls, d_assignment, settings.numReqs, settings.accessType);
                break;
            case 4:
                random_access_kernel<4><<<g_size, b_size>>>(h_pc.pdt.d_ctrls, d_pc, page_size, n_threads, settings.n_ctrls, d_assignment, settings.numReqs, settings.accessType);
                break;
            default:
                std::cout << "Invalid num reqs\n";
                break;
            }
        } else {
            switch (settings.numReqs) {
            case 1:
                sequential_access_kernel<1><<<g_size, b_size>>>(h_pc.pdt.d_ctrls, d_pc, page_size, n_threads, settings.n_ctrls, d_assignment, settings.numReqs, settings.accessType);
                break;
            case 2:
                sequential_access_kernel<2><<<g_size, b_size>>>(h_pc.pdt.d_ctrls, d_pc, page_size, n_threads, settings.n_ctrls, d_assignment, settings.numReqs, settings.accessType);
                break;
            case 3:
                sequential_access_kernel<3><<<g_size, b_size>>>(h_pc.pdt.d_ctrls, d_pc, page_size, n_threads, settings.n_ctrls, d_assignment, settings.numReqs, settings.accessType);
                break;
            case 4:
                sequential_access_kernel<4><<<g_size, b_size>>>(h_pc.pdt.d_ctrls, d_pc, page_size, n_threads, settings.n_ctrls, d_assignment, settings.numReqs, settings.accessType);
                break;
            default:
                std::cout << "Invalid num reqs\n";
                break;
            }
        }

        Event after;

        cuda_err_chk(cudaDeviceSynchronize());

        // Performance
        double elapsed = after - before;
        uint64_t ios = g_size * b_size * settings.numReqs;
        uint64_t data = ios * page_size;
        double iops = ((double)ios) / (elapsed/1000000);
        double bandwidth = (((double)data) / (elapsed / 1000000)) / (1024ULL * 1024ULL * 1024ULL);
        std::cout << std::dec << "Elapsed Time: " << elapsed << "\tNumber of Ops: "<< ios << "\tData Size (bytes): " << data << std::endl;
        std::cout << std::dec << "Ops/sec: " << iops << "\tEffective Bandwidth(GB/S): " << bandwidth << std::endl;
        //std::cout << std::dec << ctrls[0]->ns.lba_data_size << std::endl;
#endif

        if (settings.random) {
            free(assignment);
            cuda_err_chk(cudaFree(d_assignment));
        }

        cuda_err_chk(cudaFree(root));

        for (size_t i = 0 ; i < settings.n_ctrls; i++)
            delete ctrls[i];

        std::cout << "Done." << std::endl;
    } catch (const error& e) {
        fprintf(stderr, "Unexpected error: %s\n", e.what());
        return 1;
    }

    return 0;
}
