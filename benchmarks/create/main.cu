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
#include <cerrno>
#include <ctrl.h>
#include "../file/settings.h"
#include <event.h>
#include <queue.h>
#include <nvm_parallel_queue.h>
#include <page_cache.h>
#include <util.h>
#include <iostream>
#include <unistd.h>

using error = std::runtime_error;
using std::string;

const char* const ctrls_paths[] = {"/dev/libnvm0", "/dev/libnvm1", "/dev/libnvm2", "/dev/libnvm3", "/dev/libnvm4", "/dev/libnvm5", "/dev/libnvm6", "/dev/libnvm7", "/dev/libnvm8", "/dev/libnvm9", "/dev/libnvm10", "/dev/libnvm11", "/dev/libnvm12", "/dev/libnvm13", "/dev/libnvm14", "/dev/libnvm15", "/dev/libnvm16", "/dev/libnvm17", "/dev/libnvm18", "/dev/libnvm19", "/dev/libnvm20", "/dev/libnvm21", "/dev/libnvm22", "/dev/libnvm23", "/dev/libnvm24","/dev/libnvm25", "/dev/libnvm26", "/dev/libnvm27", "/dev/libnvm28", "/dev/libnvm29", "/dev/libnvm30", "/dev/libnvm31"};

static void encode_prefix(char prefix[3])
{
    static const char alphabet[] = "0123456789abcdefghijklmnopqrstuvwxyz";
    uint32_t value = (uint32_t) getpid() % (36 * 36);

    prefix[0] = alphabet[(value / 36) % 36];
    prefix[1] = alphabet[value % 36];
    prefix[2] = '\0';
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
        cuda_err_chk(cudaSetDevice(settings.cudaDevice));

        if (settings.numThreads == 0) {
            std::cerr << "threads must be greater than 0\n";
            return 1;
        }
        if (settings.blkSize == 0) {
            std::cerr << "blk_size must be greater than 0\n";
            return 1;
        }
        if ((settings.numThreads % settings.blkSize) != 0) {
            std::cerr << "threads must be a multiple of blk_size so launched threads match created files exactly\n";
            return 1;
        }

        std::vector<Controller*> ctrls(settings.n_ctrls);
        for (size_t i = 0 ; i < settings.n_ctrls; i++)
            ctrls[i] = new Controller(ctrls_paths[i], settings.nvmNamespace, settings.cudaDevice, settings.queueDepth, settings.numQueues);

        const uint64_t b_size = settings.blkSize;
        const uint64_t g_size = settings.numThreads / b_size;
        const uint64_t n_threads = settings.numThreads;

        char pci_bus_id[15];
        cuda_err_chk(cudaDeviceGetPCIBusId(pci_bus_id, 15, settings.cudaDevice));
        std::cout << pci_bus_id << std::endl;

        char prefix[3];
        encode_prefix(prefix);

        char* d_prefix = nullptr;
        uint32_t* d_mount_result = nullptr;
        uint32_t* d_result = nullptr;
        uint32_t* d_fh = nullptr;
        uint32_t mount_result;
        std::vector<uint32_t> result(n_threads);

        cuda_err_chk(cudaMalloc(&d_prefix, 2));
        cuda_err_chk(cudaMemcpy(d_prefix, prefix, 2, cudaMemcpyHostToDevice));
        cuda_err_chk(cudaMalloc(&d_mount_result, sizeof(uint32_t)));
        cuda_err_chk(cudaMalloc(&d_result, n_threads * sizeof(uint32_t)));
        cuda_err_chk(cudaMalloc(&d_fh, n_threads * sizeof(uint32_t)));

        nfs_mount<<<1, 1>>>(ctrls[0]->d_qps, d_mount_result);
        cuda_err_chk(cudaDeviceSynchronize());
        cuda_err_chk(cudaMemcpy(&mount_result, d_mount_result, sizeof(uint32_t), cudaMemcpyDeviceToHost));
        if (mount_result != 0) {
            std::cerr << "Failed to mount: errno " << mount_result << std::endl;
            return 1;
        }

        Event before;
        nfs_batch_create<<<g_size, b_size>>>(ctrls[0]->d_qps, d_result, d_fh, d_prefix, 2, 0664);
        Event after;

        cuda_err_chk(cudaDeviceSynchronize());
        cuda_err_chk(cudaMemcpy(result.data(), d_result, n_threads * sizeof(uint32_t), cudaMemcpyDeviceToHost));

        uint64_t created = 0;
        uint64_t existed = 0;
        uint64_t failed = 0;
        for (uint64_t i = 0; i < n_threads; i++) {
            if (result[i] == 0) {
                created++;
            } else if (result[i] == EEXIST) {
                existed++;
            } else {
                failed++;
            }
        }

        double elapsed = after - before;
        double create_rate = ((double) n_threads) / (elapsed / 1000000.0);

        std::cout << "Prefix: " << prefix << std::endl;
        std::cout << "Elapsed Time(us): " << elapsed << "\tThreads: " << n_threads << "\tCreates/s: " << create_rate << std::endl;
        std::cout << "Created: " << created << "\tAlready Exists: " << existed << "\tFailed: " << failed << std::endl;

        if (failed != 0) {
            for (uint64_t i = 0; i < n_threads; i++) {
                if (result[i] != 0 && result[i] != EEXIST) {
                    std::cerr << "create failed at tid " << i << ": errno " << result[i] << std::endl;
                    break;
                }
            }
        }

        cuda_err_chk(cudaFree(d_prefix));
        cuda_err_chk(cudaFree(d_mount_result));
        cuda_err_chk(cudaFree(d_result));
        cuda_err_chk(cudaFree(d_fh));

        for (size_t i = 0 ; i < settings.n_ctrls; i++)
            delete ctrls[i];
    } catch (const error& e) {
        fprintf(stderr, "Unexpected error: %s\n", e.what());
        return 1;
    }

    return 0;
}
