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
#include <chrono>
#include <thread>

// #define IO_ASYNC

#define VERIFY_INPUT "verify.in"
#define VERIFY_OUTPUT "verify.out"

using error = std::runtime_error;
using std::string;

#define FILENAME "foo"
#define TRACE_FILENAME "trace.txt"

const char* const ctrls_paths[] = {"/dev/libnvm0", "/dev/libnvm1", "/dev/libnvm2", "/dev/libnvm3", "/dev/libnvm4", "/dev/libnvm5", "/dev/libnvm6", "/dev/libnvm7", "/dev/libnvm8", "/dev/libnvm9", "/dev/libnvm10", "/dev/libnvm11", "/dev/libnvm12", "/dev/libnvm13", "/dev/libnvm14", "/dev/libnvm15", "/dev/libnvm16", "/dev/libnvm17", "/dev/libnvm18", "/dev/libnvm19", "/dev/libnvm20", "/dev/libnvm21", "/dev/libnvm22", "/dev/libnvm23", "/dev/libnvm24","/dev/libnvm25", "/dev/libnvm26", "/dev/libnvm27", "/dev/libnvm28", "/dev/libnvm29", "/dev/libnvm30", "/dev/libnvm31"};

#define SIZE (8*4096)

struct Sequence {
    std::string id;
    std::vector<size_t> block_ids;
};

void load_sequences(const std::string& trace_file, std::vector<Sequence> &sequences)
{
    std::ifstream file(trace_file);
    if (!file) {
        throw std::runtime_error("Cannot open trace file: " + trace_file);
    }

    std::string line;
    Sequence current_seq;
    
    while (std::getline(file, line)) {
        if (line.find("Seq") == 0) {  // 新序列开始
            if (!current_seq.id.empty()) {
                sequences.push_back(current_seq);
            }
            current_seq = Sequence();
            // 提取序列ID
            size_t pos = line.find("conversation id: ");
            if (pos != std::string::npos) {
                current_seq.id = line.substr(pos + 17);
                current_seq.id = current_seq.id.substr(0, current_seq.id.find(")"));
            }
        } else if (line.find("Round") == 0 && line.find("[") != std::string::npos) {  // 包含block IDs的行
            size_t start = line.find("[");
            size_t end = line.find("]");
            if (start != std::string::npos && end != std::string::npos) {
                std::string numbers = line.substr(start + 1, end - start - 1);
                std::stringstream ss(numbers);
                std::string number;
                while (std::getline(ss, number, ',')) {
                    // 去除前后空格
                    number.erase(0, number.find_first_not_of(" "));
                    number.erase(number.find_last_not_of(" ") + 1);
                    if (!number.empty()) {
                        current_seq.block_ids.push_back(std::stoul(number));
                    }
                }
            }
        }
    }
    
    if (!current_seq.id.empty()) {
        sequences.push_back(current_seq);
    }

    std::cout << "Loaded " << sequences.size() << " sequences" << std::endl;
}

__global__ __launch_bounds__(64, 32)
void process_all_sequences(Controller **ctrls, page_cache_d_t *pc, uint32_t n_threads, uint32_t fh, uint32_t io_size, uint32_t *__vec, size_t vec_size)
{
    uint32_t result, result_count;

    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t laneid = lane_id();

    uint32_t ctrl = 0;
    uint32_t queue;

    /*
    if (laneid == 0)
        queue = ctrls[ctrl]->queue_counter.fetch_add(1, simt::memory_order_relaxed) % ctrls[ctrl]->n_qps;
    queue = __shfl_sync(0xFFFFFFFF, queue, 0);
    */
    queue = tid % ctrls[0]->n_qps;

    if (tid < n_threads) {
        for (size_t vec_idx = tid; vec_idx < vec_size; vec_idx += n_threads) {
            uint32_t block_id = __vec[vec_idx];
            uint32_t offset = block_id * io_size;
            nfs_rw(ctrls[0]->d_qps + queue, pc, tid, fh, nvme_cmd_nfs_read, offset, io_size, &result, &result_count);
    
            if (result != 0 || result_count != io_size)
                printf("Error reading block %u: result %u, count %u\n", block_id, result, result_count);
        }
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

        if (n_pages < n_threads) {
            std::cerr << "Please provide enough pages. Number of pages must be greater than or equal to the number of threads!\n";
            exit(1);
        }
        
        // Create page cache
        page_cache_t h_pc(page_size, n_pages, settings.cudaDevice, ctrls[0][0], (uint64_t) 64, ctrls);
        page_cache_d_t* d_pc = (page_cache_d_t*) (h_pc.d_pc_ptr);
        std::cout << "Created page cache" << std::endl;

        // sequences
        std::vector<Sequence> sequences;
        load_sequences(TRACE_FILENAME, sequences);
        size_t vec_size = 0;
        for (const auto &seq : sequences)
            vec_size += seq.block_ids.size();
        uint64_t ios = vec_size;
        uint64_t data = ios * page_size;

        uint32_t *vec = (uint32_t *)malloc(vec_size * sizeof(uint32_t));
        size_t vec_idx = 0;
        for (const auto &seq : sequences)
            for (size_t block_id : seq.block_ids)
                vec[vec_idx++] = block_id;

        uint32_t *__vec;
        cuda_err_chk(cudaMalloc(&__vec, vec_size * sizeof(uint32_t)));
        cuda_err_chk(cudaMemcpy(__vec, vec, vec_size * sizeof(uint32_t), cudaMemcpyHostToDevice));

        // Status & file handle (assuming that only one file is accessed for now)
        uint32_t result, *__result, fh, *__fh;
        cuda_err_chk(cudaMalloc(&__result, sizeof(uint32_t)));
        cuda_err_chk(cudaMalloc(&__fh, sizeof(uint32_t)));

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
        nfs_lookup<<<1, 1>>>(ctrls[0]->d_qps, __result, __fh, __filename, filename_len);
        cuda_err_chk(cudaDeviceSynchronize());
        cuda_err_chk(cudaMemcpy(&result, __result, sizeof(uint32_t), cudaMemcpyDeviceToHost));
        cuda_err_chk(cudaMemcpy(&fh, __fh, sizeof(uint32_t), cudaMemcpyDeviceToHost));
        if (result == 0) {
            std::cout << "File found: " << FILENAME << " (handle " << fh << ')' << std::endl;
        } else if (result == ENOENT) {
            std::cout << "File not found: " << FILENAME << std::endl;
            std::cout << "Create the file if you want to run the benchmark." << std::endl;
            exit(1);
        } else {
            std::cerr << "Failed to lookup: errno " << result << std::endl;
            exit(1);
        }

        Event before;
        process_all_sequences<<<g_size, b_size>>>(h_pc.pdt.d_ctrls, d_pc, n_threads, fh, page_size, __vec, vec_size);
        Event after;

        cuda_err_chk(cudaDeviceSynchronize());

        // Performance
        double elapsed = after - before;
        double iops = ((double)ios) / (elapsed/1000000);
        double bandwidth = (((double)data) / (elapsed / 1000000)) / (1024ULL * 1024ULL * 1024ULL);
        std::cout << std::dec << "Elapsed Time(us): " << elapsed << "\tNumber of Ops: "<< ios << "\tData Size (bytes): " << data << std::endl;
        std::cout << std::dec << "IOPS: " << iops << "\tEffective Bandwidth(GB/s): " << bandwidth << std::endl;
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
