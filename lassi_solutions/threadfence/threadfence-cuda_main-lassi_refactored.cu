
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>

// Optimized two-phase reduction kernel.
// Phase 1: Each block performs a grid‐stride reduction of its assigned portion of "array"
//         using an efficient binary tree reduction in shared memory.
//         The block’s partial sum is then written to "result[blockIdx.x]".
//         A single atomic counter "count" is incremented by thread 0.
// Phase 2: The block that is last to finish (determined via the atomic counter)
//         uses its threads to perform a second reduction over the "result" array,
//         thereby obtaining the final sum in result[0] and resetting count.
//
// This refactoring avoids per-element atomic additions and minimizes global memory traffic,
// reducing overall execution time and GPU energy consumption.
__global__ void sum(
    const float* __restrict__ array,
    const int N,
    unsigned int* __restrict__ count,
    volatile float* __restrict__ result)
{
    // Dynamically allocated shared memory for block reduction.
    extern __shared__ float sdata[];

    // Each thread computes its partial sum via a grid-stride loop.
    float mySum = 0.0f;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < N;
         idx += gridDim.x * blockDim.x)
    {
        mySum += array[idx];
    }
    sdata[threadIdx.x] = mySum;
    __syncthreads();

    // Intra-block reduction: use shared memory reduction until we reach a single warp.
    // First, process the reduction loop until the active threads are 32.
    for (unsigned int stride = blockDim.x / 2; stride >= 32; stride >>= 1) {
        if (threadIdx.x < stride) {
            sdata[threadIdx.x] += sdata[threadIdx.x + stride];
        }
        __syncthreads();
    }
    // Now, use warp-level reduction for the final 32 threads.
    if (threadIdx.x < 32) {
        // Use full warp mask.
        for (int offset = 16; offset > 0; offset /= 2) {
            sdata[threadIdx.x] += __shfl_down_sync(0xffffffff, sdata[threadIdx.x], offset);
        }
    }
    // sdata[0] now holds the block's partial sum.
    if (threadIdx.x == 0) {
        result[blockIdx.x] = sdata[0];
        // Ensure the partial sum has been written to global memory.
        __threadfence();

        // Atomically increment the count of finished blocks.
        unsigned int value = atomicAdd(count, 1);
        // Determine if this is the last block to finish.
        // We encode this flag into sdata[0] for use by all threads in this block.
        sdata[0] = (value == (gridDim.x - 1)) ? 1.0f : 0.0f;
    }
    __syncthreads();

    // If this block is the last one finishing, perform final reduction over all block partial sums.
    if (sdata[0] == 1.0f) {
        float total = 0.0f;
        // Each thread sums over a portion of the "result" array (which holds partial sums).
        for (int i = threadIdx.x; i < gridDim.x; i += blockDim.x) {
            total += result[i];
        }
        sdata[threadIdx.x] = total;
        __syncthreads();

        // If gridDim.x is less than blockDim.x, initialize unused shared mem to 0.
        if (threadIdx.x >= gridDim.x) {
            sdata[threadIdx.x] = 0.0f;
        }
        __syncthreads();

        // Perform reduction over the partial totals.
        for (unsigned int stride = blockDim.x / 2; stride >= 32; stride >>= 1) {
            if (threadIdx.x < stride) {
                sdata[threadIdx.x] += sdata[threadIdx.x + stride];
            }
            __syncthreads();
        }
        if (threadIdx.x < 32) {
            for (int offset = 16; offset > 0; offset /= 2) {
                sdata[threadIdx.x] += __shfl_down_sync(0xffffffff, sdata[threadIdx.x], offset);
            }
        }
        if (threadIdx.x == 0) {
            result[0] = sdata[0];
            // Reset the counter for subsequent kernel invocations.
            *count = 0;
        }
    }
}

int main(int argc, char** argv) {
    if (argc != 3) {
        printf("Usage: %s <repeat> <array length>\n", argv[0]);
        return 1;
    }

    const int repeat = atoi(argv[1]);
    const int N = atoi(argv[2]);

    const int threadsPerBlock = 256;
    const int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Allocate host memory.
    float* h_array = (float*) malloc(N * sizeof(float));
    float h_sum;

    // Allocate device memory.
    float* d_result;
    cudaMalloc((void**)&d_result, numBlocks * sizeof(float));

    float* d_array;
    cudaMalloc((void**)&d_array, N * sizeof(float));

    unsigned int* d_count;
    cudaMalloc((void**)&d_count, sizeof(unsigned int));
    cudaMemset(d_count, 0u, sizeof(unsigned int));

    bool ok = true;
    double time = 0.0;

    // Initialize the host array with -1.f.
    for (int i = 0; i < N; i++) {
        h_array[i] = -1.f;
    }

    // Copy the constant input array from host to device only once.
    cudaMemcpy(d_array, h_array, N * sizeof(float), cudaMemcpyHostToDevice);

    // Create CUDA events for timing.
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Run the kernel "repeat" times.
    for (int n = 0; n < repeat; n++) {
        // Record the start event.
        cudaEventRecord(start, 0);

        // Launch the kernel with dynamic shared memory set to threadsPerBlock*sizeof(float).
        sum<<<numBlocks, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(d_array, N, d_count, d_result);

        // Record the stop event.
        cudaEventRecord(stop, 0);
        // Wait for the stop event to complete.
        cudaEventSynchronize(stop);

        float elapsed_ms = 0.0f;
        cudaEventElapsedTime(&elapsed_ms, start, stop);
        time += elapsed_ms;

        // Copy the final sum from device to host.
        cudaMemcpy(&h_sum, d_result, sizeof(float), cudaMemcpyDeviceToHost);

        // Verify result.
        if (h_sum != -1.f * N) {
            ok = false;
            break;
        }
    }

    // Destroy CUDA events.
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    if (ok)
        printf("Average kernel execution time: %f (ms)\n", time / repeat);

    // Clean up.
    free(h_array);
    cudaFree(d_result);
    cudaFree(d_array);
    cudaFree(d_count);

    printf("%s\n", ok ? "PASS" : "FAIL");
    return 0;
}
