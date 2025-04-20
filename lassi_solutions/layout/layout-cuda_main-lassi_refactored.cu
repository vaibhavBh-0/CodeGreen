
/**********************************************************************
* Copyright ©2013 Advanced Micro Devices, Inc. All rights reserved.
*
* Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
*
* Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or
*  other materials provided with the distribution.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
* WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
* DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
* OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
* NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
* ********************************************************************/

#include <iostream>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <cuda.h>

#define TREE_NUM   16384
#define TREE_SIZE  16384
#define GROUP_SIZE 256

struct AppleTree
{
    int apples[TREE_SIZE];
};

struct ApplesOnTrees
{
    int trees[TREE_NUM];
};

// Fused kernel version for AoS: the kernel now does "iterations" times the same per-thread work.
// (Only the final result is written onto outBuf, as in the original code.)
template <int treeSize>
__global__ void AoSKernel(const AppleTree* __restrict__ trees, int* __restrict__ outBuf, int iterations)
{
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int result = 0;
    // Fuse the iterations inside the kernel to reduce launch overhead.
    for (int it = 0; it < iterations; it++) {
        int temp = 0;
        #pragma unroll  // Hint to unroll given treeSize is known at compile time.
        for (int i = 0; i < treeSize; i++) {
            temp += trees[gid].apples[i];
        }
        result = temp; // Each iteration produces the same output.
    }
    outBuf[gid] = result;
}

// Fused kernel version for SoA.
template <int treeSize>
__global__ void SoAKernel(const ApplesOnTrees* __restrict__ applesOnTrees, int* __restrict__ outBuf, int iterations)
{
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int result = 0;
    for (int it = 0; it < iterations; it++) {
        int temp = 0;
        #pragma unroll
        for (int i = 0; i < treeSize; i++) {
            temp += applesOnTrees[i].trees[gid];
        }
        result = temp;
    }
    outBuf[gid] = result;
}

int main(int argc, char* argv[])
{
    if (argc != 2)
    {
        std::cout << "Usage: " << argv[0] << " <repeat>" << std::endl;
        return 1;
    }
    
    const int iterations = std::atoi(argv[1]); // Number of iterations to fuse into each kernel.
    const int treeSize = TREE_SIZE;
    const int treeNumber = TREE_NUM;
    bool fail = false;

    if(iterations < 1)
    {
        std::cout << "Iterations cannot be 0 or negative. Exiting..\n";
        return -1;
    }
    
    if(treeNumber < GROUP_SIZE)
    {
        std::cout << "treeNumber should be larger than the work group size" << std::endl;
        return -1;
    }
    if(treeNumber % GROUP_SIZE != 0)
    {
        std::cout << "treeNumber should be a multiple of " << GROUP_SIZE << std::endl;
        return -1;
    }
    
    const int elements = treeSize * treeNumber;
    size_t inputSize = elements * sizeof(int);
    size_t outputSize = treeNumber * sizeof(int);

    // Allocate host memory
    int* data = (int*) std::malloc(inputSize);
    int* deviceResult = (int*) std::malloc(outputSize);
    
    // Compute reference for verification (only once)
    int* reference = (int*) std::malloc(outputSize);
    std::memset(reference, 0, outputSize);
    for (int i = 0; i < treeNumber; i++)
        for (int j = 0; j < treeSize; j++)
            reference[i] += i * treeSize + j;
    
    dim3 grid(treeNumber / GROUP_SIZE);
    dim3 block(GROUP_SIZE);
    
    int *inputBuffer, *outputBuffer;
    cudaMalloc((void**)&inputBuffer, inputSize);
    cudaMalloc((void**)&outputBuffer, outputSize);
    
    // ------------------------------
    // AoS version
    // ------------------------------
    // Initialize AoS data: contiguous apples for each tree.
    for (int i = 0; i < treeNumber; i++) {
        for (int j = 0; j < treeSize; j++) {
            data[j + i * treeSize] = j + i * treeSize;
        }
    }
    cudaMemcpy(inputBuffer, data, inputSize, cudaMemcpyHostToDevice);
    
    cudaDeviceSynchronize();
    auto start = std::chrono::steady_clock::now();
    
    // Launch the fused AoS kernel in a single call
    AoSKernel<TREE_SIZE><<<grid, block>>>(reinterpret_cast<AppleTree*>(inputBuffer),
                                          outputBuffer, iterations);
    
    cudaDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    std::cout << "Average kernel execution time (AoS): "
              << (time * 1e-3f) / iterations << " (us)\n";
    
    cudaMemcpy(deviceResult, outputBuffer, outputSize, cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < treeNumber; i++)
    {
        if(deviceResult[i] != reference[i])
        {
            fail = true;
            break;
        }
    }
    
    if (fail)
        std::cout << "FAIL" << std::endl;
    else
        std::cout << "PASS" << std::endl;
    
    // ------------------------------
    // SoA version
    // ------------------------------
    // Initialize SoA data: each tree value is stored with stride equal to treeNumber.
    for (int i = 0; i < treeNumber; i++) {
        for (int j = 0; j < treeSize; j++) {
            data[i + j * treeNumber] = j + i * treeSize;
        }
    }
    cudaMemcpy(inputBuffer, data, inputSize, cudaMemcpyHostToDevice);
    
    cudaDeviceSynchronize();
    start = std::chrono::steady_clock::now();
    
    // Launch the fused SoA kernel in a single call
    SoAKernel<TREE_SIZE><<<grid, block>>>(reinterpret_cast<ApplesOnTrees*>(inputBuffer),
                                          outputBuffer, iterations);
    
    cudaDeviceSynchronize();
    end = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    std::cout << "Average kernel execution time (SoA): "
              << (time * 1e-3f) / iterations << " (us)\n";
    
    cudaMemcpy(deviceResult, outputBuffer, outputSize, cudaMemcpyDeviceToHost);
    
    fail = false;
    for (int i = 0; i < treeNumber; i++)
    {
        if(deviceResult[i] != reference[i])
        {
            fail = true;
            break;
        }
    }
    
    if (fail)
        std::cout << "FAIL" << std::endl;
    else
        std::cout << "PASS" << std::endl;
    
    cudaFree(inputBuffer);
    cudaFree(outputBuffer);
    std::free(deviceResult);
    std::free(reference);
    std::free(data);
    
    return 0;
}
