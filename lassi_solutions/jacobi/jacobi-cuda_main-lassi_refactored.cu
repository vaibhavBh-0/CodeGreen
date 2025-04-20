
#include <cstdio>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <limits>
#include <ctime>
#include <chrono>
#include <cuda.h>
#include <cstdlib>
#include <algorithm>
#include <cooperative_groups.h>
  
namespace cg = cooperative_groups;

#define N 16384
#define IDX(i, j) ((i) + (j) * N)

//------------------------------------------------
// Energy‐efficient initialization (unchanged functionality)
//------------------------------------------------
void initialize_data (float* f) {
  // Set up simple sinusoidal boundary conditions
  for (int j = 0; j < N; ++j) {
    for (int i = 0; i < N; ++i) {
      if (i == 0 || i == N-1) {
        f[IDX(i,j)] = sinf(j * 2 * M_PI / (N-1));
      }
      else if (j == 0 || j == N-1) {
        f[IDX(i,j)] = sinf(i * 2 * M_PI / (N-1));
      }
      else {
        f[IDX(i,j)] = 0.0f;
      }
    }
  }
}

//------------------------------------------------
// Optimized Jacobi kernel with energy-aware considerations:
//  - Uses shared memory to reduce global memory accesses.
//  - Performs block-level reduction using a warp-level approach for fewer synchronizations.
//  - The kernel code remains functionally equivalent.
//------------------------------------------------
__global__ void jacobi_step (float* __restrict__ f, 
                             const float* __restrict__ f_old, 
                             float* __restrict__ error) {
  // Shared memory tile with halo; dimensions chosen for a block of 16x16 threads.
  __shared__ float f_old_tile[18][18];
  
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  
  // Load interior data into shared memory (offset by 1 for halo)
  f_old_tile[threadIdx.y + 1][threadIdx.x + 1] = f_old[IDX(i, j)];
  
  // Load halo elements; each condition checked to avoid out‐of‐bound accesses.
  if (threadIdx.x == 0 && i >= 1) {
    f_old_tile[threadIdx.y + 1][0] = f_old[IDX(i - 1, j)];
  }
  if (threadIdx.x == blockDim.x - 1 && i <= N - 2) {
    f_old_tile[threadIdx.y + 1][blockDim.x + 1] = f_old[IDX(i + 1, j)];
  }
  if (threadIdx.y == 0 && j >= 1) {
    f_old_tile[0][threadIdx.x + 1] = f_old[IDX(i, j - 1)];
  }
  if (threadIdx.y == blockDim.y - 1 && j <= N - 2) {
    f_old_tile[blockDim.y + 1][threadIdx.x + 1] = f_old[IDX(i, j + 1)];
  }
  
  __syncthreads();
  
  float err = 0.0f;
  if (j >= 1 && j <= N - 2 && i >= 1 && i <= N - 2) {
    // Compute new value using the 4-point stencil from shared memory
    float new_val = 0.25f * ( f_old_tile[threadIdx.y + 1][threadIdx.x + 2] +
                              f_old_tile[threadIdx.y + 1][threadIdx.x] +
                              f_old_tile[threadIdx.y + 2][threadIdx.x + 1] +
                              f_old_tile[threadIdx.y][threadIdx.x + 1] );
    f[IDX(i,j)] = new_val;
    float diff = new_val - f_old_tile[threadIdx.y + 1][threadIdx.x + 1];
    err = diff * diff;
  }
  
  // Block-level reduction using shared memory and warp-level primitives.
  __shared__ float s_err[256]; // 16x16 block size assumed.
  int localId = threadIdx.y * blockDim.x + threadIdx.x;
  s_err[localId] = err;
  __syncthreads();
  
  int blockSize = blockDim.x * blockDim.y;
  // Perform reduction in shared memory until 32 threads remain
  for (int s = blockSize / 2; s > 32; s /= 2) {
    if (localId < s) {
      s_err[localId] += s_err[localId + s];
    }
    __syncthreads();
  }
  
  // Use warp-level reduction for the final 32 elements.
  if (localId < 32) {
    float sum = s_err[localId];
    // Use full warp mask to perform the reduction
    for (int offset = 16; offset > 0; offset /= 2) {
      sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    if (localId == 0) {
      atomicAdd(error, sum);
    }
  }
}

//------------------------------------------------
// Main function with energy-aware modifications:
//  - Uses pinned host memory for reduced data transfer overhead.
//  - Uses asynchronous operations to overlap computation and data transfers.
//  - Retains all original functionalities.
//------------------------------------------------
int main () {
  // Start total timer.
  std::clock_t start_time = std::clock();

  float* d_f = nullptr;
  float* d_f_old = nullptr;
  float* d_error = nullptr;

  // Allocate page-locked (pinned) host memory.
  float* f = nullptr;
  float* f_old = nullptr;
  cudaHostAlloc((void**)&f, N * N * sizeof(float), cudaHostAllocDefault);
  cudaHostAlloc((void**)&f_old, N * N * sizeof(float), cudaHostAllocDefault);

  // Initialize both arrays to the same boundary conditions.
  initialize_data(f);
  initialize_data(f_old);

  // Allocate device memory.
  cudaMalloc((void**)&d_f, N * N * sizeof(float));
  cudaMalloc((void**)&d_f_old, N * N * sizeof(float));
  cudaMalloc((void**)&d_error, sizeof(float));

  // Copy initial data to device.
  cudaMemcpy(d_f, f, N * N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_f_old, f_old, N * N * sizeof(float), cudaMemcpyHostToDevice);

  float error = std::numeric_limits<float>::max();
  const float tolerance = 1.e-5f;
  const int max_iters = 10000;
  int num_iters = 0;

  // Define grid and block dimensions.
  dim3 block(16, 16);
  dim3 grid(N / block.x, N / block.y);

  // Optional: set cache preference (can help energy efficiency depending on the hardware).
  cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
  
  // Create a dedicated stream for asynchronous error copy.
  cudaStream_t errorStream;
  cudaStreamCreate(&errorStream);
  
  cudaDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();

  while (error > tolerance && num_iters < max_iters) {
    // Reset the error on the device asynchronously.
    cudaMemsetAsync(d_error, 0, sizeof(float));

    // Launch Jacobi relaxation step.
    jacobi_step<<<grid, block>>>(d_f, d_f_old, d_error);

    // Instead of launching a separate kernel to swap data,
    // swap the device pointers on the host.
    std::swap(d_f, d_f_old);

    // Copy the accumulated error from device to host asynchronously using errorStream.
    cudaMemcpyAsync(&error, d_error, sizeof(float), cudaMemcpyDeviceToHost, errorStream);
    cudaStreamSynchronize(errorStream);

    // Normalize the L2 error by the number of data points and take square root.
    error = sqrtf(error / (N * N));

    // Periodically print error for monitoring.
    if (num_iters % 1000 == 0) {
      std::cout << "Error after iteration " << num_iters << " = " << error << std::endl;
    }
    ++num_iters;
  }

  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::cout << "Average execution time per iteration: " << (elapsed_ns * 1e-9f) / num_iters << " (s)\n";

  if (error <= tolerance && num_iters < max_iters) {
    std::cout << "PASS" << std::endl;
  }
  else {
    std::cout << "FAIL" << std::endl;
    return -1;
  }

  // Clean up device memory.
  cudaFree(d_f);
  cudaFree(d_f_old);
  cudaFree(d_error);

  // Destroy the error stream.
  cudaStreamDestroy(errorStream);

  // Clean up host memory.
  cudaFreeHost(f);
  cudaFreeHost(f_old);

  double duration = (std::clock() - start_time) / (double) CLOCKS_PER_SEC;
  std::cout << "Total elapsed time: " << std::setprecision(4) << duration << " seconds" << std::endl;
  return 0;
}
