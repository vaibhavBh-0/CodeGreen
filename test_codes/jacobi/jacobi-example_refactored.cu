
/*
  Refactored for energy efficiency by reducing GPU kernel launches and CPU/GPU overhead.
  Copyright (c) 2021, NVIDIA CORPORATION.
  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
*/

#include <cstdio>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <limits>
#include <ctime>
#include <chrono>
#include <cuda.h>

#define N 16384
#define IDX(i, j) ((i) + (j) * N)
#define FUSED_ITERS 10  // Must be even so that pointer roles swap back

// Initialize the 2D field with simple sinusoidal boundary conditions.
void initialize_data (float* f) {
  for (int j = 0; j < N; ++j) {
    for (int i = 0; i < N; ++i) {
      if (i == 0 || i == N-1) {
        f[IDX(i,j)] = sinf(j * 2 * M_PI / (N - 1));
      }
      else if (j == 0 || j == N-1) {
        f[IDX(i,j)] = sinf(i * 2 * M_PI / (N - 1));
      }
      else {
        f[IDX(i,j)] = 0.0f;
      }
    }
  }
}

/*
  Fused Jacobi kernel:
  - Performs FUSED_ITERS iterations in a single launch.
  - Each iteration reloads the required tile from the "current" grid and computes
    the new value into the "next" grid.
  - Only in the final iteration is the local error computed and accumulated.
  - After each iteration (except the last), current and next pointers are swapped.
*/
__global__ void jacobi_step (float* __restrict__ f, 
                             const float* __restrict__ f_old, 
                             float* __restrict__ error) {
  // Declare shared tile for a 16x16 block with halo (18x18)
  __shared__ float f_old_tile[18][18];

  // Global indices for each thread.
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  
  // Set up two pointers to perform fused iterations.
  // We start with current pointing to f_old and next to f.
  float* current = (float*) f_old;   // note: casting away const as we do only read from it.
  float* next    = f;

  float local_error = 0.0f;  // will only be accumulated in the final fused iteration

  // Fuse multiple Jacobi iterations to reduce kernel launches and synchronization overhead.
  for (int iter = 0; iter < FUSED_ITERS; iter++) {

    // Load the current tile (including halo) from global memory.
    // Interior load.
    f_old_tile[threadIdx.y+1][threadIdx.x+1] = current[IDX(i, j)];

    // Left halo.
    if (threadIdx.x == 0 && i >= 1) {
      f_old_tile[threadIdx.y+1][threadIdx.x+0] = current[IDX(i-1, j)];
    }
    // Right halo.
    if (threadIdx.x == blockDim.x - 1 && i <= N-2) {
      f_old_tile[threadIdx.y+1][threadIdx.x+2] = current[IDX(i+1, j)];
    }
    // Top halo.
    if (threadIdx.y == 0 && j >= 1) {
      f_old_tile[threadIdx.y+0][threadIdx.x+1] = current[IDX(i, j-1)];
    }
    // Bottom halo.
    if (threadIdx.y == blockDim.y - 1 && j <= N-2) {
      f_old_tile[threadIdx.y+2][threadIdx.x+1] = current[IDX(i, j+1)];
    }

    __syncthreads(); // Ensure the entire tile is loaded

    // Compute the Jacobi update and, for the last iteration, compute the error.
    if (j >= 1 && j <= N-2 && i >= 1 && i <= N-2) {
      float new_val = 0.25f * ( f_old_tile[threadIdx.y+1][threadIdx.x+2] +
                                f_old_tile[threadIdx.y+1][threadIdx.x+0] +
                                f_old_tile[threadIdx.y+2][threadIdx.x+1] +
                                f_old_tile[threadIdx.y+0][threadIdx.x+1] );
      next[IDX(i,j)] = new_val;
      // Only compute error in the final fused iteration.
      if (iter == FUSED_ITERS - 1) {
        float df = new_val - f_old_tile[threadIdx.y+1][threadIdx.x+1];
        local_error = df * df;
      }
    }
    __syncthreads(); // Ensure all threads complete the update before swapping.

    // Swap the pointers for the next iteration if not on the last fused iteration.
    if (iter < FUSED_ITERS - 1) {
      float* tmp = current;
      current = next;
      next = tmp;
    }
    __syncthreads();
  }
  
  // Perform warp-level reduction to sum the error computed in the final fused iteration.
  float err = local_error;
  for (int offset = 16 >> 1; offset > 0; offset /= 2) {
    err += __shfl_down_sync(0xffffffff, err, offset);
  }

  // Compute a unique lane id in the block.
  int thread_id = threadIdx.y * blockDim.x + threadIdx.x;
  int lane = thread_id % 32;

  // Each warp leader performs an atomicAdd to accumulate the error.
  if (lane == 0) {
    atomicAdd(error, err);
  }
}

int main () {
  std::clock_t start_time = std::clock();

  float *d_f, *d_f_old, *d_error;
  
  // Use pinned (page-locked) memory for host allocations for better transfer efficiency.
  float* f    = nullptr;
  float* f_old = nullptr;
  posix_memalign((void**)&f, 64, N * N * sizeof(float));
  posix_memalign((void**)&f_old, 64, N * N * sizeof(float));

  // Initialize both arrays with the same boundary and initial interior values.
  initialize_data(f);
  initialize_data(f_old);

  cudaMalloc((void**)&d_f,     N * N * sizeof(float));
  cudaMalloc((void**)&d_f_old, N * N * sizeof(float));
  cudaMalloc((void**)&d_error, sizeof(float));

  cudaMemcpy(d_f, f, N * N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_f_old, f_old, N * N * sizeof(float), cudaMemcpyHostToDevice);

  // Allocate pinned host memory for the error value for asynchronous copy.
  float* h_error = nullptr;
  cudaHostAlloc((void**)&h_error, sizeof(float), cudaHostAllocDefault);

  // Set initial error value and tolerance.
  float error = std::numeric_limits<float>::max();
  const float tolerance = 1.e-5f;

  const int max_iters = 10000;
  int num_iters = 0;

  // Use a 16x16 thread block.
  dim3 grid (N / 16, N / 16);
  dim3 block (16, 16);

  // Create a dedicated CUDA stream for the asynchronous error copy.
  cudaStream_t error_stream;
  cudaStreamCreate(&error_stream);

  cudaDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();

  // Main iteration loop.
  while (error > tolerance && num_iters < max_iters) {
    // Reset error on device.
    cudaMemset(d_error, 0, sizeof(float));

    // Launch the fused Jacobi kernel on error_stream.
    jacobi_step<<<grid, block, 0, error_stream>>>(d_f, d_f_old, d_error);

    // Instead of launching a separate kernel to copy the computed interior back,
    // simply swap the pointers.
    float* temp = d_f;
    d_f = d_f_old;
    d_f_old = temp;

    // Asynchronously copy the error from device to host using the dedicated stream.
    cudaMemcpyAsync(h_error, d_error, sizeof(float), cudaMemcpyDeviceToHost, error_stream);

    // Synchronize only the error_stream to ensure the copy is finished.
    cudaStreamSynchronize(error_stream);

    // Normalize the global error across all grid points and take the square root.
    error = sqrtf((*h_error) / (N * N));

    // Print progress every 1000 iterations (note: num_iters now reflects fused iterations).
    if (num_iters % 1000 == 0) {
      std::cout << "Error after iteration " << num_iters << " = " << error << std::endl;
    }
    num_iters += FUSED_ITERS;
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

  // Clean up GPU and host memory.
  cudaFree(d_f);
  cudaFree(d_f_old);
  cudaFree(d_error);
  cudaStreamDestroy(error_stream);
  free(f);
  free(f_old);
  cudaFreeHost(h_error);

  double duration = (std::clock() - start_time) / (double) CLOCKS_PER_SEC;
  std::cout << "Total elapsed time: " << std::setprecision(4) << duration << " seconds" << std::endl;

  return 0;
}
