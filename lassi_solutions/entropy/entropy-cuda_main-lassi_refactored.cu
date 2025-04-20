
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include "reference.h"

// Declare logarithm lookup table in constant memory on the device
__constant__ float d_logTableConst[26];
// Constant repeat count copied from host to device
__constant__ int d_repeat;

__global__ 
void entropy(
       float *__restrict__ d_entropy,
  const char* __restrict__ d_val, 
  int height, int width)
{
  const int x = threadIdx.x + blockIdx.x * blockDim.x;
  const int y = threadIdx.y + blockIdx.y * blockDim.y;

  // Early exit for threads outside the valid image bounds.
  if (x >= width || y >= height) return;
  
  float final_entropy = 0.0f;  // final entropy value

  // Compute the entropy once rather than repeating identical work d_repeat times.
  // Initialize per-thread local count array for values 0 to 15
  char count[16];
  for (int i = 0; i < 16; i++) count[i] = 0;
  
  // total number of valid elements in 5x5 window
  char total = 0;
  
  // 5x5 window with unrolled loops
  #pragma unroll
  for (int dy = -2; dy <= 2; dy++) {
    #pragma unroll
    for (int dx = -2; dx <= 2; dx++) {
      int xx = x + dx;
      int yy = y + dy;
      if (xx >= 0 && yy >= 0 && yy < height && xx < width) {
        count[d_val[yy * width + xx]]++;
        total++;
      }
    }
  }
  
  float entropy_val = 0.0f;
  if (total < 1) {
    total = 1;
  } else {
    for (int k = 0; k < 16; k++) {
      float p = __fdividef((float)count[k], (float)total);
      entropy_val -= p * __log2f(p);
    }
  }
  final_entropy = entropy_val;
  
  d_entropy[y * width + x] = final_entropy;
}

template<int bsize_x, int bsize_y>
__global__ void entropy_opt(
       float *__restrict__ d_entropy,
  const char* __restrict__ d_val, 
  int m, int n)
{
  // Define tile dimensions including halo (2-pixels on each side)
  const int tileWidth  = bsize_x + 4;
  const int tileHeight = bsize_y + 4;

  // Allocate shared memory tile statically (2D array)
  __shared__ char s_tile[tileHeight][tileWidth];

  // Determine block indices for global coordinates
  const int blockX = blockIdx.x * bsize_x;
  const int blockY = blockIdx.y * bsize_y;

  // Total number of elements in the shared tile
  const int totalTileElements = tileWidth * tileHeight;
  const int threadId = threadIdx.y * bsize_x + threadIdx.x;
  const int numThreads = bsize_x * bsize_y;

  // Each thread loads one or more elements of the shared tile into shared memory.
  for (int idx = threadId; idx < totalTileElements; idx += numThreads) {
    int local_x = idx % tileWidth;
    int local_y = idx / tileWidth;
    // Compute corresponding global coordinates. The tile starts at (blockX-2, blockY-2)
    int global_x = blockX - 2 + local_x;
    int global_y = blockY - 2 + local_y;
    // If within image bounds, load the pixel; otherwise mark as invalid (-1)
    if (global_x >= 0 && global_y >= 0 && global_x < n && global_y < m) {
      s_tile[local_y][local_x] = d_val[global_y * n + global_x];
    } else {
      s_tile[local_y][local_x] = -1;
    }
  }
  __syncthreads();

  // Compute the global coordinate for the thread's central pixel
  const int x = blockX + threadIdx.x;
  const int y = blockY + threadIdx.y;

  float final_entropy = 0.0f;
  
  // Compute the entropy once using the shared tile.
  // Each thread maintains a local histogram for values 0 to 15
  char count[16];
  for (int i = 0; i < 16; i++) count[i] = 0;
      
  char total = 0;
  // The thread's corresponding location in shared memory is shifted by the halo offset (2,2)
  int s_x = threadIdx.x + 2;
  int s_y = threadIdx.y + 2;
  
  // Use shared memory to fetch the 5x5 neighborhood
  #pragma unroll
  for (int dy = -2; dy <= 2; dy++) {
    #pragma unroll
    for (int dx = -2; dx <= 2; dx++) {
      int lx = s_x + dx;
      int ly = s_y + dy;
      char pixel = s_tile[ly][lx];
      if (pixel != -1) {
        count[pixel]++;
        total++;
      }
    }
  }
  
  float entropy_val = 0.0f;
  if (total < 1)
    total = 1;
  // Compute entropy using the precomputed log table in constant memory.
  for (int k = 0; k < 16; k++)
    entropy_val -= d_logTableConst[ count[k] ];
  
  entropy_val = entropy_val / total + __log2f((float)total);
  final_entropy = entropy_val;
    
  if (y < m && x < n)
    d_entropy[y * n + x] = final_entropy;
}

int main(int argc, char* argv[]) {
  if (argc != 4) {
    printf("Usage: %s <width> <height> <repeat>\n", argv[0]);
    return 1;
  }
  const int width = atoi(argv[1]); 
  const int height = atoi(argv[2]); 
  const int repeat = atoi(argv[3]); 

  const int input_bytes = width * height * sizeof(char);
  const int output_bytes = width * height * sizeof(float);

  // Use pinned host memory allocation for improved transfer performance
  char* input;
  cudaHostAlloc((void**)&input, input_bytes, cudaHostAllocDefault);
  float* output;
  cudaHostAlloc((void**)&output, output_bytes, cudaHostAllocDefault);
  float* output_ref = (float*) malloc(output_bytes);

  srand(123);
  for (int i = 0; i < height; i++)
    for (int j = 0; j < width; j++)
      input[i * width + j] = rand() % 16;

  char* d_input;
  cudaMalloc((void**)&d_input, input_bytes);
  float* d_output;
  cudaMalloc((void**)&d_output, output_bytes);

  // Create a dedicated CUDA stream for asynchronous operations
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  // Asynchronously copy input data to device using pinned memory
  cudaMemcpyAsync(d_input, input, input_bytes, cudaMemcpyHostToDevice, stream);
  // No explicit synchronization needed here because of stream serial ordering

  // Copy the repeat count into constant device memory
  cudaMemcpyToSymbol(d_repeat, &repeat, sizeof(int));

  dim3 grids((width + 15) / 16, (height + 15) / 16);
  dim3 blocks(16, 16);

  // baseline kernel (one kernel call performing the work once instead of d_repeat times)
  cudaEvent_t baseline_start, baseline_stop;
  cudaEventCreate(&baseline_start);
  cudaEventCreate(&baseline_stop);
  cudaEventRecord(baseline_start, stream);

  entropy<<< grids, blocks, 0, stream >>> (d_output, d_input, height, width);

  cudaEventRecord(baseline_stop, stream);
  cudaEventSynchronize(baseline_stop);
  float baseline_time_ms = 0.0f;
  cudaEventElapsedTime(&baseline_time_ms, baseline_start, baseline_stop);
  printf("Average kernel (baseline) execution time %f (s)\n", (baseline_time_ms * 1e-3f) / repeat);
  cudaEventDestroy(baseline_start);
  cudaEventDestroy(baseline_stop);

  // optimized kernel using constant memory for log table
  float logTable[26];
  for (int i = 0; i <= 25; i++) 
    logTable[i] = i <= 1 ? 0 : i * log2f((float)i);
  // Copy precomputed table to constant memory
  cudaMemcpyToSymbol(d_logTableConst, logTable, sizeof(logTable));

  cudaEvent_t opt_start, opt_stop;
  cudaEventCreate(&opt_start);
  cudaEventCreate(&opt_stop);
  cudaEventRecord(opt_start, stream);

  // Launch the optimized kernel.
  entropy_opt<16, 16><<< grids, blocks, 0, stream >>> (d_output, d_input, height, width);

  cudaEventRecord(opt_stop, stream);
  cudaEventSynchronize(opt_stop);
  float opt_time_ms = 0.0f;
  cudaEventElapsedTime(&opt_time_ms, opt_start, opt_stop);
  printf("Average kernel (optimized) execution time %f (s)\n", (opt_time_ms * 1e-3f) / repeat);
  cudaEventDestroy(opt_start);
  cudaEventDestroy(opt_stop);

  // Asynchronously copy output data back to host
  cudaMemcpyAsync(output, d_output, output_bytes, cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);

  // verify
  reference(output_ref, input, height, width);

  bool ok = true;
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      if (fabsf(output[i * width + j] - output_ref[i * width + j]) > 1e-3f) {
        ok = false; 
        break;
      }
    }
    if (!ok) break;
  }
  printf("%s\n", ok ? "PASS" : "FAIL");
 
  cudaFree(d_input);
  cudaFree(d_output);

  cudaFreeHost(input);
  cudaFreeHost(output);
  free(output_ref);

  // Destroy the CUDA stream
  cudaStreamDestroy(stream);
  return 0;
}
