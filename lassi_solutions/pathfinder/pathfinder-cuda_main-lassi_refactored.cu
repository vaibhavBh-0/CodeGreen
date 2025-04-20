
/***********************************************************************
 * PathFinder uses dynamic programming to find a path on a 2-D grid from
 * the bottom row to the top row with the smallest accumulated weights,
 * where each step of the path moves straight ahead or diagonally ahead.
 * It iterates row by row, each node picks a neighboring node in the
 * previous row that has the smallest accumulated weight, and adds its
 * own weight to the sum.
 *
 * This kernel uses the technique of ghost zone optimization
 ***********************************************************************/

// Other header files.
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#include <iostream>
#include <sys/time.h>
#include <cuda.h>

// halo width along one direction when advancing to the next iteration
#define HALO     1
#define STR_SIZE 256
#define DEVICE   0
#define M_SEED   9
#define IN_RANGE(x, min, max)  ((x)>=(min) && (x)<=(max))
#define CLAMP_RANGE(x, min, max) x = (x<(min)) ? min : ((x>(max)) ? max : x )
#define MIN(a, b) ((a)<=(b) ? (a) : (b))


void fatal(char *s)
{
  fprintf(stderr, "error: %s\n", s);
}

double get_time() {
  struct timeval t;
  gettimeofday(&t,NULL);
  return t.tv_sec+t.tv_usec*1e-6;
}

__global__ void pathfinder (
    const int*__restrict__ gpuWall,
    const int*__restrict__ gpuSrc,
          int*__restrict__ gpuResult,
          int*__restrict__ outputBuffer,
    const int iteration,
    const int theHalo,
    const int borderCols,
    const int cols,
    const int t)
{
  int BLOCK_SIZE = blockDim.x;
  int bx = blockIdx.x;
  int tx = threadIdx.x;
  __shared__ int prev[256];
  __shared__ int result[256];

  // Calculate the small block size.
  int small_block_cols = BLOCK_SIZE - (iteration * theHalo * 2);

  // Calculate the block boundaries
  int blkX = (small_block_cols * bx) - borderCols;
  int blkXmax = blkX + BLOCK_SIZE - 1;

  // Global thread coordinate.
  int xidx = blkX + tx;

  // Determine valid computation range within the block.
  int validXmin = (blkX < 0) ? -blkX : 0;
  int validXmax = (blkXmax > cols - 1) ? BLOCK_SIZE - 1 - (blkXmax - cols + 1) : BLOCK_SIZE - 1;

  int W = tx - 1;
  int E = tx + 1;
  W = (W < validXmin) ? validXmin : W;
  E = (E > validXmax) ? validXmax : E;

  bool isValid = IN_RANGE(tx, validXmin, validXmax);

  if (IN_RANGE(xidx, 0, cols - 1))
  {
    prev[tx] = gpuSrc[xidx];
  }

  __syncthreads();

  // Precompute invariant base index for the current t value.
  int baseIndex = cols * t;

  bool computed = false;
  // Process the first (iteration - 1) steps with full update.
  for (int i = 0; i < iteration - 1; i++)
  {
    computed = false;
    if (IN_RANGE(tx, i + 1, BLOCK_SIZE - i - 2) && isValid)
    {
      computed = true;
      int left = prev[W];
      int up = prev[tx];
      int right = prev[E];
      int shortest = MIN(left, up);
      shortest = MIN(shortest, right);
      int index = baseIndex + cols * i;
      result[tx] = shortest + gpuWall[index];

      // Debug output when required.
      if (tx == 11 && i == 0)
      {
        int bufIndex = gpuSrc[xidx];
        outputBuffer[bufIndex] = 1;
      }
    }
    __syncthreads();
    // Update shared memory values for next iteration.
    if (computed)
    {
      prev[tx] = result[tx];
    }
    __syncthreads();
  }

  // Process the final iteration without updating shared memory afterwards.
  if (iteration > 0)
  {
    int i = iteration - 1;
    computed = false;
    if (IN_RANGE(tx, i + 1, BLOCK_SIZE - i - 2) && isValid)
    {
      computed = true;
      int left = prev[W];
      int up = prev[tx];
      int right = prev[E];
      int shortest = MIN(left, up);
      shortest = MIN(shortest, right);
      int index = baseIndex + cols * i;
      result[tx] = shortest + gpuWall[index];

      if (tx == 11 && i == 0)
      {
        int bufIndex = gpuSrc[xidx];
        outputBuffer[bufIndex] = 1;
      }
    }
    __syncthreads();
  }

  // Write the final result back to global memory.
  if (computed)
  {
    gpuResult[xidx] = result[tx];
  }
}

int main(int argc, char** argv)
{
  // Program variables.
  int   rows, cols;
  int*  data;
  int** wall;
  int*  result;
  int   pyramid_height;

  if (argc == 4)
  {
    cols = atoi(argv[1]);
    rows = atoi(argv[2]);
    pyramid_height = atoi(argv[3]);
  }
  else
  {
    printf("Usage: %s <column length> <row length> <pyramid_height>\n", argv[0]);
    exit(0);
  }

  // Use pinned (page-locked) memory for data, result and outputBuffer.
  cudaMallocHost((void**)&data, sizeof(int) * rows * cols);
  cudaMallocHost((void**)&result, sizeof(int) * cols);

  wall = new int*[rows];
  for (int n = 0; n < rows; n++)
  {
    // wall[n] is set to be the nth row of the data array.
    wall[n] = data + cols * n;
  }

  int seed = M_SEED;
  srand(seed);

  for (int i = 0; i < rows; i++)
  {
    for (int j = 0; j < cols; j++)
    {
      wall[i][j] = rand() % 10;
    }
  }
#ifdef BENCH_PRINT
  for (int i = 0; i < rows; i++)
  {
    for (int j = 0; j < cols; j++)
    {
      printf("%d ", wall[i][j]);
    }
    printf("\n");
  }
#endif

  // Pyramid parameters.
  const int borderCols = pyramid_height * HALO;
  int size = rows * cols; // the size (global work size)

  // Use a block size that is a multiple of the warp size.
  int lws = 256;

  int* outputBuffer;
  cudaMallocHost((void**)&outputBuffer, sizeof(int) * 16384);
  // Initialize the output buffer to zero.
  memset(outputBuffer, 0, sizeof(int) * 16384);

  int theHalo = HALO;

  // Create a CUDA stream for asynchronous operations.
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  double offload_start = get_time();

  int* d_gpuWall;
  cudaMalloc((void**)&d_gpuWall, sizeof(int) * (size - cols));
  // Asynchronously copy the data for d_gpuWall.
  cudaMemcpyAsync(d_gpuWall, data + cols, sizeof(int) * (size - cols), cudaMemcpyHostToDevice, stream);

  int* d_gpuSrc;
  cudaMalloc((void**)&d_gpuSrc, sizeof(int) * cols);
  // Asynchronously copy the data for d_gpuSrc.
  cudaMemcpyAsync(d_gpuSrc, data, sizeof(int) * cols, cudaMemcpyHostToDevice, stream);

  int* d_gpuResult;
  cudaMalloc((void**)&d_gpuResult, sizeof(int) * cols);

  int* d_outputBuffer;
  cudaMalloc((void**)&d_outputBuffer, sizeof(int) * 16384);

  // Recompute grid dimensions using ceiling division.
  dim3 gridDim((size + lws - 1) / lws);
  dim3 blockDim(lws);

  // Ensure initial copies are completed before kernel launches.
  cudaStreamSynchronize(stream);
  double kstart = get_time();

  for (int t = 0; t < rows - 1; t += pyramid_height)
  {
    int iteration = MIN(pyramid_height, rows - t - 1);

    // Launch the kernel on the created stream.
    pathfinder<<<gridDim, blockDim, 0, stream>>>(
        d_gpuWall, d_gpuSrc, d_gpuResult, d_outputBuffer,
        iteration, theHalo, borderCols, cols, t);

    // Swap pointers for the next iteration.
    int* temp = d_gpuResult;
    d_gpuResult = d_gpuSrc;
    d_gpuSrc = temp;
  }

  // Synchronize the stream to ensure kernel execution is complete.
  cudaStreamSynchronize(stream);
  double kend = get_time();
  printf("Total kernel execution time: %lf (s)\n", kend - kstart);

  // Asynchronously copy the results back to host memory.
  cudaMemcpyAsync(result, d_gpuSrc, sizeof(int) * cols, cudaMemcpyDeviceToHost, stream);
  cudaMemcpyAsync(outputBuffer, d_outputBuffer, sizeof(int) * 16384, cudaMemcpyDeviceToHost, stream);
  // Wait for the asynchronous copies to complete.
  cudaStreamSynchronize(stream);

  cudaFree(d_gpuResult);
  cudaFree(d_gpuSrc);
  cudaFree(d_gpuWall);
  cudaFree(d_outputBuffer);

  double offload_end = get_time();
  printf("Device offloading time = %lf (s)\n", offload_end - offload_start);

  // add a null terminator at the end of the outputBuffer.
  outputBuffer[16383] = '\0';

#ifdef BENCH_PRINT
  for (int i = 0; i < cols; i++)
    printf("%d ", data[i]);
  printf("\n");
  for (int i = 0; i < cols; i++)
    printf("%d ", result[i]);
  printf("\n");
#endif

  // Memory cleanup.
  cudaFreeHost(data);
  delete[] wall;
  cudaFreeHost(result);
  cudaFreeHost(outputBuffer);

  // Destroy the stream.
  cudaStreamDestroy(stream);

  return EXIT_SUCCESS;
}
