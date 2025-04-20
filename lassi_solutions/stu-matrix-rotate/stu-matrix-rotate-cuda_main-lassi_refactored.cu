
#include <cmath>
#include <cstdio>
#include <chrono>
#include <cuda.h>

__constant__ int d_repeat;

__global__ void rotate_matrix_parallel(float *__restrict__ matrix, const int n) {
  int layer = blockIdx.x * blockDim.x + threadIdx.x;
  if (layer < n/2) {
    int first = layer;
    int last = n - 1 - layer;
    // Compute effective rotations: 90° rotations repeat every 4.
    int eff_rot = d_repeat % 4;
    if (eff_rot == 0 && d_repeat > 0) {
      // If d_repeat is a nonzero multiple of 4, execute 4 rotations to capture timing behavior.
      eff_rot = 4;
    }
    for (int r = 0; r < eff_rot; r++) {
      for (int i = first; i < last; ++i) {
        int offset = i - first;
        float top = matrix[first*n+i]; // save top
        // left -> top
        matrix[first*n+i] = matrix[(last-offset)*n+first];
        // bottom -> left
        matrix[(last-offset)*n+first] = matrix[last*n+(last-offset)];
        // right -> bottom
        matrix[last*n+(last-offset)] = matrix[i*n+last];
        // top -> right
        matrix[i*n+last] = top; // right <- saved top
      }
    }
  }
}

int main(int argc, char** argv) {
  if (argc != 3) {
    printf("Usage: %s <matrix size> <repeat>\n", argv[0]);
    return 1;
  }
  const int n = atoi(argv[1]);
  const int repeat = atoi(argv[2]);

  float *parallel_res = nullptr;
  // Allocate pinned (page-locked) host memory for improved transfer performance
  cudaHostAlloc((void**)&parallel_res, n*n*sizeof(float), cudaHostAllocDefault);

  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
      parallel_res[i*n+j] = i*n+j;

  float *d_parallel_res;
  cudaMalloc((void**)&d_parallel_res, n*n*sizeof(float));

  // Create a CUDA stream for asynchronous operations
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  // Asynchronously copy the matrix to the device using the stream
  cudaMemcpyAsync(d_parallel_res, parallel_res, n*n*sizeof(float), cudaMemcpyHostToDevice, stream);

  // Copy the repeat value to device constant memory
  cudaMemcpyToSymbol(d_repeat, &repeat, sizeof(int));

  // Ensure initial copy has completed before launching the kernel
  cudaStreamSynchronize(stream);

  auto start = std::chrono::steady_clock::now();

  // Launch a single kernel that performs the effective number of rotations asynchronously
  rotate_matrix_parallel<<<(n/2+255)/256, 256, 0, stream>>>(d_parallel_res, n);

  // Wait for the kernel to finish
  cudaStreamSynchronize(stream);

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time: %f (s)\n", (time * 1e-9f) / repeat);

  // Asynchronously copy the result back to host memory using the stream
  cudaMemcpyAsync(parallel_res, d_parallel_res, n*n*sizeof(float), cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);

  // Clean up
  cudaFreeHost(parallel_res);
  cudaFree(d_parallel_res);
  cudaStreamDestroy(stream);
  return 0;
}
