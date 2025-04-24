#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <chrono>
#include <cuda.h>

__global__ void rotate_matrix_parallel (float *matrix, const int n) {
  int layer = blockIdx.x * blockDim.x + threadIdx.x;
  if (layer < n/2) {
    int first = layer;
    int last = n - 1 - layer;
    for(int i = first; i < last; ++i) {
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


int main(int argc, char** argv) {
  if (argc != 3) {
    printf("Usage: %s <matrix size> <repeat>\n", argv[0]);
    return 1;
  }
  const int n = atoi(argv[1]);
  const int repeat = atoi(argv[2]);

    float *parallel_res = (float*) aligned_alloc(1024, n*n*sizeof(float));

  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
      parallel_res[i*n+j] = i*n+j;

  float *d_parallel_res;
  cudaMalloc((void**)&d_parallel_res, n*n*sizeof(float));
  cudaMemcpy(d_parallel_res, parallel_res, n*n*sizeof(float), cudaMemcpyHostToDevice);

  cudaDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();

  dim3 threadsPerBlock(64, 64);
  dim3 numBlocks((n + threadsPerBlock.x -1) / threadsPerBlock.x, (n+threadsPerBlock.y -1) / threadsPerBlock.y);
  
  for (int i = 0; i < repeat; i++) {
    // rotate_matrix_parallel<<<(n + threadsPerBlock.x - 1)/ threadsPerBlock.x, (n + threadsPerBlock.y - 1) / threadsPerBlock.y>>>(d_parallel_res, n);
    // rotate_matrix_parallel<<<(n +255)/256, 256>>>(d_parallel_res, n);
    // rotate_matrix_parallel<<<numBlocks, threadsPerBlock>>>(d_parallel_res, n);
    rotate_matrix_parallel<<<(n +255)/256, 64>>>(d_parallel_res, n);
  }

  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time: %f (s)\n", (time * 1e-9f) / repeat);

  cudaMemcpy(parallel_res, d_parallel_res, n*n*sizeof(float), cudaMemcpyDeviceToHost);

  free(parallel_res);
  cudaFree(d_parallel_res);
  return 0;
}
