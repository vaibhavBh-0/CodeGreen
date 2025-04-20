
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>
#include <random>

// Error checking macro to catch CUDA API errors.
#define CUDA_CHECK(call)                                                        \
  do {                                                                          \
    cudaError_t err = call;                                                     \
    if (err != cudaSuccess) {                                                   \
      fprintf(stderr, "CUDA error in %s (%s:%d): %s\n", #call, __FILE__, __LINE__, cudaGetErrorString(err)); \
      exit(EXIT_FAILURE);                                                       \
    }                                                                           \
  } while (0)

template <typename T>
void reference(
    const T* input,
    const T* dense,
    T* output,
    int embedding_dim,
    int batch_size,
    const int* offset)
{
  for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
    // cache the offset values
    const int start = offset[batch_idx];
    const int range = offset[batch_idx + 1] - start;
    for (int idx = 0; idx < embedding_dim; idx++) {
      const T dense_elem = dense[batch_idx * embedding_dim + idx];
      for (int nested_idx = idx; nested_idx < range; nested_idx += embedding_dim) {
        output[start + nested_idx] =
          input[start + nested_idx] + dense_elem;
      }
    }
  }
}

template <typename T>
__global__ void dense_esuhm(
    const T* input,
    const T* dense,
          T* output,
    int embedding_dim,
    const int* offset)
{
  const int batch_idx = blockIdx.x; // one batch per block
  const int start = offset[batch_idx];
  const int range = offset[batch_idx + 1] - start;
  const int grain_size = blockDim.x;
  const int tid = threadIdx.x;
  for (int idx = tid; idx < embedding_dim; idx += grain_size) {
    T dense_elem = dense[batch_idx * embedding_dim + idx];
    // Using a pragma to encourage unrolling the inner loop when possible.
    #pragma unroll
    for (int nested_idx = idx; nested_idx < range; nested_idx += embedding_dim) {
      output[start + nested_idx] = input[start + nested_idx] + dense_elem;
    }
  }
}

template <typename T>
__global__ void dense_esuhm2(
    const T* input,
    const T* dense,
          T* output,
    int embedding_dim,
    const int* offset)
{
  const int batch_idx = blockIdx.x;
  const int start = offset[batch_idx];
  const int range = offset[batch_idx + 1] - start;
  for (int idx = threadIdx.x; idx < embedding_dim; idx += blockDim.x) {
    T dense_elem = dense[batch_idx * embedding_dim + idx];
    #pragma unroll
    for (int nested_idx = idx; nested_idx < range; nested_idx += embedding_dim) {
      output[start + nested_idx] = input[start + nested_idx] + dense_elem;
    }
  }
}

int main(int argc, char* argv[])
{
  if (argc != 4) {
    printf("Usage: %s <number of rows> <batch size> <repeat>\n", argv[0]);
    return EXIT_FAILURE;
  }
  const int nrows = atoi(argv[1]);
  const int batch_size = atoi(argv[2]);
  const int repeat = atoi(argv[3]);
  assert(nrows > batch_size * batch_size);

  printf("Number of rows in the embedding table: %d\n", nrows);
  printf("Batch size: %d\n", batch_size);

  // Set device flag for blocking synchronization (which can reduce power consumption on CPU side)
  CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync));

  const int embed_dims[] = {768, 2048, 12288};

  for (size_t n = 0; n < sizeof(embed_dims)/sizeof(int); n++) {
    int ncols = embed_dims[n];
    printf("\nEmbedding dimension: %d\n", ncols);

    int input_size = nrows * ncols;  // same as output size
    size_t input_size_bytes = input_size * sizeof(float);

    int dense_size = batch_size * ncols;
    size_t dense_size_bytes = dense_size * sizeof(float);

    int input_offset_bytes = (batch_size + 1) * sizeof(int);

    // Allocate host memory using pinned memory (allows for faster transfers)
    float *input, *dense, *output_k1, *output_k2, *output_ref;
    CUDA_CHECK(cudaMallocHost(&input, input_size_bytes)); // embedding table
    CUDA_CHECK(cudaMallocHost(&dense, dense_size_bytes)); // dense features for the batch
    CUDA_CHECK(cudaMallocHost(&output_k1, input_size_bytes));
    CUDA_CHECK(cudaMallocHost(&output_k2, input_size_bytes));
    CUDA_CHECK(cudaMallocHost(&output_ref, input_size_bytes));
    int *input_offset;
    CUDA_CHECK(cudaMallocHost(&input_offset, input_offset_bytes));

    // Create valid offsets.
    srand(123);
    input_offset[0] = 0;
    for (int i = 1; i <= batch_size; i++)
      input_offset[i] = input_offset[i-1] + (rand() % batch_size + 1) * ncols;

    std::default_random_engine rng(123);
    std::uniform_real_distribution<float> distr(-1.f, 1.f);
    for (int i = 0; i < dense_size; i++) {
      dense[i] = distr(rng);
    }
    for (int i = 0; i < input_size; i++) {
      input[i] = distr(rng);
      output_ref[i] = 0;
    }

    // Compute reference result on host.
    reference(input, dense, output_ref, ncols, batch_size, input_offset);

    // Allocate device memory.
    float *d_input, *d_dense, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, input_size_bytes));
    CUDA_CHECK(cudaMalloc(&d_dense, dense_size_bytes));
    CUDA_CHECK(cudaMalloc(&d_output, input_size_bytes));

    int* d_input_offset;
    CUDA_CHECK(cudaMalloc(&d_input_offset, input_offset_bytes));

    // Transfer data to device.
    CUDA_CHECK(cudaMemcpy(d_input, input, input_size_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dense, dense, dense_size_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_input_offset, input_offset, input_offset_bytes, cudaMemcpyHostToDevice));

    // Use CUDA events for timing.
    cudaEvent_t event_start, event_stop;
    CUDA_CHECK(cudaEventCreate(&event_start));
    CUDA_CHECK(cudaEventCreate(&event_stop));

    // Explore different block sizes.
    for (int block_size = 128; block_size <= 1024; block_size *= 2) {
      printf("Block size: %d\n", block_size);

      // First kernel version (dense_esuhm)
      CUDA_CHECK(cudaMemset(d_output, 0, input_size_bytes));
      CUDA_CHECK(cudaDeviceSynchronize());
      CUDA_CHECK(cudaEventRecord(event_start, 0));

      for (int i = 0; i < repeat; i++)
        dense_esuhm<<<batch_size, block_size>>>(d_input, d_dense, d_output, ncols, d_input_offset);
      
      CUDA_CHECK(cudaEventRecord(event_stop, 0));
      CUDA_CHECK(cudaEventSynchronize(event_stop));
      float time_ms = 0.f;
      CUDA_CHECK(cudaEventElapsedTime(&time_ms, event_start, event_stop));
      printf("Average execution time of dense embedding kernel (k1): %f (us)\n", (time_ms * 1e3f) / repeat);
      CUDA_CHECK(cudaMemcpy(output_k1, d_output, input_size_bytes, cudaMemcpyDeviceToHost));

      // Second kernel version (dense_esuhm2)
      CUDA_CHECK(cudaMemset(d_output, 0, input_size_bytes));
      CUDA_CHECK(cudaDeviceSynchronize());
      CUDA_CHECK(cudaEventRecord(event_start, 0));
      
      for (int i = 0; i < repeat; i++)
        dense_esuhm2<<<batch_size, block_size>>>(d_input, d_dense, d_output, ncols, d_input_offset);
      
      CUDA_CHECK(cudaEventRecord(event_stop, 0));
      CUDA_CHECK(cudaEventSynchronize(event_stop));
      CUDA_CHECK(cudaEventElapsedTime(&time_ms, event_start, event_stop));
      printf("Average execution time of dense embedding kernel (k2): %f (us)\n", (time_ms * 1e3f) / repeat);
      CUDA_CHECK(cudaMemcpy(output_k2, d_output, input_size_bytes, cudaMemcpyDeviceToHost));

      // Validate results.
      bool ok = true;
      for (int i = 0; i < input_size; i++) {
        if (fabsf(output_k1[i] - output_ref[i]) > 1e-3f ||
            fabsf(output_k2[i] - output_ref[i]) > 1e-3f) {
          ok = false;
          break;
        }
      }
      printf("%s\n", ok ? "PASS" : "FAIL");
    }

    // Cleanup device memory.
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_dense));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_input_offset));

    // Cleanup host memory.
    CUDA_CHECK(cudaFreeHost(input));
    CUDA_CHECK(cudaFreeHost(dense));
    CUDA_CHECK(cudaFreeHost(output_k1));
    CUDA_CHECK(cudaFreeHost(output_k2));
    CUDA_CHECK(cudaFreeHost(output_ref));
    CUDA_CHECK(cudaFreeHost(input_offset));

    CUDA_CHECK(cudaEventDestroy(event_start));
    CUDA_CHECK(cudaEventDestroy(event_stop));
  }

  return EXIT_SUCCESS;
}
