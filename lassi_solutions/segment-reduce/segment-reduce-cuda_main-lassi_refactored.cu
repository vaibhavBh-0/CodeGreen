
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>

// Constant input value
#define VALUE 1

// Utility macro for checking CUDA call results
#define CUDA_CHECK(call)                                           \
    do {                                                           \
        cudaError_t err = call;                                    \
        if(err != cudaSuccess) {                                   \
            fprintf(stderr, "CUDA Error in %s at %s:%d: %s\n",     \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                    \
        }                                                          \
    } while(0)


// Custom functor for generating keys based on segment size.
struct KeyFunctor {
  int shift; // Precomputed log2(segment_size)
  __host__ __device__
  KeyFunctor(size_t seg_size) : shift(0) {
    size_t temp = seg_size;
    while (temp > 1) {
      temp >>= 1;
      shift++;
    }
  }

  __host__ __device__
  size_t operator()(const size_t i) const {
    return i >> shift;
  }
};

void segreduce(const size_t num_elements, const int repeat) {
  printf("num_elements = %zu\n", num_elements);

  // Allocate and initialize host input array using pinned memory.
  int *h_in = nullptr;
  CUDA_CHECK(cudaMallocHost((void**)&h_in, num_elements * sizeof(int)));
  for (size_t i = 0; i < num_elements; i++) {
    h_in[i] = VALUE;
  }

  // Allocate device memory for input values.
  int *d_in = nullptr;
  CUDA_CHECK(cudaMalloc(&d_in, num_elements * sizeof(int)));
  CUDA_CHECK(cudaMemcpy(d_in, h_in, num_elements * sizeof(int), cudaMemcpyHostToDevice));

  // Note: Removed allocation of d_keys as keys are now computed on-the-fly.
  
  // Pre-allocate output buffers that will be used for all segment sizes.
  // Since segment sizes vary from 16 to 16384, the maximum number of segments occurs when segment_size is minimum.
  size_t min_segment_size = 16;
  size_t max_segments = (num_elements + min_segment_size - 1) / min_segment_size;
  int *d_out_global = nullptr;
  int *d_keys_out_global = nullptr;
  CUDA_CHECK(cudaMalloc(&d_out_global, max_segments * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_keys_out_global, max_segments * sizeof(int)));
  
  // Allocate host output array using pinned memory.
  int *h_out_global = nullptr;
  CUDA_CHECK(cudaMallocHost((void**)&h_out_global, max_segments * sizeof(int)));

  // Create one persistent stream and persistent CUDA events for timing.
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));
  cudaEvent_t event_start, event_stop;
  CUDA_CHECK(cudaEventCreate(&event_start));
  CUDA_CHECK(cudaEventCreate(&event_stop));

  // Loop over different segment sizes; doubling each time from 16 to 16384.
  for (size_t segment_size = 16; segment_size <= 16384; segment_size *= 2) {
    // Instead of computing keys in a separate step, we use a transform iterator directly in reduce_by_key.
    KeyFunctor keyFunctor(segment_size);
    auto count_begin = thrust::make_counting_iterator((size_t)0);
    auto key_begin = thrust::make_transform_iterator(count_begin, keyFunctor);
    auto key_end = thrust::make_transform_iterator(count_begin + num_elements, keyFunctor);

    // Determine the number of segments for this iteration.
    size_t num_segments = num_elements / segment_size;

    // Record the start event on the persistent stream.
    CUDA_CHECK(cudaEventRecord(event_start, stream));
    
    // Perform reduce_by_key repeatedly using the persistent stream.
    for (int i = 0; i < repeat; i++) {
      thrust::reduce_by_key(thrust::cuda::par.on(stream),
                            key_begin,
                            key_end,
                            d_in,
                            d_keys_out_global,
                            d_out_global);
    }

    // Record the stop event on the same persistent stream.
    CUDA_CHECK(cudaEventRecord(event_stop, stream));
    // Wait for all operations in the stream to complete.
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Get elapsed time in milliseconds.
    float elapsed_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, event_start, event_stop));

    // Compute throughput in billion elements per second.
    // Convert elapsed_ms to nanoseconds: 1 ms = 1e6 ns.
    double time_ns = elapsed_ms * 1e6;
    printf("num_segments = %zu ", num_segments);
    printf("segment_size = %zu ", segment_size);
    printf("Throughput = %f (G/s)\n", 1.f * num_elements * repeat / time_ns);

    // Asynchronously copy the result corresponding to the segments from device to host.
    CUDA_CHECK(cudaMemcpyAsync(h_out_global, d_out_global, num_segments * sizeof(int),
                               cudaMemcpyDeviceToHost, stream));
    // Synchronize the persistent stream to ensure that the copy is complete before validation.
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Validate the reduction output.
    int correct_segment_sum = segment_size;
    int errors = 0;
    for (size_t i = 0; i < num_segments; i++) {
      if (h_out_global[i] != correct_segment_sum) {
        errors++;
        if (errors < 10) {
          printf("segment %zu has sum %d (expected %d)\n", i,
                 h_out_global[i], correct_segment_sum);
        }
      }
    }
    if (errors > 0) {
      printf("segmented reduction does not agree with the reference! %d errors!\n", errors);
    }
  }

  // Clean up persistent CUDA resources.
  CUDA_CHECK(cudaEventDestroy(event_start));
  CUDA_CHECK(cudaEventDestroy(event_stop));
  CUDA_CHECK(cudaStreamDestroy(stream));

  // Free device and host resources.
  CUDA_CHECK(cudaFreeHost(h_in));
  CUDA_CHECK(cudaFreeHost(h_out_global));
  CUDA_CHECK(cudaFree(d_in));
  // Note: d_keys allocation has been removed.
  CUDA_CHECK(cudaFree(d_out_global));
  CUDA_CHECK(cudaFree(d_keys_out_global));
}

int main(int argc, char* argv[]) {
  if (argc != 3) {
    printf("Usage: %s <multiplier> <repeat>\n", argv[0]);
    printf("The total number of elements is 16384 x multiplier\n");
    return 1;
  }
  const int multiplier = atoi(argv[1]);
  const int repeat = atoi(argv[2]);

  size_t num_elements = 16384 * static_cast<size_t>(multiplier);
  segreduce(num_elements, repeat);
  return 0;
}
