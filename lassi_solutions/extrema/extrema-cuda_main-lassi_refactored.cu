
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <cuda.h>

// --- 1D Kernel: Merged repeat loop with branch hoisting for clip mode ---
template<typename T>
__global__ void relextrema_1D_kernel(
    const int n,
    const int order,
    const bool clip,
    const T* __restrict__ inp,
    bool* __restrict__ results,
    const int repeat)
{
  const int tx = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;

  // Instead of launching kernel 'repeat' times, do repeat iterations here.
  if (clip) {
    for (int r = 0; r < repeat; r++) {
      for (int tid = tx; tid < n; tid += stride) {
        const T data = inp[tid];
        bool temp = true;
        // NOTE: using "o <= order" instead of "o < order+1"
        for (int o = 1; o <= order; o++) {
          // Clipped boundaries
          int plus  = tid + o;
          int minus = tid - o;
          if (plus >= n)
            plus = n - 1;
          if (minus < 0)
            minus = 0;
          temp = temp && (data > inp[plus]) && (data >= inp[minus]);
        }
        results[tid] = temp;
      }
    }
  } else {
    for (int r = 0; r < repeat; r++) {
      for (int tid = tx; tid < n; tid += stride) {
        const T data = inp[tid];
        bool temp = true;
        for (int o = 1; o <= order; o++) {
          // Wrapped boundaries
          int plus  = tid + o;
          int minus = tid - o;
          if (plus >= n)
            plus -= n;
          if (minus < 0)
            minus += n;
          temp = temp && (data > inp[plus]) && (data >= inp[minus]);
        }
        results[tid] = temp;
      }
    }
  }
}


// --- 2D Kernel: Merged repeat loop and branch hoisting for clip & axis mode ---
template<typename T>
__global__ void relextrema_2D_kernel(
  const int in_x,
  const int in_y,
  const int order,
  const bool clip,
  const int axis,
  const T* __restrict__ inp,
  bool* __restrict__ results,
  const int repeat)
{
  const int ty = blockIdx.x * blockDim.x + threadIdx.x;
  const int tx = blockIdx.y * blockDim.y + threadIdx.y;

  // proceed only if within bounds
  if (tx < in_y && ty < in_x)
  {
    const int tid = tx * in_x + ty;
    if (clip) {
      for (int r = 0; r < repeat; r++) {
        const T data = inp[tid];
        bool temp = true;
        for (int o = 1; o <= order; o++) {
          int plus, minus;
          if (axis == 0) { // vertical axis
            plus  = tx + o;
            minus = tx - o;
            if (plus >= in_y)
              plus = in_y - 1;
            if (minus < 0)
              minus = 0;
            plus  = plus  * in_x + ty;
            minus = minus * in_x + ty;
          } else { // horizontal axis
            plus  = ty + o;
            minus = ty - o;
            if (plus >= in_x)
              plus = in_x - 1;
            if (minus < 0)
              minus = 0;
            plus  = tx * in_x + plus;
            minus = tx * in_x + minus;
          }
          temp = temp && (data > inp[plus]) && (data >= inp[minus]);
        }
        results[tid] = temp;
      }
    } else {
      for (int r = 0; r < repeat; r++) {
        const T data = inp[tid];
        bool temp = true;
        for (int o = 1; o <= order; o++) {
          int plus, minus;
          if (axis == 0) { // vertical axis (wrapping)
            plus  = tx + o;
            minus = tx - o;
            if (plus >= in_y)
              plus -= in_y;
            if (minus < 0)
              minus += in_y;
            plus  = plus  * in_x + ty;
            minus = minus * in_x + ty;
          } else { // horizontal axis (wrapping)
            plus  = ty + o;
            minus = ty - o;
            if (plus >= in_x)
              plus -= in_x;
            if (minus < 0)
              minus += in_x;
            plus  = tx * in_x + plus;
            minus = tx * in_x + minus;
          }
          temp = temp && (data > inp[plus]) && (data >= inp[minus]);
        }
        results[tid] = temp;
      }
    }
  }
}


// -----------------------------------------------------------------------------
// CPU versions remain unchanged
// -----------------------------------------------------------------------------
template<typename T>
void cpu_relextrema_1D(
  const int n,
  const int order,
  const bool clip,
  const T* __restrict__ inp,
  bool* __restrict__ results)
{
  for (int tid = 0; tid < n; tid++) {
    const T data = inp[tid];
    bool temp = true;
    for (int o = 1; o <= order; o++) {
      int plus  = tid + o;
      int minus = tid - o;

      if (clip) {
        if (plus >= n) plus = n - 1;
        if (minus < 0) minus = 0;
      } else {
        if (plus >= n) plus -= n;
        if (minus < 0) minus += n;
      }
      temp = temp && (data > inp[plus]) && (data >= inp[minus]);
    }
    results[tid] = temp;
  }
}


template<typename T>
void cpu_relextrema_2D(
  const int in_x,
  const int in_y,
  const int order,
  const bool clip,
  const int axis,
  const T* __restrict__ inp,
  bool* __restrict__ results)
{
  for (int tx = 0; tx < in_y; tx++) {
    for (int ty = 0; ty < in_x; ty++) {
      const int tid = tx * in_x + ty;
      const T data = inp[tid];
      bool temp = true;
      for (int o = 1; o <= order; o++) {
        int plus, minus;
        if (axis == 0) {
          plus  = tx + o;
          minus = tx - o;
          if (clip) {
            if (plus >= in_y) plus = in_y - 1;
            if (minus < 0)   minus = 0;
          } else {
            if (plus >= in_y) plus -= in_y;
            if (minus < 0)   minus += in_y;
          }
          plus  = plus * in_x + ty;
          minus = minus * in_x + ty;
        } else {
          plus  = ty + o;
          minus = ty - o;
          if (clip) {
            if (plus >= in_x) plus = in_x - 1;
            if (minus < 0)   minus = 0;
          } else {
            if (plus >= in_x) plus -= in_x;
            if (minus < 0)   minus += in_x;
          }
          plus  = tx * in_x + plus;
          minus = tx * in_x + minus;
        }
        temp = temp && (data > inp[plus]) && (data >= inp[minus]);
      }
      results[tid] = temp;
    }
  }
}


// -----------------------------------------------------------------------------
// Test functions call the new merged GPU kernels
// -----------------------------------------------------------------------------
template <typename T>
long test_1D(const int length, const int order, const bool clip,
             const int repeat, const char* type)
{
  T* x = (T*) malloc(sizeof(T) * length);
  for (int i = 0; i < length; i++)
    x[i] = rand() % length;

  bool* cpu_r = (bool*) malloc(sizeof(bool) * length);
  bool* gpu_r = (bool*) malloc(sizeof(bool) * length);

  T* d_x;
  bool* d_result;
  cudaMalloc((void**)&d_x, length * sizeof(T));
  cudaMemcpy(d_x, x, length * sizeof(T), cudaMemcpyHostToDevice);
  cudaMalloc((void**)&d_result, length * sizeof(bool));

  const int threadsPerBlock = 256;
  const int blocks = (length + threadsPerBlock - 1) / threadsPerBlock;

  cudaDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();

  // Launch the 1D kernel once, with internal repeat loop.
  relextrema_1D_kernel<T><<<blocks, threadsPerBlock>>>(length, order, clip, d_x, d_result, repeat);

  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

  printf("Average 1D kernel (type = %s, order = %d, clip = %d) execution time: %f (s)\n",
         type, order, clip, (time * 1e-9f) / repeat);

  cudaMemcpy(gpu_r, d_result, length * sizeof(bool), cudaMemcpyDeviceToHost);
  cpu_relextrema_1D<T>(length, order, clip, x, cpu_r);

  int error = 0;
  for (int i = 0; i < length; i++) {
    if (cpu_r[i] != gpu_r[i]) {
      error = 1;
      break;
    }
  }
  cudaFree(d_x);
  cudaFree(d_result);
  free(x);
  free(cpu_r);
  free(gpu_r);
  if (error) printf("1D test: FAILED\n");
  return time;
}


template <typename T>
long test_2D(const int length_x, const int length_y, const int order,
             const bool clip, const int axis, const int repeat, const char* type)
{
  const int length = length_x * length_y;
  T* x = (T*) malloc(sizeof(T) * length);
  for (int i = 0; i < length; i++)
    x[i] = rand() % length;

  bool* cpu_r = (bool*) malloc(sizeof(bool) * length);
  bool* gpu_r = (bool*) malloc(sizeof(bool) * length);

  T* d_x;
  bool* d_result;
  cudaMalloc((void**)&d_x, length * sizeof(T));
  cudaMemcpy(d_x, x, length * sizeof(T), cudaMemcpyHostToDevice);
  cudaMalloc((void**)&d_result, length * sizeof(bool));

  dim3 threads(16, 16);
  dim3 grids((length_x + 15) / 16, (length_y + 15) / 16);

  cudaDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();

  // Launch the 2D kernel once, with internal repeat loop.
  relextrema_2D_kernel<T><<<grids, threads>>>(length_x, length_y, order, clip, axis, d_x, d_result, repeat);

  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

  printf("Average 2D kernel (type = %s, order = %d, clip = %d, axis = %d) execution time: %f (s)\n",
         type, order, clip, axis, (time * 1e-9f) / repeat);

  cudaMemcpy(gpu_r, d_result, length * sizeof(bool), cudaMemcpyDeviceToHost);
  cpu_relextrema_2D(length_x, length_y, order, clip, axis, x, cpu_r);

  int error = 0;
  for (int i = 0; i < length; i++) {
    if (cpu_r[i] != gpu_r[i]) {
      error = 1;
      break;
    }
  }
  cudaFree(d_x);
  cudaFree(d_result);
  free(x);
  free(cpu_r);
  free(gpu_r);
  if (error) printf("2D test: FAILED\n");
  return time;
}


// -----------------------------------------------------------------------------
// Main: test 1D and 2D versions for a variety of types and orders
// -----------------------------------------------------------------------------
int main(int argc, char* argv[]) {
  if (argc != 2) {
    printf("Usage: ./%s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);
  long time = 0;

  for (int order = 1; order <= 128; order = order * 2) {
    time += test_1D<int>(1000000, order, true, repeat, "int");
    time += test_1D<long>(1000000, order, true, repeat, "long");
    time += test_1D<float>(1000000, order, true, repeat, "float");
    time += test_1D<double>(1000000, order, true, repeat, "double");
  }

  for (int order = 1; order <= 128; order = order * 2) {
    // test 2D with axis = 1 (vertical)
    time += test_2D<int>(1000, 1000, order, true, 1, repeat, "int");
    time += test_2D<long>(1000, 1000, order, true, 1, repeat, "long");
    time += test_2D<float>(1000, 1000, order, true, 1, repeat, "float");
    time += test_2D<double>(1000, 1000, order, true, 1, repeat, "double");
  }

  for (int order = 1; order <= 128; order = order * 2) {
    // test 2D with axis = 0 (horizontal)
    time += test_2D<int>(1000, 1000, order, true, 0, repeat, "int");
    time += test_2D<long>(1000, 1000, order, true, 0, repeat, "long");
    time += test_2D<float>(1000, 1000, order, true, 0, repeat, "float");
    time += test_2D<double>(1000, 1000, order, true, 0, repeat, "double");
  }

  printf("\n-----------------------------------------------\n");
  printf("Total kernel execution time: %lf (s)\n", time * 1e-9);
  printf("-----------------------------------------------\n");
  return 0;
}
