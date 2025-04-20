
#include <cstdlib>
#include <iostream>
#include <cuda.h>
#include <string>

#ifndef Real_t 
#define Real_t float
#endif

// Helper to compute the initial k value used in BS2, BS3, BS4
inline size_t computeInitK(const size_t n) {
  unsigned nbits = 0;
  size_t temp = n;
  while (temp) { 
    nbits++; 
    temp >>= 1;
  }
  return 1ULL << (nbits - 1);
}

// Global cache for the precomputed init_k value.
static size_t cached_init_k = 0;

// Kernel using traditional binary search.
// Now also takes zSize for boundary checking.
// The __ldg intrinsic is used for read-only loads.
template <typename T>
__global__ void
kernel_BS (const T* __restrict__ acc_a,
           const T* __restrict__ acc_z,
           size_t* __restrict__ acc_r,
           const size_t n,
           const size_t zSize)
{ 
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= zSize) return;
  
  T z = __ldg(&acc_z[i]);
  size_t low = 0;
  size_t high = n;
  
  while (high - low > 1) {
    size_t mid = low + (high - low) / 2;
    if (z < __ldg(&acc_a[mid]))
      high = mid;
    else
      low = mid;
  }
  acc_r[i] = low;
}

//----------------------------------------------------------------------------
// Consolidated binary search logic for BS2, BS3, and BS4.
// Mode parameter: mode == 2 uses BS2's boundary test, while mode == 3 or mode == 4 
// uses BS3/BS4 logic (using a conditional selection of r or n).
template <typename T>
__device__ inline size_t bs_generic_logic(const T* __restrict__ acc_a, T z, size_t n, size_t init_k, int mode) {
  size_t k = init_k;
  size_t idx = (__ldg(&acc_a[k]) <= z) ? k : 0;
  while (k >>= 1) {
    size_t r = idx | k;
    if (mode == 2) {
      if(r < n && z >= __ldg(&acc_a[r])) {
        idx = r;
      }
    } else {  // mode == 3 or mode == 4
      size_t w = (r < n) ? r : n;
      if(z >= __ldg(&acc_a[w])) {
        idx = r;
      }
    }
  }
  return idx;
}

// Unified kernel for BS2, BS3, and BS4 using the common helper above.
template <typename T>
__global__ void
kernel_BS_generic (const T* __restrict__ acc_a,
                   const T* __restrict__ acc_z,
                   size_t* __restrict__ acc_r,
                   const size_t n,
                   const size_t zSize,
                   const size_t init_k,
                   int mode)
{
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= zSize) return;
  
  T z = __ldg(&acc_z[i]);
  acc_r[i] = bs_generic_logic(acc_a, z, n, init_k, mode);
}

// Kernel BS2 now inlines the call to bs_generic_logic with mode 2.
template <typename T>
__global__ void
kernel_BS2 (const T* __restrict__ acc_a,
            const T* __restrict__ acc_z,
            size_t* __restrict__ acc_r,
            const size_t n,
            const size_t zSize,
            const size_t init_k)
{
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= zSize) return;
  T z = __ldg(&acc_z[i]);
  acc_r[i] = bs_generic_logic(acc_a, z, n, init_k, 2);
}

// Kernel BS3 now inlines the call to bs_generic_logic with mode 3.
template <typename T>
__global__ void
kernel_BS3 (const T* __restrict__ acc_a,
            const T* __restrict__ acc_z,
            size_t* __restrict__ acc_r,
            const size_t n,
            const size_t zSize,
            const size_t init_k)
{
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= zSize) return;
  T z = __ldg(&acc_z[i]);
  acc_r[i] = bs_generic_logic(acc_a, z, n, init_k, 3);
}

// Kernel BS4 now inlines the call to bs_generic_logic with mode 4.
template <typename T>
__global__ void
kernel_BS4 (const T* __restrict__ acc_a,
            const T* __restrict__ acc_z,
            size_t* __restrict__ acc_r,
            const size_t n,
            const size_t zSize,
            const size_t init_k)
{
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= zSize) return;
  T z = __ldg(&acc_z[i]);
  acc_r[i] = bs_generic_logic(acc_a, z, n, init_k, 4);
}

// Unified helper to launch kernels with persistent stream and common timing logic.
template <typename KernelFunc, typename... Args>
void launchKernel(KernelFunc kernel, const char* kernelName, int blocks, int threadsPerBlock, int repeat, Args... args) {
  // Use a persistent stream across all kernel launches
  static cudaStream_t stream = nullptr;
  if (!stream) {
    cudaStreamCreate(&stream);
  }
  
  // Reuse persistent CUDA events across invocations instead of recreating/destroying them every time.
  static cudaEvent_t start = nullptr, stop = nullptr;
  if (!start || !stop) {
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
  }

  cudaEventRecord(start, stream);
  for (int i = 0; i < repeat; i++) {
    kernel<<<blocks, threadsPerBlock, 0, stream>>>(args...);
  }
  cudaEventRecord(stop, stream);
  cudaEventSynchronize(stop);
  
  float elapsed_ms = 0;
  cudaEventElapsedTime(&elapsed_ms, start, stop);
  std::cout << "Average kernel execution time (" << kernelName << ") " << (elapsed_ms * 1e-3f) / repeat << " (s)\n";
}

// Host functions for each version using the common launchKernel helper.
template <typename T>
void bs (const size_t aSize,
         const size_t zSize,
         const T *d_a,  // aSize elements
         const T *d_z,  // zSize elements
         size_t *d_r,
         const size_t n,
         const int repeat )
{
  int threadsPerBlock = 256;
  int blocks = (zSize + threadsPerBlock - 1) / threadsPerBlock;
  launchKernel(kernel_BS<T>, "bs1", blocks, threadsPerBlock, repeat, d_a, d_z, d_r, n, zSize);
}

template <typename T>
void bs2 (const size_t aSize,
          const size_t zSize,
          const T *d_a,  // aSize elements
          const T *d_z,  // zSize elements
          size_t *d_r,
          const size_t n,
          const int repeat )
{
  int threadsPerBlock = 256;
  int blocks = (zSize + threadsPerBlock - 1) / threadsPerBlock;
  
  // Use the already-computed global cached value for init_k.
  size_t init_k = cached_init_k;
  launchKernel(kernel_BS2<T>, "bs2", blocks, threadsPerBlock, repeat, d_a, d_z, d_r, n, zSize, init_k);
}

template <typename T>
void bs3 (const size_t aSize,
          const size_t zSize,
          const T *d_a,  // aSize elements
          const T *d_z,  // zSize elements
          size_t *d_r,
          const size_t n,
          const int repeat )
{
  int threadsPerBlock = 256;
  int blocks = (zSize + threadsPerBlock - 1) / threadsPerBlock;
  
  // Use the already-computed global cached value for init_k.
  size_t init_k = cached_init_k;
  launchKernel(kernel_BS3<T>, "bs3", blocks, threadsPerBlock, repeat, d_a, d_z, d_r, n, zSize, init_k);
}

template <typename T>
void bs4 (const size_t aSize,
          const size_t zSize,
          const T *d_a,  // aSize elements
          const T *d_z,  // zSize elements
          size_t *d_r,
          const size_t n,
          const int repeat )
{
  int threadsPerBlock = 256;
  int blocks = (zSize + threadsPerBlock - 1) / threadsPerBlock;
  
  // Use the already-computed global cached value for init_k.
  size_t init_k = cached_init_k;
  launchKernel(kernel_BS4<T>, "bs4", blocks, threadsPerBlock, repeat, d_a, d_z, d_r, n, zSize, init_k);
}

#ifdef DEBUG
void verify(Real_t *a, Real_t *z, size_t *r, size_t aSize, size_t zSize, std::string msg)
{
  for (size_t i = 0; i < zSize; ++i)
  {
    if (!(r[i] + 1 < aSize && a[r[i]] <= z[i] && z[i] < a[r[i] + 1]))
    {
      std::cout << msg << ": incorrect result:" << std::endl;
      std::cout << "index = " << i << " r[index] = " << r[i] << std::endl;
      std::cout << a[r[i]] << " <= " << z[i] << " < " << a[r[i] + 1] << std::endl;
      break;
    }
    r[i] = 0xFFFFFFFF;
  }
}
#endif

int main(int argc, char* argv[])
{
  if (argc != 3) {
    std::cout << "Usage: ./main <number of elements> <repeat>\n";
    return 1;
  }

  size_t numElem = atol(argv[1]);
  uint repeat = atoi(argv[2]);

  srand(2);
  size_t aSize = numElem;
  size_t zSize = 2 * aSize;
  Real_t *a = nullptr;
  Real_t *z = nullptr;
  size_t *r = nullptr;

  // Allocate pinned host memory using cudaHostAlloc instead of posix_memalign
  cudaHostAlloc((void**)&a, aSize * sizeof(Real_t), cudaHostAllocDefault);
  cudaHostAlloc((void**)&z, zSize * sizeof(Real_t), cudaHostAllocDefault);
  cudaHostAlloc((void**)&r, zSize * sizeof(size_t), cudaHostAllocDefault);

  size_t N = aSize - 1;
  
  // Precompute init_k once and cache it globally.
  cached_init_k = computeInitK(N);

  for (size_t i = 0; i < aSize; i++) 
    a[i] = i;

  for (size_t i = 0; i < zSize; i++) 
    z[i] = rand() % N;

  Real_t* d_a;
  Real_t* d_z;
  size_t *d_r;
  cudaMalloc((void**)&d_a, sizeof(Real_t) * aSize);
  cudaMalloc((void**)&d_z, sizeof(Real_t) * zSize);
  cudaMalloc((void**)&d_r, sizeof(size_t) * zSize);

  // Use asynchronous memory copies with a dedicated stream
  cudaStream_t memStream;
  cudaStreamCreate(&memStream);
  cudaMemcpyAsync(d_a, a, sizeof(Real_t) * aSize, cudaMemcpyHostToDevice, memStream);
  cudaMemcpyAsync(d_z, z, sizeof(Real_t) * zSize, cudaMemcpyHostToDevice, memStream);
  cudaStreamSynchronize(memStream);
  cudaStreamDestroy(memStream);

#ifdef DEBUG
  // Create one persistent verification stream to reuse for all verifications.
  cudaStream_t verifyStream;
  cudaStreamCreate(&verifyStream);
#endif

  bs(aSize, zSize, d_a, d_z, d_r, N, repeat);
#ifdef DEBUG
  {
    cudaMemcpyAsync(r, d_r, sizeof(size_t)*zSize, cudaMemcpyDeviceToHost, verifyStream);
    cudaStreamSynchronize(verifyStream);
    verify(a, z, r, aSize, zSize, "bs");
  }
#endif

  bs2(aSize, zSize, d_a, d_z, d_r, N, repeat);
#ifdef DEBUG
  {
    cudaMemcpyAsync(r, d_r, sizeof(size_t)*zSize, cudaMemcpyDeviceToHost, verifyStream);
    cudaStreamSynchronize(verifyStream);
    verify(a, z, r, aSize, zSize, "bs2");
  }
#endif

  bs3(aSize, zSize, d_a, d_z, d_r, N, repeat);
#ifdef DEBUG
  {
    cudaMemcpyAsync(r, d_r, sizeof(size_t)*zSize, cudaMemcpyDeviceToHost, verifyStream);
    cudaStreamSynchronize(verifyStream);
    verify(a, z, r, aSize, zSize, "bs3");
  }
#endif

  bs4(aSize, zSize, d_a, d_z, d_r, N, repeat);
#ifdef DEBUG
  {
    cudaMemcpyAsync(r, d_r, sizeof(size_t)*zSize, cudaMemcpyDeviceToHost, verifyStream);
    cudaStreamSynchronize(verifyStream);
    verify(a, z, r, aSize, zSize, "bs4");
  }
#endif

#ifdef DEBUG
  cudaStreamDestroy(verifyStream);
#endif

  cudaFree(d_a);
  cudaFree(d_z);
  cudaFree(d_r);
  
  // Free pinned host memory allocated by cudaHostAlloc
  cudaFreeHost(a);
  cudaFreeHost(z);
  cudaFreeHost(r);
  
  return 0;
}
