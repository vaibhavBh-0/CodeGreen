
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <cuda.h>

typedef unsigned long long int u64Int;
typedef long long int s64Int;

/* Random number generator */
#define POLY 0x0000000000000007UL
#define PERIOD 1317624576693539401L

/* NUPDATE remains 8 * TableSize */
#define NUPDATE (8 * TableSize)  /* orig: (4 * TableSize) */

/* CUDA specific parameters for init kernel */
#define K1_BLOCKSIZE 256

// CUDA specific parameters for update kernel:
#define UPDATE_THREADS_PER_BLOCK 256
#define UPDATE_NUM_BLOCKS 32

// Total number of update threads:
#define TOTAL_UPDATE_THREADS (UPDATE_THREADS_PER_BLOCK * UPDATE_NUM_BLOCKS)

// Precompute m2 lookup table on the host once and store it in constant memory.
// Declaration of constant memory variable.
__constant__ u64Int d_m2[64];

__device__
u64Int HPCC_starts(s64Int n)
{
  int i, j;
  u64Int temp, ran;
  
  // Removed unnecessary seed adjustments since n is always nonnegative and below PERIOD.
  if (n == 0) return 0x1;
  
  // Instead of computing m2 locally, we use the precomputed constant array d_m2.
  for (i = 62; i >= 0; i--)
    if ((n >> i) & 1)
      break;
  
  ran = 0x2;
  while (i > 0) {
    temp = 0;
    #pragma unroll
    for (j = 0; j < 64; j++)
      if ((ran >> j) & 1)
        temp ^= d_m2[j];
    ran = temp;
    i -= 1;
    if ((n >> i) & 1)
      ran = (ran << 1) ^ ((s64Int) ran < 0 ? POLY : 0);
  }
  
  return ran;
}

__global__ void initTable (u64Int* Table, const u64Int TableSize) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < TableSize) Table[i] = i;
}

__global__ void update (u64Int* __restrict__ Table,
                        u64Int* __restrict__ ran,
                        const u64Int TableSize)
{
  // Compute global thread index.
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  
  // Precompute the constant mask once per thread.
  const u64Int mask = TableSize - 1;
  
  // Determine per-thread iteration count.
  u64Int nUpdatesPerThread = NUPDATE / TOTAL_UPDATE_THREADS;
  
  // Initialize the random number seed for this thread using a thread-local variable.
  u64Int localRan = HPCC_starts(nUpdatesPerThread * j);
  
  for (u64Int i = 0; i < nUpdatesPerThread; i++) {
    localRan = (localRan << 1) ^ ((s64Int)localRan < 0 ? POLY : 0);
    atomicXor(&Table[localRan & mask], localRan);
  }
  
  // Removed the write-back of final random seed to global memory to reduce global memory traffic.
  // Original: ran[j] = localRan;
}

int main(int argc, char** argv) {
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  int failure;
  u64Int i;
  u64Int temp;
  double totalMem;
  u64Int *Table = NULL;
  u64Int logTableSize, TableSize;

  /* calculate local memory per node for the update table */
  totalMem = 1024 * 1024 * 512;
  totalMem /= sizeof(u64Int);

  /* calculate the size of update array (must be a power of 2) */
  for (totalMem *= 0.5, logTableSize = 0, TableSize = 1;
       totalMem >= 1.0;
       totalMem *= 0.5, logTableSize++, TableSize <<= 1)
    ; /* EMPTY */

  printf("Table size = %llu\n", TableSize);

  // Allocate pinned (page-locked) host memory to reduce transfer overhead.
  cudaError_t err = cudaHostAlloc((void**)&Table, TableSize * sizeof(u64Int), cudaHostAllocDefault);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to allocate pinned host memory for the update table %llu\n", TableSize);
    return 1;
  }

  /* Print parameters for run */
  fprintf(stdout, "Main table size   = 2^%llu = %llu words\n", logTableSize, TableSize);
  fprintf(stdout, "Number of updates = %llu\n", NUPDATE);

  u64Int* d_Table;
  cudaMalloc((void**)&d_Table, TableSize * sizeof(u64Int));

  // The unused d_ran allocation is removed.
  // u64Int *d_ran;
  // cudaMalloc((void**)&d_ran, TOTAL_UPDATE_THREADS * sizeof(u64Int));

  // Precompute the m2 lookup table on the host
  u64Int h_m2[64];
  u64Int host_temp = 0x1;
  for (int i = 0; i < 64; i++) {
    h_m2[i] = host_temp;
    host_temp = (host_temp << 1) ^ ((s64Int)host_temp < 0 ? POLY : 0);
    host_temp = (host_temp << 1) ^ ((s64Int)host_temp < 0 ? POLY : 0);
  }

  // Copy the precomputed m2 array to constant device memory
  cudaMemcpyToSymbol(d_m2, h_m2, 64 * sizeof(u64Int));

  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    /* initialize the table */
    initTable<<<(TableSize + K1_BLOCKSIZE - 1) / K1_BLOCKSIZE, K1_BLOCKSIZE>>>(d_Table, TableSize);

    /* update the table with increased occupancy */
    // Pass a NULL pointer for d_ran as it is no longer needed.
    update<<<UPDATE_NUM_BLOCKS, UPDATE_THREADS_PER_BLOCK>>>(d_Table, NULL, TableSize);
  }

  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time: %f (s)\n", (time * 1e-9f) / repeat);

  cudaMemcpy(Table, d_Table, TableSize * sizeof(u64Int), cudaMemcpyDeviceToHost);

  /* validation */
  temp = 0x1;
  for (i = 0; i < NUPDATE; i++) {
    temp = (temp << 1) ^ (((s64Int) temp < 0) ? POLY : 0);
    Table[temp & (TableSize-1)] ^= temp;
  }
  
  temp = 0;
  for (i = 0; i < TableSize; i++)
    if (Table[i] != i) {
      temp++;
    }

  fprintf(stdout, "Found %llu errors in %llu locations (%s).\n",
          temp, TableSize, (temp <= 0.01 * TableSize) ? "PASS" : "FAIL");
  if (temp <= 0.01 * TableSize) failure = 0;
  else failure = 1;

  cudaFreeHost(Table);
  cudaFree(d_Table);
  // Removed freeing of d_ran as it was never allocated.
  return failure;
}
