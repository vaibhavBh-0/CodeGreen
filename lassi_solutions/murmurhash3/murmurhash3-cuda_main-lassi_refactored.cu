
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cassert>
#include <cuda.h>

// Use CUDA events for timing so the host need not busy‐poll
// and the GPU can run more efficiently.
 
#define BLOCK_SIZE 256
#define FORCE_INLINE inline __attribute__((always_inline))
 
// Device constant for repeat count
__constant__ uint32_t gRepeat;

// Rotate left (for both device and host)
__host__ __device__
FORCE_INLINE uint64_t rotl64(uint64_t x, int8_t r)
{
    return (x << r) | (x >> (64 - r));
}
 
#define BIG_CONSTANT(x) (x##LU)
 
// Read 64 bits from pointer p at block index i (supports unaligned access)
__host__ __device__
FORCE_INLINE uint64_t getblock64(const uint8_t *p, uint32_t i)
{
    uint64_t s = 0;
    for (uint32_t n = 0; n < 8; n++) {
        s |= ((uint64_t)p[8 * i + n]) << (n * 8);
    }
    return s;
}
 
// Finalization mix - force all bits of a hash block to avalanche
__host__ __device__
FORCE_INLINE uint64_t fmix64(uint64_t k)
{
    k ^= k >> 33;
    k *= BIG_CONSTANT(0xff51afd7ed558ccd);
    k ^= k >> 33;
    k *= BIG_CONSTANT(0xc4ceb9fe1a85ec53);
    k ^= k >> 33;
    return k;
}
 
// Compute the 128-bit MurmurHash3 x64 hash (host and device)
__host__ __device__
void MurmurHash3_x64_128(const void *key, const uint32_t len,
                           const uint32_t seed, void *out)
{
    const uint8_t *data = (const uint8_t*)key;
    const uint32_t nblocks = len / 16;
 
    uint64_t h1 = seed;
    uint64_t h2 = seed;
 
    // Constants (same for host and device)
    const uint64_t c1 = BIG_CONSTANT(0x87c37b91114253d5);
    const uint64_t c2 = BIG_CONSTANT(0x4cf5ad432745937f);
 
    // Body
    for (uint32_t i = 0; i < nblocks; i++)
    {
        uint64_t k1 = getblock64(data, i * 2 + 0);
        uint64_t k2 = getblock64(data, i * 2 + 1);
 
        k1 *= c1; k1  = rotl64(k1, 31); k1 *= c2; h1 ^= k1;
 
        h1 = rotl64(h1, 27); h1 += h2; h1 = h1 * 5 + 0x52dce729;
 
        k2 *= c2; k2  = rotl64(k2, 33); k2 *= c1; h2 ^= k2;
 
        h2 = rotl64(h2, 31); h2 += h1; h2 = h2 * 5 + 0x38495ab5;
    }
 
    // Tail
    const uint8_t *tail = data + nblocks * 16;
 
    uint64_t k1 = 0;
    uint64_t k2 = 0;
 
    switch(len & 15)
    {
        case 15: k2 ^= ((uint64_t)tail[14]) << 48;
        case 14: k2 ^= ((uint64_t)tail[13]) << 40;
        case 13: k2 ^= ((uint64_t)tail[12]) << 32;
        case 12: k2 ^= ((uint64_t)tail[11]) << 24;
        case 11: k2 ^= ((uint64_t)tail[10]) << 16;
        case 10: k2 ^= ((uint64_t)tail[9]) << 8;
        case  9: k2 ^= ((uint64_t)tail[8]) << 0;
                 k2 *= c2; k2  = rotl64(k2, 33); k2 *= c1; h2 ^= k2;
 
        case  8: k1 ^= ((uint64_t)tail[7]) << 56;
        case  7: k1 ^= ((uint64_t)tail[6]) << 48;
        case  6: k1 ^= ((uint64_t)tail[5]) << 40;
        case  5: k1 ^= ((uint64_t)tail[4]) << 32;
        case  4: k1 ^= ((uint64_t)tail[3]) << 24;
        case  3: k1 ^= ((uint64_t)tail[2]) << 16;
        case  2: k1 ^= ((uint64_t)tail[1]) << 8;
        case  1: k1 ^= ((uint64_t)tail[0]) << 0;
                 k1 *= c1; k1  = rotl64(k1, 31); k1 *= c2; h1 ^= k1;
    }
 
    // Finalization mix
    h1 ^= len; h2 ^= len;
 
    h1 += h2;
    h2 += h1;
 
    h1 = fmix64(h1);
    h2 = fmix64(h2);
 
    h1 += h2;
    h2 += h1;
 
    ((uint64_t*)out)[0] = h1;
    ((uint64_t*)out)[1] = h2;
}
 
// -------------------------------------------------------------------------
// Kernel: Each thread computes the hash for one key over gRepeat iterations.
// Note: d_offsets holds cumulative byte offsets for each key.
//       keyLengths holds individual key lengths.
__global__
void MurmurHash3_x64_128_kernel(const uint8_t *__restrict__ d_keys,
                                  const uint32_t *__restrict__ d_offsets,
                                  const uint32_t *__restrict__ keyLengths,
                                  uint64_t *__restrict__ d_out,
                                  const uint32_t numKeys)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numKeys) {
        uint64_t hash[2];
        // Repeat the hash computation in a loop as specified by gRepeat.
        for (uint32_t r = 0; r < gRepeat; r++) {
            MurmurHash3_x64_128(d_keys + d_offsets[i], keyLengths[i], i, hash);
        }
        // Store the result from the final iteration.
        d_out[i * 2]     = hash[0];
        d_out[i * 2 + 1] = hash[1];
    }
}
 
int main(int argc, char** argv)
{
    if (argc != 3) {
        printf("Usage: %s <number of keys> <repeat>\n", argv[0]);
        return 1;
    }
    uint32_t numKeys = atoi(argv[1]);
    uint32_t repeat = atoi(argv[2]);
 
    srand(3);
    uint32_t i;
    // Allocate host arrays for key lengths, keys and out hash values
    uint32_t* length = (uint32_t*)malloc(sizeof(uint32_t) * numKeys);
    uint8_t** keys = (uint8_t**)malloc(sizeof(uint8_t*) * numKeys);
    uint64_t** out = (uint64_t**)malloc(sizeof(uint64_t*) * numKeys);
 
    // Generate keys and compute their hash on the host for verification
    for (i = 0; i < numKeys; i++) {
        length[i] = rand() % 10000;
        keys[i] = (uint8_t*)malloc(length[i]);
        out[i] = (uint64_t*)malloc(2 * sizeof(uint64_t));
        for (uint32_t c = 0; c < length[i]; c++) {
            keys[i][c] = static_cast<uint8_t>(c % 256);
        }
        MurmurHash3_x64_128(keys[i], length[i], i, out[i]);
#ifdef DEBUG
        printf("%lu %lu\n", out[i][0], out[i][1]);
#endif
    }
 
    // Create one-dimensional arrays for device offloading
    // d_offsets: cumulative byte offsets, keyLengths: individual key sizes
    uint32_t* h_offsets;
    cudaHostAlloc((void**)&h_offsets, sizeof(uint32_t) * (numKeys + 1), cudaHostAllocDefault);
    uint32_t total_length = 0;
    h_offsets[0] = 0;
    for (uint32_t j = 0; j < numKeys; j++) {
        total_length += length[j];
        h_offsets[j + 1] = total_length;
    }
 
    // Allocate a single contiguous host buffer for keys using pinned memory
    uint8_t* h_keys;
    cudaHostAlloc((void**)&h_keys, sizeof(uint8_t) * total_length, cudaHostAllocDefault);
    for (uint32_t j = 0; j < numKeys; j++) {
        memcpy(h_keys + h_offsets[j], keys[j], length[j]);
    }
 
    // Sanity check that key data was copied correctly
    for (uint32_t j = 0; j < numKeys; j++) {
        assert(0 == memcmp(h_keys + h_offsets[j], keys[j], length[j]));
    }
 
    // Create a CUDA stream for asynchronous operations
    cudaStream_t stream;
    cudaStreamCreate(&stream);
 
    // Allocate device memory using cudaMalloc
    uint8_t *dev_keys;
    cudaMalloc((void**)&dev_keys, sizeof(uint8_t) * total_length);
    cudaMemcpyAsync(dev_keys, h_keys, sizeof(uint8_t) * total_length, cudaMemcpyHostToDevice, stream);
 
    // d_offsets will hold the starting byte offset for each key (numKeys+1 entries)
    uint32_t *dev_offsets;
    cudaMalloc((void**)&dev_offsets, sizeof(uint32_t) * (numKeys + 1));
    cudaMemcpyAsync(dev_offsets, h_offsets, sizeof(uint32_t) * (numKeys + 1), cudaMemcpyHostToDevice, stream);
 
    // keyLengths on device: holds each key's length
    uint32_t* d_keyLengths;
    cudaMalloc((void**)&d_keyLengths, sizeof(uint32_t) * numKeys);
    cudaMemcpyAsync(d_keyLengths, length, sizeof(uint32_t) * numKeys, cudaMemcpyHostToDevice, stream);
 
    // Allocate output array on device
    uint64_t* dev_out;
    cudaMalloc((void**)&dev_out, sizeof(uint64_t) * numKeys * 2);
 
    // Ensure all asynchronous copies have completed before kernel execution
    cudaStreamSynchronize(stream);
 
    // Copy the repeat value to device constant memory
    cudaMemcpyToSymbol(gRepeat, &repeat, sizeof(uint32_t));
 
    // Use grid dimensions calculated appropriately (one thread per key)
    dim3 blockDim(BLOCK_SIZE);
    dim3 gridDim((numKeys + BLOCK_SIZE - 1) / BLOCK_SIZE);
 
    // Create CUDA events for accurate timing and to allow the host to sleep
    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
 
    cudaStreamSynchronize(stream);
    cudaEventRecord(startEvent, stream);
 
    // Launch the kernel only once. The device repeat loop covers the specified iterations.
    MurmurHash3_x64_128_kernel<<<gridDim, blockDim, 0, stream>>>(dev_keys, dev_offsets, d_keyLengths, dev_out, numKeys);
 
    cudaEventRecord(stopEvent, stream);
    cudaEventSynchronize(stopEvent);
 
    float elapsed_ms;
    cudaEventElapsedTime(&elapsed_ms, startEvent, stopEvent);
    printf("Average kernel execution time %f (s)\n", (elapsed_ms * 1e-3f) / repeat);
 
    // Copy device output back to host using pinned memory and asynchronous copy
    uint64_t* h_out;
    cudaHostAlloc((void**)&h_out, sizeof(uint64_t) * numKeys * 2, cudaHostAllocDefault);
    cudaMemcpyAsync(h_out, dev_out, sizeof(uint64_t) * numKeys * 2, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
 
    // Verify the results match the precomputed host hashes
    bool error = false;
    for (uint32_t j = 0; j < numKeys; j++) {
        if (h_out[2 * j] != out[j][0] || h_out[2 * j + 1] != out[j][1]) {
            error = true;
            break;
        }
    }
    if (error)
        printf("FAIL\n");
    else
        printf("SUCCESS\n");
 
    // Clean up host memory
    for (uint32_t j = 0; j < numKeys; j++) {
        free(out[j]);
        free(keys[j]);
    }
    free(keys);
    free(out);
    free(length);
    cudaFreeHost(h_keys);
    cudaFreeHost(h_out);
    cudaFreeHost(h_offsets);
 
    // Clean up device memory
    cudaFree(dev_keys);
    cudaFree(dev_offsets);
    cudaFree(dev_out);
    cudaFree(d_keyLengths);
 
    // Destroy CUDA events and stream
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
    cudaStreamDestroy(stream);
 
    return 0;
}
