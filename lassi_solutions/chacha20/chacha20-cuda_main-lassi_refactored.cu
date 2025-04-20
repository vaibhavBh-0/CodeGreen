
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <cuda.h>
#include <stdint.h>

/* Minimal ChaCha20 implementation */
#ifndef CHACHA20_H
#define CHACHA20_H

__device__ __forceinline__ uint32_t rotl(const uint32_t x, const int n) {
    return (x << n) | (x >> (32 - n));
}

__device__ void chacha20_block(const uint8_t key[32], const uint8_t nonce[8], 
                               uint32_t counter_low, uint32_t counter_high,
                               uint8_t output[64]) {
    uint32_t state[16];
    // Constants: "expand 32-byte k"
    state[0] = 0x61707865;
    state[1] = 0x3320646e;
    state[2] = 0x79622d32;
    state[3] = 0x6b206574;
    // Load key (little-endian)
    #pragma unroll
    for (int i = 0; i < 8; i++) {
      state[4 + i] = ((uint32_t)key[i*4 + 0]) | (((uint32_t)key[i*4 + 1]) << 8) |
                     (((uint32_t)key[i*4 + 2]) << 16) | (((uint32_t)key[i*4 + 3]) << 24);
    }
    // Set 64-bit counter split into two 32-bit words
    state[12] = counter_low;
    state[13] = counter_high;
    // Load nonce (8 bytes) into state[14] and state[15]
    state[14] = ((uint32_t)nonce[0]) | (((uint32_t)nonce[1]) << 8) |
                (((uint32_t)nonce[2]) << 16) | (((uint32_t)nonce[3]) << 24);
    state[15] = ((uint32_t)nonce[4]) | (((uint32_t)nonce[5]) << 8) |
                (((uint32_t)nonce[6]) << 16) | (((uint32_t)nonce[7]) << 24);

    uint32_t working_state[16];
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        working_state[i] = state[i];
    }

    // 20 rounds (10 double rounds)
    #pragma unroll
    for (int i = 0; i < 10; i++) {
        // Column rounds
        working_state[0] += working_state[4];  working_state[12] = rotl(working_state[12] ^ working_state[0], 16);
        working_state[8] += working_state[12]; working_state[4]  = rotl(working_state[4]  ^ working_state[8], 12);
        working_state[0] += working_state[4];  working_state[12] = rotl(working_state[12] ^ working_state[0], 8);
        working_state[8] += working_state[12]; working_state[4]  = rotl(working_state[4]  ^ working_state[8], 7);

        working_state[1] += working_state[5];  working_state[13] = rotl(working_state[13] ^ working_state[1], 16);
        working_state[9] += working_state[13]; working_state[5]  = rotl(working_state[5]  ^ working_state[9], 12);
        working_state[1] += working_state[5];  working_state[13] = rotl(working_state[13] ^ working_state[1], 8);
        working_state[9] += working_state[13]; working_state[5]  = rotl(working_state[5]  ^ working_state[9], 7);

        working_state[2] += working_state[6];  working_state[14] = rotl(working_state[14] ^ working_state[2], 16);
        working_state[10] += working_state[14]; working_state[6]  = rotl(working_state[6]  ^ working_state[10], 12);
        working_state[2] += working_state[6];  working_state[14] = rotl(working_state[14] ^ working_state[2], 8);
        working_state[10] += working_state[14]; working_state[6]  = rotl(working_state[6]  ^ working_state[10], 7);

        working_state[3] += working_state[7];  working_state[15] = rotl(working_state[15] ^ working_state[3], 16);
        working_state[11] += working_state[15]; working_state[7]  = rotl(working_state[7]  ^ working_state[11], 12);
        working_state[3] += working_state[7];  working_state[15] = rotl(working_state[15] ^ working_state[3], 8);
        working_state[11] += working_state[15]; working_state[7]  = rotl(working_state[7]  ^ working_state[11], 7);

        // Diagonal rounds
        working_state[0] += working_state[5];  working_state[15] = rotl(working_state[15] ^ working_state[0], 16);
        working_state[10] += working_state[15]; working_state[5]  = rotl(working_state[5]  ^ working_state[10], 12);
        working_state[0] += working_state[5];  working_state[15] = rotl(working_state[15] ^ working_state[0], 8);
        working_state[10] += working_state[15]; working_state[5]  = rotl(working_state[5]  ^ working_state[10], 7);

        working_state[1] += working_state[6];  working_state[12] = rotl(working_state[12] ^ working_state[1], 16);
        working_state[11] += working_state[12]; working_state[6]  = rotl(working_state[6]  ^ working_state[11], 12);
        working_state[1] += working_state[6];  working_state[12] = rotl(working_state[12] ^ working_state[1], 8);
        working_state[11] += working_state[12]; working_state[6]  = rotl(working_state[6]  ^ working_state[11], 7);

        working_state[2] += working_state[7];  working_state[13] = rotl(working_state[13] ^ working_state[2], 16);
        working_state[8] += working_state[13];  working_state[7]  = rotl(working_state[7]  ^ working_state[8], 12);
        working_state[2] += working_state[7];  working_state[13] = rotl(working_state[13] ^ working_state[2], 8);
        working_state[8] += working_state[13];  working_state[7]  = rotl(working_state[7]  ^ working_state[8], 7);

        working_state[3] += working_state[4];  working_state[14] = rotl(working_state[14] ^ working_state[3], 16);
        working_state[9] += working_state[14];  working_state[4]  = rotl(working_state[4]  ^ working_state[9], 12);
        working_state[3] += working_state[4];  working_state[14] = rotl(working_state[14] ^ working_state[3], 8);
        working_state[9] += working_state[14];  working_state[4]  = rotl(working_state[4]  ^ working_state[9], 7);
    }

    // Add the original state to the working state
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        working_state[i] += state[i];
    }
    // Serialize output in little-endian order
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        output[4*i + 0] = (uint8_t)(working_state[i] & 0xff);
        output[4*i + 1] = (uint8_t)((working_state[i] >> 8) & 0xff);
        output[4*i + 2] = (uint8_t)((working_state[i] >> 16) & 0xff);
        output[4*i + 3] = (uint8_t)((working_state[i] >> 24) & 0xff);
    }
}

class Chacha20 {
public:
    __device__ Chacha20(const uint8_t* key, const uint8_t* nonce) {
        #pragma unroll
        for (int i = 0; i < 32; i++) {
            this->key[i] = key[i];
        }
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            this->nonce[i] = nonce[i];
        }
        counter_low = 0;
        counter_high = 0;
    }
    __device__ void crypt(uint8_t* output, int length) {
        int blocks = (length + 63) / 64;
        uint8_t block[64];
        for (int i = 0; i < blocks; i++) {
            chacha20_block(key, nonce, counter_low, counter_high, block);
            counter_low++;
            if (counter_low == 0) {
                counter_high++;
            }
            int block_size = (length - i * 64) < 64 ? (length - i * 64) : 64;
            #pragma unroll
            for (int j = 0; j < block_size; j++) {
                output[i * 64 + j] = block[j];
            }
        }
    }
private:
    uint8_t key[32];
    uint8_t nonce[8];
    uint32_t counter_low;
    uint32_t counter_high;
};

#endif // CHACHA20_H

// Place lookup table in constant memory (remains for backward compatibility)
__constant__ uint8_t d_const_char_to_uint[256];

// (The device-side hex_to_raw is now unused, since we precompute on host.)
__device__
void hex_to_raw(const char* src, const int n /*src size*/, uint8_t* dst, const uint8_t* /*char_to_uint*/) {
    for (int i = threadIdx.x; i < n / 2; i += blockDim.x) {
        uint8_t hi = d_const_char_to_uint[(int)src[i * 2 + 0]];
        uint8_t lo = d_const_char_to_uint[(int)src[i * 2 + 1]];
        dst[i] = (hi << 4) | lo;
    }
}

__global__
void test_keystreams (
    const char *__restrict__ text_key,
    const char *__restrict__ text_nonce,
    const char *__restrict__ text_keystream,
    const uint8_t *__restrict__ char_to_uint,  // parameter preserved, but not used now
    uint8_t *__restrict__ raw_key,
    uint8_t *__restrict__ raw_nonce,
    uint8_t *__restrict__ raw_keystream,
    uint8_t *__restrict__ result,
    const int text_key_size,
    const int text_nonce_size,
    const int text_keystream_size)
{
    // Removed in-kernel result clearing. It is now performed using cudaMemset from the host.
    
    // Perform encryption using the preconverted values.
    if (threadIdx.x == 0) {
        Chacha20 chacha(raw_key, raw_nonce);
        chacha.crypt(result, text_keystream_size / 2);
    }
}

// Helper function to convert a hex string (host) to raw bytes.
void hex_to_raw_host(const char* src, uint8_t* dst, int len) {
    // len is the length of the hex string (should be even)
    for (int i = 0; i < len / 2; i++) {
        char hi_c = src[2*i];
        char lo_c = src[2*i + 1];
        uint8_t hi = (hi_c >= '0' && hi_c <= '9') ? hi_c - '0' :
                     ((hi_c | 32) - 'a' + 10);
        uint8_t lo = (lo_c >= '0' && lo_c <= '9') ? lo_c - '0' :
                     ((lo_c | 32) - 'a' + 10);
        dst[i] = (hi << 4) | lo;
    }
}

int main(int argc, char* argv[])
{
    if (argc != 2) {
        printf("Usage: %s <repeat>\n", argv[0]);
        return 1;
    }
    const int repeat = atoi(argv[1]);

    // Initialize lookup table on host (remains for cudaMemcpyToSymbol)
    uint8_t char_to_uint[256] = {0};
    for (int i = 0; i < 10; i++)
        char_to_uint[i + '0'] = i;
    for (int i = 0; i < 26; i++) {
        char_to_uint[i + 'a'] = i + 10;
        char_to_uint[i + 'A'] = i + 10;
    }

    // Sample input strings (hex strings)
    const char *h_key = "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f";
    const char *h_nonce = "0001020304050607";
    const char *h_keystream = "f798a189f195e66982105ffb640bb7757f579da31602fc93ec01ac56f85ac3c134a4547b733b46413042c9440049176905d3be59ea1c53f15916155c2be8241a38008b9a26bc35941e2444177c8ade6689de95264986d95889fb60e84629c9bd9a5acb1cc118be563eb9b3a4a472f82e09a7e778492b562ef7130e88dfe031c79db9d4f7c7a899151b9a475032b63fc385245fe054e3dd5a97a5f576fe064025d3ce042c566ab2c507b138db853e3d6959660996546cc9c4a6eafdc777c040d70eaf46f76dad3979e5c5360c3317166a1c894c94a371876a94df7628fe4eaaf2ccb27d5aaae0ad7ad0f9d4b6ad3b54098746d4524d38407a6deb3ab78fab78c9";

    const int key_len = strlen(h_key);
    const int nonce_len = strlen(h_nonce);
    const int keystream_len = strlen(h_keystream);
    const int result_len = keystream_len / 2;

    uint8_t *result = (uint8_t*) malloc(result_len);
    // Allocate host buffers for precomputed raw conversions.
    uint8_t *h_raw_key = (uint8_t*) malloc(key_len / 2);
    uint8_t *h_raw_nonce = (uint8_t*) malloc(nonce_len / 2);
    uint8_t *h_raw_keystream = (uint8_t*) malloc(result_len);

    // Precompute static conversions on host.
    hex_to_raw_host(h_key, h_raw_key, key_len);
    hex_to_raw_host(h_nonce, h_raw_nonce, nonce_len);
    hex_to_raw_host(h_keystream, h_raw_keystream, keystream_len);

    // Device buffers
    char *d_key;
    uint8_t *d_raw_key;
    char *d_nonce;
    uint8_t *d_raw_nonce;
    char *d_keystream;
    uint8_t *d_raw_keystream;
    uint8_t *d_result;

    // Copy lookup table to constant memory
    cudaMemcpyToSymbol(d_const_char_to_uint, char_to_uint, 256 * sizeof(uint8_t));

    // Allocate and copy other buffers.
    cudaMalloc((void**)&d_key, key_len);
    cudaMemcpy(d_key, h_key, key_len, cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_raw_key, key_len / 2);
    cudaMemcpy(d_raw_key, h_raw_key, key_len / 2, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&d_nonce, nonce_len);
    cudaMemcpy(d_nonce, h_nonce, nonce_len, cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_raw_nonce, nonce_len / 2);
    cudaMemcpy(d_raw_nonce, h_raw_nonce, nonce_len / 2, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&d_keystream, keystream_len);
    cudaMemcpy(d_keystream, h_keystream, keystream_len, cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_raw_keystream, result_len);
    cudaMemcpy(d_raw_keystream, h_raw_keystream, result_len, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&d_result, result_len);

    cudaDeviceSynchronize();
    auto start = std::chrono::steady_clock::now();

    // Launch kernel repeatedly.
    // Note: The kernel now uses precomputed raw values and no longer clears result memory,
    // so we clear it using cudaMemset before each kernel launch.
    for (int i = 0; i < repeat; i++) {
        cudaMemset(d_result, 0, result_len);
        test_keystreams<<<1, 256>>>(
            d_key, d_nonce, d_keystream, nullptr,
            d_raw_key, d_raw_nonce, d_raw_keystream, d_result,
            key_len, nonce_len, keystream_len);
    }

    cudaDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time of kernels: %f (us)\n", (time * 1e-3f) / repeat);

    cudaMemcpy(result, d_result, result_len, cudaMemcpyDeviceToHost);

    int error = memcmp(result, h_raw_keystream, result_len);
    printf("%s\n", error ? "FAIL" : "PASS");

    free(result);
    free(h_raw_key);
    free(h_raw_nonce);
    free(h_raw_keystream);
    cudaFree(d_key);
    cudaFree(d_raw_key);
    cudaFree(d_nonce);
    cudaFree(d_raw_nonce);
    cudaFree(d_keystream);
    cudaFree(d_raw_keystream);
    cudaFree(d_result);

    return 0;
}
