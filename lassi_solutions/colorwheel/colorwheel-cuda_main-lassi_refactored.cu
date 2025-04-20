
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda.h>

#define RY  15
#define YG   6
#define GC   4
#define CB  11
#define BM  13
#define MR   6
#define MAXCOLS  (RY + YG + GC + CB + BM + MR)
typedef unsigned char uchar;

// __constant__ memory storage for the precomputed color wheel
__constant__ int constantCW[MAXCOLS * 3];

// Precomputed reciprocal constant for 255.f.
__device__ __host__ inline float inv255() { return 1.0f / 255.0f; }

// This version of computeColor is used for host (and correctness checking).
// It recomputes the color wheel for each call so that the results are identical.
__host__ __device__
void computeColor(float fx, float fy, uchar *pix)
{
  int cw[MAXCOLS][3];
  int k = 0;
  for (int i = 0; i < RY; i++) { cw[k][0] = 255; cw[k][1] = 255 * i / RY; cw[k][2] = 0; k++; }
  for (int i = 0; i < YG; i++) { cw[k][0] = 255 - 255 * i / YG; cw[k][1] = 255; cw[k][2] = 0; k++; }
  for (int i = 0; i < GC; i++) { cw[k][0] = 0; cw[k][1] = 255; cw[k][2] = 255 * i / GC; k++; }
  for (int i = 0; i < CB; i++) { cw[k][0] = 0; cw[k][1] = 255 - 255 * i / CB; cw[k][2] = 255; k++; }
  for (int i = 0; i < BM; i++) { cw[k][0] = 255 * i / BM; cw[k][1] = 0; cw[k][2] = 255; k++; }
  for (int i = 0; i < MR; i++) { cw[k][0] = 255; cw[k][1] = 0; cw[k][2] = 255 - 255 * i / MR; k++; }
  
  float rad = sqrtf(fx * fx + fy * fy);
  float a = atan2f(-fy, -fx) / (float)M_PI;
  float fk = (a + 1.f) / 2.f * (MAXCOLS - 1);
  int k0 = (int)fk;
  int k1 = (k0 + 1) % MAXCOLS;
  float f = fk - k0;
  float inv = inv255();
  for (int b = 0; b < 3; b++) {
    float col0 = cw[k0][b] * inv;
    float col1 = cw[k1][b] * inv;
    float col = (1.f - f) * col0 + f * col1;
    if (rad <= 1)
      col = 1.f - rad * (1.f - col); // increase saturation with radius
    else
      col *= .75f; // out of range
    pix[2 - b] = (int)(255.f * col);
  }
}

// This device–only version uses the precomputed constant memory color wheel.
// Although it still accepts a "cw" pointer to preserve the function signature,
// it reads from constantCW instead.
__device__ inline void computeColorShared(float fx, float fy, uchar *pix, const int* cw)
{
  // Use fast math intrinsics for lower latency and energy consumption.
  float rad = __fsqrt_rn(fx * fx + fy * fy);
  float a = atan2f(-fy, -fx) / (float)M_PI;
  float fk = (a + 1.f) / 2.f * (MAXCOLS - 1);
  int k0 = (int)fk;
  int k1 = (k0 + 1) % MAXCOLS;
  float f = fk - k0;
  float inv = inv255();
  for (int b = 0; b < 3; b++) {
    float col0 = constantCW[k0 * 3 + b] * inv;
    float col1 = constantCW[k1 * 3 + b] * inv;
    float col = (1.f - f) * col0 + f * col1;
    if (rad <= 1)
      col = 1.f - rad * (1.f - col);
    else
      col *= 0.75f;
    pix[2 - b] = (int)(255.f * col);
  }
}

__global__
void color(uchar* pix, int size, int half_size, float range, float truerange)
{
  int y = blockDim.y * blockIdx.y + threadIdx.y;
  int x = blockDim.x * blockIdx.x + threadIdx.x;

  if (y < size && x < size) {
    float fx = ((float)x / half_size) * range - range;
    float fy = ((float)y / half_size) * range - range;
    if (x == half_size || y == half_size) return; // leave coordinate axes black
    size_t idx = (y * size + x) * 3;
    computeColorShared(fx / truerange, fy / truerange, pix + idx, constantCW);
  }
}

int main(int argc, char **argv)
{
  if (argc != 4) {
    printf("Usage: %s <range> <size> <repeat>\n", argv[0]);
    exit(1);
  }
  const float truerange = atof(argv[1]);
  const int size = atoi(argv[2]);
  const int repeat = atoi(argv[3]);

  float range = 1.04f * truerange;
  const int half_size = size / 2;

  size_t imgSize = size * size * 3;
  uchar *pix, *res;
  cudaHostAlloc((void**)&pix, imgSize, cudaHostAllocDefault);
  cudaHostAlloc((void**)&res, imgSize, cudaHostAllocDefault);
  memset(pix, 0, imgSize);

  for (int y = 0; y < size; y++) {
    for (int x = 0; x < size; x++) {
      float fx = ((float)x / half_size) * range - range;
      float fy = ((float)y / half_size) * range - range;
      if (x == half_size || y == half_size) continue; // coordinate axes remain black
      size_t idx = (y * size + x) * 3;
      computeColor(fx / truerange, fy / truerange, pix + idx);
    }
  }

  int hostCW[MAXCOLS * 3];
  int k = 0;
  for (int i = 0; i < RY; i++) {
    hostCW[k * 3 + 0] = 255;
    hostCW[k * 3 + 1] = 255 * i / RY;
    hostCW[k * 3 + 2] = 0;
    k++;
  }
  for (int i = 0; i < YG; i++) {
    hostCW[k * 3 + 0] = 255 - 255 * i / YG;
    hostCW[k * 3 + 1] = 255;
    hostCW[k * 3 + 2] = 0;
    k++;
  }
  for (int i = 0; i < GC; i++) {
    hostCW[k * 3 + 0] = 0;
    hostCW[k * 3 + 1] = 255;
    hostCW[k * 3 + 2] = 255 * i / GC;
    k++;
  }
  for (int i = 0; i < CB; i++) {
    hostCW[k * 3 + 0] = 0;
    hostCW[k * 3 + 1] = 255 - 255 * i / CB;
    hostCW[k * 3 + 2] = 255;
    k++;
  }
  for (int i = 0; i < BM; i++) {
    hostCW[k * 3 + 0] = 255 * i / BM;
    hostCW[k * 3 + 1] = 0;
    hostCW[k * 3 + 2] = 255;
    k++;
  }
  for (int i = 0; i < MR; i++) {
    hostCW[k * 3 + 0] = 255;
    hostCW[k * 3 + 1] = 0;
    hostCW[k * 3 + 2] = 255 - 255 * i / MR;
    k++;
  }
  cudaMemcpyToSymbol(constantCW, hostCW, sizeof(hostCW));

  printf("Start execution on a device\n");
  uchar *d_pix;
  cudaMalloc((void**)&d_pix, imgSize);
  cudaMemset(d_pix, 0, imgSize);

  dim3 grids((size + 15) / 16, (size + 15) / 16);
  dim3 blocks(16, 16);

  cudaDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    color<<<grids, blocks>>>(d_pix, size, half_size, range, truerange);
  }

  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time : %f (ms)\n", (time * 1e-6f) / repeat);

  cudaMemcpy(res, d_pix, imgSize, cudaMemcpyDeviceToHost);

  int fail = memcmp(pix, res, imgSize);
  if (fail) {
    int max_error = 0;
    for (size_t i = 0; i < imgSize; i++) {
       int e = abs(res[i] - pix[i]);
       if (e > max_error) max_error = e;
    }
    printf("Maximum error between host and device results: %d\n", max_error);
  }
  else {
    printf("%s\n", "PASS");
  }
  
  cudaFree(d_pix);
  cudaFreeHost(pix);
  cudaFreeHost(res);
  return 0;
}
