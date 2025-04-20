
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <chrono>
#include <cuda.h>

#define NUM 512
#define BLOCK_SIZE 128

#define DOUBLE

#ifdef DOUBLE
#define Real double

#define ZERO 0.0
#define ONE 1.0
#define TWO 2.0
#define FOUR 4.0

#define SMALL 1.0e-10;

/** Reynolds number */
const Real Re_num = 1000.0;

/** SOR relaxation parameter */
const Real omega = 1.7;

/** Discretization mixture parameter (gamma) */
const Real mix_param = 0.9;

/** Safety factor for time step modification */
const Real tau = 0.5;

/** Body forces in x- and y- directions */
const Real gx = 0.0;
const Real gy = 0.0;

/** Domain size (non-dimensional) */
#define xLength 1.0
#define yLength 1.0
#else
#define Real float

#undef fmin
#define fmin fminf
#undef fmax
#define fmax fmaxf
#undef fabs
#define fabs fabsf
#undef sqrt
#define sqrt sqrtf

#define ZERO 0.0f
#define ONE 1.0f
#define TWO 2.0f
#define FOUR 4.0f
#define SMALL 1.0e-10f;

const Real Re_num = 1000.0f;
const Real omega = 1.7f;
const Real mix_param = 0.9f;
const Real tau = 0.5f;
const Real gx = 0.0f;
const Real gy = 0.0f;
#define xLength 1.0f
#define yLength 1.0f
#endif

const Real dx = xLength / NUM;
const Real dy = yLength / NUM;

#define NUM_2 (NUM >> 1)

#define u(I, J) u[((I) * ((NUM) + 2)) + (J)]
#define v(I, J) v[((I) * ((NUM) + 2)) + (J)]
#define F(I, J) F[((I) * ((NUM) + 2)) + (J)]
#define G(I, J) G[((I) * ((NUM) + 2)) + (J)]
#define pres_red(I, J) pres_red[((I) * ((NUM_2) + 2)) + (J)]
#define pres_black(I, J) pres_black[((I) * ((NUM_2) + 2)) + (J)]

///////////////////////////////////////////////////////////////////////////////
__host__
void set_BCs_host (Real* u, Real* v) 
{
  int ind;
  for (ind = 0; ind < NUM + 2; ++ind) {
    u(0, ind) = ZERO;
    v(0, ind) = -v(1, ind);
    u(NUM, ind) = ZERO;
    v(NUM + 1, ind) = -v(NUM, ind);
    u(ind, 0) = -u(ind, 1);
    v(ind, 0) = ZERO;
    u(ind, NUM + 1) = TWO - u(ind, NUM);
    v(ind, NUM) = ZERO;

    if (ind == NUM) {
      u(0, 0) = ZERO;
      v(0, 0) = -v(1, 0);
      u(0, NUM + 1) = ZERO;
      v(0, NUM + 1) = -v(1, NUM + 1);
      u(NUM, 0) = ZERO;
      v(NUM + 1, 0) = -v(NUM, 0);
      u(NUM, NUM + 1) = ZERO;
      v(NUM + 1, NUM + 1) = -v(NUM, NUM + 1);
      u(0, 0) = -u(0, 1);
      v(0, 0) = ZERO;
      u(NUM + 1, 0) = -u(NUM + 1, 1);
      v(NUM + 1, 0) = ZERO;
      u(0, NUM + 1) = TWO - u(0, NUM);
      v(0, NUM) = ZERO;
      u(NUM + 1, NUM + 1) = TWO - u(NUM + 1, NUM);
      v(ind, NUM + 1) = ZERO;
    }
  }
}

///////////////////////////////////////////////////////////////////////////////
__global__
void set_BCs (Real* __restrict__ u, Real* __restrict__ v) 
{
  int ind = (blockIdx.x * blockDim.x) + threadIdx.x + 1;
  u(0, ind) = ZERO;
  v(0, ind) = -v(1, ind);
  u(NUM, ind) = ZERO;
  v(NUM + 1, ind) = -v(NUM, ind);
  u(ind, 0) = -u(ind, 1);
  v(ind, 0) = ZERO;
  u(ind, NUM + 1) = TWO - u(ind, NUM);
  v(ind, NUM) = ZERO;

  if (ind == NUM) {
    u(0, 0) = ZERO;
    v(0, 0) = -v(1, 0);
    u(0, NUM + 1) = ZERO;
    v(0, NUM + 1) = -v(1, NUM + 1);
    u(NUM, 0) = ZERO;
    v(NUM + 1, 0) = -v(NUM, 0);
    u(NUM, NUM + 1) = ZERO;
    v(NUM + 1, NUM + 1) = -v(NUM, NUM + 1);
    u(0, 0) = -u(0, 1);
    v(0, 0) = ZERO;
    u(NUM + 1, 0) = -u(NUM + 1, 1);
    v(NUM + 1, 0) = ZERO;
    u(0, NUM + 1) = TWO - u(0, NUM);
    v(0, NUM) = ZERO;
    u(NUM + 1, NUM + 1) = TWO - u(NUM + 1, NUM);
    v(ind, NUM + 1) = ZERO;
  }
}

///////////////////////////////////////////////////////////////////////////////
__global__ 
void calculate_F (const Real dt,
                  const Real* __restrict__ u,
                  const Real* __restrict__ v,
                        Real* __restrict__ F) 
{  
  int row = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int col = blockIdx.y * blockDim.y + threadIdx.y + 1;
  
  // Return if thread is out-of-bound
  if (row > NUM || col > NUM) return;

  if (col == NUM) {
    F(0, row) = u(0, row);
    F(NUM, row) = u(NUM, row);
  } else {
    Real u_ij = u(col, row);
    Real u_ip1j = u(col + 1, row);
    Real u_ijp1 = u(col, row + 1);
    Real u_im1j = u(col - 1, row);
    Real u_ijm1 = u(col, row - 1);

    Real v_ij = v(col, row);
    Real v_ip1j = v(col + 1, row);
    Real v_ijm1 = v(col, row - 1);
    Real v_ip1jm1 = v(col + 1, row - 1);

    Real du2dx, duvdy, d2udx2, d2udy2;

    du2dx = (((u_ij + u_ip1j) * (u_ij + u_ip1j) - (u_im1j + u_ij) * (u_im1j + u_ij))
        + mix_param * (fabs(u_ij + u_ip1j) * (u_ij - u_ip1j)
          - fabs(u_im1j + u_ij) * (u_im1j - u_ij))) / (FOUR * dx);
    duvdy = ((v_ij + v_ip1j) * (u_ij + u_ijp1) - (v_ijm1 + v_ip1jm1) * (u_ijm1 + u_ij)
        + mix_param * (fabs(v_ij + v_ip1j) * (u_ij - u_ijp1)
          - fabs(v_ijm1 + v_ip1jm1) * (u_ijm1 - u_ij))) / (FOUR * dy);
    d2udx2 = (u_ip1j - (TWO * u_ij) + u_im1j) / (dx * dx);
    d2udy2 = (u_ijp1 - (TWO * u_ij) + u_ijm1) / (dy * dy);

    F(col, row) = u_ij + dt * (((d2udx2 + d2udy2) / Re_num) - du2dx - duvdy + gx);
  }
}

///////////////////////////////////////////////////////////////////////////////
__global__ 
void calculate_G (const Real dt,
                  const Real* __restrict__ u,
                  const Real* __restrict__ v,
                        Real* __restrict__ G) 
{
  int row = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int col = blockIdx.y * blockDim.y + threadIdx.y + 1;
  
  // Return if thread is out-of-bound
  if (row > NUM || col > NUM) return;

  if (row == NUM) {
    G(col, 0) = v(col, 0);
    G(col, NUM) = v(col, NUM);
  } else {
    Real u_ij = u(col, row);
    Real u_ijp1 = u(col, row + 1);
    Real u_im1j = u(col - 1, row);
    Real u_im1jp1 = u(col - 1, row + 1);

    Real v_ij = v(col, row);
    Real v_ijp1 = v(col, row + 1);
    Real v_ip1j = v(col + 1, row);
    Real v_ijm1 = v(col, row - 1);
    Real v_im1j = v(col - 1, row);

    Real dv2dy, duvdx, d2vdx2, d2vdy2;

    dv2dy = ((v_ij + v_ijp1) * (v_ij + v_ijp1) - (v_ijm1 + v_ij) * (v_ijm1 + v_ij)
        + mix_param * (fabs(v_ij + v_ijp1) * (v_ij - v_ijp1)
          - fabs(v_ijm1 + v_ij) * (v_ijm1 - v_ij))) / (FOUR * dy);
    duvdx = ((u_ij + u_ijp1) * (v_ij + v_ip1j) - (u_im1j + u_im1jp1) * (v_im1j + v_ij)
        + mix_param * (fabs(u_ij + u_ijp1) * (v_ij - v_ip1j) 
          - fabs(u_im1j + u_im1jp1) * (v_im1j - v_ij))) / (FOUR * dx);
    d2vdx2 = (v_ip1j - (TWO * v_ij) + v_im1j) / (dx * dx);
    d2vdy2 = (v_ijp1 - (TWO * v_ij) + v_ijm1) / (dy * dy);

    G(col, row) = v_ij + dt * (((d2vdx2 + d2vdy2) / Re_num) - dv2dy - duvdx + gy);
  }
}

///////////////////////////////////////////////////////////////////////////////
// Optimized in-kernel reduction using warp-level primitives for sum_pressure
__global__ 
void sum_pressure (const Real* __restrict__ pres_red,
                   const Real* __restrict__ pres_black, 
                         Real* __restrict__ pres_sum)
{
  // Calculate 2D index (grid dimensions as before)
  int row = (blockIdx.x * blockDim.x) + threadIdx.x + 1;
  int col = (blockIdx.y * blockDim.y) + threadIdx.y + 1;

  // Each thread computes its own squared pressure sum.
  Real pres_r = pres_red(col, row);
  Real pres_b = pres_black(col, row);
  Real sum = (pres_r * pres_r) + (pres_b * pres_b);

  // Warp-level reduction using shuffle.
  unsigned int mask = 0xffffffff;
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    sum += __shfl_down_sync(mask, sum, offset);
  }

  __shared__ Real shared_sum[BLOCK_SIZE / 32];  // one value per warp
  int lane = threadIdx.x % warpSize;
  int warpId = threadIdx.x / warpSize;
  if (lane == 0) {
    shared_sum[warpId] = sum;
  }
  __syncthreads();

  if (warpId == 0) {
    sum = (threadIdx.x < (blockDim.x / warpSize)) ? shared_sum[lane] : 0;
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
      sum += __shfl_down_sync(mask, sum, offset);
    }
    if (lane == 0) {
      pres_sum[blockIdx.y + (gridDim.y * blockIdx.x)] = sum;
    }
  }
}

///////////////////////////////////////////////////////////////////////////////
__global__ 
void set_horz_pres_BCs (Real* __restrict__ pres_red, Real* __restrict__ pres_black) 
{
  int col = (blockIdx.x * blockDim.x) + threadIdx.x + 1;
  col = (col * 2) - 1;
  int n2 = NUM >> 1;
  pres_black(col, 0) = pres_red(col, 1);
  pres_red(col + 1, 0) = pres_black(col + 1, 1);
  pres_red(col, n2 + 1) = pres_black(col, n2);
  pres_black(col + 1, n2 + 1) = pres_red(col + 1, n2);
}

//////////////////////////////////////////////////////////////////////////////
__global__
void set_vert_pres_BCs (Real* __restrict__ pres_red, Real* __restrict__ pres_black) 
{
  int row = (blockIdx.x * blockDim.x) + threadIdx.x + 1;
  int n2 = NUM >> 1;
  pres_black(0, row) = pres_red(1, row);
  pres_red(0, row) = pres_black(1, row);
  pres_black(NUM + 1, row) = pres_red(NUM, row);
  pres_red(NUM + 1, row) = pres_black(NUM, row);
}

///////////////////////////////////////////////////////////////////////////////
__global__
void red_kernel (const Real dt,
                 const Real* __restrict__ F, 
                 const Real* __restrict__ G,
                 const Real* __restrict__ pres_black,
                       Real* __restrict__ pres_red) 
{
  int row = (blockIdx.x * blockDim.x) + threadIdx.x + 1;
  int col = (blockIdx.y * blockDim.y) + threadIdx.y + 1;
  int n2 = NUM >> 1;      

  Real p_ij = pres_red(col, row);
  Real p_im1j = pres_black(col - 1, row);
  Real p_ip1j = pres_black(col + 1, row);
  Real p_ijm1 = pres_black(col, row - (col & 1));
  Real p_ijp1 = pres_black(col, row + ((col + 1) & 1));

  Real rhs = (((F(col, (2 * row) - (col & 1))
          - F(col - 1, (2 * row) - (col & 1))) / dx)
      + ((G(col, (2 * row) - (col & 1))
          - G(col, (2 * row) - (col & 1) - 1)) / dy)) / dt;

  pres_red(col, row) = p_ij * (ONE - omega) + omega * 
    (((p_ip1j + p_im1j) / (dx * dx)) + ((p_ijp1 + p_ijm1) / (dy * dy)) - 
     rhs) / ((TWO / (dx * dx)) + (TWO / (dy * dy)));
}

///////////////////////////////////////////////////////////////////////////////
__global__ 
void black_kernel (const Real dt,
                   const Real* __restrict__ F, 
                   const Real* __restrict__ G,
                   const Real* __restrict__ pres_red, 
                         Real* __restrict__ pres_black) 
{
  int row = (blockIdx.x * blockDim.x) + threadIdx.x + 1;
  int col = (blockIdx.y * blockDim.y) + threadIdx.y + 1;
  int n2 = NUM >> 1;

  Real p_ij = pres_black(col, row);
  Real p_im1j = pres_red(col - 1, row);
  Real p_ip1j = pres_red(col + 1, row);
  Real p_ijm1 = pres_red(col, row - ((col + 1) & 1));
  Real p_ijp1 = pres_red(col, row + (col & 1));

  Real rhs = (((F(col, (2 * row) - ((col + 1) & 1))
          - F(col - 1, (2 * row) - ((col + 1) & 1))) / dx)
      + ((G(col, (2 * row) - ((col + 1) & 1))
          - G(col, (2 * row) - ((col + 1) & 1) - 1)) / dy)) / dt;

  pres_black(col, row) = p_ij * (ONE - omega) + omega * 
    (((p_ip1j + p_im1j) / (dx * dx)) + ((p_ijp1 + p_ijm1) / (dy * dy)) - 
     rhs) / ((TWO / (dx * dx)) + (TWO / (dy * dy)));
}

///////////////////////////////////////////////////////////////////////////////
__global__
void calc_residual (const Real dt,
                    const Real* __restrict__ F,
                    const Real* __restrict__ G, 
                    const Real* __restrict__ pres_red,
                    const Real* __restrict__ pres_black,
                          Real* __restrict__ res_array)
{
  int row = (blockIdx.x * blockDim.x) + threadIdx.x + 1;
  int col = (blockIdx.y * blockDim.y) + threadIdx.y + 1;
  int n2 = NUM >> 1;

  Real p_ij, p_im1j, p_ip1j, p_ijm1, p_ijp1, rhs, res, res2;

  p_ij = pres_red(col, row);
  p_im1j = pres_black(col - 1, row);
  p_ip1j = pres_black(col + 1, row);
  p_ijm1 = pres_black(col, row - (col & 1));
  p_ijp1 = pres_black(col, row + ((col + 1) & 1));

  rhs = (((F(col, (2 * row) - (col & 1)) - F(col - 1, (2 * row) - (col & 1))) / dx)
      +  ((G(col, (2 * row) - (col & 1)) - G(col, (2 * row) - (col & 1) - 1)) / dy)) / dt;

  res = ((p_ip1j - (TWO * p_ij) + p_im1j) / (dx * dx))
    + ((p_ijp1 - (TWO * p_ij) + p_ijm1) / (dy * dy)) - rhs;

  p_ij = pres_black(col, row);
  p_im1j = pres_red(col - 1, row);
  p_ip1j = pres_red(col + 1, row);
  p_ijm1 = pres_red(col, row - ((col + 1) & 1));
  p_ijp1 = pres_red(col, row + (col & 1));

  rhs = (((F(col, (2 * row) - ((col + 1) & 1)) - F(col - 1, (2 * row) - ((col + 1) & 1))) / dx)
      +  ((G(col, (2 * row) - ((col + 1) & 1)) - G(col, (2 * row) - ((col + 1) & 1) - 1)) / dy)) / dt;

  res2 = ((p_ip1j - (TWO * p_ij) + p_im1j) / (dx * dx))
    + ((p_ijp1 - (TWO * p_ij) + p_ijm1) / (dy * dy)) - rhs;

  __shared__ Real sum_cache[BLOCK_SIZE];
  sum_cache[threadIdx.x] = (res * res) + (res2 * res2);
  __syncthreads();

  int i = BLOCK_SIZE >> 1;
  while (i != 0) {
    if (threadIdx.x < i) {
      sum_cache[threadIdx.x] += sum_cache[threadIdx.x + i];
    }
    __syncthreads();
    i >>= 1;
  }
  if (threadIdx.x == 0) {
    res_array[blockIdx.y + (gridDim.y * blockIdx.x)] = sum_cache[0];
  }
} 

///////////////////////////////////////////////////////////////////////////////
// New device-side reduction kernel for final residual sum.
__global__
void final_reduce(const Real* __restrict__ in, Real* __restrict__ out, int n)
{
    __shared__ Real sdata[BLOCK_SIZE];
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    Real sum = 0;
    if(idx < n)
        sum = in[idx];
    if(idx + blockDim.x < n)
        sum += in[idx + blockDim.x];
    sdata[tid] = sum;
    __syncthreads();

    for (unsigned int s = blockDim.x/2; s > 0; s >>= 1) {
        if(tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if(tid == 0)
        out[blockIdx.x] = sdata[0];
}

///////////////////////////////////////////////////////////////////////////////
__global__ 
void calculate_u (const Real dt,
                  const Real* __restrict__ F, 
                  const Real* __restrict__ pres_red,
                  const Real* __restrict__ pres_black, 
                        Real* __restrict__ u,
                        Real* __restrict__ max_u)
{
  int row = (blockIdx.x * blockDim.x) + threadIdx.x + 1;
  int col = (blockIdx.y * blockDim.y) + threadIdx.y + 1;
  __shared__ Real max_cache[BLOCK_SIZE];
  max_cache[threadIdx.x] = ZERO;
  int n2 = NUM >> 1;
  Real new_u = ZERO;

  if (col != NUM) {
    Real p_ij, p_ip1j, new_u2;
    p_ij = pres_red(col, row);
    p_ip1j = pres_black(col + 1, row);
    new_u = F(col, (2 * row) - (col & 1)) - (dt * (p_ip1j - p_ij) / dx);
    u(col, (2 * row) - (col & 1)) = new_u;
    p_ij = pres_black(col, row);
    p_ip1j = pres_red(col + 1, row);
    new_u2 = F(col, (2 * row) - ((col + 1) & 1)) - (dt * (p_ip1j - p_ij) / dx);
    u(col, (2 * row) - ((col + 1) & 1)) = new_u2;
    new_u = fmax(fabs(new_u), fabs(new_u2));
    if ((2 * row) == NUM) {
      new_u = fmax(new_u, fabs( u(col, NUM + 1) ));
    }
  } else {
    new_u = fmax(fabs( u(NUM, (2 * row)) ), fabs( u(0, (2 * row)) ));
    new_u = fmax(fabs( u(NUM, (2 * row) - 1) ), new_u);
    new_u = fmax(fabs( u(0, (2 * row) - 1) ), new_u);
    new_u = fmax(fabs( u(NUM + 1, (2 * row)) ), new_u);
    new_u = fmax(fabs( u(NUM + 1, (2 * row) - 1) ), new_u);
  }

  max_cache[threadIdx.x] = new_u;
  __syncthreads();
  int i = BLOCK_SIZE >> 1;
  while (i != 0) {
    if (threadIdx.x < i) {
      max_cache[threadIdx.x] = fmax(max_cache[threadIdx.x], max_cache[threadIdx.x + i]);
    }
    __syncthreads();
    i >>= 1;
  }
  if (threadIdx.x == 0) {
    max_u[blockIdx.y + (gridDim.y * blockIdx.x)] = max_cache[0];
  }
}

///////////////////////////////////////////////////////////////////////////////
__global__ 
void calculate_v (const Real dt,
                  const Real* __restrict__ G, 
                  const Real* __restrict__ pres_red,
                  const Real* __restrict__ pres_black, 
                        Real* __restrict__ v,
                        Real* __restrict__ max_v)
{
  int row = (blockIdx.x * blockDim.x) + threadIdx.x + 1;
  int col = (blockIdx.y * blockDim.y) + threadIdx.y + 1;
  __shared__ Real max_cache[BLOCK_SIZE];
  max_cache[threadIdx.x] = ZERO;
  int n2 = NUM >> 1;
  Real new_v = ZERO;

  if (row != n2) {
    Real p_ij, p_ijp1, new_v2;
    p_ij = pres_red(col, row);
    p_ijp1 = pres_black(col, row + ((col + 1) & 1));
    new_v = G(col, (2 * row) - (col & 1)) - (dt * (p_ijp1 - p_ij) / dy);
    v(col, (2 * row) - (col & 1)) = new_v;
    p_ij = pres_black(col, row);
    p_ijp1 = pres_red(col, row + (col & 1));
    new_v2 = G(col, (2 * row) - ((col + 1) & 1)) - (dt * (p_ijp1 - p_ij) / dy);
    v(col, (2 * row) - ((col + 1) & 1)) = new_v2;
    new_v = fmax(fabs(new_v), fabs(new_v2));
    if (col == NUM) {
      new_v = fmax(new_v, fabs( v(NUM + 1, (2 * row)) ));
    }
  } else {
    if ((col & 1) == 1) {
      Real p_ij = pres_red(col, row);
      Real p_ijp1 = pres_black(col, row + ((col + 1) & 1));
      new_v = G(col, (2 * row) - (col & 1)) - (dt * (p_ijp1 - p_ij) / dy);
      v(col, (2 * row) - (col & 1)) = new_v;
    } else {
      Real p_ij = pres_black(col, row);
      Real p_ijp1 = pres_red(col, row + (col & 1));
      new_v = G(col, (2 * row) - ((col + 1) & 1)) - (dt * (p_ijp1 - p_ij) / dy);
      v(col, (2 * row) - ((col + 1) & 1)) = new_v;
    }
    new_v = fabs(new_v);
    new_v = fmax(fabs( v(col, NUM) ), new_v);
    new_v = fmax(fabs( v(col, 0) ), new_v);
    new_v = fmax(fabs( v(col, NUM + 1) ), new_v);
  }
  max_cache[threadIdx.x] = new_v;
  __syncthreads();
  int i = BLOCK_SIZE >> 1;
  while (i != 0) {
    if (threadIdx.x < i) {
      max_cache[threadIdx.x] = fmax(max_cache[threadIdx.x], max_cache[threadIdx.x + i]);
    }
    __syncthreads();
    i >>= 1;
  }
  if (threadIdx.x == 0) {
    max_v[blockIdx.y + (gridDim.y * blockIdx.x)] = max_cache[0];
  }
}

///////////////////////////////////////////////////////////////////////////////
int main (int argc, char *argv[])
{
  int iter = 0;
  const int it_max = 1000000;
  const Real tol = 0.001;
  const Real time_start = 0.0;
  const Real time_end = 0.001;
  Real dt = 0.02;

  int size = (NUM + 2) * (NUM + 2);
  int size_pres = ((NUM / 2) + 2) * (NUM + 2);

  Real* F;
  Real* u;
  Real* G;
  Real* v;

  F = (Real *) calloc (size, sizeof(Real));
  u = (Real *) calloc (size, sizeof(Real));
  G = (Real *) calloc (size, sizeof(Real));
  v = (Real *) calloc (size, sizeof(Real));

  for (int i = 0; i < size; ++i) {
    F[i] = ZERO;
    u[i] = ZERO;
    G[i] = ZERO;
    v[i] = ZERO;
  }

  Real* pres_red;
  Real* pres_black;

  pres_red = (Real *) calloc (size_pres, sizeof(Real));
  pres_black = (Real *) calloc (size_pres, sizeof(Real));

  for (int i = 0; i < size_pres; ++i) {
    pres_red[i] = ZERO;
    pres_black[i] = ZERO;
  }

  printf("Problem size: %d x %d \n", NUM, NUM);

  dim3 block_bcs (BLOCK_SIZE, 1);
  dim3 grid_bcs (NUM / BLOCK_SIZE, 1);

  // Use 2D configuration for the compute kernels calculate_F and calculate_G
  const int BLOCK_DIM = 16;
  dim3 block_F (BLOCK_DIM, BLOCK_DIM);
  dim3 grid_F ((NUM + BLOCK_DIM - 1) / BLOCK_DIM, (NUM + BLOCK_DIM - 1) / BLOCK_DIM);

  dim3 block_G = block_F;
  dim3 grid_G = grid_F;

  dim3 block_pr (BLOCK_SIZE, 1);
  dim3 grid_pr (NUM / (2 * BLOCK_SIZE), NUM);

  dim3 block_hpbc (BLOCK_SIZE, 1);
  dim3 grid_hpbc (NUM / (2 * BLOCK_SIZE), 1);

  dim3 block_vpbc (BLOCK_SIZE, 1);
  dim3 grid_vpbc (NUM / (2 * BLOCK_SIZE), 1);

  int size_res = grid_pr.x * grid_pr.y;
  Real* res;
  cudaHostAlloc((void**)&res, size_res * sizeof(Real), cudaHostAllocDefault);

  Real* max_u_arr;
  Real* max_v_arr;
  int size_max = grid_pr.x * grid_pr.y;
  cudaHostAlloc((void**)&max_u_arr, size_max * sizeof(Real), cudaHostAllocDefault);
  cudaHostAlloc((void**)&max_v_arr, size_max * sizeof(Real), cudaHostAllocDefault);

  Real* pres_sum;
  cudaHostAlloc((void**)&pres_sum, size_res * sizeof(Real), cudaHostAllocDefault);

  set_BCs_host (u, v);

  Real max_u = SMALL;
  Real max_v = SMALL;
  #pragma unroll
  for (int col = 0; col < NUM + 2; ++col) {
    #pragma unroll
    for (int row = 1; row < NUM + 2; ++row) {
      max_u = fmax(max_u, fabs( u(col, row) ));
    }
  }

  #pragma unroll
  for (int col = 1; col < NUM + 2; ++col) {
    #pragma unroll
    for (int row = 0; row < NUM + 2; ++row) {
      max_v = fmax(max_v, fabs( v(col, row) ));
    }
  }

  Real* u_d;
  Real* F_d;
  Real* v_d;
  Real* G_d;

  Real* pres_red_d;
  Real* pres_black_d;
  Real* pres_sum_d;
  Real* res_d;

  Real* max_u_d;
  Real* max_v_d;

  cudaMalloc ((void**) &u_d, size * sizeof(Real));
  cudaMalloc ((void**) &F_d, size * sizeof(Real));
  cudaMalloc ((void**) &v_d, size * sizeof(Real));
  cudaMalloc ((void**) &G_d, size * sizeof(Real));

  cudaMalloc ((void**) &pres_red_d, size_pres * sizeof(Real));
  cudaMalloc ((void**) &pres_black_d, size_pres * sizeof(Real));

  cudaMalloc ((void**) &pres_sum_d, size_res * sizeof(Real));
  cudaMalloc ((void**) &res_d, size_res * sizeof(Real));
  cudaMalloc ((void**) &max_u_d, size_max * sizeof(Real));
  cudaMalloc ((void**) &max_v_d, size_max * sizeof(Real));

  Real* red_tmp1;
  cudaMalloc((void**)&red_tmp1, size_res * sizeof(Real));
  Real* res_final_d;
  cudaMalloc((void**)&res_final_d, sizeof(Real));

  cudaMemcpy (u_d, u, size * sizeof(Real), cudaMemcpyHostToDevice);
  cudaMemcpy (F_d, F, size * sizeof(Real), cudaMemcpyHostToDevice);
  cudaMemcpy (v_d, v, size * sizeof(Real), cudaMemcpyHostToDevice);
  cudaMemcpy (G_d, G, size * sizeof(Real), cudaMemcpyHostToDevice);
  cudaMemcpy (pres_red_d, pres_red, size_pres * sizeof(Real), cudaMemcpyHostToDevice);
  cudaMemcpy (pres_black_d, pres_black, size_pres * sizeof(Real), cudaMemcpyHostToDevice);

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  Real time = time_start;
  Real dt_Re = 0.5 * Re_num / ((1.0 / (dx * dx)) + (1.0 / (dy * dy)));

  cudaDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();

  while (time < time_end) {
    dt = fmin((dx / max_u), (dy / max_v));
    dt = tau * fmin(dt_Re, dt);
    if ((time + dt) >= time_end) {
      dt = time_end - time;
    }
    // Launch the 2D kernels calculate_F and calculate_G with the new configurations
    calculate_F <<<grid_F, block_F>>> (dt, u_d, v_d, F_d);
    calculate_G <<<grid_G, block_G>>> (dt, u_d, v_d, G_d);

    // Previously a reduction was performed every iteration.
    // Now we let the SOR loop run in batches.
    sum_pressure <<<grid_pr, block_pr>>> (pres_red_d, pres_black_d, pres_sum_d);
    cudaMemcpyAsync (pres_sum, pres_sum_d, size_res * sizeof(Real), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    Real p0_norm = ZERO;
    #pragma unroll
    for (int i = 0; i < size_res; ++i) {
      p0_norm += pres_sum[i];
    }
    p0_norm = sqrt(p0_norm / ((Real)(NUM * NUM)));
    if (p0_norm < 0.0001) {
      p0_norm = 1.0;
    }

    Real norm_L2 = 1.0;
    const int batch_interval = 5;  // number of SOR iterations per batch
    iter = 0;
    while (iter < it_max) {
      // Perform a batch of SOR iterations before checking convergence:
      for (int batch = 0; batch < batch_interval && iter < it_max; batch++, iter++) {
        set_horz_pres_BCs <<<grid_hpbc, block_hpbc>>> (pres_red_d, pres_black_d);
        set_vert_pres_BCs <<<grid_vpbc, block_vpbc>>> (pres_red_d, pres_black_d);
        red_kernel <<<grid_pr, block_pr>>> (dt, F_d, G_d, pres_black_d, pres_red_d);
        black_kernel <<<grid_pr, block_pr>>> (dt, F_d, G_d, pres_red_d, pres_black_d);
      }
      // Check convergence once per batch
      calc_residual <<<grid_pr, block_pr>>> (dt, F_d, G_d, pres_red_d, pres_black_d, res_d);
      int n = size_res;
      int threads = BLOCK_SIZE;
      int blocks = (n + threads * 2 - 1) / (threads * 2);
      final_reduce<<<blocks, threads, 0, stream>>>(res_d, red_tmp1, n);
      int current_size = blocks;
      Real* in_ptr = red_tmp1;
      while (current_size > 1) {
          int nb = (current_size + threads * 2 - 1) / (threads * 2);
          final_reduce<<<nb, threads, 0, stream>>>(in_ptr, red_tmp1, current_size);
          current_size = nb;
      }
      Real res_val;
      cudaMemcpyAsync(&res_val, red_tmp1, sizeof(Real), cudaMemcpyDeviceToHost, stream);
      cudaStreamSynchronize(stream);
      norm_L2 = sqrt(res_val / ((Real)(NUM * NUM))) / p0_norm;
      if (norm_L2 < tol) {
        break;
      }
    }

    printf("Time = %f, delt = %e, iter = %i, res = %e\n", time + dt, dt, iter, norm_L2);
    calculate_u <<<grid_pr, block_pr>>> (dt, F_d, pres_red_d, pres_black_d, u_d, max_u_d);
    calculate_v <<<grid_pr, block_pr>>> (dt, G_d, pres_red_d, pres_black_d, v_d, max_v_d);
    cudaMemcpyAsync (max_u_arr, max_u_d, size_max * sizeof(Real), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync (max_v_arr, max_v_d, size_max * sizeof(Real), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    max_v = SMALL;
    max_u = SMALL;
    #pragma unroll
    for (int i = 0; i < size_max; ++i) {
      Real test_u = max_u_arr[i];
      max_u = fmax(max_u, test_u);
      Real test_v = max_v_arr[i];
      max_v = fmax(max_v, test_v);
    }
    set_BCs <<<grid_bcs, block_bcs>>> (u_d, v_d);
    time += dt;
  }

  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto elapsed_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("\nTotal execution time of the iteration loop: %f (s)\n", elapsed_time * 1e-9f);

  cudaMemcpy (u, u_d, size * sizeof(Real), cudaMemcpyDeviceToHost);
  cudaMemcpy (v, v_d, size * sizeof(Real), cudaMemcpyDeviceToHost);
  cudaMemcpy (pres_red, pres_red_d, size_pres * sizeof(Real), cudaMemcpyDeviceToHost);
  cudaMemcpy (pres_black, pres_black_d, size_pres * sizeof(Real), cudaMemcpyDeviceToHost);

  FILE * pfile;
  pfile = fopen("velocity_gpu.dat", "w");
  fprintf(pfile, "#x\ty\tu\tv\n");
  if (pfile != NULL) {
    for (int row = 0; row < NUM; ++row) {
      for (int col = 0; col < NUM; ++col) {

        Real u_ij = u[(col * NUM) + row];
        Real u_im1j;
        if (col == 0) {
          u_im1j = 0.0;
        } else {
          u_im1j = u[(col - 1) * NUM + row];
        }
        u_ij = (u_ij + u_im1j) / 2.0;

        Real v_ij = v[(col * NUM) + row];
        Real v_ijm1;
        if (row == 0) {
          v_ijm1 = 0.0;
        } else {
          v_ijm1 = v[(col * NUM) + row - 1];
        }
        v_ij = (v_ij + v_ijm1) / 2.0;

        fprintf(pfile, "%f\t%f\t%f\t%f\n", ((Real)col + 0.5) * dx, ((Real)row + 0.5) * dy, u_ij, v_ij);
      }
    }
  }

  fclose(pfile);

  cudaFree(u_d);
  cudaFree(v_d);
  cudaFree(F_d);
  cudaFree(G_d);
  cudaFree(pres_red_d);
  cudaFree(pres_black_d);
  cudaFree(max_u_d);
  cudaFree(max_v_d);
  cudaFree(pres_sum_d);
  cudaFree(res_d);
  cudaFree(red_tmp1);
  cudaFree(res_final_d);

  free(pres_red);
  free(pres_black);
  free(u);
  free(v);
  free(F);
  free(G);
  cudaFreeHost(max_u_arr);
  cudaFreeHost(max_v_arr);
  cudaFreeHost(res);
  cudaFreeHost(pres_sum);

  cudaStreamDestroy(stream);

  return 0;
}
