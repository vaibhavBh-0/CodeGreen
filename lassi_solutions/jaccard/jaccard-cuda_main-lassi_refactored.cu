
/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cuda.h>
#include <stdio.h>
#include <algorithm>
#include <iostream>
#include <vector>
#include <chrono>

using namespace std;

#define MAX_KERNEL_THREADS 256
#define mask 0xFFFFFFFF

// float or double 
typedef float vtype;
typedef vector<vector<vtype>> matrix;

template<typename T>
__device__
T parallel_prefix_sum(const int n, const int *ind, const T *w)
{
  T sum = 0.0;
  T last;

  int mn = (((n+blockDim.x-1)/blockDim.x)*blockDim.x); // n in multiple of blockDim.x
  for (int i = threadIdx.x; i < mn; i += blockDim.x) {
    // All threads (especially the last one) must always participate
    // in the shfl instruction, otherwise their sum will be undefined.
    // So, the loop stopping condition is based on multiple of n in loop increments,
    // so that all threads enter into the loop and inside we make sure we do not
    // read out of bounds memory checking for the actual size n.

    // check if the thread is valid
    bool valid = i < n;

    // Notice that the last thread is used to propagate the prefix sum.
    // For all the threads, in the first iteration the last is 0, in the following
    // iterations it is the value at the last thread of the previous iterations.

    // get the value of the last thread
    last = __shfl_sync(mask, sum, blockDim.x-1, blockDim.x);

    // if you are valid read the value from memory, otherwise set your value to 0
    sum = (valid) ? w[ind[i]] : 0.0;

    // do prefix sum (of size warpSize=blockDim.x =< 32)
    for (int j = 1; j < blockDim.x; j *= 2) {
      T v = __shfl_up_sync(mask, sum, j, blockDim.x);
      if (threadIdx.x >= j) sum += v;
    }
    // shift by last
    sum += last;
    // notice that no __threadfence or __syncthreads are needed in this implementation
  }
  // get the value of the last thread (to all threads)
  last = __shfl_sync(mask, sum, blockDim.x-1, blockDim.x);

  return last;
}

// Volume of neighboors (*weight_s)
template<bool weighted, typename T>
__global__ void 
jaccard_row_sum(const int n,
                const int *__restrict__ csrPtr,
                const int *__restrict__ csrInd,
                const T *__restrict__ w,
                      T *__restrict__ work)
{
  for (int row = threadIdx.y + blockIdx.y * blockDim.y; row < n; row += gridDim.y * blockDim.y) {
    int start = csrPtr[row];
    int end   = csrPtr[row+1];
    int length = end - start;
    // compute row sums 
    if (weighted) {
      T sum = parallel_prefix_sum(length, csrInd + start, w);
      if (threadIdx.x == 0) work[row] = sum;
    }
    else {
      work[row] = (T)length;
    }
  }
}

// Volume of intersections (*weight_i) and cumulated volume of neighboors (*weight_s)
// Note the number of columns is constrained by the number of rows
template<bool weighted, typename T>
__global__ void
jaccard_is(const int n, const int e,
           const int *__restrict__ csrPtr,
           const int *__restrict__ csrInd,
           const T *__restrict__ v,
           const T *__restrict__ work,
                 T *__restrict__ weight_i,
                 T *__restrict__ weight_s)
{
  // Iterate over rows
  for (int row = threadIdx.z + blockIdx.z * blockDim.z; row < n; row += gridDim.z * blockDim.z) {
    // Iterate over the neighbors in this row
    for (int j = csrPtr[row] + threadIdx.y + blockIdx.y * blockDim.y;
             j < csrPtr[row+1]; j += gridDim.y * blockDim.y) {
      int col = csrInd[j];
      // find which row has fewer elements (call it reference row)
      int Ni = csrPtr[row+1] - csrPtr[row];
      int Nj = csrPtr[col+1] - csrPtr[col];
      int ref = (Ni < Nj) ? row : col;
      int cur = (Ni < Nj) ? col : row;

      // compute new sum weights
      weight_s[j] = work[row] + work[col];

      // Only one thread per work-item does the intersection merge
      if (threadIdx.x == 0) {
        T local_sum = 0;
        int i_ptr = csrPtr[ref];      // pointer in reference row
        int j_ptr = csrPtr[cur];        // pointer in current row
        int ref_end = csrPtr[ref+1];
        int cur_end = csrPtr[cur+1];

        // Two-pointer merge for intersection of the two sorted lists
        while (i_ptr < ref_end && j_ptr < cur_end) {
          int ref_col = csrInd[i_ptr];
          int cur_col = csrInd[j_ptr];
          if (ref_col == cur_col) {
            T ref_val = weighted ? v[ref_col] : (T)1.0;
            local_sum += ref_val;
            i_ptr++;
            j_ptr++;
          } else if (ref_col < cur_col) {
            i_ptr++;
          } else {
            j_ptr++;
          }
        }
        // perform a single atomic update per this j index
        if (local_sum != 0)
          atomicAdd(&weight_i[j], local_sum);
      }
    }
  }
}

template<bool weighted, typename T>
__global__ void 
jaccard_jw(const int e,
           const T *__restrict__ csrVal,
           const T gamma,
           const T *__restrict__ weight_i,
           const T *__restrict__ weight_s,
                 T *__restrict__ weight_j)
{
  for (int j = threadIdx.x + blockIdx.x * blockDim.x; j < e; j += gridDim.x * blockDim.x) {
    T Wi =  weight_i[j];
    T Ws =  weight_s[j];
    weight_j[j] = (gamma * csrVal[j]) * (Wi / (Ws - Wi));
  }
}

// Original fill kernel (kept for reference; no longer used in the iteration loop)
template<bool weighted, typename T>
__global__ void 
fill(const int e, T* w, const T value)
{
  for (int j = threadIdx.x + blockIdx.x * blockDim.x; j < e; j += gridDim.x * blockDim.x) {
    // e.g. w[0] is the weight of a non-zeron element when csr_ind[i] equals 0. 
    // So multiple non-zero elements on different rows of a matrix may share 
    // the same weight value
    w[j] = weighted ? (T)(j + 1) / e : value;
  }
}

// New fused fill kernel: initializes both d_weight_j and d_weight_i in one launch.
template<bool weighted, typename T>
__global__ void 
fused_fill(const int e, T* d_weight_j, T* d_weight_i)
{
  for (int j = threadIdx.x + blockIdx.x * blockDim.x; j < e; j += gridDim.x * blockDim.x) {
    d_weight_j[j] = weighted ? (T)(j + 1) / e : (T)1.0;
    d_weight_i[j] = (T)0.0;
  }
}

template <bool weighted, typename T>
void jaccard_weight (const int iteration, const int n, const int e, 
                     int* csr_ptr, int* csr_ind, T* csr_val)
{
  const T gamma = (T)0.46;  // arbitrary

  T *d_weight_i,
    *d_weight_s,
    *d_weight_j,
    *d_work;
  int *d_csrInd;
  int *d_csrPtr;
  T *d_csrVal;

#ifdef DEBUG
  T* weight_i = (T*) malloc(sizeof(T) * e);
  T* weight_s = (T*) malloc(sizeof(T) * e);
  T* work = (T*) malloc(sizeof(T) * n);
#endif
  T* weight_j = (T*) malloc(sizeof(T) * e);

  cudaMalloc((void**)&d_work, sizeof(T) * n);
  cudaMalloc((void**)&d_weight_i, sizeof(T) * e);
  cudaMalloc((void**)&d_weight_s, sizeof(T) * e);
  cudaMalloc((void**)&d_weight_j, sizeof(T) * e);
  cudaMalloc((void**)&d_csrVal, sizeof(T) * e);
  cudaMalloc((void**)&d_csrPtr, sizeof(int) * (n+1));
  cudaMalloc((void**)&d_csrInd, sizeof(int) * e);

  cudaMemcpy(d_csrPtr, csr_ptr, sizeof(int) * (n+1), cudaMemcpyHostToDevice);
  cudaMemcpy(d_csrInd, csr_ind, sizeof(int) * e, cudaMemcpyHostToDevice);
  cudaMemcpy(d_csrVal, csr_val, sizeof(T) * e, cudaMemcpyHostToDevice);

  cudaDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();

  // Precompute kernel launch configurations outside the iteration loop.
  // fused_fill kernel configuration:
  dim3 fill_block(MAX_KERNEL_THREADS, 1, 1);
  dim3 fill_grid((e + MAX_KERNEL_THREADS - 1) / MAX_KERNEL_THREADS, 1, 1);

  // jaccard_row_sum kernel configuration:
  const int y = 4;
  dim3 row_block(64 / y, y, 1);   // 16 x 4 x 1
  dim3 row_grid(1, (n + row_block.y - 1) / row_block.y, 1);
  
  // jaccard_is kernel configuration:
  dim3 is_block(32 / y, y, 8);    // 8 x 4 x 8
  dim3 is_grid(1, 1, (n + is_block.z - 1) / is_block.z);
  
  // jaccard_jw kernel configuration:
  int jw_block_x = (e < MAX_KERNEL_THREADS) ? e : MAX_KERNEL_THREADS;
  dim3 jw_block(jw_block_x, 1, 1);
  dim3 jw_grid((e + jw_block.x - 1) / jw_block.x, 1, 1);

  for (int i = 0; i < iteration; i++) {
    // Use the fused fill kernel to initialize both d_weight_j and d_weight_i
    fused_fill<weighted, T><<<fill_grid, fill_block>>>(e, d_weight_j, d_weight_i);

#ifdef DEBUG
    {
      T* weight_j_debug = (T*) malloc(sizeof(T) * e);
      cudaMemcpy(weight_j_debug, d_weight_j, sizeof(T) * e, cudaMemcpyDeviceToHost);
      for (int i = 0; i < e; i++) printf("wj: %d %f\n", i, weight_j_debug[i]);
      free(weight_j_debug);
    }
#endif

    // compute row sum with prefix sum
    jaccard_row_sum<weighted, T><<<row_grid, row_block>>>(n, d_csrPtr, d_csrInd, d_weight_j, d_work);

#ifdef DEBUG
    {
      T* work_debug = (T*) malloc(sizeof(T) * n);
      cudaMemcpy(work_debug, d_work, sizeof(T) * n, cudaMemcpyDeviceToHost);
      for (int i = 0; i < n; i++) printf("work: %d %f\n", i, work_debug[i]);
      free(work_debug);
    }
#endif

    // compute volume of intersections (*weight_i) and cumulative volume of neighbors (*weight_s)
    jaccard_is<weighted, T><<<is_grid, is_block>>>(n, e, d_csrPtr, d_csrInd, d_weight_j, d_work, d_weight_i, d_weight_s);

#ifdef DEBUG
    {
      T* weight_i_debug = (T*) malloc(sizeof(T) * e);
      T* weight_s_debug = (T*) malloc(sizeof(T) * e);
      cudaMemcpy(weight_i_debug, d_weight_i, sizeof(T) * e, cudaMemcpyDeviceToHost);
      cudaMemcpy(weight_s_debug, d_weight_s, sizeof(T) * e, cudaMemcpyDeviceToHost);
      for (int i = 0; i < e; i++) printf("wi: %d %f\n", i, weight_i_debug[i]);
      for (int i = 0; i < e; i++) printf("ws: %d %f\n", i, weight_s_debug[i]);
      free(weight_i_debug);
      free(weight_s_debug);
    }
#endif

    // compute jaccard weights
    jaccard_jw<weighted, T><<<jw_grid, jw_block>>>(e, d_csrVal, gamma, d_weight_i, d_weight_s, d_weight_j);
  }

  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  cout << "Average execution time of kernels: " << (time * 1e-9f) / iteration << " (s)\n";

  cudaMemcpy(weight_j, d_weight_j, sizeof(T) * e, cudaMemcpyDeviceToHost);
#ifdef DEBUG
  // verify using known values when weighted is true
  float error;

  if (weighted)
    error = std::fabs(weight_j[0] - 0.306667) +
            std::fabs(weight_j[1] - 0.000000) +
            std::fabs(weight_j[2] - 3.680000) +
            std::fabs(weight_j[3] - 1.380000) +
            std::fabs(weight_j[4] - 0.788571) +
            std::fabs(weight_j[5] - 0.460000);
  else
    error = std::fabs(weight_j[0] - 0.230000) +
            std::fabs(weight_j[1] - 0.000000) +
            std::fabs(weight_j[2] - 3.680000) +
            std::fabs(weight_j[3] - 1.380000) +
            std::fabs(weight_j[4] - 0.920000) +
            std::fabs(weight_j[5] - 0.460000);

  if (error > 1e-5) {
    for (int i = 0; i < e; i++) printf("wj: %d %f\n", i, weight_j[i]);
    printf("FAIL");
  } else {
    printf("PASS");
  }
  printf("\n");
#endif

  cudaFree(d_work);
  cudaFree(d_weight_i);
  cudaFree(d_weight_s);
  cudaFree(d_weight_j);
  cudaFree(d_csrInd);
  cudaFree(d_csrVal);
  cudaFree(d_csrPtr);
  free(weight_j);
#ifdef DEBUG
  free(weight_i);
  free(weight_s);
  free(work);
#endif
}

// Utilities
void printMatrix(const matrix& M)
{
  int m = M.size();
  int n = M[0].size();
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++)
      cout << M[i][j] << " ";
    cout << endl;
  }
}

template <typename T>
void printVector(const vector<T>& V, char* msg)
{
  cout << msg << "[ ";
  for_each(V.begin(), V.end(), [](int a) { cout << a << " "; });
  cout << "]" << endl;
}

// Reference: https://www.geeksforgeeks.org/sparse-matrix-representations-set-3-csr/
int main(int argc, char** argv)
{
  int iteration = 10;

#ifdef DEBUG
  matrix M  = {
    { 0, 0, 0, 1},
    { 5, 8, 0, 0},
    { 0, 0, 3, 0},
    { 0, 6, 0, 1}
  };
#else

  int numRow = atoi(argv[1]);
  int numCol = atoi(argv[2]);
  iteration = atoi(argv[3]);

  srand(2);

  matrix M;
  vector<vtype> rowElems(numCol);
  for (int r = 0; r < numRow; r++) {
    for (int c = 0; c < numCol; c++)
      rowElems[c] = rand() % 10;
    M.push_back(rowElems);
  }
#endif

  int row = M.size();
  int col = M[0].size();
  printf("Number of matrix rows and cols: %d %d\n", row, col);
  vector<vtype> csr_val;
  vector<int> csr_ptr = {0}; // require -std=c++11
  vector<int> csr_ind;
  int nnz = 0; // count Number of non-zero elements in each row

  for (int i = 0; i < row; i++) {
    for (int j = 0; j < col; j++) {
      if (M[i][j] != (vtype)0) {
        csr_val.push_back(M[i][j]);
        csr_ind.push_back(j);
        nnz++;
      }
    }
    csr_ptr.push_back(nnz);
  }

  // print when the matrix is small
  if (row <= 16 && col <= 16) {
    printMatrix(M);
    printVector(csr_val, (char*)"values = ");
    printVector(csr_ptr, (char*)"row pointer = ");
    printVector(csr_ind, (char*)"col indices = ");
  }

  jaccard_weight<true, vtype>(iteration, row, nnz, csr_ptr.data(), csr_ind.data(), csr_val.data());
  jaccard_weight<false, vtype>(iteration, row, nnz, csr_ptr.data(), csr_ind.data(), csr_val.data());

  return 0;
}
