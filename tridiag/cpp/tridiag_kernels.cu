#pragma once

#include <cuda_runtime.h>
#include "include/device_utils.cu.h"
#include "include/constants.hpp"
#include "include/pbbKernels.cu.h"

__device__ inline void filltup4(DTYPE* a, DTYPE* b, DTYPE* c, unsigned int index, volatile typename tuple4op<DTYPE>::RedElTp* shared, unsigned int n)
{
  pbbtuple4<DTYPE> tup;
  if (threadIdx.x == 0 || threadIdx.x >= n)
  {
    tup.a = 1;
    tup.b = 0;
    tup.c = 0;
    tup.d = 1;
  }
  else
  {
    tup.a = b[index];
    tup.b = -a[index]*c[index-1];
    tup.c = 1;
    tup.d = 0;
  }
  shared[threadIdx.x] = tup;
}

__device__ inline void filltup2_1(DTYPE* a, DTYPE* b, DTYPE* d, unsigned int index, volatile typename tuple2op<DTYPE>::RedElTp* shared, unsigned int n)
{
  pbbtuple2<DTYPE> tup;
  if (threadIdx.x == 0 || threadIdx.x >= n)
  {
    tup.a = 0;
    tup.b = 1;
  }
  else
  {
    tup.a = d[index];
    tup.b = -a[index]/b[threadIdx.x-1];
  }
  shared[threadIdx.x] = tup;
}

__device__ inline void filltup2_2(DTYPE* b, DTYPE* c, DTYPE* d, unsigned int datastart, volatile typename tuple2op<DTYPE>::RedElTp* shared, unsigned int n)
{
  int newIdx = n - threadIdx.x - 1;
  pbbtuple2<DTYPE> tup;
  if (newIdx <= 0)
  {
    tup.a = 0;
    tup.b = 1;
  }
  else
  {
    DTYPE b_tmp = b[newIdx];
    tup.a = d[newIdx]/b_tmp;
    tup.b = -c[datastart + newIdx]/b_tmp;
  }
  __syncthreads();
  shared[threadIdx.x] = tup;
}

__global__ void tridiag_shared(DTYPE* a, DTYPE* b, DTYPE* c, DTYPE* d, DTYPE* out, unsigned int n)
{

  extern __shared__ DTYPE shared[];
  DTYPE* b_tmp = shared;
  DTYPE* d_tmp = shared+blockDim.x;
  volatile typename tuple4op<DTYPE>::RedElTp* tuple4ptr = reinterpret_cast<typename tuple4op<DTYPE>::RedElTp*>(shared);
  volatile typename tuple2op<DTYPE>::RedElTp* tuple2ptr = reinterpret_cast<typename tuple2op<DTYPE>::RedElTp*>(d_tmp);

  const unsigned int datastart = blockIdx.x * n; 
  const unsigned int index = datastart + threadIdx.x; 
  
  // if (threadIdx.x < n)
  filltup4(a, b, c, index, tuple4ptr, n);
  __syncthreads();

  typename tuple4op<DTYPE>::RedElTp tup4 = scanIncBlock<tuple4op<DTYPE>>(tuple4ptr, threadIdx.x);
  DTYPE b0 = b[datastart];
  __syncthreads();
  if (threadIdx.x < n)
    b_tmp[threadIdx.x] = (tup4.a*b0+tup4.b) / (tup4.c*b0 + tup4.d);
  __syncthreads();
  // if (threadIdx.x < n)
  filltup2_1(a, b_tmp, d, index, tuple2ptr, n);
  __syncthreads();
  typename tuple2op<DTYPE>::RedElTp tup2 = scanIncBlock<tuple2op<DTYPE>>(tuple2ptr, threadIdx.x);
  DTYPE d0 = d[datastart];
  __syncthreads();
  if (threadIdx.x < n)
    d_tmp[threadIdx.x] = tup2.a + tup2.b*d0;
  __syncthreads();
  DTYPE d_div_b = d_tmp[n-1] / b_tmp[n-1];
  __syncthreads();
  filltup2_2(b_tmp, c, d_tmp, datastart, tuple2ptr, n);
  __syncthreads();
  tup2 = scanIncBlock<tuple2op<DTYPE>>(tuple2ptr, threadIdx.x);
  
  __syncthreads();
  if (threadIdx.x < n)
    out[datastart + n - threadIdx.x - 1] = tup2.a + tup2.b * d_div_b;
}


__global__ void recurrence1_no_const(DTYPE* a, DTYPE* b, DTYPE* c, unsigned int num_chunks, unsigned int n)
{
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx>=num_chunks)
        return;
    const unsigned int chunk_start = idx * n;

    for (int i = chunk_start + 1 ; i < chunk_start + n ; i++)
    {
        b[i] -= a[i]*c[i-1]/b[i-1];
    }

}

__global__ void recurrence1(DTYPE* a, DTYPE* b, DTYPE* c, unsigned int num_chunks)
{
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx>=num_chunks)
        return;
    const unsigned int chunk_start = idx * TRIDIAG_INNER_DIM;
    // const unsigned int chunk_end = chunk_start + n;
    
    DTYPE as[TRIDIAG_INNER_DIM-1];
    DTYPE bs[TRIDIAG_INNER_DIM];
    DTYPE cs[TRIDIAG_INNER_DIM-1];
    
    #pragma unroll
    for (int i = 0 ; i < TRIDIAG_INNER_DIM ; i++)
    {
        int loc = chunk_start + i;
        as[i] = a[loc+1];
        bs[i] = b[loc];  
        cs[i] = c[loc];
    }
    #pragma unroll
    for (int i = 0 ; i < TRIDIAG_INNER_DIM -1  ; i++)
    {
        bs[i+1] -= as[i]*cs[i]/bs[i];
    }
    #pragma unroll
    for (int i = 0 ; i < TRIDIAG_INNER_DIM ; i++)
    {
        b[chunk_start + i] = bs[i];
    }
}
__global__ void create_tuple4_r1(DTYPE *a, DTYPE *b, DTYPE *c,
                                 tuple4<DTYPE> *tups, unsigned int total_size,
                                 unsigned int n) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_size)
    return;
  tuple4<DTYPE> t;
  if (idx % n != 0) {
    t.a = b[idx];
    t.b = -(a[idx] * c[idx - 1]);
    t.c = 1;
    t.d = 0;
  }
  tups[idx] = t;
}

__global__
void generate_keys(unsigned int* keys, unsigned int total_size, unsigned int n)
{
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx>=total_size)
        return;
    keys[idx] = idx / n;
}

__global__ void get_first_elem_in_chunk(DTYPE *in, DTYPE *out,
                                        unsigned int num_chunks,
                                        unsigned int n) {
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_chunks)
    return;
  out[idx] = in[idx * n];
}

__global__ void combine_tuple4_r1(tuple4<DTYPE> *tups, unsigned int *keys,
                                  DTYPE *b, DTYPE *b0s, unsigned int total_size,
                                  unsigned int n) {
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_size)
    return;
  tuple4<DTYPE> t = tups[idx];
  DTYPE b0 = b0s[keys[idx]];
  b[idx] = (t.a * b0 + t.b) / (t.c * b0 + t.d);
}

__global__ void create_tuple2_r2(tuple2<DTYPE> *tups, DTYPE *a, DTYPE *b,
                                 DTYPE *d, unsigned int total_size,
                                 unsigned int n) {
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_size)
    return;
  tuple2<DTYPE> t;
  if (idx % n == 0) {
    t.a = 0;
    t.b = 1;

  } else {
    t.a = d[idx];
    t.b = -a[idx] / b[idx - 1];
  }
  tups[idx] = t;
}

__global__ void combine_tuple2_r2(tuple2<DTYPE> *tups, unsigned int *keys,
                                  DTYPE *d, DTYPE *d0s, unsigned int total_size,
                                  unsigned int n) {
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_size)
    return;
  tuple2<DTYPE> t = tups[idx];
  DTYPE d0 = d0s[keys[idx]];
  d[idx] = t.a + t.b * d0;
}

__global__ void get_last_yb_div_in_chunk(DTYPE *d, DTYPE *b, DTYPE *lastDiv,
                                         unsigned int num_chunks,
                                         unsigned int n) {
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_chunks)
    return;
  const int n1 = idx * n + (n - 1);
  lastDiv[idx] = d[n1] / b[n1];
}

__global__ void create_tuple2_r3(tuple2<DTYPE> *tups, unsigned int *keys,
                                 DTYPE *b, DTYPE *c, DTYPE *d,
                                 unsigned int total_size, unsigned int n) {
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_size)
    return;
  const unsigned int revIdx = n * keys[idx] + (n - (idx % n) - 1);
  tuple2<DTYPE> t;
  if (idx % n == 0) {
    t.a = 0;
    t.b = 1;

  } else {
    DTYPE rb = b[revIdx];
    t.a = d[revIdx] / rb;
    t.b = -c[revIdx] / rb;
  }
  tups[idx] = t;
}

__global__ void combine_tuple2_and_reverse_r3(tuple2<DTYPE> *tups,
                                              unsigned int *keys,
                                              DTYPE *lastDivs, DTYPE *d,
                                              unsigned int total_size,
                                              unsigned int n) {
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_size)
    return;
  unsigned int k = keys[idx];
  const unsigned int revIdx = n * k + (n - (idx % n) - 1);
  tuple2<DTYPE> t = tups[idx];
  d[revIdx] = t.a + t.b * lastDivs[k];
}

__global__
void execute_no_const(
    const DTYPE *a,
    const DTYPE *b,
    DTYPE *c,
    DTYPE *d,
    DTYPE *solution,
    int total_size,
    int n
){
    const size_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * n;

    if (idx >= total_size) {
        return;
    }

    c[idx] /= b[idx];
    d[idx] /= b[idx];
    DTYPE norm_factor;
    #pragma unroll
    for (ptrdiff_t j = 1; j < n; ++j) {
        norm_factor = 1.0 / (b[idx+j] - a[idx+j] * c[idx + j-1]);
        c[idx + j] = c[idx+j] * norm_factor;
        d[idx + j] = (d[idx+j] - a[idx+j] * d[idx + j-1]) * norm_factor;
    }

    solution[idx + n-1] = d[idx + n-1];
    #pragma unroll
    for (ptrdiff_t j=n-2; j >= 0; --j) {
        solution[idx + j] = d[idx + j] - c[idx + j] * solution[idx + j+1];
    }
}

__global__
void execute(
    const DTYPE *a,
    const DTYPE *b,
    const DTYPE *c,
    const DTYPE *d,
    DTYPE *solution,
    int total_size
){
    const size_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * TRIDIAG_INNER_DIM;

    if (idx >= total_size) {
        return;
    }

    DTYPE cp[TRIDIAG_INNER_DIM];
    DTYPE dp[TRIDIAG_INNER_DIM];

    cp[0] = c[idx] / b[idx];
    dp[0] = d[idx] / b[idx];
    DTYPE norm_factor;
    #pragma unroll
    for (ptrdiff_t j = 1; j < TRIDIAG_INNER_DIM; ++j) {
        norm_factor = 1.0 / (b[idx+j] - a[idx+j] * cp[j-1]);
        cp[j] = c[idx+j] * norm_factor;
        dp[j] = (d[idx+j] - a[idx+j] * dp[j-1]) * norm_factor;
    }

    solution[idx + TRIDIAG_INNER_DIM-1] = dp[TRIDIAG_INNER_DIM-1];
    #pragma unroll
    for (ptrdiff_t j=TRIDIAG_INNER_DIM-2; j >= 0; --j) {
        solution[idx + j] = dp[j] - cp[j] * solution[idx + j+1];
    }
}

__global__
void transpose4(
  const DTYPE* a,
  const DTYPE* b,
  const DTYPE* c,
  const DTYPE* d,
  DTYPE* a_t,
  DTYPE* b_t,
  DTYPE* c_t,
  DTYPE* d_t,
  int xdim,
  int ydim,
  int total_size
)
{
  __shared__ DTYPE tile[4*TILE_DIM][TILE_DIM+1];

  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;
  
  if (x < xdim)
  {
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
    {
        int index = (y+j)*xdim + x;
        if (index < total_size)
        {  
          tile[threadIdx.y+j][threadIdx.x] = a[index];
          tile[TILE_DIM + threadIdx.y+j][threadIdx.x] = b[index];
          tile[2 * TILE_DIM + threadIdx.y+j][threadIdx.x] = c[index];
          tile[3 * TILE_DIM + threadIdx.y+j][threadIdx.x] = d[index];
        }
    }
  }
  __syncthreads();

  x = blockIdx.y * TILE_DIM + threadIdx.x;  // transpose block offset
  y = blockIdx.x * TILE_DIM + threadIdx.y;
  if (x < ydim)
  {
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
    {
      int index = (y+j)*ydim + x;
      if (index < total_size)
      {
        
        a_t[index] = tile[threadIdx.x][threadIdx.y + j];
        b_t[index] = tile[TILE_DIM + threadIdx.x][threadIdx.y + j];
        c_t[index] = tile[2 * TILE_DIM + threadIdx.x][threadIdx.y + j];
        d_t[index] = tile[3 * TILE_DIM + threadIdx.x][threadIdx.y + j];
      }
    }
  }
}

__global__
void transpose(
  const DTYPE* m,
  DTYPE* m_t,
  int xdim,
  int ydim,
  int total_size
)
{
  __shared__ DTYPE tile[TILE_DIM][TILE_DIM+1];

  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;
  
  if (x < xdim)
  {
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
    {
        int index = (y+j)*xdim + x;
        if (index < total_size)
          tile[threadIdx.y+j][threadIdx.x] = m[index];
    }
  }
  __syncthreads();

  x = blockIdx.y * TILE_DIM + threadIdx.x;  // transpose block offset
  y = blockIdx.x * TILE_DIM + threadIdx.y;
  if (x < ydim)
  {
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
    {
      int index = (y+j)*ydim + x;
      if (index < total_size)
        m_t[index] = tile[threadIdx.x][threadIdx.y + j];
    }
  }
}

__global__
void execute_coalesced(
    const DTYPE *a,
    const DTYPE *b,
    DTYPE *c,
    DTYPE *d,
    DTYPE *solution,
    int n,
    int num_chunks
){

  const unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx >= num_chunks) {
      return;
  }
  DTYPE b0 = b[idx];
  c[idx] /= b0;
  d[idx] /= b0;

  DTYPE norm_factor;
  unsigned int indj = idx;
  DTYPE ai;
  DTYPE cm1;
  DTYPE dm1;

  for (int j = 0; j < n-1; ++j) {
      // c and d from last iteration
      cm1 = c[indj];
      dm1 = d[indj];
      // jump to next chunk
      indj += num_chunks;
      ai = a[indj];
      norm_factor = 1.0f / (b[indj] - ai * cm1);
      c[indj] = c[indj] * norm_factor;
      d[indj] = (d[indj] - ai * dm1) * norm_factor;
  }
  int lastIndx = idx + num_chunks*(n-1);
  solution[lastIndx] = d[lastIndx];
  for (int j=0; j < n-1; ++j) {
      lastIndx -= num_chunks;
      solution[lastIndx] = d[lastIndx] - c[lastIndx] * solution[lastIndx + num_chunks];
  }
}

__global__
void execute_coalesced_const(
    const DTYPE *a,
    const DTYPE *b,
    const DTYPE *c,
    const DTYPE *d,
    DTYPE *solution,
    int num_chunks
){
  const unsigned int n = TRIDIAG_INNER_DIM;

  const unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx >= num_chunks) {
      return;
  }

  DTYPE cp[n];
  DTYPE dp[n];

  cp[0] = c[idx] / b[idx];
  dp[0] = d[idx] / b[idx];

  #pragma unroll
  for (int j = 1; j < n; ++j) {
      unsigned int indj = idx+(j*num_chunks);
      const DTYPE norm_factor = (b[indj] - a[indj] * cp[j-1]);
      cp[j] = c[indj] / norm_factor;
      dp[j] = (d[indj] - a[indj] * dp[j-1]) / norm_factor;
  }

  solution[idx + num_chunks*(n-1)] = dp[n-1];
  #pragma unroll
  for (int j=n-2; j >= 0; --j) {
      solution[idx + num_chunks*j] = dp[j] - cp[j] * solution[idx + num_chunks*(j+1)];
  }
}