#pragma once

#include <cuda_runtime.h>
#include "include/data_structures.hpp"
#include "include/constants.hpp"

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