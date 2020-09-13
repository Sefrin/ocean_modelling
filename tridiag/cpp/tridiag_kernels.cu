#pragma once

#include<cuda_runtime.h>

#include "data_structures.h"
__global__ 
void map1(double* a, double* b, double* c, tuple4* tups, int total_size, int n)
{
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx>=total_size)
        return;
    tuple4 t;
    if (idx % n == 0)
    {
        t.a = 1;
        t.b = 0;
        t.c = 0;
        t.d = 1;
    }
    else
    {
        t.a = b[idx];
        t.b = -a[idx] * c[idx-1];
        t.c = 1;
        t.d = 0;
    }
    tups[idx] = t;
}

__global__
void generate_keys(unsigned int* keys, int total_size, int n)
{
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx>=total_size)
        return;
    keys[idx] = idx / n;
}

__global__ void get_first(double* in, double* out, int num_chunks, int n)
{
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx>=num_chunks)
        return;
    out[idx] = in[idx * n];
}

__global__ void map2(tuple4* tups, double* b, double* b0s, int total_size, int n)
{
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx>=total_size)
        return;
    tuple4 t = tups[idx];
    double b0 = b0s[idx / n];
    b[idx] = (t.a*b0 + t.b) / (t.c*b0 + t.d);
    
}

__global__
void map3(tuple2* tups, double* a, double* b, double* d, int total_size, int n)
{
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx>=total_size)
        return;
    tuple2 t;
    if (idx % n == 0)
    {
        t.a = 0;
        t.b = 1;

    }
    else
    {
        t.a = d[idx];
        t.b = -a[idx]/b[idx-1];
    }
    tups[idx] = t;
}

__global__ void map4(tuple2* tups, double* d, double* d0s, int total_size, int n)
{
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx>=total_size)
        return;
    tuple2 t = tups[idx];
    double d0 = d0s[idx / n];
    d[idx] = t.a + t.b*d0;
}

__global__ void getLastDiv(double* d, double* b, double* lastDiv, int num_chunks, int n)
{
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx>=num_chunks)
        return;
    const int n1 = idx * n + (n-1);
    lastDiv[idx] = d[n1]/b[n1]; 
}

__global__ void map5(tuple2* tups, double* b, double* c, double* d, int total_size, int n)
{
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx>=total_size)
        return;
    const unsigned int revIdx = n * (idx / n) + (n - (idx%n) - 1);
    tuple2 t;
    if (idx % n == 0)
    {
        t.a = 0;
        t.b = 1;

    }
    else
    {
        t.a = d[revIdx]/b[revIdx];
        t.b = -c[revIdx]/b[revIdx];
    }
    tups[idx] = t;
}

__global__ void map6(tuple2* tups, double* lastDivs, double* d, int total_size, int n)
{
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx>=total_size)
        return;
    const unsigned int revIdx = n * (idx / n) + (n - (idx%n) - 1);
    tuple2 t = tups[idx];
    d[revIdx] =  t.a + t.b*lastDivs[idx / n];
}

__global__
void execute(
    const double *a,
    const double *b,
    const double *c,
    const double *d,
    double *solution,
    int total_size
){
    const int m = 115;
    const size_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * m;

    if (idx >= total_size) {
        return;
    }

    double cp[m];
    double dp[m];

    cp[0] = c[idx] / b[idx];
    dp[0] = d[idx] / b[idx];

    for (ptrdiff_t j = 1; j < m; ++j) {
        const double norm_factor = b[idx+j] - a[idx+j] * cp[j-1];
        cp[j] = c[idx+j] / norm_factor;
        dp[j] = (d[idx+j] - a[idx+j] * dp[j-1]) / norm_factor;
    }

    solution[idx + m-1] = dp[m-1];
    for (ptrdiff_t j=m-2; j >= 0; --j) {
        solution[idx + j] = dp[j] - cp[j] * solution[idx + j+1];
    }
}