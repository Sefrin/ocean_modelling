#include <iostream>
#include <cstdlib>

#include <time.h> 
#include <cuda_runtime.h>
#include "tridiag_kernels.cu"
#include <stdio.h>
#include <random>
#include "data_structures.h"

#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <chrono>
#include <sys/time.h>
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

int timeval_substract(struct timeval* result, struct timeval* t2, struct timeval* t1)
{
  unsigned int resolution=1000000;
  long int diff = (t2->tv_usec + resolution * t2->tv_sec) - (t1->tv_usec + resolution * t1->tv_sec);
  result->tv_sec = diff / resolution;
  result->tv_usec = diff % resolution;
  return (diff<0);
}

void tridiag_naive(double* a, double* b, double* c, double* d, double* out, int num_chunks, int n)
{
    for (int chunk = 0 ; chunk < num_chunks ; chunk++)
    {
        int chunk_start = chunk * n;

        for (int i = chunk_start+1 ; i < chunk_start + n ; i++)
        {
       
            double w = a[i] / b[i - 1];
            b[i] += -w * c[i - 1];
            d[i] += -w * d[i - 1];
        }
        out[chunk_start+n-1] = d[chunk_start+n-1] / b[chunk_start+n-1];

    
        for (int i = chunk_start+n-2; i >= chunk_start; i--)
        {
            out[i] = (d[i] - c[i] * out[i + 1]) / b[i];
        }
        
    }
}


inline void tridiag_parallel(double* a, double* b, double* c, double* d, double* out, int num_chunks, int n, int total_size)
{
    
    int num_blocks = (num_chunks + BLOCK_SIZE-1) / BLOCK_SIZE; 
    execute<<<num_blocks, BLOCK_SIZE>>>(a, b, c, d, out, total_size);


}




inline void tridiag_thrust(double* a, double* b, double* c, double* d, tuple4* tups, tuple2* tups2, unsigned int* keys, double* firstBuf, int num_chunks, int n, int total_size)
{
    // recurrence 1
    int num_blocks = (total_size + BLOCK_SIZE-1) / BLOCK_SIZE;
    // map1<<<num_blocks, BLOCK_SIZE>>>(a,b,c,tups,total_size,n);

    int num_blocks_chunk = (num_chunks + BLOCK_SIZE-1) / BLOCK_SIZE;
    generate_keys<<<num_blocks, BLOCK_SIZE>>>(keys, total_size, n);

    // get_first<<<num_blocks_chunk, BLOCK_SIZE>>>(b, firstBuf, num_chunks, n);

    // auto assOp1 = [] __device__ (tuple4 a, tuple4 b) {
    //     double value = 1.0/(a.a * b.a);
    //     tuple4 t;
    //     t.a = (b.a*a.a + b.b*a.c)*value;
    //     t.b = (b.a*a.b + b.b*a.d)*value;
    //     t.c = (b.c*a.a + b.d*a.c)*value;
    //     t.d = (b.c*a.b + b.d*a.d)*value;
    //     return t; 
    // };
    thrust::device_ptr<unsigned int> keys_ptr(keys);
    thrust::equal_to<unsigned int> eq;
    // thrust::device_ptr<tuple4> tup_ptr(tups);
    // thrust::device_ptr<tuple4> scan_ptr(scan1);
    // tuple4 init_tup;
    // init_tup.a = 1;
    // init_tup.b = 0;
    // init_tup.c = 0;
    // init_tup.d = 1;
    // thrust::inclusive_scan_by_key(keys_ptr, keys_ptr + total_size, tup_ptr, tup_ptr, eq, assOp1);
    recurrence1<<<num_blocks_chunk, BLOCK_SIZE>>>(a,b,c,num_chunks);
    // gpuErrchk(cudaDeviceSynchronize());
    // gpuErrchk(cudaPeekAtLastError());
    map2<<<num_blocks, BLOCK_SIZE>>>(tups, keys, b, firstBuf, total_size, n);

    // recurrence2
    
    get_first<<<num_blocks_chunk, BLOCK_SIZE>>>(d, firstBuf, num_chunks, n);

    map3<<<num_blocks, BLOCK_SIZE>>>(tups2, a, b, d, total_size, n);

    auto assOp2 = [] __device__ (tuple2 a, tuple2 b) {
        tuple2 t;
        t.a = b.a + b.b*a.a;
        t.b = a.b*b.b; 
        return t; 
    };
    thrust::device_ptr<tuple2> tup_ptr2(tups2);
    // thrust::device_ptr<tuple2> scan_ptr2(scan2);
    // tuple2 init_tup2;
    // init_tup2.a = 0;
    // init_tup2.b = 1;
    thrust::inclusive_scan_by_key(keys_ptr, keys_ptr + total_size, tup_ptr2, tup_ptr2, eq, assOp2);

    map4<<<num_blocks, BLOCK_SIZE>>>(tups2, keys, d, firstBuf, total_size, n);

    // recurrence 3
    getLastDiv<<<num_blocks_chunk, BLOCK_SIZE>>>(d, b, firstBuf, num_chunks, n);

    map5<<<num_blocks, BLOCK_SIZE>>>(tups2, keys, b, c, d, total_size, n);

    thrust::inclusive_scan_by_key(keys_ptr, keys_ptr + total_size, tup_ptr2, tup_ptr2, eq, assOp2);

    map6<<<num_blocks, BLOCK_SIZE>>>(tups2, keys, firstBuf, d, total_size, n);

}



int main()
{
    const int num_chunks = 57600;
    const int n = M;
    const int total_size = n*num_chunks;
    double* a = new double[total_size];
    double* b = new double[total_size];
    double* b_seq = new double[total_size];
    double* c = new double[total_size];
    double* d = new double[total_size];
    double* d_seq = new double[total_size];
    double* out_naive = new double[total_size];
    double* out_parallel = new double[total_size];
    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(0.0,1.0);
    for (int i = 0 ; i < total_size ; i++)
    {
        a[i] = distribution(generator);
        b[i] = distribution(generator);
        c[i] = distribution(generator);
        d[i] = distribution(generator);
        b_seq[i] = b[i];
        d_seq[i] = d[i];
    }

    tridiag_naive(a, b_seq, c, d_seq, out_naive, num_chunks, n);

    double* a_dev;
    double* b_dev;
    double* c_dev;
    double* d_dev;
    double* out_dev;
    const int mem_size = total_size*sizeof(double);
    tuple4* tups;
    tuple2* tups2;
    unsigned int* keys;
    double* firstBuf;
    cudaMalloc((void**)&a_dev, mem_size);
    cudaMalloc((void**)&b_dev, mem_size);
    cudaMalloc((void**)&c_dev, mem_size);
    cudaMalloc((void**)&d_dev, mem_size);
    cudaMalloc((void**)&out_dev, mem_size);
    cudaMemcpy(a_dev, a, mem_size, cudaMemcpyHostToDevice);
    cudaMemcpy(b_dev, b, mem_size, cudaMemcpyHostToDevice);
    cudaMemcpy(c_dev, c, mem_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dev, d, mem_size, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&tups, total_size*sizeof(tuple4));
    cudaMalloc((void**)&tups2, total_size*sizeof(tuple2));
    cudaMalloc((void**)&keys, total_size*sizeof(unsigned int));
    cudaMalloc((void**)&firstBuf, num_chunks*sizeof(double));
    const int GPU_RUNS = 20;
    // unsigned long int elapsed; struct timeval t_start, t_end, t_diff;
    cudaDeviceSynchronize();
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0 ; i < GPU_RUNS ; i++)
        tridiag_parallel(a_dev, b_dev, c_dev, d_dev, out_dev, num_chunks, n, total_size);
    cudaDeviceSynchronize();
    auto finish = std::chrono::high_resolution_clock::now();
    // -----------------------------
    
    // timeval_substract(&t_diff, &t_end, &t_start);
    auto elapsed =  std::chrono::duration_cast<std::chrono::microseconds>(finish - start).count() / GPU_RUNS; //(t_diff.tv_sec*1e6+t_diff.tv_usec) / GPU_RUNS;
    printf("GPU took %lu microseconds (%.2fms)\n",elapsed,elapsed/1000.0);
    cudaDeviceSynchronize();
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0 ; i < GPU_RUNS ; i++)
        tridiag_thrust(a_dev, b_dev, c_dev, d_dev, tups, tups2, keys, firstBuf, num_chunks, n, total_size);
    cudaDeviceSynchronize();
    finish = std::chrono::high_resolution_clock::now();
    // -----------------------------
  
    // timeval_substract(&t_diff, &t_end, &t_start);

    elapsed = std::chrono::duration_cast<std::chrono::microseconds>(finish - start).count() / GPU_RUNS; //t_diff.tv_sec*1e6+t_diff.tv_usec) / GPU_RUNS;
    printf("thrust took %lu microseconds (%.2fms)\n",elapsed,elapsed/1000.0);

    int device;
    cudaGetDevice(&device);

    struct cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);
    std::cout << props.name << std::endl;

    cudaMemcpy(out_parallel, d_dev, mem_size, cudaMemcpyDeviceToHost);

    bool valid = true;
    for (unsigned int i = 0 ; i < total_size ; i++)
    {
        
        if (fabs(out_naive[i] - out_parallel[i]) >= 0.01)
        {
            std::cout << i << std::endl;
            std::cout << out_naive[i] << std::endl;
            std::cout << out_parallel[i] << std::endl;
            std::cout << fabs(out_naive[i] - out_parallel[i]) << std::endl;
            valid = false;
            break;
        }
    }
    if (valid)
        std::cout << "VALID\n" << std::endl;
    else
        std::cout << "INVALID\n" << std::endl;
    cudaFree(a_dev);
    cudaFree(b_dev);
    cudaFree(c_dev);
    cudaFree(d_dev);
    return 0;
}

