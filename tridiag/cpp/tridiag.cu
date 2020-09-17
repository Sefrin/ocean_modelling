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
#include <string>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

#define BENCH(PROGRAM, NAME, GPU_RUNS) {unsigned long int elapsed; struct timeval t_start, t_end, t_diff;\
    cudaDeviceSynchronize(); \
    gettimeofday(&t_start, NULL);   \
    for (int i = 0 ; i < GPU_RUNS ; i++)  \
        PROGRAM; \
    cudaDeviceSynchronize(); \
    gettimeofday(&t_end, NULL); \
    timeval_substract(&t_diff, &t_end, &t_start); \
    elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec) / GPU_RUNS; \
    printf(NAME " took %lu microseconds (%.2fms)\n",elapsed,elapsed/1000.0);}

int timeval_substract(struct timeval* result, struct timeval* t2, struct timeval* t1)
{
  unsigned int resolution=1000000;
  long int diff = (t2->tv_usec + resolution * t2->tv_sec) - (t1->tv_usec + resolution * t1->tv_sec);
  result->tv_sec = diff / resolution;
  result->tv_usec = diff % resolution;
  return (diff<0);
}

void tridiag_naive(DTYPE* a, DTYPE* b, DTYPE* c, DTYPE* d, DTYPE* out, int num_chunks, int n)
{
    for (int chunk = 0 ; chunk < num_chunks ; chunk++)
    {
        int chunk_start = chunk * n;

        for (int i = chunk_start+1 ; i < chunk_start + n ; i++)
        {
       
            DTYPE w = a[i] / b[i - 1];
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


inline void tridiag_parallel(DTYPE* a, DTYPE* b, DTYPE* c, DTYPE* d, DTYPE* out, int num_chunks, int n, int total_size, bool use_const = false)
{
    
    int num_blocks = (num_chunks + BLOCK_SIZE-1) / BLOCK_SIZE; 
    if (use_const)
    {
        execute<<<num_blocks, BLOCK_SIZE>>>(a, b, c, d, out, total_size);
        SYNCPEEK
    }
    else
    {
        execute_no_const<<<num_blocks, BLOCK_SIZE>>>(a, b, c, d, out, total_size, n);
        SYNCPEEK
    }
    

}

inline void tridiag_thrust_seqRec1(DTYPE* a, DTYPE* b, DTYPE* c, DTYPE* d, tuple4<DTYPE>* tups, tuple2<DTYPE>* tups2, unsigned int* keys, DTYPE* firstBuf, int num_chunks, int n, int total_size, bool use_const = false)
{
  
    int num_blocks = (total_size + BLOCK_SIZE-1) / BLOCK_SIZE;
    int num_blocks_chunk = (num_chunks + BLOCK_SIZE-1) / BLOCK_SIZE;
    generate_keys<<<num_blocks, BLOCK_SIZE>>>(keys, total_size, n);
    SYNCPEEK
    // recurrence 1
    if (use_const)
    {
        recurrence1<<<num_blocks_chunk, BLOCK_SIZE>>>(a,b,c,num_chunks);
        SYNCPEEK
    }
    else
    {
        recurrence1_no_const<<<num_blocks_chunk, BLOCK_SIZE>>>(a,b,c,num_chunks, n);
        SYNCPEEK
    }

    // recurrence2
    
    get_first_elem<<<num_blocks_chunk, BLOCK_SIZE>>>(d, firstBuf, num_chunks, n);
    SYNCPEEK
    map3<<<num_blocks, BLOCK_SIZE>>>(tups2, a, b, d, total_size, n);
    SYNCPEEK
    auto assOp2 = [] __device__ (tuple2<DTYPE> a, tuple2<DTYPE> b) {
        tuple2<DTYPE> t;
        t.a = b.a + b.b*a.a;
        t.b = a.b*b.b; 
        return t; 
    };
    thrust::device_ptr<tuple2<DTYPE>> tup_ptr2(tups2);
    // thrust::device_ptr<tuple2<DTYPE>> scan_ptr2(scan2);
    thrust::device_ptr<unsigned int> keys_ptr(keys);
    thrust::equal_to<unsigned int> eq;
    thrust::inclusive_scan_by_key(keys_ptr, keys_ptr + total_size, tup_ptr2, tup_ptr2, eq, assOp2);
    SYNCPEEK
    map4<<<num_blocks, BLOCK_SIZE>>>(tups2, keys, d, firstBuf, total_size, n);
    SYNCPEEK
    // recurrence 3
    getLastDiv<<<num_blocks_chunk, BLOCK_SIZE>>>(d, b, firstBuf, num_chunks, n);
    SYNCPEEK
    map5<<<num_blocks, BLOCK_SIZE>>>(tups2, keys, b, c, d, total_size, n);
    SYNCPEEK
    thrust::inclusive_scan_by_key(keys_ptr, keys_ptr + total_size, tup_ptr2, tup_ptr2, eq, assOp2);
    SYNCPEEK
    map6<<<num_blocks, BLOCK_SIZE>>>(tups2, keys, firstBuf, d, total_size, n);
    SYNCPEEK
}



inline void tridiag_thrust(DTYPE* a, DTYPE* b, DTYPE* c, DTYPE* d, tuple4<DTYPE>* tups, tuple2<DTYPE>* tups2, unsigned int* keys, DTYPE* firstBuf, int num_chunks, int n, int total_size)
{
    // recurrence 1
    int num_blocks = (total_size + BLOCK_SIZE-1) / BLOCK_SIZE;
    int num_blocks_chunk = (num_chunks + BLOCK_SIZE-1) / BLOCK_SIZE;

    firstMap<<<num_blocks, BLOCK_SIZE>>>(a,b,c,tups,total_size,n);
    SYNCPEEK

    
    generate_keys<<<num_blocks, BLOCK_SIZE>>>(keys, total_size, n);
    SYNCPEEK

    get_first_elem<<<num_blocks_chunk, BLOCK_SIZE>>>(b, firstBuf, num_chunks, n);
    SYNCPEEK

    auto assOp1 = [] __device__ (tuple4<DTYPE> a, tuple4<DTYPE> b) {
        DTYPE value = 1.0/(a.a * b.a);
        tuple4<DTYPE> t;
        t.a = (b.a*a.a + b.b*a.c)*value;
        t.b = (b.a*a.b + b.b*a.d)*value;
        t.c = (b.c*a.a + b.d*a.c)*value;
        t.d = (b.c*a.b + b.d*a.d)*value;
        return t; 
    };
    thrust::device_ptr<unsigned int> keys_ptr(keys);
    thrust::equal_to<unsigned int> eq;
    thrust::device_ptr<tuple4<DTYPE>> tup_ptr(tups);
    // thrust::device_ptr<tuple4<DTYPE>> scan_ptr(scan1);
    thrust::inclusive_scan_by_key(keys_ptr, keys_ptr + total_size, tup_ptr, tup_ptr, eq, assOp1);
    SYNCPEEK
    map2<<<num_blocks, BLOCK_SIZE>>>(tups, keys, b, firstBuf, total_size, n);
    SYNCPEEK
    // recurrence2
    
    get_first_elem<<<num_blocks_chunk, BLOCK_SIZE>>>(d, firstBuf, num_chunks, n);
    SYNCPEEK
    map3<<<num_blocks, BLOCK_SIZE>>>(tups2, a, b, d, total_size, n);
    SYNCPEEK
    auto assOp2 = [] __device__ (tuple2<DTYPE> a, tuple2<DTYPE> b) {
        tuple2<DTYPE> t;
        t.a = b.a + b.b*a.a;
        t.b = a.b*b.b; 
        return t; 
    };
    thrust::device_ptr<tuple2<DTYPE>> tup_ptr2(tups2);
    // thrust::device_ptr<tuple2<DTYPE>> scan_ptr2(scan2);
    thrust::inclusive_scan_by_key(keys_ptr, keys_ptr + total_size, tup_ptr2, tup_ptr2, eq, assOp2);
    SYNCPEEK
    map4<<<num_blocks, BLOCK_SIZE>>>(tups2, keys, d, firstBuf, total_size, n);
    SYNCPEEK
    // recurrence 3
    getLastDiv<<<num_blocks_chunk, BLOCK_SIZE>>>(d, b, firstBuf, num_chunks, n);
    SYNCPEEK
    map5<<<num_blocks, BLOCK_SIZE>>>(tups2, keys, b, c, d, total_size, n);
    SYNCPEEK
    thrust::inclusive_scan_by_key(keys_ptr, keys_ptr + total_size, tup_ptr2, tup_ptr2, eq, assOp2);
    SYNCPEEK
    map6<<<num_blocks, BLOCK_SIZE>>>(tups2, keys, firstBuf, d, total_size, n);
    SYNCPEEK
}

void verify(DTYPE* out_naive, DTYPE* out_parallel, unsigned int total_size)
{
    bool valid = true;
    double totalerr = 0;
    double maxabs = 0;
    for (unsigned int i = 0 ; i < total_size ; i++)
    {
        double abs = fabs(out_naive[i] - out_parallel[i]);
        totalerr += abs;

        if (abs > maxabs)
                maxabs = abs;
        if (abs >= 0.01)
        {
            
            // std::cout << i << std::endl;
            // std::cout << out_naive[i] << std::endl;
            // std::cout << out_parallel[i] << std::endl;
            // std::cout << fabs(out_naive[i] - out_parallel[i]) << std::endl;
            valid = false;
        }
    }
    if (valid)
        std::cout << "VALID - Total abs error: " << totalerr << " - Max abs error: " << maxabs << "\n" << std::endl;
    else
        std::cout << "INVALID - Total abs error: " << totalerr << " - Max abs error: " << maxabs << "\n" << std::endl;
}

int main(int argc, char** argv)
{
    if (argc < 3)
        return -1;
    
    int num_chunks = std::atoi(argv[1]); // 57600
    int n = std::atoi(argv[2]); // 115
    std::cout << "Running with " << num_chunks << " chunks and N = " << n << std::endl;
    const unsigned int total_size = n*num_chunks;
    std::cout << total_size << std::endl;
    DTYPE* a = new DTYPE[total_size];
    DTYPE* b = new DTYPE[total_size];
    DTYPE* b_seq = new DTYPE[total_size];
    DTYPE* c = new DTYPE[total_size];
    DTYPE* d = new DTYPE[total_size];
    DTYPE* d_seq = new DTYPE[total_size];
    DTYPE* out_naive = new DTYPE[total_size];
    DTYPE* out_parallel = new DTYPE[total_size];
    std::default_random_engine generator;
    std::uniform_real_distribution<DTYPE> distribution(0.0,1.0);
    for (int i = 0 ; i < total_size ; i++)
    {
        a[i] = distribution(generator);
        b[i] = distribution(generator);
        c[i] = distribution(generator);
        d[i] = distribution(generator);
        b_seq[i] = b[i];
        d_seq[i] = d[i];
    }

    // compute reference solution on cpu
    tridiag_naive(a, b_seq, c, d_seq, out_naive, num_chunks, n);

    DTYPE* a_dev;
    DTYPE* b_dev;
    DTYPE* c_dev;
    DTYPE* d_dev;
    DTYPE* out_dev;
    tuple4<DTYPE>* tups;
    tuple2<DTYPE>* tups2;
    unsigned int* keys;
    DTYPE* firstBuf;

    const int mem_size = total_size*sizeof(DTYPE);

    gpuErrchk(cudaMalloc((void**)&a_dev, mem_size));
    gpuErrchk(cudaMalloc((void**)&b_dev, mem_size));
    gpuErrchk(cudaMalloc((void**)&c_dev, mem_size));
    gpuErrchk(cudaMalloc((void**)&d_dev, mem_size));
    gpuErrchk(cudaMalloc((void**)&out_dev, mem_size));
    gpuErrchk(cudaMalloc((void**)&tups, total_size*sizeof(tuple4<DTYPE>)));
    gpuErrchk(cudaMalloc((void**)&tups2, total_size*sizeof(tuple2<DTYPE>)));
    gpuErrchk(cudaMalloc((void**)&keys, total_size*sizeof(unsigned int)));
    gpuErrchk(cudaMalloc((void**)&firstBuf, num_chunks*sizeof(DTYPE)));


    // verify results for each method:
    gpuErrchk(cudaMemcpy(a_dev, a, mem_size, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(b_dev, b, mem_size, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(c_dev, c, mem_size, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_dev, d, mem_size, cudaMemcpyHostToDevice));
    tridiag_parallel(a_dev, b_dev, c_dev, d_dev, out_dev, num_chunks, n, total_size);
    gpuErrchk(cudaMemcpy(out_parallel, out_dev, mem_size, cudaMemcpyDeviceToHost));
    std::cout << "Naive parallel version: ";
    verify(out_naive, out_parallel, total_size);

    gpuErrchk(cudaMemcpy(a_dev, a, mem_size, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(b_dev, b, mem_size, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(c_dev, c, mem_size, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_dev, d, mem_size, cudaMemcpyHostToDevice));
    tridiag_parallel(a_dev, b_dev, c_dev, d_dev, out_dev, num_chunks, n, total_size, true);
    gpuErrchk(cudaMemcpy(out_parallel, out_dev, mem_size, cudaMemcpyDeviceToHost));
    std::cout << "Naive parallel version with const M: ";
    verify(out_naive, out_parallel, total_size);

    gpuErrchk(cudaMemcpy(a_dev, a, mem_size, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(b_dev, b, mem_size, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(c_dev, c, mem_size, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_dev, d, mem_size, cudaMemcpyHostToDevice));
    tridiag_thrust(a_dev, b_dev, c_dev, d_dev, tups, tups2, keys, firstBuf, num_chunks, n, total_size);
    gpuErrchk(cudaMemcpy(out_parallel, d_dev, mem_size, cudaMemcpyDeviceToHost));
    std::cout << "Flat version: ";
    verify(out_naive, out_parallel, total_size);

    gpuErrchk(cudaMemcpy(a_dev, a, mem_size, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(b_dev, b, mem_size, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(c_dev, c, mem_size, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_dev, d, mem_size, cudaMemcpyHostToDevice));
    tridiag_thrust_seqRec1(a_dev, b_dev, c_dev, d_dev, tups, tups2, keys, firstBuf, num_chunks, n, total_size);
    gpuErrchk(cudaMemcpy(out_parallel, d_dev, mem_size, cudaMemcpyDeviceToHost));
    std::cout << "Flat version with sequential first recurrence: ";
    verify(out_naive, out_parallel, total_size);

    gpuErrchk(cudaMemcpy(a_dev, a, mem_size, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(b_dev, b, mem_size, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(c_dev, c, mem_size, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_dev, d, mem_size, cudaMemcpyHostToDevice));
    tridiag_thrust_seqRec1(a_dev, b_dev, c_dev, d_dev, tups, tups2, keys, firstBuf, num_chunks, n, total_size, true);
    gpuErrchk(cudaMemcpy(out_parallel, d_dev, mem_size, cudaMemcpyDeviceToHost));
    std::cout << "Flat version with sequential first recurrence using const M: ";
    verify(out_naive, out_parallel, total_size);


    const int GPU_RUNS = 20;
    
    BENCH(tridiag_parallel(a_dev, b_dev, c_dev, d_dev, out_dev, num_chunks, n, total_size), "Primitive parallel version", GPU_RUNS);
    BENCH(tridiag_parallel(a_dev, b_dev, c_dev, d_dev, out_dev, num_chunks, n, total_size, true), "Primitive parallel version using const", GPU_RUNS);
    BENCH(tridiag_thrust(a_dev, b_dev, c_dev, d_dev, tups, tups2, keys, firstBuf, num_chunks, n, total_size), "Flat version", GPU_RUNS);
    BENCH(tridiag_thrust_seqRec1(a_dev, b_dev, c_dev, d_dev, tups, tups2, keys, firstBuf, num_chunks, n, total_size), "Flat version with sequential first recurrence", GPU_RUNS);
    BENCH(tridiag_thrust_seqRec1(a_dev, b_dev, c_dev, d_dev, tups, tups2, keys, firstBuf, num_chunks, n, total_size, true), "Flat version with sequential first recurrence using const", GPU_RUNS);
 

    gpuErrchk(cudaFree(a_dev));
    gpuErrchk(cudaFree(b_dev));
    gpuErrchk(cudaFree(c_dev));
    gpuErrchk(cudaFree(d_dev));
    gpuErrchk(cudaFree(tups));
    gpuErrchk(cudaFree(tups2));
    gpuErrchk(cudaFree(keys));
    gpuErrchk(cudaFree(firstBuf));
    return 0;
}

