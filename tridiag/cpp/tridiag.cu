#include <iostream>
#include <cstdlib>
#include <chrono>
#include <sys/time.h>
#include <string>
#include <stdio.h>
#include <random>
#include <time.h> 
#include <fstream>
// CUDA / THRUST
#include <cuda_runtime.h>
#include <thrust/scan.h>
#include <thrust/device_ptr.h>
// Project internal
#include "tridiag_kernels.cu"
#include "include/device_utils.cu.h"


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

#define BENCH(PROGRAM, NAME, GPU_RUNS) { struct timeval t_start, t_end, t_diff;\
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

void verify(DTYPE* out_naive, DTYPE* out_parallel, unsigned int total_size)
{
    bool valid = true;
    double totalerr = 0;
    double maxabs = 0;
    double maxrel = 0;
    for (unsigned int i = 0 ; i < total_size ; i++)
    {
        double abs = fabs(out_naive[i] - out_parallel[i]);
        totalerr += abs;
        double rel = abs / out_naive[i];
        if (rel > maxrel)
                maxrel = rel;
        if (abs > maxabs)
                maxabs = abs;
        if (rel >= 0.000001)
        {
            valid = false;
        }
    }
    if (valid)
        std::cout << "\nVALID \n- Total abs error: " << totalerr << "\n - Max abs error: " << maxabs << "\n - Max rel error: " << maxrel << "\n" << std::endl;
    else
        std::cout << "\nINVALID \n- Total abs error: " << totalerr << "\n - Max abs error: " << maxabs << "\n- Max rel error: " << maxrel << "\n" << std::endl;
}

void transposeMats(DTYPE* a, DTYPE* b, DTYPE* c, DTYPE* d,
    DTYPE* a_t, DTYPE* b_t, DTYPE* c_t, DTYPE* d_t,
     int xdim, int ydim, int total_size)
{
    int num_b_x = (xdim + TILE_DIM -1) / TILE_DIM;
    int num_b_y = (ydim + TILE_DIM -1) / TILE_DIM;

    dim3 dimGrid(num_b_x, num_b_y, 1);
    dim3 dimBlock(TILE_DIM, BLOCK_ROWS, 1);
    transpose4<<<dimGrid, dimBlock>>>(a, b, c, d, a_t, b_t, c_t, d_t, xdim, ydim, total_size);
    SYNCPEEK

}
void transposeMat(DTYPE* m, DTYPE* m_t, int xdim, int ydim, int total_size)
{
    int num_b_x = (xdim + TILE_DIM -1) / TILE_DIM;
    int num_b_y = (ydim + TILE_DIM -1) / TILE_DIM;

    dim3 dimGrid(num_b_x, num_b_y, 1);
    dim3 dimBlock(TILE_DIM, BLOCK_ROWS, 1);
    transpose<<<dimGrid, dimBlock>>>(m, m_t, xdim, ydim, total_size);
    SYNCPEEK

}


void tridiag_naive(DTYPE* a, DTYPE* b, DTYPE* c, DTYPE* d, DTYPE* out, unsigned int num_chunks, unsigned int n)
{
    for (unsigned int chunk = 0 ; chunk < num_chunks ; chunk++)
    {
        unsigned int chunk_start = chunk * n;

        for (unsigned int i = chunk_start+1 ; i < chunk_start + n ; i++)
        {
            DTYPE w = a[i] / b[i - 1];
            b[i] += -w * c[i - 1];
            d[i] += -w * d[i - 1];
        }
        out[chunk_start+n-1] = d[chunk_start+n-1] / b[chunk_start+n-1];
        for (int i = chunk_start+n-2; i >= (int)chunk_start; i--)
        {
            out[i] = (d[i] - c[i] * out[i + 1]) / b[i];
        }
        
    }
}


inline void tridiag_parallel(DTYPE* a, DTYPE* b, DTYPE* c, DTYPE* d, DTYPE* out, unsigned int num_chunks, unsigned int n, unsigned int total_size, bool use_const = false)
{
    
    unsigned int num_blocks = (num_chunks + BLOCK_SIZE-1) / BLOCK_SIZE; 
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

inline void tridiag_parallel_coalesced(DTYPE* a_dev, DTYPE* b_dev, DTYPE* c_dev, DTYPE* d_dev,
    DTYPE* a_dev_t, DTYPE* b_dev_t, DTYPE* c_dev_t, DTYPE* d_dev_t, DTYPE* solution, DTYPE* solution_t,
     int n, int num_chunks, int total_size)
{
    transposeMats(a_dev, b_dev, c_dev, d_dev, a_dev_t, b_dev_t, c_dev_t, d_dev_t, n, num_chunks, total_size);
    unsigned int num_blocks_chunk = (num_chunks + BLOCK_SIZE-1) / BLOCK_SIZE;
    execute_coalesced_const<<<num_blocks_chunk, BLOCK_SIZE>>>(a_dev_t, b_dev_t, c_dev_t, d_dev_t, solution, num_chunks);
    SYNCPEEK
    transposeMat(solution, solution_t, num_chunks, n, total_size);
}

inline void tridiag_parallel_coalesced_no_const(DTYPE* a_dev, DTYPE* b_dev, DTYPE* c_dev, DTYPE* d_dev,
    DTYPE* a_dev_t, DTYPE* b_dev_t, DTYPE* c_dev_t, DTYPE* d_dev_t, DTYPE* solution, DTYPE* solution_t,
     int n, int num_chunks, int total_size)
{
    transposeMats(a_dev, b_dev, c_dev, d_dev, a_dev_t, b_dev_t, c_dev_t, d_dev_t, n, num_chunks, total_size);
    unsigned int num_blocks_chunk = (num_chunks + BLOCK_SIZE-1) / BLOCK_SIZE;
    execute_coalesced<<<num_blocks_chunk, BLOCK_SIZE>>>(a_dev_t, b_dev_t, c_dev_t, d_dev_t, solution, n, num_chunks);
    SYNCPEEK
    transposeMat(solution, solution_t, num_chunks, n, total_size);
}

inline void tridiag_thrust_seqRec1(DTYPE* a, DTYPE* b, DTYPE* c, DTYPE* d, tuple4<DTYPE>* tups, tuple2<DTYPE>* tups2, unsigned int* keys, DTYPE* firstBuf, unsigned int num_chunks, unsigned int n, unsigned int total_size, bool use_const = false)
{
  
    unsigned int num_blocks = (total_size + BLOCK_SIZE-1) / BLOCK_SIZE;
    unsigned int num_blocks_chunk = (num_chunks + BLOCK_SIZE-1) / BLOCK_SIZE;
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
    
    get_first_elem_in_chunk<<<num_blocks_chunk, BLOCK_SIZE>>>(d, firstBuf, num_chunks, n);
    SYNCPEEK

    create_tuple2_r2<<<num_blocks, BLOCK_SIZE>>>(tups2, a, b, d, total_size, n);
    SYNCPEEK

    auto assOp2 = [] __device__ (tuple2<DTYPE> a, tuple2<DTYPE> b) {
        tuple2<DTYPE> t;
        t.a = b.a + b.b*a.a;
        t.b = a.b*b.b; 
        return t; 
    };

    thrust::device_ptr<tuple2<DTYPE>> tup_ptr2(tups2);
    thrust::device_ptr<unsigned int> keys_ptr(keys);
    thrust::equal_to<unsigned int> eq;
    thrust::inclusive_scan_by_key(keys_ptr, keys_ptr + total_size, tup_ptr2, tup_ptr2, eq, assOp2);
    SYNCPEEK

    combine_tuple2_r2<<<num_blocks, BLOCK_SIZE>>>(tups2, keys, d, firstBuf, total_size, n);
    SYNCPEEK

    // recurrence 3
    get_last_yb_div_in_chunk<<<num_blocks_chunk, BLOCK_SIZE>>>(d, b, firstBuf, num_chunks, n);
    SYNCPEEK

    create_tuple2_r3<<<num_blocks, BLOCK_SIZE>>>(tups2, keys, b, c, d, total_size, n);
    SYNCPEEK

    thrust::inclusive_scan_by_key(keys_ptr, keys_ptr + total_size, tup_ptr2, tup_ptr2, eq, assOp2);
    SYNCPEEK

    combine_tuple2_and_reverse_r3<<<num_blocks, BLOCK_SIZE>>>(tups2, keys, firstBuf, d, total_size, n);
    SYNCPEEK
}


inline void tridiag_shared(DTYPE* a, DTYPE* b, DTYPE* c, DTYPE* d, DTYPE* out, int num_chunks, int n)
{
    if (n > 1024)
    {
        return;
    }
    // round blocksize to next multiple of 32 bigger than n
    const int blocksize = ((n + 31) / 32) * 32;
    int num_blocks = num_chunks;
    size_t shared_mem_cnt = 4*blocksize*sizeof(DTYPE);
    tridiag_shared<<<num_blocks, blocksize, shared_mem_cnt>>>(a,b,c,d,out,n);
    SYNCPEEK
}


inline unsigned long int tridiag_thrust(DTYPE* a, DTYPE* b, DTYPE* c, DTYPE* d, tuple4<DTYPE>* tups, tuple2<DTYPE>* tups2, unsigned int* keys, DTYPE* firstBuf, int num_chunks, int n, int total_size)
{
    // recurrence 1
    int num_blocks = (total_size + BLOCK_SIZE-1) / BLOCK_SIZE;
    int num_blocks_chunk = (num_chunks + BLOCK_SIZE-1) / BLOCK_SIZE;

    unsigned long int mem_usage = 0;
    int dsize = sizeof(DTYPE);
    mem_usage += (3*dsize + 4*dsize) * total_size;
    // MEMUSAGE: R: 3*DTYPE * total_size  W: 4*DTYPE * total_size
    create_tuple4_r1<<<num_blocks, BLOCK_SIZE>>>(a, b, c, tups, total_size, n);
    SYNCPEEK

    // MEMUSAGE: R: total_size * 4 bytes W: 0
    mem_usage += (sizeof(uint)) * total_size;
    generate_keys<<<num_blocks, BLOCK_SIZE>>>(keys, total_size, n);
    SYNCPEEK


    // R: num_chunks * DTYPE, W: num_chunks * DTYPE
    mem_usage += (2*dsize) * num_chunks;
    get_first_elem_in_chunk<<<num_blocks_chunk, BLOCK_SIZE>>>(b, firstBuf, num_chunks, n);
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
    // auto assOp1 = [] __device__ (tuple4<DTYPE> a, tuple4<DTYPE> b) {
    //     tuple4<DTYPE> t;
    //     t.a = (1 + b.b*a.c/(a.a * b.a));
    //     t.b = (a.b/(a.a) + b.b*a.d/(a.a * b.a));
    //     t.c = (b.c/b.a) + b.d*a.c/(a.a * b.a);
    //     t.d = (b.c*a.b/(a.a * b.a) + b.d*a.d/(a.a * b.a));
    //     return t; 
    // };
    thrust::device_ptr<unsigned int> keys_ptr(keys);
    thrust::equal_to<unsigned int> eq;
    thrust::device_ptr<tuple4<DTYPE>> tup_ptr(tups);
    // MEMUSAGE: R: (4*uint + 4*DTYPE) totalsize, W: 4*DTYPE
    mem_usage += (sizeof(uint) + 4*dsize + 4*dsize) * total_size;
    thrust::inclusive_scan_by_key(keys_ptr, keys_ptr + total_size, tup_ptr, tup_ptr, eq, assOp1);
    SYNCPEEK

    mem_usage += (4*dsize + sizeof(uint) + dsize + dsize) * total_size;
    combine_tuple4_r1<<<num_blocks, BLOCK_SIZE>>>(tups, keys, b, firstBuf, total_size, n);
    SYNCPEEK
    // recurrence2

    mem_usage += (2*dsize) * num_chunks;
    get_first_elem_in_chunk<<<num_blocks_chunk, BLOCK_SIZE>>>(d, firstBuf, num_chunks, n);
    SYNCPEEK

    mem_usage += (3*dsize + 2*dsize) * total_size;
    create_tuple2_r2<<<num_blocks, BLOCK_SIZE>>>(tups2, a, b, d, total_size, n);
    SYNCPEEK

    auto assOp2 = [] __device__ (tuple2<DTYPE> a, tuple2<DTYPE> b) {
        tuple2<DTYPE> t;
        t.a = b.a + b.b*a.a;
        t.b = a.b*b.b; 
        return t; 
    };
    thrust::device_ptr<tuple2<DTYPE>> tup_ptr2(tups2);
    mem_usage += (2*dsize + sizeof(uint) + 2*dsize) * total_size;
    thrust::inclusive_scan_by_key(keys_ptr, keys_ptr + total_size, tup_ptr2, tup_ptr2, eq, assOp2);
    SYNCPEEK

    mem_usage += (2*dsize + sizeof(uint) + dsize + dsize) * total_size;
    combine_tuple2_r2<<<num_blocks, BLOCK_SIZE>>>(tups2, keys, d, firstBuf, total_size, n);
    SYNCPEEK
    // recurrence 3
    mem_usage += (3*dsize) * total_size;
    get_last_yb_div_in_chunk<<<num_blocks_chunk, BLOCK_SIZE>>>(d, b, firstBuf, num_chunks, n);
    SYNCPEEK

    mem_usage += (sizeof(uint) + 3*dsize + 2*dsize) * total_size;
    create_tuple2_r3<<<num_blocks, BLOCK_SIZE>>>(tups2, keys, b, c, d, total_size, n);
    SYNCPEEK

    mem_usage += (2*dsize + sizeof(uint) + 2*dsize) * total_size;
    thrust::inclusive_scan_by_key(keys_ptr, keys_ptr + total_size, tup_ptr2, tup_ptr2, eq, assOp2);
    SYNCPEEK
    mem_usage += (sizeof(uint) + 2*dsize + dsize + dsize) * total_size;
    combine_tuple2_and_reverse_r3<<<num_blocks, BLOCK_SIZE>>>(tups2, keys, firstBuf, d, total_size, n);
    SYNCPEEK
    return mem_usage;
}

void save_dataset(const DTYPE* a, const DTYPE* b, const DTYPE* c, const DTYPE* d, int num_chunks, int n)
{
    std::cout << "writing dataset" << std::endl;
    std::ofstream dataset("../dataset.test", std::ios::out | std::ios::binary);

    const char* f32 = " f32";
    const char* f64 = " f64";
    std::uint64_t num_chunks_64 = num_chunks;
    std::uint64_t n64 = n;
    dataset << "b" << (char)2 << (char)2;
    if (sizeof(DTYPE) == sizeof(double))
    {
        dataset << f64;
    }
    else if (sizeof(DTYPE) == sizeof(float))
    {
        dataset << f32;
    }
    else
    {
        return;
    }
    std::cout << "writing dimensions" << std::endl;
    dataset.write((char*) &num_chunks_64, 8);
    dataset.write((char*) &n64, 8);
    for (int i = 0 ; i < num_chunks ; i++)
    {
        for (int j = 0; j<n; j++)
        {
            dataset.write((char*) (a + (i*n + j)), sizeof(DTYPE));
        }
    }
    dataset.close();
}



int main(int argc, char** argv)
{
    if (argc < 3)
        return -1;
    
    unsigned int num_chunks = std::atoi(argv[1]); // 57600
    unsigned int n = std::atoi(argv[2]); // 115
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
    generator.seed(12);//std::chrono::system_clock::now().time_since_epoch().count());
    std::uniform_real_distribution<DTYPE> dominant(5,15);
    std::uniform_real_distribution<DTYPE> nondominant(0.005,0.015);
    for (unsigned int i = 0 ; i < total_size ; i++)
    {
        a[i] = nondominant(generator);
        b[i] = dominant(generator); // distribution(generator);
        c[i] = nondominant(generator);// distribution(generator);
        d[i] = dominant(generator); // distribution(generator);
        b_seq[i] = b[i];
        d_seq[i] = d[i];
    }
    // save_dataset(a,b,c,d,num_chunks,n);
    // compute reference solution on cpu
    tridiag_naive(a, b_seq, c, d_seq, out_naive, num_chunks, n);
    
    DTYPE* a_dev;
    DTYPE* b_dev;
    DTYPE* c_dev;
    DTYPE* d_dev;
    DTYPE* out_dev;
    DTYPE* a_dev_t;
    DTYPE* b_dev_t;
    DTYPE* c_dev_t;
    DTYPE* d_dev_t;
    DTYPE* out_dev_t;
    tuple4<DTYPE>* tups;
    tuple2<DTYPE>* tups2;
    unsigned int* keys;
    DTYPE* firstBuf;

    const unsigned int mem_size = total_size*sizeof(DTYPE);
    std::cout << "Need " << (float)(10*mem_size
        + total_size*sizeof(tuple4<DTYPE>) 
        + total_size*sizeof(tuple2<DTYPE>)
        + total_size*sizeof(unsigned int)
        + num_chunks*sizeof(DTYPE)) / 1048576.0 << "MB." << std::endl;
    gpuErrchk(cudaMalloc((void**)&a_dev, mem_size));
    gpuErrchk(cudaMalloc((void**)&b_dev, mem_size));
    gpuErrchk(cudaMalloc((void**)&c_dev, mem_size));
    gpuErrchk(cudaMalloc((void**)&d_dev, mem_size));
    gpuErrchk(cudaMalloc((void**)&out_dev, mem_size));
    gpuErrchk(cudaMalloc((void**)&a_dev_t, mem_size));
    gpuErrchk(cudaMalloc((void**)&b_dev_t, mem_size));
    gpuErrchk(cudaMalloc((void**)&c_dev_t, mem_size));
    gpuErrchk(cudaMalloc((void**)&d_dev_t, mem_size));
    gpuErrchk(cudaMalloc((void**)&out_dev_t, mem_size));
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
    std::cout << "Naive parallel version with const N: ";
    verify(out_naive, out_parallel, total_size);

    gpuErrchk(cudaMemcpy(a_dev, a, mem_size, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(b_dev, b, mem_size, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(c_dev, c, mem_size, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_dev, d, mem_size, cudaMemcpyHostToDevice));
    tridiag_parallel_coalesced_no_const(a_dev, b_dev, c_dev, d_dev, a_dev_t, b_dev_t, c_dev_t, d_dev_t, out_dev, out_dev_t, n, num_chunks, total_size);
    gpuErrchk(cudaMemcpy(out_parallel, out_dev_t, mem_size, cudaMemcpyDeviceToHost));
    std::cout << "Naive parallel coalesced version: ";
    verify(out_naive, out_parallel, total_size);

    gpuErrchk(cudaMemcpy(a_dev, a, mem_size, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(b_dev, b, mem_size, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(c_dev, c, mem_size, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_dev, d, mem_size, cudaMemcpyHostToDevice));
    tridiag_parallel_coalesced(a_dev, b_dev, c_dev, d_dev, a_dev_t, b_dev_t, c_dev_t, d_dev_t, out_dev, out_dev_t, n, num_chunks, total_size);
    gpuErrchk(cudaMemcpy(out_parallel, out_dev_t, mem_size, cudaMemcpyDeviceToHost));
    std::cout << "Naive parallel coalesced version with const N: ";
    verify(out_naive, out_parallel, total_size);

    gpuErrchk(cudaMemcpy(a_dev, a, mem_size, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(b_dev, b, mem_size, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(c_dev, c, mem_size, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_dev, d, mem_size, cudaMemcpyHostToDevice));
    unsigned long int flat_mem = tridiag_thrust(a_dev, b_dev, c_dev, d_dev, tups, tups2, keys, firstBuf, num_chunks, n, total_size);
    gpuErrchk(cudaMemcpy(out_parallel, d_dev, mem_size, cudaMemcpyDeviceToHost));
    std::cout << "Flat version: " << std::endl;
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
    std::cout << "Flat version with sequential first recurrence using const N: ";
    verify(out_naive, out_parallel, total_size);

    gpuErrchk(cudaMemcpy(a_dev, a, mem_size, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(b_dev, b, mem_size, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(c_dev, c, mem_size, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_dev, d, mem_size, cudaMemcpyHostToDevice));
    tridiag_shared(a_dev, b_dev, c_dev, d_dev, out_dev, num_chunks, n);
    gpuErrchk(cudaMemcpy(out_parallel, out_dev, mem_size, cudaMemcpyDeviceToHost));
    std::cout << "Flat version using shared mem: ";
    verify(out_naive, out_parallel, total_size);




    const int GPU_RUNS = 20;
    unsigned long int elapsed;
    BENCH(tridiag_parallel(a_dev, b_dev, c_dev, d_dev, out_dev, num_chunks, n, total_size), "Primitive parallel version", GPU_RUNS);
    BENCH(tridiag_parallel(a_dev, b_dev, c_dev, d_dev, out_dev, num_chunks, n, total_size, true), "Primitive parallel version using const", GPU_RUNS);
    BENCH(tridiag_parallel_coalesced_no_const(a_dev, b_dev, c_dev, d_dev, a_dev_t, b_dev_t, c_dev_t, d_dev_t, out_dev, out_dev_t, n, num_chunks, total_size), "Primitive parallel coalesced version", GPU_RUNS);
    BENCH(tridiag_parallel_coalesced(a_dev, b_dev, c_dev, d_dev, a_dev_t, b_dev_t, c_dev_t, d_dev_t, out_dev, out_dev_t, n, num_chunks, total_size), "Primitive parallel coalesced version using const", GPU_RUNS);
    BENCH(tridiag_thrust(a_dev, b_dev, c_dev, d_dev, tups, tups2, keys, firstBuf, num_chunks, n, total_size), "Flat version", GPU_RUNS);
    

    double elapsed_sec = (double) elapsed / 1000000.0;
    std::cout << "Bandwith: " << ((double)flat_mem / (1024*1024*1024))  / elapsed_sec << "GB/s" << std::endl;
    BENCH(tridiag_thrust_seqRec1(a_dev, b_dev, c_dev, d_dev, tups, tups2, keys, firstBuf, num_chunks, n, total_size), "Flat version with sequential first recurrence", GPU_RUNS);
    BENCH(tridiag_thrust_seqRec1(a_dev, b_dev, c_dev, d_dev, tups, tups2, keys, firstBuf, num_chunks, n, total_size, true), "Flat version with sequential first recurrence using const", GPU_RUNS);
    BENCH(tridiag_shared(a_dev, b_dev, c_dev, d_dev, out_dev, num_chunks, n), "Flat version using shared mem", GPU_RUNS);

    gpuErrchk(cudaFree(a_dev));
    gpuErrchk(cudaFree(b_dev));
    gpuErrchk(cudaFree(c_dev));
    gpuErrchk(cudaFree(d_dev));
    gpuErrchk(cudaFree(out_dev));
    gpuErrchk(cudaFree(a_dev_t));
    gpuErrchk(cudaFree(b_dev_t));
    gpuErrchk(cudaFree(c_dev_t));
    gpuErrchk(cudaFree(d_dev_t));
    gpuErrchk(cudaFree(out_dev_t));
    gpuErrchk(cudaFree(tups));
    gpuErrchk(cudaFree(tups2));
    gpuErrchk(cudaFree(keys));
    gpuErrchk(cudaFree(firstBuf));
    return 0;
}

