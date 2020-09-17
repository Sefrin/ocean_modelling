#define BLOCK_SIZE 256
#define TRIDIAG_INNER_DIM 115
// #define TOTAL_SIZE 6624000
#define DTYPE float
#define ERRCHK 0

#if ERRCHK
#define SYNCPEEK gpuErrchk(cudaDeviceSynchronize()); gpuErrchk(cudaPeekAtLastError()); 
#else
#define SYNCPEEK ;
#endif