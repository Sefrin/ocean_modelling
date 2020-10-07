#define BLOCK_SIZE 256
#define TRIDIAG_INNER_DIM 115
// #define TOTAL_SIZE 6624000
#define lgWARP      5
#define WARP        (1<<lgWARP)
#define DTYPE double
#define ERRCHK 1
#define TILE_DIM 32
#define BLOCK_ROWS 8
#if ERRCHK
#define SYNCPEEK gpuErrchk(cudaDeviceSynchronize()); gpuErrchk(cudaPeekAtLastError()); 
#else
#define SYNCPEEK ;
#endif