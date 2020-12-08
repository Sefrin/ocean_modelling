
#include <array>
#include <cstddef>
#include <bit>
#include "cuda_superbee_kernels.h"
#include "cuda_utils.h"
#include <iostream>
template <typename DTYPE>
__device__ inline DTYPE limiter(DTYPE cr)
{
    return max(0.0, max(min(1.0, 2.0 * cr), min(2.0, cr)));
}

template <typename DTYPE>
__device__ inline DTYPE calcFlux
    (
        const DTYPE velfac,
        const DTYPE velS,
        const DTYPE dt_tracer,
        const DTYPE dx,
        const DTYPE varS,
        const DTYPE varSM1,
        const DTYPE varSP1,
        const DTYPE varSP2,
        const DTYPE maskS,
        const DTYPE maskSM1,
        const DTYPE maskSP1
    )
{
        const DTYPE scaledVel =  velfac * velS;
        const DTYPE uCFL = abs(scaledVel * dt_tracer / dx);
        const DTYPE rjp = (varSP2 - varSP1) * maskSP1;
        const DTYPE rj = (varSP1 - varS) * maskS;
        const DTYPE rjm = (varS - varSM1) * maskSM1;
        DTYPE cr;
        DTYPE divisor = rj;
        const DTYPE epsilon = 1e-20;
        if (abs(divisor) < epsilon)
        {
            divisor = epsilon;
        }
        if (velS > 0)
        {
            cr = rjm / divisor;
        }
        else
        {
            cr = rjp / divisor;
        }
        cr = limiter(cr);

        return scaledVel * (varSP1 + varS) * 0.5 - abs(scaledVel) * ((1.0 - cr) + uCFL * cr) * rj * 0.5;
}

#define TILE 8

template <typename DTYPE>
__global__
void SuperbeeKernelTiled(
    const DTYPE *var,
    const DTYPE *u_wgrid,
    const DTYPE *v_wgrid,
    const DTYPE *w_wgrid,
    const DTYPE *maskW,
    const DTYPE *dxt,
    const DTYPE *dyt,
    const DTYPE *dzw,
    const DTYPE *cost,
    const DTYPE *cosu,
    const DTYPE *dt_tracer,
    DTYPE *flux_east,
    DTYPE *flux_north,
    DTYPE *flux_top,
    int dim1,
    int dim2,
    int dim3
){
    const int EXT_TILE = TILE+3;
    __shared__ DTYPE shared[2][EXT_TILE][EXT_TILE][EXT_TILE];

    const int loadSize = (EXT_TILE) * (EXT_TILE) * (EXT_TILE);
    
    const int dim1Stride = dim2 * dim3;
    const int dim2Stride = dim3;
    const int dim3Stride = 1;


    // We need to load more elements than we have threads, so it's easier to have the flat indices
    const int flatThreadIdx = threadIdx.x + (blockDim.x * threadIdx.y) + (threadIdx.z * blockDim.x * blockDim.y);
    // dim3 offset into global mem (-1 to account for stencil start)
    int global_startX = (blockIdx.x * blockDim.x - 1);
    // dim2 offset into global mem
    int global_startY = (blockIdx.y * blockDim.y - 1);
    // dim1 offset into global mem
    int global_startZ = (blockIdx.z * blockDim.z - 1);


    for (int i = flatThreadIdx ; i < loadSize ; i += blockDim.x * blockDim.y * blockDim.z)
    {
        // local offset
        int blockX = i % (EXT_TILE);
        
        // local offset
        int blockY = (i / (EXT_TILE)) % (EXT_TILE);

        // local offset
        int blockZ =  i / ((EXT_TILE) * (EXT_TILE));
        
        int globalX = global_startX + blockX;
        int globalY = global_startY + blockY;
        int globalZ = global_startZ + blockZ;
        // check bounds
        bool blockXvalid = globalX >= 0 && globalX < dim3;
        bool blockYvalid = globalY >= 0 && globalY < dim2;
        bool blockZvalid = globalZ >= 0 && globalZ < dim1;

        if (blockXvalid && blockYvalid && blockZvalid)
        {
            int globalIndx = globalZ * dim1Stride + globalY * dim2Stride + globalX;

            shared[0][blockZ][blockY][blockX] = var[globalIndx];
            shared[1][blockZ][blockY][blockX] = maskW[globalIndx];
        }
        else
        {
            shared[0][blockZ][blockY][blockX] = 0;
            shared[1][blockZ][blockY][blockX] = 0;
        }
    }
    __syncthreads();


    int global_d1 = blockIdx.z * blockDim.z + threadIdx.z;
    int global_d2 = blockIdx.y * blockDim.y + threadIdx.y;
    int global_d3 = blockIdx.x * blockDim.x + threadIdx.x;
    
    int local_d1 = threadIdx.z + 1;
    int local_d2 = threadIdx.y + 1;
    int local_d3 = threadIdx.x + 1;
    
    int flatResultIdx = global_d1*dim1Stride + global_d2 * dim2Stride + global_d3 * dim3Stride;

    const DTYPE varS = shared[0][local_d1][local_d2][local_d3];
    const DTYPE maskWs = shared[1][local_d1][local_d2][local_d3];
    
    DTYPE adv_fe = 0;
    DTYPE adv_fn = 0;
    DTYPE adv_ft = 0;

    if (global_d1 > 0 && global_d1 < dim1-2 && global_d2 > 1 && global_d2 < dim2-2 && global_d3 < dim3)
    {
        const DTYPE velS = u_wgrid[flatResultIdx];
        const DTYPE varSM1 = shared[0][local_d1-1][local_d2][local_d3];
        const DTYPE varSP1 = shared[0][local_d1+1][local_d2][local_d3];
        const DTYPE varSP2 = shared[0][local_d1+2][local_d2][local_d3];
        const DTYPE maskWm1 = shared[1][local_d1-1][local_d2][local_d3];
        const DTYPE maskWp1 = shared[1][local_d1+1][local_d2][local_d3];
        const DTYPE maskwp2 = shared[1][local_d1+2][local_d2][local_d3];
     
        DTYPE maskUtr = maskWs * maskWp1;
        DTYPE maskUtrP1 = maskWp1 * maskwp2;
        DTYPE maskUtrM1 = maskWm1 * maskWs;

        const DTYPE dx = cost[global_d2] * dxt[global_d1];
        adv_fe = calcFlux<DTYPE>(1, velS, *dt_tracer, dx, varS, varSM1, varSP1, varSP2, maskUtr, maskUtrM1, maskUtrP1);
    }
    if (global_d2 > 0 && global_d2 < dim2-2 && global_d1 > 1 && global_d1 < dim1-2 && global_d3 < dim3)
    {
        const DTYPE velS = v_wgrid[flatResultIdx];
        const DTYPE varSM1 = shared[0][local_d1][local_d2-1][local_d3];
        const DTYPE varSP1 = shared[0][local_d1][local_d2+1][local_d3];
        const DTYPE varSP2 = shared[0][local_d1][local_d2+2][local_d3];
        const DTYPE maskWm1 = shared[1][local_d1][local_d2-1][local_d3];
        const DTYPE maskWp1 = shared[1][local_d1][local_d2+1][local_d3];
        const DTYPE maskwp2 = shared[1][local_d1][local_d2+2][local_d3];

        DTYPE maskVtr = maskWs * maskWp1;
        DTYPE maskVtrP1 = maskWp1 * maskwp2;
        DTYPE maskVtrM1 = maskWm1 * maskWs;

        const DTYPE dx = cost[global_d2] * dyt[global_d2];
        adv_fn = calcFlux<DTYPE>(cosu[global_d2], velS, *dt_tracer, dx, varS, varSM1, varSP1, varSP2, maskVtr, maskVtrM1, maskVtrP1);
    }
    if (global_d3 < dim3-1 && global_d1 > 1 && global_d1 < dim1-2 && global_d2 > 1 && global_d2 < dim2-2)
    {
        const DTYPE velS = w_wgrid[flatResultIdx];

        DTYPE varSM1 = shared[0][local_d1][local_d2][local_d3-1];
        DTYPE maskWm1 = shared[1][local_d1][local_d2][local_d3-1];

        DTYPE varSP2 = shared[0][local_d1][local_d2][local_d3+2];
        DTYPE maskwp2 = shared[1][local_d1][local_d2][local_d3+2];

        const DTYPE varSP1 = shared[0][local_d1][local_d2][local_d3+1];
        const DTYPE maskWp1 = shared[1][local_d1][local_d2][local_d3+1];

        DTYPE maskWtr = maskWs * maskWp1;
        DTYPE maskWtrP1 = maskWp1 * maskwp2;
        DTYPE maskWtrM1 = maskWm1 * maskWs;

        const DTYPE dx = dzw[global_d3];
        adv_ft = calcFlux<DTYPE>(1, velS, *dt_tracer, dx, varS, varSM1, varSP1, varSP2, maskWtr, maskWtrM1, maskWtrP1);
    }
    if (global_d1 < dim1 && global_d2 < dim2 && global_d3 < dim3)
    {
        flux_east[flatResultIdx] = adv_fe;
        flux_north[flatResultIdx] = adv_fn;
        flux_top[flatResultIdx] = adv_ft;
    }
}




template <typename DTYPE>
__global__
void SuperbeeKernel(
    const DTYPE *var,
    const DTYPE *u_wgrid,
    const DTYPE *v_wgrid,
    const DTYPE *w_wgrid,
    const DTYPE *maskW,
    const DTYPE *dxt,
    const DTYPE *dyt,
    const DTYPE *dzw,
    const DTYPE *cost,
    const DTYPE *cosu,
    const DTYPE *dt_tracer,
    DTYPE *flux_east,
    DTYPE *flux_north,
    DTYPE *flux_top,
    int dim1,
    int dim2,
    int dim3
){
    const int dim1Stride = dim2 * dim3;
    const int dim2Stride = dim3;
    const int dim3Stride = 1;
    for (std::int64_t index = blockIdx.x * blockDim.x + threadIdx.x; index < dim1Stride * dim1;
        index += blockDim.x * gridDim.x) {

        int x = index / dim1Stride;
        int y = (index / dim3) % dim2;
        int z = index % dim3;
        const int s = index;

        const DTYPE varS = var[s];
        const DTYPE maskWs = maskW[s];
        
        DTYPE adv_fe = 0;
        DTYPE adv_fn = 0;
        DTYPE adv_ft = 0;
        if (x > 0 && x < dim1-2 && y > 1 && y < dim2-2)
        {
            const DTYPE velS = u_wgrid[s];
            DTYPE maskUtr = 0;
            DTYPE maskUtrP1 = 0;
            DTYPE maskUtrM1 = 0;
            const int s1m1 = index - dim1Stride;
            const int s1p1 = index + dim1Stride;
            const int s1p2 = index + 2*dim1Stride;
            const DTYPE maskWm1 = maskW[s1m1];
            const DTYPE maskWp1 = maskW[s1p1];
            const DTYPE maskwp2 = maskW[s1p2];
            const DTYPE varSM1 = var[s1m1];
            const DTYPE varSP1 = var[s1p1];
            const DTYPE varSP2 = var[s1p2];
            if (x < dim1-1)
            {
                maskUtr = maskWs * maskWp1;
                maskUtrP1 = maskWp1 * maskwp2;
                maskUtrM1 = maskWm1 * maskWs;
            }
            const DTYPE dx = cost[y] * dxt[x];
            adv_fe = calcFlux<DTYPE>(1, velS, *dt_tracer, dx, varS, varSM1, varSP1, varSP2, maskUtr, maskUtrM1, maskUtrP1);
        }
        if (y > 0 && y < dim2-2 && x > 1 && x < dim1-2)
        {
            const DTYPE velS = v_wgrid[s];
            DTYPE maskVtr = 0;
            DTYPE maskVtrP1 = 0;
            DTYPE maskVtrM1 = 0;
            const int s1m1 = index - dim2Stride;
            const int s1p1 = index + dim2Stride;
            const int s1p2 = index + 2*dim2Stride;
            const DTYPE maskWm1 = maskW[s1m1];
            const DTYPE maskWp1 = maskW[s1p1];
            const DTYPE maskwp2 = maskW[s1p2];
            const DTYPE varSM1 = var[s1m1];
            const DTYPE varSP1 = var[s1p1];
            const DTYPE varSP2 = var[s1p2];
            if (y < dim2-1)
            {
                maskVtr = maskWs * maskWp1;
                maskVtrP1 = maskWp1 * maskwp2;
                maskVtrM1 = maskWm1 * maskWs;
            }
            const DTYPE dx = cost[y] * dyt[y];
            adv_fn = calcFlux<DTYPE>(cosu[y], velS, *dt_tracer, dx, varS, varSM1, varSP1, varSP2, maskVtr, maskVtrM1, maskVtrP1);
        }
        if (z < dim3-1 && x > 1 && x < dim1-2 && y > 1 && y < dim2-2)
        {
            const DTYPE velS = w_wgrid[s];
            DTYPE maskWtr = 0;
            DTYPE maskWtrP1 = 0;
            DTYPE maskWtrM1 = 0;
            const int s1m1 = index - dim3Stride;
            const int s1p1 = index + dim3Stride;
            const int s1p2 = index + 2*dim3Stride;
            DTYPE maskWm1 = 0;
            DTYPE varSM1 = 0 ;
            if (z != 0)
            {
                maskWm1 = maskW[s1m1];
                varSM1 = var[s1m1];
            }
            DTYPE maskwp2 = 0;
            DTYPE varSP2 = 0;
            if (z < dim3-2)
            {
                maskwp2 = maskW[s1p2];
                varSP2 = var[s1p2];
            }
            const DTYPE varSP1 = var[s1p1];
            const DTYPE maskWp1 = maskW[s1p1];
            if (z < dim3-1)
            {
                maskWtr = maskWs * maskWp1;
                maskWtrP1 = maskWp1 * maskwp2;
                maskWtrM1 = maskWm1 * maskWs;
            }
            const DTYPE dx = dzw[z];
            adv_ft = calcFlux<DTYPE>(1, velS, *dt_tracer, dx, varS, varSM1, varSP1, varSP2, maskWtr, maskWtrM1, maskWtrP1);
    }

    flux_east[index] = adv_fe;
    flux_north[index] = adv_fn;
    flux_top[index] = adv_ft;
    }
}

struct SuperbeeDescriptor {
  std::int64_t dim1;
  std::int64_t dim2;
  std::int64_t dim3;
};

// Unpacks a descriptor object from a byte string.
template <typename T>
const T* UnpackDescriptor(const char* opaque, std::size_t opaque_len) {
  if (opaque_len != sizeof(T)) {
    throw std::runtime_error("Descriptor was not encoded correctly.");
  }
  return reinterpret_cast<const T*>(opaque);
}

template <typename DTYPE>
void CudaSuperbeeTiled(cudaStream_t stream, void** buffers, const char* opaque,
                      std::size_t opaque_len) {

  const auto& descriptor =
      *UnpackDescriptor<SuperbeeDescriptor>(opaque, opaque_len);
  const std::int64_t xdim = descriptor.dim1;
  const std::int64_t ydim = descriptor.dim2;
  const std::int64_t zdim = descriptor.dim3;
  const std::int64_t allDims = xdim*ydim*zdim;
 
  const DTYPE* var = reinterpret_cast<const DTYPE*>(buffers[0]);
  const DTYPE* u_wgrid = reinterpret_cast<const DTYPE*>(buffers[1]);
  const DTYPE* v_wgrid = reinterpret_cast<const DTYPE*>(buffers[2]);
  const DTYPE* w_wgrid = reinterpret_cast<const DTYPE*>(buffers[3]);
  const DTYPE* maskW = reinterpret_cast<const DTYPE*>(buffers[4]);
  const DTYPE* dxt = reinterpret_cast<const DTYPE*>(buffers[5]);
  const DTYPE* dyt = reinterpret_cast<const DTYPE*>(buffers[6]);
  const DTYPE* dzw = reinterpret_cast<const DTYPE*>(buffers[7]);
  const DTYPE* cost = reinterpret_cast<const DTYPE*>(buffers[8]);
  const DTYPE* cosu = reinterpret_cast<const DTYPE*>(buffers[9]);
  const DTYPE* dt_tracer = reinterpret_cast<const DTYPE*>(buffers[10]);
  DTYPE* flux_east = reinterpret_cast<DTYPE*>(buffers[11]); // output1
  DTYPE* flux_north = reinterpret_cast<DTYPE*>(buffers[12]); // output2
  DTYPE* flux_top = reinterpret_cast<DTYPE*>(buffers[13]); // output3
                        
  dim3 blocksize(TILE, TILE, TILE);
  int numblocks1 = (xdim + TILE-1) / TILE;
  int numblocks2 = (ydim + TILE-1) / TILE;
  int numblocks3 = (zdim + TILE-1) / TILE;

  dim3 gridsize(numblocks3, numblocks2, numblocks1);
  SuperbeeKernelTiled<DTYPE><<<gridsize, blocksize, 0, stream>>>(
      var
    , u_wgrid
    , v_wgrid
    , w_wgrid
    , maskW
    , dxt
    , dyt
    , dzw
    , cost
    , cosu
    , dt_tracer
    , flux_east
    , flux_north
    , flux_top
    , xdim
    , ydim
    , zdim
    );
  gpuErrchk(cudaPeekAtLastError());
}

template <typename DTYPE>
void CudaSuperbee(cudaStream_t stream, void** buffers, const char* opaque,
                      std::size_t opaque_len) {

  const auto& descriptor =
      *UnpackDescriptor<SuperbeeDescriptor>(opaque, opaque_len);
  const std::int64_t dim1 = descriptor.dim1;
  const std::int64_t dim2 = descriptor.dim2;
  const std::int64_t dim3 = descriptor.dim3;
  const std::int64_t allDims = dim1*dim2*dim3;
 
  const DTYPE* var = reinterpret_cast<const DTYPE*>(buffers[0]);
  const DTYPE* u_wgrid = reinterpret_cast<const DTYPE*>(buffers[1]);
  const DTYPE* v_wgrid = reinterpret_cast<const DTYPE*>(buffers[2]);
  const DTYPE* w_wgrid = reinterpret_cast<const DTYPE*>(buffers[3]);
  const DTYPE* maskW = reinterpret_cast<const DTYPE*>(buffers[4]);
  const DTYPE* dxt = reinterpret_cast<const DTYPE*>(buffers[5]);
  const DTYPE* dyt = reinterpret_cast<const DTYPE*>(buffers[6]);
  const DTYPE* dzw = reinterpret_cast<const DTYPE*>(buffers[7]);
  const DTYPE* cost = reinterpret_cast<const DTYPE*>(buffers[8]);
  const DTYPE* cosu = reinterpret_cast<const DTYPE*>(buffers[9]);
  const DTYPE* dt_tracer = reinterpret_cast<const DTYPE*>(buffers[10]);
  DTYPE* flux_east = reinterpret_cast<DTYPE*>(buffers[11]); // output1
  DTYPE* flux_north = reinterpret_cast<DTYPE*>(buffers[12]); // output2
  DTYPE* flux_top = reinterpret_cast<DTYPE*>(buffers[13]); // output3
                        
  const int BLOCK_SIZE = 128;
  const std::int64_t grid_dim =
      std::min<std::int64_t>(1024, (allDims + BLOCK_SIZE - 1) / BLOCK_SIZE);
      
  SuperbeeKernel<DTYPE><<<grid_dim, BLOCK_SIZE, 0, stream>>>(
      var
    , u_wgrid
    , v_wgrid
    , w_wgrid
    , maskW
    , dxt
    , dyt
    , dzw
    , cost
    , cosu
    , dt_tracer
    , flux_east
    , flux_north
    , flux_top
    , dim1
    , dim2
    , dim3
    );
  gpuErrchk(cudaPeekAtLastError());
}

void CudaSuperbeeFloat(cudaStream_t stream, void** buffers, const char* opaque,
                      std::size_t opaque_len) {
  CudaSuperbeeTiled<float>(stream, buffers, opaque, opaque_len);
}
void CudaSuperbeeDouble(cudaStream_t stream, void** buffers, const char* opaque,
                      std::size_t opaque_len) {
  CudaSuperbeeTiled<double>(stream, buffers, opaque, opaque_len);
}

