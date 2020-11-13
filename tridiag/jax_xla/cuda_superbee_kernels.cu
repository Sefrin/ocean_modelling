
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
  CudaSuperbee<float>(stream, buffers, opaque, opaque_len);
}
void CudaSuperbeeDouble(cudaStream_t stream, void** buffers, const char* opaque,
                      std::size_t opaque_len) {
  CudaSuperbee<double>(stream, buffers, opaque, opaque_len);
}

