/* Copyright 2019 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <array>
#include <cstddef>

#include "cuda_tridiag_kernels.h"
#include "gpu_kernel_helpers.h"
#include "kernel_helpers.h"
#include "cuda_transpose.h"

namespace jax {
namespace {

template <typename DTYPE>
__global__
void TridiagKernel(
    const DTYPE *a,
    const DTYPE *b,
    DTYPE *c,
    DTYPE *d,
    DTYPE *solution,
    int n,
    int num_chunks
){
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if ( idx < num_chunks ) {
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
}

}  // namespace

struct TridiagDescriptor {
  std::int64_t total_size;
  std::int64_t num_systems;
  std::int64_t system_depth;
};

std::string BuildCudaTridiagDescriptor(std::int64_t total_size, std::int64_t num_systems, std::int64_t system_depth) {
  return PackDescriptorAsString(TridiagDescriptor{total_size, num_systems, system_depth});
}

template <typename DTYPE>
void CudaTridiag(cudaStream_t stream, void** buffers, const char* opaque,
                      std::size_t opaque_len) {
  
  const auto& descriptor =
      *UnpackDescriptor<TridiagDescriptor>(opaque, opaque_len);
  const int num_systems = descriptor.num_systems;
  const int system_depth = descriptor.system_depth;
  // const int total_size = descriptor.total_size;


  const DTYPE* a = reinterpret_cast<const DTYPE*>(buffers[0]);
  const DTYPE* b = reinterpret_cast<const DTYPE*>(buffers[1]);
  DTYPE* c = reinterpret_cast<DTYPE*>(buffers[2]); // should be const
  DTYPE* d = reinterpret_cast<DTYPE*>(buffers[3]); // should be const
  DTYPE* out = reinterpret_cast<DTYPE*>(buffers[4]); // output
  // DTYPE* a_t = out + total_size; // a transpose
  // DTYPE* b_t = a_t + total_size; // b transpose
  // DTYPE* c_t = b_t + total_size; // c transpose
  // DTYPE* d_t = c_t + total_size; // d transpose
  // DTYPE* out_t = d_t + total_size; // out transpose
  


  const int BLOCK_SIZE = 256;
  // transposeMats(a, b, c, d, a_t, b_t, c_t, d_t, system_depth, num_systems, total_size);
  const std::int64_t grid_dim =
      std::min<std::int64_t>(1024, (num_systems + BLOCK_SIZE - 1) / BLOCK_SIZE);
  TridiagKernel<DTYPE><<<grid_dim, BLOCK_SIZE, 0, stream>>>(a, b, c, d, out, system_depth, num_systems);
  // transposeMat(out_t, out, num_systems, system_depth, total_size);
  ThrowIfError(cudaGetLastError());
}

void CudaTridiagFloat(cudaStream_t stream, void** buffers, const char* opaque,
                      std::size_t opaque_len) {
  CudaTridiag<float>(stream, buffers, opaque, opaque_len);
}
void CudaTridiagDouble(cudaStream_t stream, void** buffers, const char* opaque,
                      std::size_t opaque_len) {
  CudaTridiag<double>(stream, buffers, opaque, opaque_len);
}


}  // namespace jax
