#pragma once
#include <cstddef>
#include <string>

#include <cuda_runtime.h>

std::string BuildCudaSuperbeeDescriptor(std::int64_t dim1, std::int64_t dim2, std::int64_t dim3);

void CudaSuperbeeFloat(cudaStream_t stream, void** buffers, const char* opaque,
                      std::size_t opaque_len);
void CudaSuperbeeDouble(cudaStream_t stream, void** buffers, const char* opaque,
                      std::size_t opaque_len);

