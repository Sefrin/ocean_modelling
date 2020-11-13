from cpython.pycapsule cimport PyCapsule_New

from libc.stdint cimport int64_t

cdef extern from "cuda_runtime_api.h":
    ctypedef void* cudaStream_t

cdef extern from "cuda_superbee_kernels.h":
    void CudaSuperbeeFloat(cudaStream_t stream, void** buffers, const char* opaque, size_t opaque_len)
    void CudaSuperbeeDouble(cudaStream_t stream, void** buffers, const char* opaque, size_t opaque_len)


cdef struct SuperbeeDescriptor:
    int64_t dim1
    int64_t dim2
    int64_t dim3


cpdef bytes build_superbee_descriptor(int64_t dim1, int64_t dim2, int64_t dim3):
    cdef SuperbeeDescriptor desc = SuperbeeDescriptor(
        dim1, dim2, dim3
    )
    return bytes((<char*> &desc)[:sizeof(SuperbeeDescriptor)])

gpu_custom_call_targets = {}

cdef register_custom_call_target(fn_name, void* fn):
    cdef const char* name = "xla._CUSTOM_CALL_TARGET"
    gpu_custom_call_targets[fn_name] = PyCapsule_New(fn, name, NULL)

register_custom_call_target(b"cuda_superbee_double", <void*>(CudaSuperbeeDouble))
register_custom_call_target(b"cuda_superbee_float", <void*>(CudaSuperbeeFloat))