#!/usr/bin/env python
import math
import numpy as np
import numba as nb
import numba.cuda
from tke import tke
from scipy.linalg import lapack
shape = (360, 160, 115)
import cupy

from string import Template

def tdma_naive(a, b, c, d):
    """
    Solves many tridiagonal matrix systems with diagonals a, b, c and RHS vectors d.
    """
    assert a.shape == b.shape and a.shape == c.shape and a.shape == d.shape

    n = a.shape[-1]

    for i in range(1, n):
        w = a[..., i] / b[..., i - 1]
        b[..., i] += -w * c[..., i - 1]
        d[..., i] += -w * d[..., i - 1]

    out = np.empty_like(a)
    out[..., -1] = d[..., -1] / b[..., -1]

    for i in range(n - 2, -1, -1):
        out[..., i] = (d[..., i] - c[..., i] * out[..., i + 1]) / b[..., i]

    return out

order_kernel = Template('''
extern "C" __global__
    void order(
        const ${DTYPE} * __restrict__ a,
        const ${DTYPE} * __restrict__ b,
        const ${DTYPE} * __restrict__ c,
        const ${DTYPE} * __restrict__ d,
        ${DTYPE} *a_out,
        ${DTYPE} *b_out,
        ${DTYPE} *c_out,
        ${DTYPE} *d_out
    ){
        
        const unsigned int source_idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (source_idx >= ${SIZE})
            return;

        const unsigned int m = ${SYS_DEPTH};
        const unsigned int stride = ${STRIDE};
        const unsigned int seg_idx = source_idx / m;
        const unsigned int seg_offset = source_idx % m;
        const unsigned int target_idx = seg_offset * stride + seg_idx;

        a_out[target_idx] = a[source_idx]; 
        b_out[target_idx] = b[source_idx];
        c_out[target_idx] = c[source_idx];
        d_out[target_idx] = d[source_idx];
    }

''').substitute(
        DTYPE='double',
        SYS_DEPTH=shape[-1],
        SIZE=np.product(shape),
        STRIDE=shape[0]*shape[1]
)

order_back_kernel = Template('''
extern "C" __global__
    void order_back(
        const ${DTYPE} * __restrict__ out,
        ${DTYPE} *o_out
    ){

        const unsigned int target_idx = blockIdx.x * blockDim.x + threadIdx.x;
        
        if (target_idx >= ${SIZE})
            return;
        const unsigned int m = ${SYS_DEPTH};
        const unsigned int stride = ${STRIDE};
        const unsigned int seg_idx = target_idx / m;
        const unsigned int seg_offset = target_idx % m;
        const unsigned int source_idx = seg_offset * stride + seg_idx;
        o_out[target_idx] = out[source_idx];
    }

''').substitute(
        DTYPE='double',
        SYS_DEPTH=shape[-1],
        SIZE=np.product(shape),
        STRIDE=shape[0]*shape[1]
)

kernel = Template('''
    extern "C" __global__
    void execute(
        const ${DTYPE} * __restrict__ a,
        const ${DTYPE} * __restrict__ b,
        const ${DTYPE} * __restrict__ c,
        const ${DTYPE} * __restrict__ d,
        ${DTYPE} *solution
    ){
        const unsigned int m = ${SYS_DEPTH};
        const unsigned int total_size = ${SIZE};
        const unsigned int stride = total_size / m;
        const unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
        if (idx >= stride) {
            return;
        }

        ${DTYPE} cp[${SYS_DEPTH}];
        ${DTYPE} dp[${SYS_DEPTH}];

        cp[0] = c[idx] / b[idx];
        dp[0] = d[idx] / b[idx];

        for (ptrdiff_t j = 1; j < m; ++j) {
            unsigned int indj = idx+(j*stride);
            const ${DTYPE} norm_factor = b[indj] - a[indj] * cp[j-1];
            cp[j] = c[indj] / norm_factor;
            dp[j] = (d[indj] - a[indj] * dp[j-1]) / norm_factor;
        }

        solution[idx + stride*(m-1)] = dp[m-1];
        for (ptrdiff_t j=m-2; j >= 0; --j) {
            solution[idx + stride*j] = dp[j] - cp[j] * solution[idx + stride*(j+1)];
        }
    }
    ''').substitute(
        DTYPE='float',
        SYS_DEPTH=shape[-1],
        SIZE=np.product(shape),
        STRIDE=shape[0]*shape[1]
)

tdma_cupy_kernel = cupy.RawKernel(kernel, 'execute', backend=u'nvcc')
cupy_order_kernel = cupy.RawKernel(order_kernel, 'order', backend=u'nvcc')
cupy_order_back_kernel = cupy.RawKernel(order_back_kernel, 'order_back', backend=u'nvcc')


def tdma_cupy(a, b, c, d, blocksize=256):
    assert a.shape == b.shape and a.shape == c.shape and a.shape == d.shape

    a, b, c, d = (cupy.asarray(k) for k in (a, b, c, d))
    
    a_tmp, b_tmp, c_tmp, d_tmp, out, o_out = (cupy.empty(s.shape, dtype=s.dtype) for s in (a,b,c,d,a,a))

    cupy_order_kernel(
        (math.ceil(a.size  / blocksize),),
        (blocksize,),
        (a, b, c, d, a_tmp, b_tmp, c_tmp, d_tmp)
    )

    tdma_cupy_kernel(
        (math.ceil(a.size / a.shape[-1] / blocksize),),
        (blocksize,),
        (a_tmp, b_tmp, c_tmp, d_tmp, o_out)
    )
    
    cupy_order_back_kernel(
        (math.ceil(a.size / blocksize),),
        (blocksize,),
        (o_out, out)
    )
    return out



np.random.seed(17)
a, b, c, d = np.random.randn(4, *shape)
res_naive = tdma_naive(a, b, c, d)


np.random.seed(17)
a, b, c, d = np.random.randn(4, *shape)
out = tdma_cupy(a, b, c, d)
try:
    out = out.get()
except AttributeError:
    pass

np.testing.assert_allclose(out, res_naive)
print('✔️')