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

kernel_old = Template('''
extern "C" __global__
void execute(
    const ${DTYPE} *a,
    const ${DTYPE} *b,
    const ${DTYPE} *c,
    const ${DTYPE} *d,
    ${DTYPE} *solution
){
    const size_t m = ${SYS_DEPTH};
    const size_t total_size = ${SIZE};
    const size_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * m;

    if (idx >= total_size) {
        return;
    }

    ${DTYPE} cp[${SYS_DEPTH}];
    ${DTYPE} dp[${SYS_DEPTH}];

    cp[0] = c[idx] / b[idx];
    dp[0] = d[idx] / b[idx];

    for (ptrdiff_t j = 1; j < m; ++j) {
        const ${DTYPE} norm_factor = b[idx+j] - a[idx+j] * cp[j-1];
        cp[j] = c[idx+j] / norm_factor;
        dp[j] = (d[idx+j] - a[idx+j] * dp[j-1]) / norm_factor;
    }

    solution[idx + m-1] = dp[m-1];
    for (ptrdiff_t j=m-2; j >= 0; --j) {
        solution[idx + j] = dp[j] - cp[j] * solution[idx + j+1];
    }
}
''').substitute(
    DTYPE='double',
    SYS_DEPTH=shape[-1],
    SIZE=np.product(shape)
)

tdma_cupy_kernel_old = cupy.RawKernel(kernel_old, 'execute')


def tdma_cupy(a, b, c, d,  blocksize=256):
    assert a.shape == b.shape and a.shape == c.shape and a.shape == d.shape
    a, b, c, d = (cupy.asarray(k) for k in (a, b, c, d))
    out = cupy.empty(a.shape, dtype=a.dtype)
    
    
    
    stream = cupy.cuda.stream.Stream()
    # tdma_cupy_kernel_old(
    #         (math.ceil((a.size / a.shape[-1]) / blocksize),),
    #         (blocksize,),
    #         (a, b, c, d, out)
    #     )
    
    with stream:
        for i in range(5):    
            start_time = time.time()         
            tdma_cupy_kernel_old(
                (math.ceil((a.size / a.shape[-1]) / blocksize),),
                (blocksize,),
                (a, b, c, d, out)
            )
            stream.synchronize()
            elapsed_time = time.time() - start_time
            print(str(1000*elapsed_time) + "ms")
    
    return out

np.random.seed(17)
a, b, c, d = np.random.randn(4, *shape)
res_naive = tdma_naive(a, b, c, d)


np.random.seed(17)
a, b, c, d = np.random.randn(4, *shape)
import time


out = tdma_cupy(a, b, c, d)
try:
    out = out.get()
except AttributeError:
    pass

np.testing.assert_allclose(out, res_naive)
print('✔️')