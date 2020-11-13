import cupy
from string import Template
import jax
import jax.numpy as jnp
import numpy as np
from jax.config import config
import math
config.update('jax_enable_x64', True)

@jax.jit
def where(mask, a, b):
    return jnp.where(mask, a, b)


@jax.jit
def _calc_cr(rjp, rj, rjm, vel):
    """
    Calculates cr value used in superbee advection scheme
    """
    eps = 1e-20  # prevent division by 0
    return where(vel > 0., rjm, rjp) / where(jnp.abs(rj) < eps, eps, rj)


@jax.jit
def pad_z_edges(arr):
    arr_shape = list(arr.shape)
    arr_shape[2] += 2
    out = jnp.zeros(arr_shape, arr.dtype)
    out = jax.ops.index_update(
        out, jax.ops.index[:, :, 1:-1], arr
    )
    return out


@jax.jit
def limiter(cr):
    return jnp.maximum(0., jnp.maximum(jnp.minimum(1., 2 * cr), jnp.minimum(2., cr)))


def _adv_superbee(vel, var, mask, dx, axis, cost, cosu, dt_tracer):
    velfac = 1
    if axis == 0:
        sm1, s, sp1, sp2 = ((slice(1 + n, -2 + n or None), slice(2, -2), slice(None))
                            for n in range(-1, 3))
        dx = cost[jnp.newaxis, 2:-2, jnp.newaxis] * \
            dx[1:-2, jnp.newaxis, jnp.newaxis]
    elif axis == 1:
        sm1, s, sp1, sp2 = ((slice(2, -2), slice(1 + n, -2 + n or None), slice(None))
                            for n in range(-1, 3))
        dx = (cost * dx)[jnp.newaxis, 1:-2, jnp.newaxis]
        velfac = cosu[jnp.newaxis, 1:-2, jnp.newaxis]
    elif axis == 2:
        vel, var, mask = (pad_z_edges(a) for a in (vel, var, mask))
        sm1, s, sp1, sp2 = ((slice(2, -2), slice(2, -2), slice(1 + n, -2 + n or None))
                            for n in range(-1, 3))
        dx = dx[jnp.newaxis, jnp.newaxis, :-1]
    else:
        raise ValueError('axis must be 0, 1, or 2')
    uCFL = jnp.abs(velfac * vel[s] * dt_tracer / dx)
    rjp = (var[sp2] - var[sp1]) * mask[sp1]
    rj = (var[sp1] - var[s]) * mask[s]
    rjm = (var[s] - var[sm1]) * mask[sm1]
    cr = limiter(_calc_cr(rjp, rj, rjm, vel[s]))
    return velfac * vel[s] * (var[sp1] + var[s]) * 0.5 - jnp.abs(velfac * vel[s]) * ((1. - cr) + uCFL * cr) * rj * 0.5


_adv_superbee = jax.jit(_adv_superbee, static_argnums=(4,))


@jax.jit
def adv_flux_superbee_wgrid(var, u_wgrid, v_wgrid, w_wgrid, maskW, dxt, dyt, dzw, cost, cosu, dt_tracer):
    """
    Calculates advection of a tracer defined on Wgrid
    """
    maskUtr = jnp.zeros_like(maskW)
    maskUtr = jax.ops.index_update(
        maskUtr, jax.ops.index[:-1, :, :],
        maskW[1:, :, :] * maskW[:-1, :, :]
    )

    adv_fe = jnp.zeros_like(maskW)
    adv_fe = jax.ops.index_update(
        adv_fe, jax.ops.index[1:-2, 2:-2, :],
        _adv_superbee(u_wgrid, var, maskUtr, dxt, 0, cost, cosu, dt_tracer)
    )

    maskVtr = jnp.zeros_like(maskW)
    maskVtr = jax.ops.index_update(
        maskVtr, jax.ops.index[:, :-1, :],
        maskW[:, 1:, :] * maskW[:, :-1, :]

    )
    adv_fn = jnp.zeros_like(maskW)
    adv_fn = jax.ops.index_update(
        adv_fn, jax.ops.index[2:-2, 1:-2, :],
        _adv_superbee(v_wgrid, var, maskVtr, dyt, 1, cost, cosu, dt_tracer)
    )

    maskWtr = jnp.zeros_like(maskW)
    maskWtr = jax.ops.index_update(
        maskWtr, jax.ops.index[:, :, :-1],
        maskW[:, :, 1:] * maskW[:, :, :-1]
    )
    adv_ft = jnp.zeros_like(maskW)
    adv_ft = jax.ops.index_update(
        adv_ft, jax.ops.index[2:-2, 2:-2, :-1],
        _adv_superbee(w_wgrid, var, maskWtr, dzw, 2, cost, cosu, dt_tracer)
    )

    return adv_fe, adv_fn, adv_ft








kernel = Template('''
__device__ inline ${DTYPE} limiter(${DTYPE} cr)
{
    return max(0.0, max(min(1.0, 2.0 * cr), min(2.0, cr)));
}

__device__ inline ${DTYPE} calcFlux
    (
        const ${DTYPE} velfac,
        const ${DTYPE} velS,
        const ${DTYPE} dt_tracer,
        const ${DTYPE} dx,
        const ${DTYPE} varS,
        const ${DTYPE} varSM1,
        const ${DTYPE} varSP1,
        const ${DTYPE} varSP2,
        const ${DTYPE} maskS,
        const ${DTYPE} maskSM1,
        const ${DTYPE} maskSP1
    )
{
        const ${DTYPE} scaledVel =  velfac * velS;
        const ${DTYPE} uCFL = abs(scaledVel * dt_tracer / dx);
        const ${DTYPE} rjp = (varSP2 - varSP1) * maskSP1;
        const ${DTYPE} rj = (varSP1 - varS) * maskS;
        const ${DTYPE} rjm = (varS - varSM1) * maskSM1;
        ${DTYPE} cr;
        ${DTYPE} divisor = rj;
        const ${DTYPE} epsilon = 1e-20;
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

extern "C" __global__
void superbee(
    const ${DTYPE} *var,
    const ${DTYPE} *u_wgrid,
    const ${DTYPE} *v_wgrid,
    const ${DTYPE} *w_wgrid,
    const ${DTYPE} *maskW,
    const ${DTYPE} *dxt,
    const ${DTYPE} *dyt,
    const ${DTYPE} *dzw,
    const ${DTYPE} *cost,
    const ${DTYPE} *cosu,
    const ${DTYPE} dt_tracer,
    ${DTYPE} *flux_east,
    ${DTYPE} *flux_north,
    ${DTYPE} *flux_top,
    int dim1,
    int dim2,
    int dim3
){
//   maskW[:-1, :, :] * maskW[1:, :, :]

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    const int dim1Stride = dim2 * dim3;
    const int dim2Stride = dim3;
    const int dim3Stride = 1;
    if (index >= dim1Stride*dim1)
        return;

    int x = index / dim1Stride;
    int y = (index / dim3) % dim2;
    int z = index % dim3;
    const int s = index;

    const ${DTYPE} varS = var[s];
    const ${DTYPE} maskWs = maskW[s];
    
    ${DTYPE} adv_fe = 0;
    ${DTYPE} adv_fn = 0;
    ${DTYPE} adv_ft = 0;
    if (x > 0 && x < dim1-2 && y > 1 && y < dim2-2)
    {
        const ${DTYPE} velS = u_wgrid[s];
        ${DTYPE} maskUtr = 0;
        ${DTYPE} maskUtrP1 = 0;
        ${DTYPE} maskUtrM1 = 0;
        const int s1m1 = index - dim1Stride;
        const int s1p1 = index + dim1Stride;
        const int s1p2 = index + 2*dim1Stride;
        const ${DTYPE} maskWm1 = maskW[s1m1];
        const ${DTYPE} maskWp1 = maskW[s1p1];
        const ${DTYPE} maskwp2 = maskW[s1p2];
        const ${DTYPE} varSM1 = var[s1m1];
        const ${DTYPE} varSP1 = var[s1p1];
        const ${DTYPE} varSP2 = var[s1p2];
        if (x < dim1-1)
            maskUtr = maskWs * maskWp1;
            maskUtrP1 = maskWp1 * maskwp2;
            maskUtrM1 = maskWm1 * maskWs;
        const ${DTYPE} dx = cost[y] * dxt[x];
        adv_fe = calcFlux(1, velS, dt_tracer, dx, varS, varSM1, varSP1, varSP2, maskUtr, maskUtrM1, maskUtrP1);
    }
    if (y > 0 && y < dim2-2 && x > 1 && x < dim1-2)
    {
        const ${DTYPE} velS = v_wgrid[s];
        ${DTYPE} maskVtr = 0;
        ${DTYPE} maskVtrP1 = 0;
        ${DTYPE} maskVtrM1 = 0;
        const int s1m1 = index - dim2Stride;
        const int s1p1 = index + dim2Stride;
        const int s1p2 = index + 2*dim2Stride;
        const ${DTYPE} maskWm1 = maskW[s1m1];
        const ${DTYPE} maskWp1 = maskW[s1p1];
        const ${DTYPE} maskwp2 = maskW[s1p2];
        const ${DTYPE} varSM1 = var[s1m1];
        const ${DTYPE} varSP1 = var[s1p1];
        const ${DTYPE} varSP2 = var[s1p2];
        if (y < dim2-1)
            maskVtr = maskWs * maskWp1;
            maskVtrP1 = maskWp1 * maskwp2;
            maskVtrM1 = maskWm1 * maskWs;
        const ${DTYPE} dx = cost[y] * dyt[y];
        adv_fn = calcFlux(cosu[y], velS, dt_tracer, dx, varS, varSM1, varSP1, varSP2, maskVtr, maskVtrM1, maskVtrP1);
    }
    if (z < dim3-1 && x > 1 && x < dim1-2 && y > 1 && y < dim2-2)
    {
        const ${DTYPE} velS = w_wgrid[s];
        ${DTYPE} maskWtr = 0;
        ${DTYPE} maskWtrP1 = 0;
        ${DTYPE} maskWtrM1 = 0;
        const int s1m1 = index - dim3Stride;
        const int s1p1 = index + dim3Stride;
        const int s1p2 = index + 2*dim3Stride;
        ${DTYPE} maskWm1 = 0;
        ${DTYPE} varSM1 = 0 ;
        if (z != 0)
        {
            maskWm1 = maskW[s1m1];
            varSM1 = var[s1m1];
        }
        ${DTYPE} maskwp2 = 0;
        ${DTYPE} varSP2 = 0;
        if (z < dim3-2)
        {
            maskwp2 = maskW[s1p2];
            varSP2 = var[s1p2];
        }
        const ${DTYPE} varSP1 = var[s1p1];
        const ${DTYPE} maskWp1 = maskW[s1p1];
        if (z < dim3-1)
            maskWtr = maskWs * maskWp1;
            maskWtrP1 = maskWp1 * maskwp2;
            maskWtrM1 = maskWm1 * maskWs;
        const ${DTYPE} dx = dzw[z];
        adv_ft = calcFlux(1, velS, dt_tracer, dx, varS, varSM1, varSP1, varSP2, maskWtr, maskWtrM1, maskWtrP1);
   }

   flux_east[index] = adv_fe;
   flux_north[index] = adv_fn;
   flux_top[index] = adv_ft;
}
''').substitute(
    DTYPE='double'
)

superbee_kernel = cupy.RawKernel(kernel, 'superbee')


def adv_superbee_cupy(var, u_wgrid, v_wgrid, w_wgrid, maskW, dxt, dyt, dzw, cost, cosu, dt_tracer, out1, out2, out3,  blocksize=256):
    dim1, dim2, dim3 = var.shape
    total_size = dim1 * dim2 * dim3
    numblocks = int((total_size + blocksize - 1) / blocksize)


    superbee_kernel(
        (numblocks,),
        (blocksize,),
        (var, u_wgrid, v_wgrid, w_wgrid, maskW, dxt, dyt, dzw, cost, cosu, dt_tracer, out1, out2, out3, dim1, dim2, dim3)
    )

    return out1, out2, out3






import time







if __name__ == "__main__":
    arg1 = jnp.load("arg1.txt.npy")
    print("var: " + str(arg1.shape))
    arg2 = jnp.load("arg2.txt.npy")
    print("u_wgrid: " + str(arg2.shape))
    arg3 = jnp.load("arg3.txt.npy")
    print("v_wgrid: " + str(arg3.shape))
    arg4 = jnp.load("arg4.txt.npy")
    print("w_wgrid: " + str(arg4.shape))
    arg5 = jnp.load("arg5.txt.npy")
    print("maskW: " + str(arg5.shape))
    arg6 = jnp.load("arg6.txt.npy")
    print("dxt: " + str(arg6.shape))
    arg7 = jnp.load("arg7.txt.npy")
    print("dyt: " + str(arg7.shape))
    arg8 = jnp.load("arg8.txt.npy")
    print("dzw: " + str(arg8.shape))
    arg9 = jnp.load("arg9.txt.npy")
    print("cost: " + str(arg9.shape))
    arg10 = jnp.load("arg10.txt.npy")
    print("cosu: " + str(arg10.shape))
    arg11 = jnp.load("arg11.txt.npy")
    print("dt_tracer: " + str(arg11.shape))
    res1 = jnp.load("res1.txt.npy")
    res2 = jnp.load("res2.txt.npy")
    res3 = jnp.load("res3.txt.npy")
    jax_arg1 = jnp.array(arg1)
    jax_arg2 = jnp.array(arg2)
    jax_arg3 = jnp.array(arg3)
    jax_arg4 = jnp.array(arg4)
    jax_arg5 = jnp.array(arg5)
    jax_arg6 = jnp.array(arg6)
    jax_arg7 = jnp.array(arg7)
    jax_arg8 = jnp.array(arg8)
    jax_arg9 = jnp.array(arg9)
    jax_arg10 = jnp.array(arg10)
    jax_arg11 = jnp.array(arg11)
    jaxfun = jax.jit(adv_flux_superbee_wgrid, backend="gpu")
    start = time.time()
    jaxres1, jaxres2, jaxres3 = jaxfun(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11)
    jaxres1.block_until_ready()
    end = time.time()
    print("jax first call took: " + str(end - start))

    jax_arg1 = jnp.array(arg2)
    jax_arg2 = jnp.array(arg1)
    jax_arg3 = jnp.array(arg4)
    jax_arg4 = jnp.array(arg3)
    jax_arg5 = jnp.array(arg5)
    jax_arg6 = jnp.array(arg6)
    jax_arg7 = jnp.array(arg7)
    jax_arg8 = jnp.array(arg8)
    jax_arg9 = jnp.array(arg9)
    jax_arg10 = jnp.array(arg10)
    jax_arg11 = jnp.array(arg11)
    jaxfun = jax.jit(adv_flux_superbee_wgrid, backend="gpu")
    start = time.time()
    jaxres1, jaxres2, jaxres3 = jaxfun(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11)
    jaxres1.block_until_ready()
    end = time.time()
    print("jax second call took: " + str(end - start))


    np.testing.assert_allclose(jnp.asarray(jaxres1), res1)
    np.testing.assert_allclose(jnp.asarray(jaxres2), res2)
    np.testing.assert_allclose(jnp.asarray(jaxres3), res3)

    cupy_arg1 = cupy.asarray(arg2)
    cupy_arg2 = cupy.asarray(arg1)
    cupy_arg3 = cupy.asarray(arg4)
    cupy_arg4 = cupy.asarray(arg3)
    cupy_arg5 = cupy.asarray(arg5)
    cupy_arg6 = cupy.asarray(arg6)
    cupy_arg7 = cupy.asarray(arg7)
    cupy_arg8 = cupy.asarray(arg8)
    cupy_arg9 = cupy.asarray(arg9)
    cupy_arg10 = cupy.asarray(arg10)
    dt_tracer = np.asscalar(arg11)
    print(type(dt_tracer))
    out1 = cupy.empty(arg5.shape, dtype=arg5.dtype)
    out2 = cupy.empty(arg5.shape, dtype=arg5.dtype)
    out3 = cupy.empty(arg5.shape, dtype=arg5.dtype)
    start = time.time()

    flux_east, flux_north, flux_top = adv_superbee_cupy(cupy_arg1 ,cupy_arg2 ,cupy_arg3 ,cupy_arg4 ,cupy_arg5 ,cupy_arg6 ,cupy_arg7 ,cupy_arg8 ,cupy_arg9 ,cupy_arg10 ,dt_tracer ,out1 ,out2 ,out3)
    cupy.cuda.Stream.null.synchronize()
    end = time.time()
    print("cupy first call took: " + str(end - start))

    cupy_arg1 = cupy.asarray(arg1)
    cupy_arg2 = cupy.asarray(arg2)
    cupy_arg3 = cupy.asarray(arg3)
    cupy_arg4 = cupy.asarray(arg4)
    cupy_arg5 = cupy.asarray(arg5)
    cupy_arg6 = cupy.asarray(arg6)
    cupy_arg7 = cupy.asarray(arg7)
    cupy_arg8 = cupy.asarray(arg8)
    cupy_arg9 = cupy.asarray(arg9)
    cupy_arg10 = cupy.asarray(arg10)
    dt_tracer = np.asscalar(arg11)

    start = time.time()

    flux_east, flux_north, flux_top = adv_superbee_cupy(cupy_arg1 ,cupy_arg2 ,cupy_arg3 ,cupy_arg4 ,cupy_arg5 ,cupy_arg6 ,cupy_arg7 ,cupy_arg8 ,cupy_arg9 ,cupy_arg10 ,dt_tracer ,out1 ,out2 ,out3)
    cupy.cuda.Stream.null.synchronize()
    end = time.time()
    print("cupy second call took: " + str(end - start))

    print(np.allclose(flux_east.get(), res1))
    print(np.allclose(flux_north.get(), res2))
    print(np.allclose(flux_top.get(), res3))
    np.testing.assert_allclose(flux_east.get(), res1)
    np.testing.assert_allclose(flux_north.get(), res2)
    np.testing.assert_allclose(flux_top.get(), res3)
