
import numpy as np
import jax
import jax.numpy as jnp
from jax.config import config
config.update('jax_enable_x64', True)
from turbulent_kinetic_energy import tke_jax_old # import prepare_inputs, run
import math
def where(mask, a, b):
    return np.where(mask, a, b)

def solve_implicit(ks, a, b, c, d, b_edge=None, d_edge=None):
    land_mask = (ks >= 0)[:, :, np.newaxis]
    edge_mask = land_mask & (np.arange(a.shape[2])[np.newaxis, np.newaxis, :]
                             == ks[:, :, np.newaxis])
    water_mask = land_mask & (np.arange(a.shape[2])[np.newaxis, np.newaxis, :]
                              >= ks[:, :, np.newaxis])

    a_tri = water_mask * a * np.logical_not(edge_mask)
    b_tri = where(water_mask, b, 1.)
    if b_edge is not None:
        b_tri = where(edge_mask, b_edge, b_tri)
    c_tri = water_mask * c
    d_tri = water_mask * d
    if d_edge is not None:
        d_tri = where(edge_mask, d_edge, d_tri)

    return solve_tridiag(a_tri, b_tri, c_tri, d_tri), water_mask

def solve_tridiag(a, b, c, d):
    """
    Solves a tridiagonal matrix system with diagonals a, b, c and RHS vector d.
    """
    assert a.shape == b.shape and a.shape == c.shape and a.shape == d.shape

    def compute_primes(last_primes, x):
        last_cp, last_dp = last_primes
        a, b, c, d = x
        cp = c / (b - a * last_cp)
        dp = (d - a * last_dp) / (b - a * last_cp)
        new_primes = jnp.stack((cp, dp))
        return new_primes, new_primes

    diags_stacked = jnp.stack(
        [arr.transpose((2, 0, 1)) for arr in (a, b, c, d)],
        axis=1
    )
    _, primes = jax.lax.scan(compute_primes, jnp.zeros((2, *a.shape[:-1])), diags_stacked)

    def backsubstitution(last_x, x):
        cp, dp = x
        new_x = dp - cp * last_x
        return new_x, new_x

    _, sol = jax.lax.scan(backsubstitution, jnp.zeros(a.shape[:-1]), primes[::-1])
    return sol[::-1].transpose((1, 2, 0))

def _calc_cr(rjp, rj, rjm, vel):
    """
    Calculates cr value used in superbee advection scheme
    """
    eps = 1e-20  # prevent division by 0
    return where(vel > 0., rjm, rjp) / where(jnp.abs(rj) < eps, eps, rj)

def pad_z_edges(arr):
    arr_shape = list(arr.shape)
    arr_shape[2] += 2
    out = jnp.zeros(arr_shape, arr.dtype)
    out = jax.ops.index_update(
        out, jax.ops.index[:, :, 1:-1], arr
    )
    return out

# def limiter(cr):
#     return jnp.maximum(0., jnp.maximum(jnp.minimum(1., 2 * cr), jnp.minimum(2., cr)))


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


def limiter(cr):
    return max(0.0, max(min(1.0, 2.0 * cr), min(2.0, cr)))


def calcFlux(velfac, velS, dt_tracer, dx, varS, varSM1, varSP1, varSP2, maskS, maskSM1, maskSP1):
    scaledVel =  velfac * velS
    uCFL = abs(scaledVel * dt_tracer / dx)
    rjp = (varSP2 - varSP1) * maskSP1
    rj = (varSP1 - varS) * maskS
    rjm = (varS - varSM1) * maskSM1
    divisor = rj
    epsilon = 1e-20
    if abs(divisor) < epsilon:
        divisor = epsilon
    if velS > 0:
        cr = rjm / divisor
    else:
        cr = rjp / divisor
    
    cr = limiter(cr)
    return scaledVel * (varSP1 + varS) * 0.5 - abs(scaledVel) * ((1.0 - cr) + uCFL * cr) * rj * 0.5

def adv_flux_superbee_wgrid(var, u_wgrid, v_wgrid, w_wgrid, maskW, dxt, dyt, dzw, cost, cosu, dt_tracer):
    """
    Calculates advection of a tracer defined on Wgrid
    """
    # maskUtr = np.zeros_like(maskW)
    # maskUtr = jax.ops.index_update(
    #     maskUtr, jax.ops.index[:-1, :, :],
    #     maskW[1:, :, :] * maskW[:-1, :, :]
    # )
    # adv_fe = jax.ops.index_update(
    #     adv_fe, jax.ops.index[1:-2, 2:-2, :],
    #     _adv_superbee(u_wgrid, var, maskUtr, dxt, 0, cost, cosu, dt_tracer)
    # )
        # maskVtr = np.zeros_like(maskW)
    # maskVtr = jax.ops.index_update(
    #     maskVtr, jax.ops.index[:, :-1, :],
    #     maskW[:, 1:, :] * maskW[:, :-1, :]

    # )
    adv_fn = np.zeros_like(maskW)
    # adv_fn = jax.ops.index_update(
    #     adv_fn, jax.ops.index[2:-2, 1:-2, :],
    #     _adv_superbee(v_wgrid, var, maskVtr, dyt, 1, cost, cosu, dt_tracer)
    # )
    #     maskWtr = np.zeros_like(maskW)
    # maskWtr = jax.ops.index_update(
    #     maskWtr, jax.ops.index[:, :, :-1],
    #     maskW[:, :, 1:] * maskW[:, :, :-1]
    # )
    adv_ft = np.zeros_like(maskW)
    # adv_ft = jax.ops.index_update(
    #     adv_ft, jax.ops.index[2:-2, 2:-2, :-1],
    #     _adv_superbee(w_wgrid, var, maskWtr, dzw, 2, cost, cosu, dt_tracer)
    # )
    adv_fe = np.zeros_like(maskW)
    for x in range(adv_fe.shape[0]):
        for y in range(adv_fe.shape[1]):
            for z in range(adv_fe.shape[2]):
                if x >= 1 and x < adv_fe.shape[0] - 2 and y >= 2 and y < adv_fe.shape[1] - 2:
                    dx = cost[y]
                    velS = u_wgrid[x,y,z]
                    maskUtr = 0
                    maskUtrP1 = 0
                    maskUtrM1 = 0
                    maskWm1 = maskW[x-1,y,z]
                    maskWs = maskW[x,y,z]
                    maskWp1 = maskW[x+1,y,z]
                    maskwp2 = maskW[x+2,y,z]

                    varSM1 = var[x-1,y,z]
                    varS = var[x,y,z]
                    varSP1 = var[x+1,y,z]
                    varSP2 = var[x+2,y,z]
                    if x < adv_fe.shape[0]-1:
                        maskUtr = maskWs * maskWp1
                        maskUtrP1 = maskWp1 * maskwp2
                        maskUtrM1 = maskWm1 * maskWs
                    
                    dx = cost[y] * dxt[x]
                    adv_fe[x,y,z] = calcFlux(1, velS, dt_tracer, dx, varS, varSM1, varSP1, varSP2, maskUtr, maskUtrM1, maskUtrP1)
                if y >= 1 and y < adv_fn.shape[1] - 2 and x >= 2 and x < adv_fn.shape[0]-2:
                    velS = v_wgrid[x,y,z]
                    maskVtr = 0
                    maskVtrP1 = 0
                    maskVtrM1 = 0
                    maskWm1 = maskW[x,y-1,z]
                    maskWs = maskW[x,y,z]
                    maskWp1 = maskW[x,y+1,z]
                    maskwp2 = maskW[x,y+2,z]
                    varSM1 = var[x,y-1,z]
                    varS = var[x,y,z]
                    varSP1 = var[x,y+1,z]
                    varSP2 = var[x,y+2,z]
                    if y < adv_fn.shape[1]-1:    
                        maskVtr = maskWs * maskWp1
                        maskVtrP1 = maskWp1 * maskwp2
                        maskVtrM1 = maskWm1 * maskWs
                    dx = cost[y] * dyt[y]
                    adv_fn[x,y,z] = calcFlux(cosu[y], velS, dt_tracer, dx, varS, varSM1, varSP1, varSP2, maskVtr, maskVtrM1, maskVtrP1)
                if z < adv_ft.shape[2]-1 and x >= 2 and x < adv_ft.shape[0]-2 and y >= 2 and y < adv_ft.shape[1]-2:
                    velS = w_wgrid[x,y,z]
                    maskWtr = 0
                    maskWtrP1 = 0
                    maskWtrM1 = 0
                    maskWm1 = 0
                    varSM1 = 0 
                    if z != 0:
                        maskWm1 = maskW[x,y,z-1]
                        varSM1 = var[x,y,z-1]
                    maskwp2 = 0
                    varSP2 = 0
                    if (z < adv_ft.shape[2]-2):
                        maskwp2 = maskW[x,y,z+2]
                        varSP2 = var[x,y,z+2]
                    
                    varSP1 = var[x,y,z+1]
                    maskWp1 = maskW[x,y,z+1]
                    maskWtr = maskWs * maskWp1
                    maskWtrP1 = maskWp1 * maskwp2
                    maskWtrM1 = maskWm1 * maskWs
                    dx = dzw[z]
                    adv_ft[x,y,z] = calcFlux(1, velS, dt_tracer, dx, varS, varSM1, varSP1, varSP2, maskWtr, maskWtrM1, maskWtrP1)

    return adv_fe, adv_fn, adv_ft

def integrate_tke(u, v, w, maskU, maskV, maskW, dxt, dxu, dyt, dyu, dzt, dzw, cost, cosu, kbot, kappaM, mxl, forc, forc_tke_surface, tke, dtke):
    tau = 0
    taup1 = 1
    taum1 = 2

    dt_tracer = 1.
    dt_mom = 1.
    AB_eps = 0.1
    alpha_tke = 1.
    c_eps = 0.7
    K_h_tke = 2000.

    flux_east = np.zeros_like(maskU)
    flux_north = np.zeros_like(maskU)
    flux_top = np.zeros_like(maskU)

    sqrttke = np.sqrt(np.maximum(0., tke[:, :, :, tau]))

    """
    integrate Tke equation on W grid with surface flux boundary condition
    """
    dt_tke = dt_mom  # use momentum time step to prevent spurious oscillations

    """
    vertical mixing and dissipation of TKE
    """
    ks = kbot - 1 # [2:-2, 2:-2]

    print("Init empty")
    a_tri = np.zeros((maskU.shape[0], maskU.shape[1], maskU.shape[2])) # [2:-2, 2:-2]) shapes match better if we ignore slicing
    b_tri = np.zeros((maskU.shape[0], maskU.shape[1], maskU.shape[2])) # [2:-2, 2:-2])
    c_tri = np.zeros((maskU.shape[0], maskU.shape[1], maskU.shape[2])) # [2:-2, 2:-2])
    d_tri = np.zeros((maskU.shape[0], maskU.shape[1], maskU.shape[2])) # [2:-2, 2:-2])
    delta = np.zeros((maskU.shape[0], maskU.shape[1], maskU.shape[2])) # [2:-2, 2:-2])
    b_tri_edge = np.zeros((maskU.shape[0], maskU.shape[1], maskU.shape[2])) # [2:-2, 2:-2])
    

    # delta = jax.ops.index_update(
    #     delta, jax.ops.index[:, :, :-1],
    #     dt_tke / dzt[np.newaxis, np.newaxis, 1:] * alpha_tke * 0.5 \
    #     * (kappaM[2:-2, 2:-2, :-1] + kappaM[2:-2, 2:-2, 1:])
    # )
    print("Init delta")
    for x in range(delta.shape[0]):
        for y in range(delta.shape[1]):
            for z in range(delta.shape[2]):
                if x >= 2 and x < delta.shape[0] - 2 and y >= 2 and y < delta.shape[1] - 2 and z < delta.shape[2]-1:
                    delta[x,y,z] = dt_tke / dzt[z+1] * alpha_tke * 0.5 \
                        * (kappaM[x, y, z] + kappaM[x, y, z+1])
                # else:
                #     # not necessary if we assume 0 init
                #     delta[x,y,z] = 0
        


    # a_tri = jax.ops.index_update(
    #     a_tri, jax.ops.index[:, :, 1:-1],
    #     -delta[:, :, :-2] /
    #     dzw[np.newaxis, np.newaxis, 1:-1]
    # )
    # a_tri = jax.ops.index_update(
    #     a_tri, jax.ops.index[:, :, -1],
    #     -delta[:, :, -2] / (0.5 * dzw[-1])
    # )
    print("Init attri")
    for x in range(a_tri.shape[0]):
        for y in range(a_tri.shape[1]):
            for z in range(a_tri.shape[2]):
                if x >= 2 and x < a_tri.shape[0] - 2 and y >= 2 and y < a_tri.shape[1] - 2:
                    if z > 0 and z < a_tri.shape[2]-1:
                        a_tri[x,y,z] = -delta[x,y,z-1] / dzw[z]
                    elif z == a_tri.shape[2] - 1:
                        a_tri[x,y,z] = -delta[x, y, z-1] / (0.5 * dzw[z])

    # b_tri = jax.ops.index_update(
    #     b_tri, jax.ops.index[:, :, 1:-1],
    #     1 + (delta[:, :, 1:-1] + delta[:, :, :-2]) / dzw[np.newaxis, np.newaxis, 1:-1] \
    #     + dt_tke * c_eps \
    #     * sqrttke[2:-2, 2:-2, 1:-1] / mxl[2:-2, 2:-2, 1:-1]
    # )
    # b_tri = jax.ops.index_update(
    #     b_tri, jax.ops.index[:, :, -1],
    #      1 + delta[:, :, -2] / (0.5 * dzw[-1]) \
    #     + dt_tke * c_eps / mxl[2:-2, 2:-2, -1] * sqrttke[2:-2, 2:-2, -1]
    # )
    print("Init b_tri")
    for x in range(b_tri.shape[0]):
        for y in range(b_tri.shape[1]):
            for z in range(b_tri.shape[2]):
                if x >= 2 and x < b_tri.shape[0] - 2 and y >= 2 and y < b_tri.shape[1] - 2:
                    if z > 0 and z < b_tri.shape[2]-1:
                        b_tri[x,y,z] = 1 + (delta[x, y, z] + delta[x, y, z-1]) / dzw[z] \
                            + dt_tke * c_eps \
                            * sqrttke[x, y, z] / mxl[x, y, z]
                    elif z == b_tri.shape[2]-1:
                        b_tri[x,y,z] =  1 + delta[x, y, z-1] / (0.5 * dzw[z]) \
                            + dt_tke * c_eps / mxl[x,y,z] * sqrttke[x,y,z]
                    else:
                        # not necessary if we assume 0 init
                        b_tri[x,y,z] = 0
                else:
                    # not necessary if we assume 0 init
                    b_tri[x,y,z] = 0

#   b_tri_edge = 1 + delta / dzw[np.newaxis, np.newaxis, :] \
#         + dt_tke * c_eps / mxl[2:-2, 2:-2, :] * sqrttke[2:-2, 2:-2, :]
    print("Init b_tri_edge")
    for x in range(b_tri_edge.shape[0]):
        for y in range(b_tri_edge.shape[1]):
            for z in range(b_tri_edge.shape[2]):
                if x >= 2 and x < b_tri_edge.shape[0] - 2 and y >= 2 and y < b_tri_edge.shape[1] - 2:
                    b_tri_edge[x,y,z] = 1 + delta[x,y,z] / dzw[z] \
                        + dt_tke * c_eps / mxl[x, y, z] * sqrttke[x, y, z] #mxl and sqrttke
                else:
                        # not necessary if we assume 0 init
                        b_tri[x,y,z] = 0


    # c_tri = jax.ops.index_update(
    #     c_tri, jax.ops.index[:, :, :-1],
    #      -delta[:, :, :-1] / dzw[np.newaxis, np.newaxis, :-1]
    # )
    print("Init c_tri")
    for x in range(c_tri.shape[0]):
        for y in range(c_tri.shape[1]):
            for z in range(c_tri.shape[2]):
                if x >= 2 and x < c_tri.shape[0] - 2 and y >= 2 and y < c_tri.shape[1] - 2:
                    if z < c_tri.shape[2]-1:
                        c_tri[x,y,z] = -delta[x,y,z] / dzw[z]

    # d_tri = tke[2:-2, 2:-2, :, tau] + dt_tke * forc[2:-2, 2:-2, :]
    # d_tri = jax.ops.index_add(
    #     d_tri, jax.ops.index[:, :, -1],
    #     dt_tke * forc_tke_surface[2:-2, 2:-2] / (0.5 * dzw[-1])
    # )
    print("Init d_tri")
    for x in range(d_tri.shape[0]):
        for y in range(d_tri.shape[1]):
            for z in range(d_tri.shape[2]):
                if x >= 2 and x < d_tri.shape[0] - 2 and y >= 2 and y < d_tri.shape[1] - 2:
                    d_tri[x,y,z] = tke[x,y,z,tau] + dt_tke * forc[x,y,z]
                    if z == d_tri.shape[2]-1:
                        d_tri[x,y,z] += dt_tke * forc_tke_surface[x,y] / (0.5 * dzw[z])
    
    # so far so good#
    print("Init masks and edge")
    # edge_mask = np.zeros(a_tri.shape)
    # water_mask = np.zeros(a_tri.shape)


    for x in range(a_tri.shape[0]):
        for y in range(a_tri.shape[1]):
            land_mask = ks[x,y] >= 0
            for z in range(a_tri.shape[2]):
                if x >= 2 and x < a_tri.shape[0] - 2 and y >= 2 and y < a_tri.shape[1] - 2:
                    edge_mask = land_mask and (z == ks[x, y])
                    water_mask = land_mask and (z >= ks[x, y])
                    if edge_mask:
                        a_tri[x,y,z] = 0 #water_mask * a_tri[x,y,z] * np.logical_not(edge_mask)
                    if not water_mask:
                        a_tri[x,y,z] = 0
                        b_tri[x,y,z] = 1.
                        c_tri[x,y,z] = 0
                        d_tri[x,y,z] = 0
                    if b_tri_edge is not None:
                        if edge_mask:
                            b_tri[x,y,z] = b_tri_edge[x,y,z]
    

                # if d_edge is not None:
                #     if edge_mask:
                #         d_tri[x,y,z] = d_edge[x,y,z]

    # if d_edge is not None:
    #     d_tri = where(edge_mask, d_edge, d_tri)
    print("solve tridiag")
    a_tri_jax = jnp.array(a_tri)
    b_tri_jax = jnp.array(b_tri)
    c_tri_jax = jnp.array(c_tri)
    d_tri_jax = jnp.array(d_tri)
    a_tri_jax.block_until_ready()
    b_tri_jax.block_until_ready()
    c_tri_jax.block_until_ready()
    d_tri_jax.block_until_ready()
    sol = solve_tridiag(a_tri_jax, b_tri_jax, c_tri_jax, d_tri_jax)
    print("solve tridiag done")
    sol.block_until_ready()
    sol = np.array(sol)
    # tke = jax.ops.index_update(
    #     tke, jax.ops.index[2:-2, 2:-2, :, taup1],
    #     where(water_mask, sol, tke[2:-2, 2:-2, :, taup1])
    # )
    print("integrate tridiag sol")
    for x in range(a_tri.shape[0]):
        for y in range(a_tri.shape[1]):
            for z in range(a_tri.shape[2]):
                water_mask = (ks[x,y] >= 0) and (z >= ks[x, y])
                if x >= 2 and x < c_tri.shape[0] - 2 and y >= 2 and y < c_tri.shape[1] - 2:
                    if water_mask:
                        tke[x,y,z,taup1] = sol[x,y,z]


    """
    Add TKE if surface density flux drains TKE in uppermost box
    """
    #mask = tke[2:-2, 2:-2, -1, taup1] < 0.0

        # tke_surf_corr = jax.ops.index_update(
        # tke_surf_corr, jax.ops.index[2:-2, 2:-2],
        # where(mask,
        #       -tke[2:-2, 2:-2, -1, taup1] * 0.5 * dzw[-1] / dt_tke,
        #       0.)
        #   )

        # tke = jax.ops.index_update(
        #     tke, jax.ops.index[2:-2, 2:-2, -1, taup1],
        #     np.maximum(0., tke[2:-2, 2:-2, -1, taup1])
        # )
    print("correct surf")
    tke_surf_corr = np.zeros((maskU.shape[0], maskU.shape[1]))
    for x in range(tke_surf_corr.shape[0]):
        for y in range(tke_surf_corr.shape[1]):
            if x >= 2 and x < tke_surf_corr.shape[0] - 2 and y >= 2 and y < tke_surf_corr.shape[1] - 2:
                tke_val = tke[x, y, tke.shape[2]-1, taup1]
                if tke_val < 0.0:
                    tke_surf_corr[x,y] = -tke_val * 0.5 * dzw[dzw.shape[0]-1] / dt_tke
                    tke[x, y,tke.shape[2]-1, taup1] = 0   
                else:
                    tke_surf_corr[x,y] = 0
  
    
    # """
    # add tendency due to lateral diffusion
    # """
    # flux_east = jax.ops.index_update(
    #     flux_east, jax.ops.index[:-1, :, :],
    #     K_h_tke * (tke[1:, :, :, tau] - tke[:-1, :, :, tau])
    #     / (cost[np.newaxis, :, np.newaxis] * dxu[:-1, np.newaxis, np.newaxis]) * maskU[:-1, :, :]
    # )
    print("lateral diffusion east")
    for x in range(flux_east.shape[0]):
        for y in range(flux_east.shape[1]):
            for z in range(flux_east.shape[2]):
                if x < flux_east.shape[0]-1:
                    flux_east[x,y,z] = K_h_tke * (tke[x+1, y, z, tau] - tke[x, y, z, tau]) \
                        / (cost[y] * dxu[x]) * maskU[x, y, z]


    # flux_north = jax.ops.index_update(
    #     flux_north, jax.ops.index[:, :-1, :],
    #     K_h_tke * (tke[:, 1:, :, tau] - tke[:, :-1, :, tau]) \
    #     / dyu[np.newaxis, :-1, np.newaxis] * maskV[:, :-1, :] * cosu[np.newaxis, :-1, np.newaxis]
    # )
    print("lateral diffusion north")
    for x in range(flux_north.shape[0]):
        for y in range(flux_north.shape[1]):
            for z in range(flux_north.shape[2]):
                if y < flux_north.shape[1]-1:
                    flux_north[x,y,z] = K_h_tke * (tke[x, y+1, z, tau] - tke[x, y, z, tau]) \
                        / dyu[y] * maskV[x, y, z] * cosu[y]


    # tke = jax.ops.index_add(
    #     tke, jax.ops.index[2:-2, 2:-2, :, taup1],
    #     dt_tke * maskW[2:-2, 2:-2, :] *
    #     ((flux_east[2:-2, 2:-2, :] - flux_east[1:-3, 2:-2, :])
    #         / (cost[np.newaxis, 2:-2, np.newaxis] * dxt[2:-2, np.newaxis, np.newaxis])
    #         + (flux_north[2:-2, 2:-2, :] - flux_north[2:-2, 1:-3, :])
    #         / (cost[np.newaxis, 2:-2, np.newaxis] * dyt[np.newaxis, 2:-2, np.newaxis]))
    # )
    print("add lateral diffusion")
    for x in range(tke.shape[0]):
        for y in range(tke.shape[1]):
            for z in range(tke.shape[2]):
                if x >= 2 and x < tke.shape[0] - 2 and y >= 2 and y < tke.shape[1] - 2:
                    tke[x,y,z,taup1] += dt_tke * maskW[x, y, z] * \
                        ((flux_east[x,y,z] - flux_east[x-1, y, z])
                        / (cost[y] * dxt[x])
                        + (flux_north[x,y,z] - flux_north[x, y-1, z])
                        / (cost[y] * dyt[y]))

    """
    add tendency due to advection
    """
    flux_east, flux_north, flux_top = adv_flux_superbee_wgrid(
        tke[:, :, :, tau], u[..., tau], v[..., tau], w[..., tau],
        maskW, dxt, dyt, dzw,
        cost, cosu, dt_tracer
    )

    # dtke = jax.ops.index_update(
    #     dtke, jax.ops.index[2:-2, 2:-2, :, tau],
    #     maskW[2:-2, 2:-2, :] * (-(flux_east[2:-2, 2:-2, :] - flux_east[1:-3, 2:-2, :])
    #     / (cost[jnp.newaxis, 2:-2, jnp.newaxis] * dxt[2:-2, jnp.newaxis, jnp.newaxis])
    #     - (flux_north[2:-2, 2:-2, :] - flux_north[2:-2, 1:-3, :])
    #     / (cost[jnp.newaxis, 2:-2, jnp.newaxis] * dyt[jnp.newaxis, 2:-2, jnp.newaxis]))
    # )

    print("Adding to dtke")
    for x in range(dtke.shape[0]):
        for y in range(dtke.shape[1]):
            for z in range(dtke.shape[2]):
                if x >= 2 and x < dtke.shape[0] - 2 and y >= 2 and y < dtke.shape[1] - 2:
                    dtke[x,y,z,tau] = maskW[x,y,z] * (-(flux_east[x,y,z] - flux_east[x-1, y, z]) \
                        / (cost[y] * dxt[x]) \
                        - (flux_north[x,y,z] - flux_north[x, y-1, z])                      \
                        / (cost[y] * dyt[y]))
                if z == 0:
                    dtke[x,y,z,tau] -= flux_top[x, y, 0] / dzw[0]
                if z >= 1 and z < dtke.shape[2]-1:
                    dtke[x,y,z,tau] -= (flux_top[x, y, z] - flux_top[x, y, z-1]) / dzw[z]
                if  z == dtke.shape[2]-1:
                    dtke[x,y,z,tau] -= (flux_top[x, y, z] - flux_top[x, y, z-1]) / \
                                            (0.5 * dzw[z])

                tke[x,y,z, taup1] += dt_tracer * ((1.5 + AB_eps) * dtke[x, y, z, tau] - (0.5 + AB_eps) * dtke[x, y, z, taum1])
    

    # dtke = jax.ops.index_add(
    #     dtke, jax.ops.index[:, :, 0, tau],
    #     -flux_top[:, :, 0] / dzw[0]
    # )
    # dtke = jax.ops.index_add(
    #     dtke, jax.ops.index[:, :, 1:-1, tau],
    #     -(flux_top[:, :, 1:-1] - flux_top[:, :, :-2]) / dzw[1:-1]
    # )
    # dtke = jax.ops.index_add(
    #     dtke, jax.ops.index[:, :, -1, tau],
    #     -(flux_top[:, :, -1] - flux_top[:, :, -2]) / \
    #     (0.5 * dzw[-1])
    # )

    # """
    # Adam Bashforth time stepping
    # """
    # tke = jax.ops.index_add(
    #     tke, jax.ops.index[:, :, :, taup1],
    #     dt_tracer * ((1.5 + AB_eps) * dtke[:, :, :, tau] - (0.5 + AB_eps) * dtke[:, :, :, taum1])
    # )

    return tke, dtke, tke_surf_corr, sol, a_tri, b_tri, c_tri, d_tri, delta, flux_east, flux_north


# def prepare_inputs(*inputs, device):
#     out = [np.array(k) for k in inputs]
#     for o in out:
#         o.block_until_ready()
#     return out


# def run(*inputs, device='cpu'):
#     outputs = integrate_tke(*inputs)
#     for o in outputs:
#         o.block_until_ready()
#     return outputs

def generate_inputs(size):

    np.random.seed(17)

    shape = (
        math.ceil(2 * size ** (1/3)),
        math.ceil(2 * size ** (1/3)),
        math.ceil(0.25 * size ** (1/3)),
    )

    # masks
    maskU, maskV, maskW = ((np.random.rand(*shape) < 0.8).astype('float64') for _ in range(3))

    # 1d arrays
    dxt, dxu = (np.random.randn(shape[0]) for _ in range(2))
    dyt, dyu = (np.random.randn(shape[1]) for _ in range(2))
    dzt, dzw = (np.random.randn(shape[2]) for _ in range(2))
    cost, cosu = (np.random.randn(shape[1]) for _ in range(2))

    # 2d arrays
    kbot = np.random.randint(0, shape[2], size=shape[:2])
    forc_tke_surface = np.random.randn(*shape[:2])

    # 3d arrays
    kappaM, mxl, forc = (np.random.randn(*shape) for _ in range(3))

    # 4d arrays
    u, v, w, tke, dtke = (np.random.randn(*shape, 3) for _ in range(5))

    return (
        u, v, w,
        maskU, maskV, maskW,
        dxt, dxu, dyt, dyu, dzt, dzw,
        cost, cosu,
        kbot,
        kappaM, mxl, forc,
        forc_tke_surface,
        tke, dtke
    )

args = generate_inputs(40000)

copy_args = [np.copy(arg) for arg in args]
results1 = integrate_tke(*copy_args)
results2 = tke_jax_old.run(*tke_jax_old.prepare_inputs(*args, device="cpu"))

for o in results2:
    o.block_until_ready()
print(np.testing.assert_allclose(np.array(results2[9]), results1[9]))
print(np.testing.assert_allclose(np.array(results2[10]), results1[10]))

print(np.testing.assert_allclose(np.array(results2[0]), results1[0]))
print(np.testing.assert_allclose(np.array(results2[1]), results1[1]))
print(np.testing.assert_allclose(np.array(results2[2]), results1[2]))


np.testing.assert_allclose(np.array(results2[3]), results1[3][2:-2,2:-2])
np.testing.assert_allclose(np.array(results2[4]), results1[4][2:-2,2:-2])
np.testing.assert_allclose(np.array(results2[5]), results1[5][2:-2,2:-2])
np.testing.assert_allclose(np.array(results2[6]), results1[6][2:-2,2:-2])
np.testing.assert_allclose(np.array(results2[7]), results1[7][2:-2,2:-2])
np.testing.assert_allclose(np.array(results2[8]), results1[8][2:-2,2:-2])
