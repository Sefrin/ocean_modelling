import numpy as np
import jax.numpy as jnp

from jax import abstract_arrays
from jax.core import Primitive
from jax.lib import xla_client
from jax.interpreters import xla
from jax.lax import transpose, reshape
from . import cuda_tridiag

def tridiag(a, b, c, d):
    if not a.shape == b.shape == c.shape == d.shape:
        raise ValueError('all inputs must have identical shape')

    if not a.dtype == b.dtype == c.dtype == d.dtype:
        raise ValueError('all inputs must have the same dtype')
    
    # transpose inputs
    dim_count = len(a.shape)
    new_shape = (dim_count-1,) + tuple(range(0,dim_count-1)) # (2,0,1) - (115,n,m)
   
    old_shape = tuple(range(1,dim_count)) + (0,)              #(1,2,0) 
   
    a_t = transpose(a, new_shape)
    b_t = transpose(b, new_shape)
    c_t = transpose(c, new_shape)
    d_t = transpose(d, new_shape)

    res = tridiag_p.bind(a_t, b_t, c_t, d_t)

    return transpose(res, old_shape)

def tridiag_impl(*args, **kwargs):
    return xla.apply_primitive(tridiag_p, *args, **kwargs)

def _tridiag_gpu_translation_rule(computation_builder, a, b, c, d):
    return cuda_tridiag.tridiag(computation_builder, a, b, c, d)

def tridiag_abstract_eval(a, b, c, d):
    return abstract_arrays.ShapedArray(a.shape, a.dtype)

if cuda_tridiag:
    tridiag_p = Primitive('tridiag')
    tridiag_p.def_impl(tridiag_impl)
    tridiag_p.def_abstract_eval(tridiag_abstract_eval)
    xla.backend_specific_translations['gpu'][tridiag_p] = _tridiag_gpu_translation_rule