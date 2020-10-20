import numpy as np
import jax.numpy as jnp
import functools
import itertools
import operator

from jax import abstract_arrays
from jax.core import Primitive
from jax.lib import xla_client
from jax.interpreters import xla
import tridiag_cuda

try:
    for _name, _value in tridiag_cuda.gpu_custom_call_targets.items():
        xla_client.register_custom_call_target(_name, _value, platform="gpu")
except ImportError:
  print("could not import cuda_tridiag_kernels. Are .so files present?")
  pass

_prod = lambda xs: functools.reduce(operator.mul, xs, 1)

# TODO(phawkins): remove after we no longer need to support old jax releases.
def _unpack_builder(c):
    # If `c` is a ComputationBuilder object, extracts the underlying XlaBuilder.
    return getattr(c, "_builder", c)

def _tridiag(builder, a, b, c, d):
    """Tridiag kernel for GPU."""
    
    builder = _unpack_builder(builder)
    a_shape = builder.get_shape(a)
    dtype = a_shape.element_type()
    dims = a_shape.dimensions()

    supported_dtypes = (
        np.dtype(np.float32),
        np.dtype(np.float64),
    )
    
    if dtype not in supported_dtypes:
        raise TypeError('Tridiag only supports {} arrays, got: {}'.format(supported_dtypes, dtype))

    total_size = _prod(dims)
    system_depth = dims[-1]

    num_systems = int(total_size / system_depth)

    if dtype is np.dtype(np.float32):
        kernel = b'cuda_tridiag_float'
    elif dtype is np.dtype(np.float64):
        kernel = b'cuda_tridiag_double'
    else:
        raise RuntimeError('got unrecognized dtype')

    opaque = tridiag_cuda.build_tridiag_descriptor(total_size, num_systems, system_depth)
    num_dims = len(dims)
    shape_tup = tuple(range(num_dims-2, -1, -1)) + (num_dims-1,)
    shape = xla_client.Shape.array_shape(dtype, dims, shape_tup) # transpose here for coalesced access!
    return xla_client.ops.CustomCallWithLayout(
        builder, kernel,
        operands=(a, b, c, d),
        shape_with_layout=shape,
        operand_shapes_with_layout=(shape,) * 4,
        opaque=opaque)


def tridiag(a, b, c, d):
    if not a.shape == b.shape == c.shape == d.shape:
        raise ValueError('all inputs must have identical shape')
    if not a.dtype == b.dtype == c.dtype == d.dtype:
        raise ValueError('all inputs must have the same dtype')
    return tridiag_p.bind(a, b, c, d) #transpose(res, (0,2,1))

def tridiag_impl(*args, **kwargs):
    return xla.apply_primitive(tridiag_p, *args, **kwargs)

def _tridiag_gpu_translation_rule(computation_builder, a, b, c, d):
    return _tridiag(computation_builder, a, b, c, d)

def tridiag_abstract_eval(a, b, c, d):
    return abstract_arrays.ShapedArray(a.shape, a.dtype)

tridiag_p = Primitive('tridiag')
tridiag_p.def_impl(tridiag_impl)
tridiag_p.def_abstract_eval(tridiag_abstract_eval)
xla.backend_specific_translations['gpu'][tridiag_p] = _tridiag_gpu_translation_rule