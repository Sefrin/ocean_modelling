import numpy as np
import jax.numpy as jnp
import functools
import itertools
import operator

from jax import abstract_arrays
from jax.core import Primitive
from jax.lib import xla_client
from jax.interpreters import xla
import superbee_cuda

try:
    for _name, _value in superbee_cuda.gpu_custom_call_targets.items():
        xla_client.register_custom_call_target(_name, _value, platform="gpu")
except ImportError:
  print("could not import cuda_superbee_kernels. Are .so files present?")
  pass

_prod = lambda xs: functools.reduce(operator.mul, xs, 1)

# TODO(phawkins): remove after we no longer need to support old jax releases.
def _unpack_builder(c):
    # If `c` is a ComputationBuilder object, extracts the underlying XlaBuilder.
    return getattr(c, "_builder", c)

def _superbee(builder, var, u_wgrid, v_wgrid, w_wgrid, maskW, dxt, dyt, dzw, cost, cosu, dt_tracer):
    """Superbee kernel for GPU."""
    
    builder = _unpack_builder(builder)
    a_shape = builder.get_shape(var)
    dtype = a_shape.element_type()
    dims = a_shape.dimensions()
    
    supported_dtypes = (
        np.dtype(np.float32),
        np.dtype(np.float64),
    )
    
    if dtype not in supported_dtypes:
        raise TypeError('Superbee only supports {} arrays, got: {}'.format(supported_dtypes, dtype))



    if dtype is np.dtype(np.float32):
        kernel = b'cuda_superbee_float'
    elif dtype is np.dtype(np.float64):
        kernel = b'cuda_superbee_double'
    else:
        raise RuntimeError('got unrecognized dtype')

    opaque = superbee_cuda.build_superbee_descriptor(dims[0], dims[1], dims[2])
    
    out_shape = xla_client.Shape.array_shape(dtype, dims, (2,1,0))
    x_shape = xla_client.Shape.array_shape(dtype, (dims[0],), (0,))
    y_shape = xla_client.Shape.array_shape(dtype, (dims[1],), (0,))
    z_shape = xla_client.Shape.array_shape(dtype, (dims[2],), (0,))
    scalar = xla_client.Shape.array_shape(dtype, (), ())
    return xla_client.ops.CustomCallWithLayout(
        builder, kernel,
        operands=(var, u_wgrid, v_wgrid, w_wgrid, maskW, dxt, dyt, dzw, cost, cosu, dt_tracer),
        shape_with_layout=xla_client.Shape.tuple_shape([out_shape, out_shape, out_shape]),
        operand_shapes_with_layout=(out_shape,
                                    out_shape,
                                    out_shape,
                                    out_shape,
                                    out_shape,
                                    x_shape,
                                    y_shape,
                                    z_shape,
                                    x_shape,
                                    x_shape,
                                    scalar
                                    ),
        opaque=opaque)


def superbee(var, u_wgrid, v_wgrid, w_wgrid, maskW, dxt, dyt, dzw, cost, cosu, dt_tracer):
    if not var.shape == u_wgrid.shape == v_wgrid.shape == w_wgrid.shape:
        raise ValueError('first five inputs must have identical shape')
    if not dxt.shape[0] == var.shape[0] == cost.shape[0] == cosu.shape[0]:
        raise ValueError('dxt, cost, cosu must have same shape as x dim of var')
    if not dyt.shape[0] == var.shape[1]:
        raise ValueError('dyt must have same shape as y dim of var')
    if not dzw.shape[0] == var.shape[2]:
        raise ValueError('dzw must have same shape as z dim of var')

    if not var.dtype == u_wgrid.dtype == v_wgrid.dtype == w_wgrid.dtype \
        == maskW.dtype == dxt.dtype == dyt.dtype == dzw.dtype == cost.dtype \
        == cosu.dtype == dt_tracer.dtype:
        raise ValueError('all inputs must have the same dtype')
    
    return superbee_p.bind(var, u_wgrid, v_wgrid, w_wgrid, maskW, dxt, dyt, dzw, cost, cosu, dt_tracer) 

def superbee_impl(*args, **kwargs):
    return xla.apply_primitive(superbee_p, *args, **kwargs)

def _superbee_gpu_translation_rule(computation_builder, var, u_wgrid, v_wgrid, w_wgrid, maskW, dxt, dyt, dzw, cost, cosu, dt_tracer):
    return _superbee(computation_builder, var, u_wgrid, v_wgrid, w_wgrid, maskW, dxt, dyt, dzw, cost, cosu, dt_tracer)

def superbee_abstract_eval(var, u_wgrid, v_wgrid, w_wgrid, maskW, dxt, dyt, dzw, cost, cosu, dt_tracer):

    aarr = abstract_arrays.ShapedArray(var.shape, var.dtype)
    return (aarr,) * 3

superbee_p = Primitive('superbee')
superbee_p.multiple_results = True
superbee_p.def_impl(superbee_impl)
superbee_p.def_abstract_eval(superbee_abstract_eval)
xla.backend_specific_translations['gpu'][superbee_p] = _superbee_gpu_translation_rule