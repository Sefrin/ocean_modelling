# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import functools
import itertools
import operator

import numpy as np
from jax.lib import xla_client
from jax.lax import transpose
try:
    from . import cuda_tridiag_kernels
    for _name, _value in cuda_tridiag_kernels.registrations().items():
        xla_client.register_custom_call_target(_name, _value, platform="gpu")
except ImportError:
  print("could not import cuda_tridiag_kernels. Are .so files present?")
  pass

_prod = lambda xs: functools.reduce(operator.mul, xs, 1)

# TODO(phawkins): remove after we no longer need to support old jax releases.
def _unpack_builder(c):
    # If `c` is a ComputationBuilder object, extracts the underlying XlaBuilder.
    return getattr(c, "_builder", c)

def tridiag(builder, a, b, c, d):
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

    b_shape = builder.get_shape(b)
    assert b_shape.element_type() == dtype
    assert dims == b_shape.dimensions(), (dims, b_shape)
    c_shape = builder.get_shape(c)
    assert c_shape.element_type() == dtype
    assert dims == c_shape.dimensions(), (dims, c_shape)
    d_shape = builder.get_shape(d)
    assert d_shape.element_type() == dtype
    assert dims == d_shape.dimensions(), (dims, d_shape)

    total_size = _prod(dims)
    system_depth = dims[0]

    num_systems = int(total_size / system_depth)


    if dtype is np.dtype(np.float32):
        kernel = b'cuda_tridiag_float'
    elif dtype is np.dtype(np.float64):
        kernel = b'cuda_tridiag_double'
    else:
        raise RuntimeError('got unrecognized dtype')

    opaque = cuda_tridiag_kernels.cuda_tridiag_descriptor(total_size, num_systems, system_depth)
    
    in_shape = xla_client.Shape.array_shape(dtype, dims, (0,1,2))
    out_shape = xla_client.Shape.array_shape(dtype, dims, (0,1,2))
    return xla_client.ops.CustomCallWithLayout(
        builder, kernel,
        operands=(a, b, c, d),
        shape_with_layout=out_shape,
        operand_shapes_with_layout=(in_shape,) * 4,
        opaque=opaque)
