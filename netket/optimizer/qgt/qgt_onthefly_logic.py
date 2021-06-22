# Copyright 2021 The NetKet Authors - All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import jax
import jax.numpy as jnp
from functools import partial
from netket.stats import subtract_mean
from netket.utils import mpi
import netket.jax as nkjax
from netket.jax import tree_conj, tree_dot, tree_cast, tree_axpy

# Stochastic Reconfiguration with jvp and vjp

# Here O (Oks) is the jacobian (derivatives w.r.t. params) of the vectorised (in x) log wavefunction (forward_fn) evaluated at all samples.
# instead of computing (and storing) the full jacobian matrix jvp and vjp are used to implement the matrix vector multiplications with it.
# Expectation values are then just the mean over the leading dimension.


def O_vjp(vjp_fun, w):
    res, = vjp_fun(w)
    return jax.tree_map(lambda x: mpi.mpi_sum_jax(x)[0], res)  # allreduce w/ MPI.SUM


def O_vjp_rc(vjp_fun, w):
    res_r, = vjp_fun(w)
    res_i, = vjp_fun(-1.0j * w)
    res = jax.tree_multimap(jax.lax.complex, res_r, res_i)
    return jax.tree_map(lambda x: mpi.mpi_sum_jax(x)[0], res)  # allreduce w/ MPI.SUM


def O_mean(forward_fn, params, samples, holomorphic=True):
    r"""
    compute \langle O \rangle
    i.e. the mean of the rows of the jacobian of forward_fn
    """

    # determine the output type of the forward pass
    dtype = jax.eval_shape(forward_fn, params, samples).dtype
    w = jnp.ones(samples.shape[0], dtype=dtype) * (
        1.0 / (samples.shape[0] * mpi.n_nodes)
    )

    homogeneous = nkjax.tree_ishomogeneous(params)
    real_params = not nkjax.tree_leaf_iscomplex(params)
    real_out = not nkjax.is_complex_dtype(dtype)

    # only support R->C, R->R, holomorphic C->C
    assert homogeneous and (real_params or holomorphic)

    _, vjp_fun = jax.vjp(lambda p: forward_fn(p, samples), params)

    if real_params and not real_out:
        # R->C
        return O_vjp_rc(vjp_fun, w)
    else:
        # R->R and holomorphic C->C
        return O_vjp(vjp_fun, w)


def OH_w(vjp_fun, w):
    r"""
    compute  O^H w
    (where ^H is the hermitian transpose)
    """

    # O^H w = (w^H O)^H
    # The transposition of the 1D arrays is omitted in the implementation:
    # (w^H O)^H -> (w* O)*

    # TODO The allreduce in O_vjp could be deferred until after the tree_cast
    # where the amount of data to be transferred would potentially be smaller
    # CHECK the tree_cast is not needed because the vjp returns the right dtypes
    # anyway
    return tree_conj(O_vjp(vjp_fun, w.conjugate()))


def Odagger_O_v(jvp_fun, vjp_fun, v, *, center=False):
    r"""
    if center=False (default):
        compute \langle O^\dagger O \rangle v

    else (center=True):
        compute \langle O^\dagger \Delta O \rangle v
        where \Delta O = O - \langle O \rangle
    """

    # w is an array of size n_samples; each MPI rank has its own slice
    w = jvp_fun(v)
    # w /= n_samples (elementwise):
    w = w * (1.0 / (w.size * mpi.n_nodes))

    if center:
        w = subtract_mean(w)  # w/ MPI

    return OH_w(vjp_fun, w)


Odagger_DeltaO_v = partial(Odagger_O_v, center=True)
