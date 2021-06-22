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

from typing import Callable, Optional, Union, Tuple, Any
from functools import partial

import jax
from jax import numpy as jnp

from netket.utils import struct
from netket.utils.types import PyTree
import netket.jax as nkjax

from .qgt_onthefly_logic import O_mean, Odagger_O_v

from ..linear_operator import LinearOperator, Uninitialized


def QGTOnTheFly(vstate=None, centered: bool = True, holomorphic: bool = True, diag_shift: float = 0.0) -> "QGTOnTheFlyT":
    
    if vstate is None:
        return partial(QGTOnTheFly, **kwargs)

    return QGTOnTheFlyT(
        mat_vec=QGT_mat_vec_builder(
            apply_fun=vstate._apply_fun,
            params=vstate.parameters,
            samples=vstate.samples,
            model_state=vstate.model_state,
            centered=centered,
            holomorphic=holomorphic,
            diag_shift=diag_shift
        )
    )

@partial(jax.jit, static_argnums=(0,4,5))
def QGT_mat_vec_builder(apply_fun: Callable[[PyTree, jnp.ndarray], jnp.ndarray],
                        params: PyTree,
                        samples: jnp.ndarray,
                        model_state: Optional[PyTree],
                        centered: bool,
                        holomorphic: bool,
                        diag_shift: float):
    def forward_fn(W, σ):
        return apply_fun({"params": W, **model_state}, σ)
    
    # For centered=True and networks other than R→R, R→C, and holomorphic C→C
    # the params pytree must be disassembled into real and imaginary parts
    should_disassemble = False
    
    if centered:
        # Calculate and subtract ⟨O⟩ from the gradients by redefining forward_fn
        # If needed, disassemble mixed-dtype/non-holomorphic params into real
        homogeneous = nkjax.tree_ishomogeneous(params)
        real_params = not nkjax.tree_leaf_iscomplex(params)
        if not (homogeneous and (real_params or holomorphic)):
            # everything except R->R, holomorphic C->C and R->C
            should_disassemble = True
            params, reassemble = nkjax.tree_to_real(params)
            _forward_fn = forward_fn

            def forward_fn(p, x):
                return _forward_fn(reassemble(p), x)

        omean = O_mean(forward_fn, params, samples, holomorphic=holomorphic)

        _forward_fn = forward_fn

        def forward_fn(p,x):
            return _forward_fn(p, x) - tree_dot(p, omean)
    
    _, O_jvp = jax.linearize(lambda p: forward_fn(p, samples), (params,))
    _, O_vjp = jax.vjp(lambda p: forward_fn(p, samples), params)

    @jax.jit
    def mat_vec(v):
        if should_disassemble:
            v, _ = nkjax.tree_to_real(v)
        res = Odagger_O_v(O_jvp, O_vjp, v, center = not centered)
        if should_disassemble:
            res = reassemble(res)
        res = tree_axpy(diag_shift, v, res) # res += diag_shift * v
        return res

    return mat_vec

@struct.dataclass
class QGTOnTheFlyT(LinearOperator):
    mat_vec: Callable[PyTree,PyTree] = struct.field(
        pytree_node=False, default=Uninitialized
    )
    """Jitted function performing the matrix-vector product on PyTree inputs
    and outputs."""

    params: PyTree = Uninitialized
    """The first input to apply_fun (parameters of the ansatz).
    Stored only to access the pytree structure of inputs/outputs."""

    def __matmul__(self, vec):
        # if hasa ndim it's an array and not a pytree
        if hasattr(vec, "ndim"):
            if not vec.ndim == 1:
                raise ValueError("Unsupported mat-vec for batches of vectors")
            # If the input is a vector
            if not nkjax.tree_size(self.params) == vec.size:
                raise ValueError(
                    f"Size mismatch between number of parameters ({nkjax.tree_size(S.params)}) and vector size {vec.size}."
                )

            _, unravel = nkjax.tree_ravel(self.params)
            vec = unravel(vec)
            ravel_result = True
        else:
            ravel_result = False

        vec = tree_cast(vec, self.params)

        vec = self.mat_vec(vec)
        
        if ravel_result:
            res, _ = nkjax.tree_ravel(res)

        return res

    def _solve(self, solve_fun, y: PyTree, *, x0: Optional[PyTree], **kwargs) -> PyTree:
        y = tree_cast(y, self.params)
        return _solve(self.mat_vec, solve_fun, y, x0=x0, **kwargs)

    def to_dense(self) -> jnp.ndarray:
        return _to_dense(self.mat_vec)


########################################################################################
#####                                  QGT Logic                                   #####
########################################################################################

def solve(mat_vec: Callable, solve_fun: Callable, y: PyTree, *, x0: Optional[PyTree], **kwargs) -> PyTree:
    if x0 is None:
        x0 = jax.tree_map(jnp.zeros_like, y)

    out, info = solve_fun(self, y, x0=x0)
    return out, info

def _to_dense(mat_vec: Callable) -> jnp.ndarray:
    Npars = nkjax.tree_size(self.params)
    I = jax.numpy.eye(Npars)
    out = jax.vmap(mat_vec, in_axes=0)(I)

    if nkjax.is_complex(out):
        out = out.T

    return out
