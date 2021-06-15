import netket as nk
from netket.optimizer import qgt
from flax.core import unfreeze
from flax import linen as nn
from flax.linen.initializers import normal, variance_scaling
import jax
from jax import numpy as jnp
from jax.scipy.sparse.linalg import cg
import numpy as np
from typing import Any
from netket.utils.types import NNInitFunc
from datetime import datetime

# Benchmark starts here

side = 10
n_nodes = side * side
keys = jax.random.split(jax.random.PRNGKey(0), 5)

graph = nk.graph.Square(side)
hilbert = nk.hilbert.Spin(s=0.5, N=n_nodes)
symm_group = graph.automorphisms()

pure_jax = True  # removes every remembrance of `symm_group` from the GCNN
if pure_jax:
    symm = nk.utils.HashableArray(np.asarray(symm_group))
    pt = nk.utils.HashableArray(np.asarray(symm_group.product_table.ravel()))
    machine = nk.models.GCNN(
        symmetries=symm,
        flattened_product_table=pt,
        layers=8,
        features=10,
        bias_init=normal(0.01),
    )
else:
    machine = nk.models.GCNN(
        symmetries=symm_group, layers=8, features=10, bias_init=normal(0.01)
    )

# Create a variational state to run QGT on
n_samples = 200
sa = nk.sampler.MetropolisExchange(hilbert=hilbert, graph=graph, d_max=2)
vstate = nk.variational.MCState(sampler=sa, model=machine, n_samples=n_samples)
vstate.init(seed=0)
# We don't actually want to perform a rather slow sampling
vstate._samples = hilbert.random_state(key=keys[0], size=n_samples)

qgt_ = qgt.QGTOnTheFly(vstate=vstate, diag_shift=0.01)

# Generate a random RHS of the same pytree shape as the parameters
vec, unravel = nk.jax.tree_ravel(vstate.parameters)
vec1 = jax.random.normal(keys[1], shape=vec.shape, dtype=vec.dtype)
rhs1 = unravel(vec1)
vec2 = jax.random.normal(keys[2], shape=vec.shape, dtype=vec.dtype)
rhs2 = unravel(vec2)
vecp = vec * jax.random.normal(keys[3], shape=vec.shape, dtype=vec.dtype)
pars = unravel(vecp)

start = datetime.now()
lhs = jax.tree_map(lambda x: x.block_until_ready(), qgt_.solve(cg, rhs1))
end = datetime.now()
print(f"First time took {(end-start).total_seconds()} seconds")

# See what jit hath wrought us
vstate._samples = hilbert.random_state(key=keys[4], size=n_samples)
vstate._parameters = pars
qgt_ = qgt.QGTOnTheFly(vstate=vstate, diag_shift=0.01)

start = datetime.now()
lhs = jax.tree_map(lambda x: x.block_until_ready(), qgt_.solve(cg, rhs2))
end = datetime.now()
print(f"Second time took {(end-start).total_seconds()} seconds")
