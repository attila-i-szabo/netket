import netket as nk
from netket.optimizer import qgt
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
n_nodes = side*side
key_sample, key_rhs = jax.random.split(jax.random.PRNGKey(0), 2)

graph = nk.graph.Square(side)
hilbert = nk.hilbert.Spin(s=0.5, N=n_nodes)
symm_group = graph.automorphisms()

pure_jax = True # removes every remembrance of `symm_group` from the GCNN
if pure_jax:
    symm = nk.utils.HashableArray(jnp.asarray(symm_group))
    pt = nk.utils.HashableArray(jnp.asarray(symm_group.product_table.ravel()))
    machine = nk.models.GCNN(symmetries = symm, flattened_product_table=pt, layers=8, features=5, bias_init=normal(0.01))
else:
    machine = nk.models.GCNN(symmetries = symm_group, layers=8, features=5, bias_init=normal(0.01))

# Create a variational state to run QGT on
n_samples = 2000
sa = nk.sampler.MetropolisExchange(hilbert = hilbert, graph=graph, d_max=2)
vstate = nk.variational.MCState(sampler=sa, model=machine, n_samples=n_samples)
vstate.init(seed = 0)
# We don't actually want to perform a rather slow sampling
vstate._samples = hilbert.random_state(key = key_sample, size = n_samples)

qgt = qgt.QGTOnTheFly(vstate = vstate, diag_shift=0.01)

# Generate a random RHS of the same pytree shape as the parameters
vec, unravel = nk.jax.tree_ravel(vstate.parameters)
vec = jax.random.normal(key_rhs, shape = vec.shape, dtype = vec.dtype)
rhs = unravel(vec)

start = datetime.now()
lhs = qgt.solve(cg, rhs)
end = datetime.now()
print(f'took {(end-start).total_seconds()} seconds')
