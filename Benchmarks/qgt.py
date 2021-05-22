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


class FFNN(nn.Module):
    n_layers: int
    width: int
    dtype: Any = np.float64
    activation: Any = jax.nn.selu
    kernel_init: NNInitFunc = variance_scaling(1.0, "fan_in", "normal")
    bias_init: NNInitFunc = normal(0.01)

    def setup(self):
        self.layers = [
            nk.nn.Dense(
                features=self.width,
                use_bias=True,
                dtype=self.dtype,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
            )
            for layer in range(self.n_layers)
        ]

    @nn.compact
    def __call__(self, x_in):
        x = x_in
        for layer in self.layers:
            x = layer(x)
            x = self.activation(x)
        return jnp.sum(x, axis=-1) / (x.shape[-1]) ** 0.5


# Benchmark starts here

n_nodes = 100
key_sample, key_rhs = jax.random.split(jax.random.PRNGKey(0), 2)

graph = nk.graph.Chain(n_nodes)
hilbert = nk.hilbert.Spin(s=0.5, N=n_nodes)
machine = FFNN(n_layers=8, width=500)

# Create a variational state to run QGT on
n_samples = 2000
sa = nk.sampler.MetropolisExchange(hilbert=hilbert, graph=graph, d_max=2)
vstate = nk.variational.MCState(sampler=sa, model=machine, n_samples=n_samples)
vstate.init(seed=0)
# We don't actually want to perform a rather slow sampling
vstate._samples = hilbert.random_state(key=key_sample, size=n_samples)

qgt = qgt.QGTOnTheFly(vstate=vstate, diag_shift=0.01)

# Generate a random RHS of the same pytree shape as the parameters
vec, unravel = nk.jax.tree_ravel(vstate.parameters)
vec = jax.random.normal(key_rhs, shape=vec.shape, dtype=vec.dtype)
rhs = unravel(vec)

start = datetime.now()
lhs = qgt.solve(cg, rhs)
end = datetime.now()
print(f"took {(end-start).total_seconds()} seconds")
