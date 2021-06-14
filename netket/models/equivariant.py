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

from typing import Union, Optional, Tuple, Any, Callable, Iterable

import numpy as np

import jax
from jax import numpy as jnp
from flax import linen as nn

from netket.utils import HashableArray
from netket.utils.types import PRNGKeyT, Shape, DType, Array, NNInitFunc
from netket.utils.group import PermutationGroup


from netket import nn as nknn
from netket.nn.initializers import lecun_complex, zeros, variance_scaling


class GCNN(nn.Module):
    """Implements a Group Convolutional Neural Network (G-CNN) that outputs a wavefunction
    that is invariant over a specified symmetry group.

    The G-CNN is described in ` Cohen et. {\it al} <http://proceedings.mlr.press/v48/cohenc16.pdf>`_
    and applied to quantum many-body problems in ` Roth et. {\it al} <https://arxiv.org/pdf/2104.05085.pdf>`_.

    The G-CNN alternates convolution operations with pointwise non-linearities. The first
    layer is symmetrized linear transform given by DenseSymm, while the other layers are
    G-convolutions given by DenseEquivariant. The hidden layers of the G-CNN are related by
    the following equation:

    .. math ::

        {\bf f}^{i+1}_h = \Gamma( \sum_h W_{g^{-1} h} {\bf f}^i_h).

    """

    symmetries: Union[HashableArray, PermutationGroup]
    """A group of symmetry operations (or array of permutation indices) over which the layer should be invariant.

    Numpy/Jax arrays must be wrapped into an :class:`netket.utils.HashableArray`. 
    """
    layers: int
    """Number of layers (not including sum layer over output)."""
    features: Union[Tuple, int]
    """Number of features in each layer starting from the input. If a single number is given,
    all layers will have the same number of features."""
    flattened_product_table: Optional[HashableArray] = None
    """Flattened product table generated by PermutationGroup.product_table.ravel()
    that specifies the product of the group with its inverse"""
    dtype: Any = np.float64
    """The dtype of the weights."""
    activation: Any = jax.nn.selu
    """The nonlinear activation function between hidden layers."""
    output_activation: Any = None
    """The nonlinear activation before the output."""
    use_bias: bool = True
    """if True uses a bias in all layers."""
    precision: Any = None
    """numerical precision of the computation see `jax.lax.Precision`for details."""
    kernel_init: NNInitFunc = variance_scaling(1.0, "fan_in", "normal")
    """Initializer for the Dense layer matrix."""
    bias_init: NNInitFunc = zeros
    """Initializer for the hidden bias."""

    def setup(self):

        self.n_symm = np.asarray(self.symmetries).shape[0]

        if self.flattened_product_table is None and not isinstance(
            self.symmetries, PermutationGroup
        ):
            raise AttributeError(
                "product table must be specified if symmetries are given as an array"
            )

        if self.flattened_product_table is None:
            flat_pt = HashableArray(self.symmetries.product_table.ravel())
        else:
            flat_pt = self.flattened_product_table

        if not np.asarray(flat_pt).shape[0] == np.square(self.n_symm):
            raise ValueError("Flattened product table must have shape [n_symm*n_symm]")

        if isinstance(self.features, int):
            feature_dim = [self.features for layer in range(self.layers)]
        else:
            if not len(self.features) == self.layers:
                raise ValueError(
                    """Length of vector specifying feature dimensions must be the same as the number of layers"""
                )
            feature_dim = self.features

        self.dense_symm = nknn.DenseSymm(
            symmetries=self.symmetries,
            features=feature_dim[0],
            dtype=self.dtype,
            use_bias=self.use_bias,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            precision=self.precision,
        )

        self.equivariant_layers = [
            nknn.DenseEquivariant(
                symmetry_info=flat_pt,
                in_features=feature_dim[layer],
                out_features=feature_dim[layer + 1],
                use_bias=self.use_bias,
                dtype=self.dtype,
                precision=self.precision,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
            )
            for layer in range(self.layers - 1)
        ]

    @nn.compact
    def __call__(self, x_in):
        x = self.dense_symm(x_in)
        for layer in range(self.layers - 1):
            x = self.activation(x)
            x = self.equivariant_layers[layer](x)

        if not self.output_activation == None:
            x = self.output_activation(x)
        # variance scaling for output layer
        x = jnp.sum(x, axis=-1) / np.sqrt(x.shape[-1])

        return x
