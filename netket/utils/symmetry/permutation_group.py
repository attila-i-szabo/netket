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

# Ignore false-positives for redefined `product` functions:
# pylint: disable=function-redefined

from plum import dispatch
import numpy as np
from dataclasses import dataclass
from typing import Optional

from .semigroup import Identity, Element
from .group import Group

from netket.utils import HashableArray, struct
from netket.utils.types import Array, DType, Shape


class Permutation(Element):
    def __init__(self, permutation: Array, name: Optional[str] = None):
        """
        Creates a `Permutation` from an array of preimages of :code:`range(N)`

        Arguments:
            permutation: a 1D array listing :math:`g^{-1}(x)` for all :math:`0\le x < N` (i.e., `V[permutation]` permutes the elements of `V` as desired)

        Returns:
            a `Permutation` object encoding the same permutation
        """
        self.permutation = HashableArray(np.asarray(permutation))
        self.name = name

    def __call__(self, x):
        return x[..., self.permutation]

    def __hash__(self):
        return hash(self.permutation)

    def __eq__(self, other):
        if isinstance(other, Permutation):
            return self.permutation == other.permutation
        else:
            return False

    def __repr__(self):
        if self.name is not None:
            return self.name
        else:
            return f"Permutation({self.__array__().tolist()})"

    def __array__(self, dtype: DType = None):
        return np.asarray(self.permutation, dtype)


@dispatch
def product(p: Permutation, q: Permutation):
    name = None if p.name is None and q.name is None else f"{p} @ {q}"
    return Permutation(p(np.asarray(q)), name)


@struct.dataclass()
class PermutationGroup(Group):
    """
    Collection of permutation operations acting on sequences of length :code:`degree`.

    Group elements need not all be of type :ref:`netket.utils.symmetry.Permutation`,
    only act as such on a sequence when called. Currently, however, only `Identity`
    and `Permutation` have canonical forms implemented.

    The class can contain elements that are distinct as objects (e.g.,
    :code:`Identity()` and :code:`Translation((0,))`) but have identical action.
    Those can be removed by calling :code:`remove_duplicates`.
    """

    degree: int
    """Number of elements the permutations act on."""

    def __post_init__(self):
        super().__post_init__()
        myhash = hash((super().__hash__(), hash(self.degree)))
        object.__setattr__(self, "_PermutationGroup__hash", myhash)

    def __matmul__(self, other) -> "PermutationGroup":
        if not isinstance(other, PermutationGroup):
            raise ValueError(
                "Incompatible groups (`PermutationGroup` and something else)"
            )
        elif self.degree != other.degree:
            raise ValueError(
                "Incompatible groups (`PermutationGroup`s of different degree)"
            )

        return PermutationGroup(super().__matmul__(other).elems, self.degree)

    def _canonical(self, x: Element) -> Array:
        if isinstance(x, Identity):
            return np.arange(self.degree, dtype=int)
        elif isinstance(x, Permutation):
            return np.asarray(x.permutation)
        else:
            raise ValueError(
                "`PermutationGroup` only supports `Identity` and `Permutation` elements"
            )

    def to_array(self) -> Array:
        """
        Convert the abstract group operations to an array of permutation indices,
        such that the `i`-th row contains the indices corresponding to the `i`-th group
        element. That is, `self.to_array()[i, j]` is :math:`g_i^{-1}(j)`) and
        (for :code:`G = self`)::
            V = np.arange(G.degree)
            assert np.all(G(V) == V[..., G.to_array()])
        """
        return self._canonical_array()

    def __array__(self, dtype=None) -> Array:
        return np.asarray(self.to_array(), dtype=dtype)

    def remove_duplicates(self, *, return_inverse=False) -> "PermutationGroup":
        """
        Returns a new :code:`PermutationGroup` with duplicate elements (that is, elements which
        represent identical permutations) removed.

        Arguments:
            return_inverse: If True, also return indices to reconstruct the original
                group from the result.

        Returns:
            group: the permutation group with duplicate elements removed.
            return_inverse: Indices to reconstruct the original group from the result.
                Only returned if `return_inverse` is True.
        """
        if return_inverse:
            group, inverse = super().remove_duplicates(return_inverse=True)
        else:
            group = super().remove_duplicates(return_inverse=False)

        pgroup = PermutationGroup(group.elems, self.degree)

        if return_inverse:
            return pgroup, inverse
        else:
            return pgroup

    @struct.property_cached
    def inverse(self) -> Array:
        perm_array = self.to_array()
        n_symm = len(perm_array)
        inverse = np.zeros([n_symm], dtype=int)
        for i, perm1 in enumerate(perm_array):
            for j, perm2 in enumerate(perm_array):
                perm_sq = perm1[perm2]
                if np.all(perm_sq == np.arange(len(perm_sq))):
                    inverse[i] = j

        return inverse

    @struct.property_cached
    def product_table(self) -> Array:
        perms = self.to_array()
        inverse = perms[self.inverse].squeeze()
        n_symm = len(perms)
        product_table = np.zeros([n_symm, n_symm], dtype=int)

        inv_t = inverse.transpose()
        perms_t = perms.transpose()
        inv_elements = perms_t[inv_t].reshape(-1, n_symm * n_symm).transpose()

        perms = [HashableArray(element) for element in perms]
        inv_perms = [HashableArray(element) for element in inv_elements]

        inverse_index_mapping = {element: index for index, element in enumerate(perms)}

        inds = [
            (index, inverse_index_mapping[element])
            for index, element in enumerate(inv_perms)
            if element in inverse_index_mapping
        ]

        inds = np.asarray(inds)

        product_table[inds[:, 0] // n_symm, inds[:, 0] % n_symm] = inds[:, 1]

        return product_table

    @property
    def shape(self) -> Shape:
        """Tuple `(<# of group elements>, <degree>)`, same as :code:`self.to_array().shape`."""
        return (len(self), self.degree)

    def __hash__(self):
        # pylint: disable=no-member
        return self.__hash
