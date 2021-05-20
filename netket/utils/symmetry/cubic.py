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

import numpy as np
from itertools import permutations

from .semigroup import Identity
from .point_group import PGSymmetry, PointGroup
from .axial import (
    cuboid,
    cuboid_rotations,
    _rotation,
    reflections as _refl_group,
    inversions as _inv_group,
)

from netket.utils.types import Array
from typing import Tuple

T = PointGroup(
    [
        Identity(),
        _rotation(120, [1, 1, 1]),
        _rotation(120, [1, -1, -1]),
        _rotation(120, [-1, 1, -1]),
        _rotation(120, [-1, -1, 1]),
        _rotation(-120, [1, 1, 1]),
        _rotation(-120, [1, -1, -1]),
        _rotation(-120, [-1, 1, -1]),
        _rotation(-120, [-1, -1, 1]),
        _rotation(180, [0, 0, 1]),
        _rotation(180, [0, 1, 0]),
        _rotation(180, [1, 0, 0]),
    ],
    ndim=3,
)
"""Rotational symmetries of a tetrahedron with vertices (1,1,1), (1,-1,-1), (-1,1,-1), (-1,-1,1)."""

tetrahedral_rotations = T

Td = _refl_group([1, 1, 0]) @ T
"""Symmetry group of a tetrahedron with vertices (1,1,1), (1,-1,-1), (-1,1,-1), (-1,-1,1)."""

tetrahedral = Td

Th = _inv_group @ T
"""Pyritohedral symmetry group generated by T and inversion."""

pyritohedral = Th

# NB the first factor isn't an actual point group but this is fine
# we only use it to generate a coset of T in O
O = PointGroup([Identity(), _rotation(90, [0, 0, 1])], ndim=3) @ T
"""Rotational symmetries of a cube/octahedron aligned with the Cartesian axes."""

octahedral_rotations = cubic_rotations = O

Oh = _inv_group @ O
"""Symmetry group of a cube/octahedron aligned with the Cartesian axes."""

octahedral = cubic = Oh


def _perm_symm(perm: Tuple) -> PGSymmetry:
    n = len(perm)
    M = np.zeros((n, n))
    M[range(n), perm] = 1
    return PGSymmetry(M)


def _axis_reflection(axis: int, ndim: int) -> PGSymmetry:
    M = np.eye(ndim)
    M[axis, axis] = -1
    return PGSymmetry(M)


def hypercubic(ndim: int) -> PointGroup:
    """
    Returns the symmetry group of an `ndim` dimensional hypercube as a `PointGroup`.
    Isomorphic to, but listed in a different order from,
    * `planar.square` if `ndim==2`
    * `cubic.cubic` if `ndim==3`
    """
    result = PointGroup([_perm_symm(i) for i in permutations(range(ndim))], ndim=ndim)
    for i in range(ndim):
        result = result @ PointGroup([Identity(), _axis_reflection(i, ndim)], ndim=ndim)
    return result
