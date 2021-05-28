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
    rotation as _rotation,
    reflection_group as _refl_group,
    inversion_group as _inv_group,
)

from netket.utils.types import Array
from typing import Tuple


def T() -> PointGroup:
    """Rotational symmetries of a tetrahedron with vertices (1,1,1), (1,-1,-1), (-1,1,-1), (-1,-1,1)."""
    return PointGroup(
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


tetrahedral_rotations = T


def Td() -> PointGroup:
    """Symmetry group of a tetrahedron with vertices (1,1,1), (1,-1,-1), (-1,1,-1), (-1,-1,1)."""
    return _refl_group([1, 1, 0]) @ T()


tetrahedral = Td


def Th() -> PointGroup:
    """Pyritohedral symmetry group generated by T and inversion."""
    return _inv_group() @ T()


pyritohedral = Th


def O() -> PointGroup:
    """Rotational symmetries of a cube/octahedron aligned with the Cartesian axes."""
    # NB the first factor isn't an actual point group but this is fine
    # we only use it to generate a coset of T in O
    return PointGroup([Identity(), _rotation(90, [0, 0, 1])], ndim=3) @ T()


octahedral_rotations = cubic_rotations = O


def Oh() -> PointGroup:
    """Symmetry group of a cube/octahedron aligned with the Cartesian axes."""
    return _inv_group() @ O()


octahedral = cubic = Oh
