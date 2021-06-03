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

from netket.utils.types import Array

from .point_group import PGSymmetry, PointGroup
from .semigroup import Identity


def rotation(angle: float) -> PGSymmetry:
    """
    Returns a 2D rotation by `angle` degrees.

    Args:
       angle: the angle of rotation in degrees.
    """
    angle = np.radians(angle)
    return PGSymmetry(
        np.asarray([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    )


def C(n: int) -> PointGroup:
    """
    Returns the Cₙ `PointGroup` of 2D rotational symmetries.

    Arguments:
        n: the index of the rotation group (the smallest rotation angle is 360°/n)

    Returns:
        a `PointGroup` implementing Cₙ
    """
    return PointGroup(
        [Identity()] + [rotation(360 / n * i) for i in range(1, n)], ndim=2
    )


rotation_group = C


def reflection(angle: float) -> PGSymmetry:
    """
    Returns a 2D reflection across an axis at angle `angle` to the +x direction.

    Args:
        angle: the angle between the +x axis and the reflection axis.
    """
    axis = np.radians(angle) * 2  # the mirror matrix is written in terms of 2φ
    return PGSymmetry(
        np.asarray([[np.cos(axis), np.sin(axis)], [np.sin(axis), -np.cos(axis)]])
    )


def reflection_group(angle: float) -> PointGroup:
    """
    Returns the Z₂ `PointGroup` containing the identity and a reflection across an
    axis at angle `angle` to the +x.

    Args:
        angle: the angle between the +x axis and the reflection axis.
    """
    return PointGroup([Identity(), reflection(angle)], ndim=2)


def glide(trans: Array, origin: Array = (0, 0)) -> PGSymmetry:
    """
    Returns a 2D glide composed of translation by `trans` and reflection across
    its direction.

    Arguments:
        trans: translation vector
        origin: a point on the glide axis, defaults to the origin
    """
    axis = np.arctan2(trans[1], trans[0]) * 2
    W = np.asarray([[np.cos(axis), np.sin(axis)], [np.sin(axis), -np.cos(axis)]])
    w = np.asarray(trans) + (np.eye(2) - W) @ np.asarray(origin)
    return PGSymmetry(W, w)


def glide_group(trans: Array, origin: Array = (0, 0)) -> PGSymmetry:
    """
    Returns the Z_2 `PointGroup`containing the identity and a  2D glide composed
    of translation by `trans` and reflection across its direction.

    Arguments:
        trans: translation vector
        origin: a point on the glide axis, defaults to the origin
    The output is only a valid `PointGroup` after supplying a `unit_cell`
    consistent with the glide axis; otherwise, operations like `product_table`
    will fail.
    """
    return PointGroup([Identity(), glide(trans, origin)], ndim=2)


def D(n: int, axis: float = 0) -> PointGroup:
    """
    Returns the 2D dihedral `PointGroup` :math:`D_n` generated by a 360°/n rotation
    and a reflection.

    Arguments:
        n: index of the dihedral group
        axis: optional, the angle of one reflection axis with the +x direction
            (default: 0)

    Returns:
        a `PointGroup` object implementing :math:`D_n`
    """
    return reflection_group(axis) @ C(n)


dihedral = D


def rectangle() -> PointGroup:
    """
    The symmetry group of a rectangle aligned with the Cartesian axes (Vierergruppe).
    """
    return D(2)


def square() -> PointGroup:
    """The symmetry group of a square aligned with the Cartesian axes."""
    return D(4)
