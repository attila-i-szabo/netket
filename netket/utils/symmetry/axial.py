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

from .semigroup import Identity
from .point_group import PGSymmetry, PointGroup
from netket.utils.types import Array
import numpy as np


def _rotation(angle: float, axis: Array) -> PGSymmetry:
    """Returns a rotation by `angle` degrees around `axis`."""
    angle = np.radians(angle)
    axis = np.asarray(axis) / np.linalg.norm(axis)
    return PGSymmetry(
        np.cos(angle) * np.eye(3)
        + np.sin(angle) * np.cross(np.eye(3), axis)
        + (1 - np.cos(angle)) * np.outer(axis, axis)
    )


def C(n: int, axis: Array = np.array([0, 0, 1])) -> PointGroup:
    """
    Returns the :math:`C_n` `PointGroup` of rotations around a given axis.

    Arguments:
        n: the index of the rotation group (the smallest rotation angle is 360°/n)
        axis: the axis of rotations, need not be normalised, defaults to the z-axis

    Returns:
        a `PointGroup` implementing :math:`C_n`
    """
    return PointGroup(
        [Identity()] + [_rotation(360 / n * i, axis) for i in range(1, n)], dim=3
    )


rotations = C

_inversion = PGSymmetry(-np.eye(3))

inversions = PointGroup([Identity(), _inversion], dim=3)
"""
:math:`\mathbb{Z}_2` `PointGroup` containing the identity and inversion across the origin.
"""


def _reflection(axis: Array) -> PGSymmetry:
    """Returns a reflection across a plane whose normal is `axis`"""
    axis = np.asarray(axis) / np.linalg.norm(axis)
    return PGSymmetry(np.eye(3) - 2 * np.outer(axis, axis))


def reflections(axis: Array) -> PointGroup:
    """
    Returns the :math:`\mathbb{Z}_2` `PointGroup` containing the identity and a
    reflection across a plane with normal `axis`
    """
    return PointGroup([Identity(), _reflection(axis)], dim=3)


def Ch(n: int, axis: Array = np.array([0, 0, 1])) -> PointGroup:
    """
    Returns the reflection group :math:`C_{nh}` generated by an *n*-fold rotation axis
    and a reflection across the plane normal to the same axis.

    Arguments:
        n: index of the group
        axis: the axis of rotations and normal to the mirror plane, need not be normalised, defaults to the z-axis

    Returns:
        a `PointGroup` object implementing :math:`C_{nh}`
    """
    return reflections(axis) @ C(n, axis)


reflection_group = Ch


def Cv(
    n: int, axis: Array = np.array([0, 0, 1]), axis2=np.array([1, 0, 0])
) -> PointGroup:
    """
    Returns the pyramidal group :math:`C_{nv}` generated by an *n*-fold rotation axis
    and a reflection across a plane that contains the same axis.

    Arguments:
        n: index of the group
        axis: the axis of rotations, need not be normalised, defaults to the z-axis
        axis2: normal of the generating mirror plane, need not be normalised, must be perpendicular to `axis`, defaults to the x-axis

    Returns:
        a `PointGroup` object implementing :math:`C_{nv}`
    """
    assert np.isclose(np.dot(axis, axis2), 0.0)
    return reflections(axis2) @ C(n, axis)


pyramidal = Cv


def _rotoreflection(angle: float, axis: Array) -> PGSymmetry:
    """Returns a rotoreflection by `angle` degrees around `axis`."""
    angle = np.radians(angle)
    axis = axis / np.linalg.norm(axis)
    rot_matrix = (
        np.cos(angle) * np.eye(3)
        + np.sin(angle) * np.cross(np.eye(3), axis)
        + (1 - np.cos(angle)) * np.outer(axis, axis)
    )
    refl_matrix = np.eye(3) - 2 * np.outer(axis, axis)
    return PGSymmetry(refl_matrix @ rot_matrix)


def S(n: int, axis: Array = np.array([0, 0, 1])) -> PointGroup:
    """
    Returns the :math:`S_n` `PointGroup` of rotoreflections around a given axis.

    Arguments:
        n: the index of the rotoreflection group (the smallest rotation angle is 360°/n)
        axis: the axis, need not be normalised, defaults to the z-axis

    Returns:
        a `PointGroup` implementing :math:`S_n`
    """
    if n % 2 == 1:
        return Ch(n, axis)
    else:
        return PointGroup(
            [Identity()]
            + [
                (
                    _rotoreflection(360 / n * i, axis)
                    if i % 2 == 1
                    else _rotation(360 / n * i, axis)
                )
                for i in range(1, n)
            ],
            dim=3,
        )


rotoreflections = S


def D(
    n: int, axis: Array = np.array([0, 0, 1]), axis2=np.array([1, 0, 0])
) -> PointGroup:
    """
    Returns the dihedral group :math:`D_n` generated by an *n*-fold rotation axis
    and a twofold rotation axis perpendicular to the former.

    Arguments:
        n: index of the group
        axis: the n-fold rotation axis, need not be normalised, defaults to the z-axis
        axis2: generating twofold rotation axis, need not be normalised, must be perpendicular to `axis`, defaults to the x-axis

    Returns:
        a `PointGroup` object implementing :math:`D_n`
    """
    assert np.isclose(np.dot(axis, axis2), 0.0)
    return C(2, axis2) @ C(n, axis)


dihedral = D

cuboid_rotations = D(2)
"""Rotational symmetries of a cuboid with edges aligned with the Cartesian axes."""


def Dh(
    n: int, axis: Array = np.array([0, 0, 1]), axis2=np.array([1, 0, 0])
) -> PointGroup:
    """
    Returns the prismatic group :math:`D_{nh}` generated by an *n*-fold rotation axis,
    a twofold rotation axis perpendicular to the former, and a mirror plane normal
    to the *n*-fold axis.

    Arguments:
        n: index of the group
        axis: the n-fold rotation axis and normal to the mirror plane, need not be normalised, defaults to the z-axis
        axis2: generating twofold rotation axis, need not be normalised, must be perpendicular to `axis`, defaults to the x-axis

    Returns:
        a `PointGroup` object implementing :math:`D_{nh}`
    """
    return reflections(axis) @ D(n, axis, axis2)


prismatic = Dh

cuboid = Dh(2)
"""Symmetry group of a cuboid with edges aligned with the Cartesian axes."""


def Dd(
    n: int, axis: Array = np.array([0, 0, 1]), axis2=np.array([1, 0, 0])
) -> PointGroup:
    """
    Returns the antiprismatic group :math:`D_{nd}` generated by a 2*n*-fold rotoreflection
    axis and a reflection across a plane that contains the same axis.

    Arguments:
        n: index of the group
        axis: the rotoreflection axis, need not be normalised, defaults to the z-axis
        axis2: normal of the generating mirror plane, need not be normalised, must be perpendicular to `axis`, defaults to the x-axis

    Returns:
        a `PointGroup` object implementing :math:`D_{nd}`
    """
    assert np.isclose(np.dot(axis, axis2), 0.0)
    return reflections(axis2) @ S(2 * n, axis)


antiprismatic = Dd
