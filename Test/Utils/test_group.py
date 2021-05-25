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

import pytest

import netket as nk
import numpy as np

from netket.utils import group

from itertools import product

from .. import common

pytestmark = common.skipif_mpi

# Tests for group.py and overrides in subclasses

planar_families = [group.planar.C, group.planar.D]
planars = [fn(n) for fn in planar_families for n in range(1, 9)]
uniaxial_families = [group.axial.C, group.axial.Ch, group.axial.S]
uniaxials = [
    fn(n, axis=np.random.standard_normal(3))
    for fn in uniaxial_families
    for n in range(1, 9)
]
impropers = [
    group.axial.inversions,
    group.axial.reflections(axis=np.random.standard_normal(3)),
]
biaxial_families = [group.axial.Cv, group.axial.D, group.axial.Dh, group.axial.Dd]
axes1 = np.random.standard_normal((32, 3))
axes2 = np.cross(axes1, np.random.standard_normal((32, 3)))
biaxials = [
    fn(n, axis=axes1[i], axis2=axes2[i])
    for i, (fn, n) in enumerate(product(biaxial_families, range(1, 9)))
]
cubics = [group.cubic.T, group.cubic.Td, group.cubic.Th, group.cubic.O, group.cubic.Oh]
perms = [
    nk.graph.Hypercube(2, n_dim=3).automorphisms(),
    nk.graph.Square(4).automorphisms(),
]
groups = planars + uniaxials + biaxials + impropers + cubics + perms


def equal(a, b):
    return np.all(a == b)


@pytest.mark.parametrize("grp", groups)
def test_inverse(grp):
    inv = grp.inverse
    for i, j in enumerate(inv):
        assert equal(grp._canonical(grp[i] @ grp[j]), grp._canonical(group.Identity()))


@pytest.mark.parametrize("grp", groups)
def test_product_table(grp):
    pt = grp.product_table
    # u = g^-1 h  ->  gu = h
    for i in range(len(grp)):
        for j in range(len(grp)):
            assert equal(grp._canonical(grp[i] @ grp[pt[i, j]]), grp._canonical(grp[j]))


@pytest.mark.parametrize("grp", groups)
def test_conjugacy_table(grp):
    ct = grp.conjugacy_table
    inv = grp.inverse
    for i in range(len(grp)):
        for j, jinv in enumerate(inv):
            assert equal(
                grp._canonical(grp[jinv] @ grp[i] @ grp[j]),
                grp._canonical(grp[ct[i, j]]),
            )


# Conjugacy class sizes and irrep dimensions taken from
# https://en.wikipedia.org/wiki/List_of_character_tables_for_chemically_important_3D_point_groups
details = [
    (group.planar.C(3), [1, 1, 1], [1, 1, 1]),
    (group.planar.C(4), [1, 1, 1, 1], [1, 1, 1, 1]),
    (group.planar.D(3), [1, 2, 3], [1, 1, 2]),
    (group.planar.D(4), [1, 2, 1, 2, 2], [1, 1, 1, 1, 2]),
    (group.planar.D(6), [1, 2, 2, 1, 3, 3], [1, 1, 1, 1, 2, 2]),
    (group.axial.C(4), [1, 1, 1, 1], [1, 1, 1, 1]),
    (group.axial.Ch(4), [1, 1, 1, 1] * 2, [1, 1, 1, 1] * 2),
    (group.axial.Cv(4), [1, 2, 1, 2, 2], [1, 1, 1, 1, 2]),
    (group.axial.S(4), [1, 1, 1, 1], [1, 1, 1, 1]),
    (group.axial.D(4), [1, 2, 1, 2, 2], [1, 1, 1, 1, 2]),
    (group.axial.Dh(4), [1, 2, 1, 2, 2] * 2, [1, 1, 1, 1, 2] * 2),
    (group.axial.Dd(4), [1, 2, 2, 2, 1, 4, 4], [1, 1, 1, 1, 2, 2, 2]),
    (group.cubic.T, [1, 4, 4, 3], [1, 1, 1, 3]),
    (group.cubic.Td, [1, 8, 3, 6, 6], [1, 1, 2, 3, 3]),
    (group.cubic.Th, [1, 4, 4, 3] * 2, [1, 1, 1, 3] * 2),
    (group.cubic.O, [1, 6, 3, 8, 6], [1, 1, 2, 3, 3]),
    (group.cubic.Oh, [1, 6, 3, 8, 6] * 2, [1, 1, 2, 3, 3] * 2),
]


@pytest.mark.parametrize("grp,cls,dims", details)
def test_conjugacy_class(grp, cls, dims):
    classes, _, _ = grp.conjugacy_classes
    class_sizes = classes.sum(axis=1)

    assert equal(np.sort(class_sizes), np.sort(cls))


@pytest.mark.parametrize("grp,cls,dims", details)
def test_character_table(grp, cls, dims):
    classes, _, _ = grp.conjugacy_classes
    class_sizes = classes.sum(axis=1)
    cht = grp.character_table_by_class

    # check that dimensions match and are sorted
    assert np.allclose(cht[:, 0], np.sort(dims))

    # check orthogonality of characters
    assert np.allclose(
        cht @ np.diag(class_sizes) @ cht.T.conj(), np.eye(len(class_sizes)) * len(grp)
    )

    # check orthogonality of columns of the character table
    column_prod = cht.T.conj() @ cht
    assert np.allclose(column_prod, np.diag(np.diag(column_prod)))


# Test for naming and generating 2D and 3D PGSymmetries

names = [
    (
        group.planar._planar_rotation(47),
        np.asarray([[0.6819983601, -0.7313537016], [0.7313537016, 0.6819983601]]),
        "Rot(47°)",
    ),
    (
        group.planar._reflection(78),
        np.asarray([[-0.9135454576, 0.4067366431], [0.4067366431, 0.9135454576]]),
        "Refl(78°)",
    ),
    (
        group.axial._rotation(34, [1, 1, 2]),
        np.asarray(
            [
                [0.8575313105, -0.4280853559, 0.2852770227],
                [0.4850728317, 0.8575313105, -0.1713020711],
                [-0.1713020711, 0.2852770227, 0.9430125242],
            ]
        ),
        "Rot(34°)[1,1,2]",
    ),
    (
        group.axial._reflection([1, 4, 2]),
        np.asarray([[19, -8, -4], [-8, -11, -16], [-4, -16, 13]]) / 21,
        "Refl[1,4,2]",
    ),
    (
        group.axial._rotoreflection(8, [2, 3, 1]),
        np.asarray(
            [
                [0.4216200491, -0.8901676053, -0.1727372824],
                [-0.8157764537, -0.2891899754, -0.5008771663],
                [-0.3959107372, -0.3520948631, 0.8481060638],
            ]
        ),
        "RotoRefl(8°)[2,3,1]",
    ),
    (group.axial._inversion, -np.eye(3), "Inv()"),
]


@pytest.mark.parametrize("symm,W,name", names)
def test_naming(symm, W, name):
    assert np.allclose(symm.matrix, W)
    assert str(symm) == name