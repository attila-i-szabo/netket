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

import os
import warnings
import types
from textwrap import dedent

from netket.utils.config_flags import config
from netket.utils import module_version

_mpi4py_loaded = False
_mpi4jax_loaded = False
mpi4jax_available = False

try:
    if (
        not config.netket_mpi or config.netket_experimental_sharding  # type: ignore
    ):  # pragma: no cover  # type: ignore
        # if mpi is disabled trigger import error
        # and follow the no-mpi code path
        raise ImportError

    from mpi4py import MPI

    _mpi4py_loaded = True
    mpi4py_available = True

    # We don't use the standard communicator because Jax and
    # np/python must use different ones to avoid desync issues
    # so we insulate also from the user-space.
    MPI_py_comm = MPI.COMM_WORLD.Create(MPI.COMM_WORLD.Get_group())
    MPI_jax_comm = MPI.COMM_WORLD.Create(MPI.COMM_WORLD.Get_group())

    n_nodes = MPI_py_comm.Get_size()
    node_number = MPI_py_comm.Get_rank()
    rank = MPI_py_comm.Get_rank()

    import mpi4jax

    _mpi4jax_loaded = True
    mpi4jax_available = True

    # If we got here, mpi4jax is loaded and we are using MPI.
    # Let's check if we have multiple GPUs available. If that is the case, as netket
    # does not support multiple GPUs per rank in MPI mode, we fix that only one gpu
    # is avaialble per rank, by using local ranks.
    if config.netket_mpi_autodetect_local_gpu:  # type: ignore
        from .gpu_autorank_util import autoset_default_gpu

        autoset_default_gpu(MPI_py_comm)  # type: ignore
        del autoset_default_gpu


except ImportError:
    mpi4py_available = False
    mpi4jax_available = False
    MPI_py_comm = None  # type: ignore
    MPI_jax_comm = None  # type: ignore
    n_nodes = 1
    node_number = 0
    rank = 0

    FakeMPI = types.ModuleType("FakeMPI", "FakeMPI Module")
    FakeMPI.COMM_WORLD = None

    MPI = FakeMPI

    # Try to detect if we are running under MPI and warn that mpi4py is not installed
    if config.netket_mpi_warning and not config.netket_experimental_sharding:  # type: ignore
        _MPI_ENV_VARIABLES = [
            "OMPI_COMM_WORLD_SIZE",
            "I_MPI_HYDRA_HOST_FILE",
            "MPI_LOCALRANKID",
            "MPI_LOCALNRANKS",
        ]
        for varname in _MPI_ENV_VARIABLES:
            if varname in os.environ:
                warnings.warn(
                    dedent(
                        f"""
                    MPI WARNING: It seems you might be running Python with MPI, but
                    dependencies required by NetKet to enable MPI support are missing
                    or cannot be loaded, so MPI support is disabled.

                    NetKet will not take advantage of MPI, and every MPI rank will
                    execute the same code independently.

                    MPI dependencies are:
                      - mpi4py>=3.0.1     ....... {"available" if _mpi4py_loaded else "missing"}
                      - mpi4jax>=0.5.11   ....... {"available" if _mpi4jax_loaded else "missing"}

                    To enable MPI support, install the missing dependencies.
                    To learn more about MPI and NetKet consult the documentation at
                    https://www.netket.org/docs/getting_started.html

                    To disable this warning, set the environment variable `NETKET_MPI_WARNING=0`
                    """
                    )
                )

if mpi4py_available:
    _MIN_MPI4JAX_VERSION = (0, 3, 1)

    if module_version(mpi4jax) < _MIN_MPI4JAX_VERSION:  # type: ignore
        raise ImportError(
            f"Netket is only compatible with mpi4jax >= {_MIN_MPI4JAX_VERSION} "
            f"(you have mpi4jax == {mpi4jax.__version__}). "  # type: ignore
            "Please update it to a more recent version by running "
            "(`pip install -U mpi4jax`)."
        )


del os, warnings, dedent, config, module_version
