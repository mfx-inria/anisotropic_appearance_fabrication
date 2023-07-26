"""
Provide a method to probe available backend. Source:
https://github.com/google/jax/issues/6459
"""

import os
import sys

import jax
from jax._src.lib import xla_client


def backends() -> list[str]:
    """Probe available backend.

    Returns
    -------
    list[str]
        A list of the available backend from the following list: `['cpu',
        'gpu', 'tpu']`
    """
    backends = []
    for backend in ['cpu', 'gpu', 'tpu']:
        try:
            jax.devices(backend)
        except RuntimeError:
            pass
        else:
            backends.append(backend)
    return backends


def get_cpu_and_gpu_devices() -> tuple[xla_client.Device, xla_client.Device]:
    """Return the cpu and gpu devices.

    The program exits if the CPU is not an available backend.

    Returns
    -------
    tuple[xla_client.Device, xla_client.Device]
        out1
            The CPU device.
        out2
            The GPU device, if available, and if the JAX_PLATFORM environment
            variable is not equal to 'cpu'. Otherwise, it is the CPU device.
    """
    JAX_PLATFORM = os.environ.get('JAX_PLATFORMS')
    backend_list = backends()
    cpu_in_backends = 'cpu' in backend_list
    if not cpu_in_backends:
        sys.exit("JAX is not able to find the cpu backend.")
    gpu_in_backends = 'gpu' in backend_list

    device_cpu = jax.devices('cpu')[0]
    device_gpu = device_cpu
    if not JAX_PLATFORM == 'cpu' and gpu_in_backends:
        device_gpu = jax.devices('gpu')[0]
    return device_cpu, device_gpu
