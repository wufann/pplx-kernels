# pyright: reportCallIssue=false

import logging
import os

import torch

try:
    _lib_path = os.path.join(os.path.dirname(__file__), "libpplx_kernels.so")
    torch.ops.load_library(_lib_path)
    _ops = torch.ops.pplx_kernels
except OSError:
    from types import SimpleNamespace

    _ops = SimpleNamespace()
    logging.exception("Error loading pplx-kernels")
