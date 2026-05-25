"""Scribe — local interview transcription with speaker identification."""

from .engine import (
    gpu_backend,
    gpu_runtime_version,
    gpu_vendor,
    has_gpu,
    is_cuda,
    is_mps,
    is_rocm,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "gpu_backend",
    "gpu_runtime_version",
    "gpu_vendor",
    "has_gpu",
    "is_cuda",
    "is_mps",
    "is_rocm",
]
