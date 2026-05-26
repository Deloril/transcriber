"""Scribe — local interview transcription with speaker identification."""

from .engine import (
    apply_rocm_runtime_workarounds,
    gpu_backend,
    gpu_runtime_version,
    gpu_vendor,
    has_gpu,
    is_cuda,
    is_mps,
    is_rdna2,
    is_rocm,
    needs_hsa_override,
    patch_pyannote_lstm_dropout,
    recommended_hsa_override_value,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "apply_rocm_runtime_workarounds",
    "gpu_backend",
    "gpu_runtime_version",
    "gpu_vendor",
    "has_gpu",
    "is_cuda",
    "is_mps",
    "is_rdna2",
    "is_rocm",
    "needs_hsa_override",
    "patch_pyannote_lstm_dropout",
    "recommended_hsa_override_value",
]
