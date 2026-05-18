"""Generic Python bindings for NVIDIA cuda-checkpoint (multi-GPU capable).

Works with any CUDA process — vLLM, TensorRT-LLM, SGLang, PyTorch training, etc.

Two layers:
  1. CudaCheckpointAPI — direct ctypes bindings to cuCheckpointProcess* 4-step API
  2. MultiGPUCheckpointer — parallel checkpoint/restore across multiple CUDA PIDs
"""

from cuda_checkpoint.api import CudaCheckpointAPI
from cuda_checkpoint.multi_gpu import MultiGPUCheckpointer
from cuda_checkpoint.discover import discover_cuda_pids, find_cuda_pids_for_process

__all__ = [
    "CudaCheckpointAPI",
    "MultiGPUCheckpointer",
    "discover_cuda_pids",
    "find_cuda_pids_for_process",
]
