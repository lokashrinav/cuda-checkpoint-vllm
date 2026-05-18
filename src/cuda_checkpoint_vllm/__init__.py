"""vLLM-specific integration for cuda-checkpoint.

Builds on the generic cuda_checkpoint package with vLLM-aware orchestration:
  - VLLMCheckpointer: sleep/wake optimization, V0 NCCL reinit, V1 passthrough
  - CLI: vllm-ckpt command for sidecar deployment
  - Auto-discovery of vllm serve processes
"""

from cuda_checkpoint_vllm.orchestrator import VLLMCheckpointer
from cuda_checkpoint_vllm.discovery import find_vllm_server

__all__ = ["VLLMCheckpointer", "find_vllm_server"]
