"""vLLM-specific process discovery."""

from cuda_checkpoint.discover import find_process_by_name


def find_vllm_server() -> int:
    """Auto-discover a running vllm serve process.

    Searches for processes matching vllm's OpenAI API server module.
    Returns the PID of the oldest matching process.
    """
    return find_process_by_name("vllm.entrypoints.openai.api_server")
