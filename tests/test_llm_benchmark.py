import pytest

from src.llm.benchmark import format_gpu_offload_status, gpu_offload_status, require_gpu_offload


def test_gpu_offload_status_reads_newer_llama_cpp_logs():
    log_text = """
load_backend: loaded Vulkan backend from C:\\Users\\pssh\\llama.cpp\\ggml-vulkan.dll
llama_prepare_model_devices: using device Vulkan0 (AMD Radeon(TM) 8060S Graphics)
load_tensors: offloaded 35/35 layers to GPU
"""

    status = gpu_offload_status(log_text)

    assert status["has_vulkan"] is True
    assert status["has_required_device"] is True
    assert status["offloaded_layers"] == 35
    assert status["total_layers"] == 35
    require_gpu_offload(log_text)
    assert format_gpu_offload_status(log_text) == "Vulkan0 (AMD Radeon(TM) 8060S Graphics), offloaded 35/35 layers"


def test_gpu_offload_status_counts_verbose_layer_assignments():
    log_text = """
ggml_vulkan: Found 1 Vulkan devices:
ggml_vulkan: 0 = AMD Radeon(TM) 8060S Graphics
load_tensors: layer   0 assigned to device Vulkan0, is_swa = 1
load_tensors: layer   1 assigned to device Vulkan0, is_swa = 1
"""

    status = gpu_offload_status(log_text)

    assert status["offloaded_layers"] == 2
    assert status["total_layers"] is None
    require_gpu_offload(log_text)


def test_require_gpu_offload_rejects_cpu_only_logs():
    log_text = """
load_backend: loaded CPU backend from C:\\Users\\pssh\\llama.cpp\\ggml-cpu-zen4.dll
load_tensors: offloaded 0/35 layers to GPU
"""

    with pytest.raises(RuntimeError, match="did not prove Vulkan layer offload"):
        require_gpu_offload(log_text)
