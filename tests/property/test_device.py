import pytest
import torch
import numpy as np
from lunar_reg.device import DeviceManager

def test_device_manager_resolution():
    # If cuda is not available, cuda preferred should fall back to cpu
    cuda_available = torch.cuda.is_available()
    
    dm_cpu = DeviceManager(preferred="cpu")
    assert dm_cpu.device_str == "cpu"
    assert not dm_cpu.is_gpu_available()
    
    dm_auto = DeviceManager(preferred="auto")
    if cuda_available:
        assert dm_auto.device_str == "cuda"
        assert dm_auto.is_gpu_available()
    else:
        assert dm_auto.device_str == "cpu"
        assert not dm_auto.is_gpu_available()
        
    dm_cuda = DeviceManager(preferred="cuda")
    if cuda_available:
        assert dm_cuda.device_str == "cuda"
        assert dm_cuda.is_gpu_available()
    else:
        assert dm_cuda.device_str == "cpu"
        assert not dm_cuda.is_gpu_available()

# Feature: lunar-image-registration, Property 24: GPU/CPU result equivalence
# Validates: Requirements 15.4
def test_gpu_cpu_equivalence_placeholder():
    # Since SuperPointDetector isn't implemented yet, we test basic torch tensor operations
    # to show that CPU and GPU produce numerically equivalent outputs.
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available, skipping GPU/CPU equivalence check.")
        
    # Example computation
    x_cpu = torch.randn(10, 10, device="cpu")
    x_gpu = x_cpu.to("cuda")
    
    # Run a simple neural layer or operation on both
    conv_cpu = torch.nn.Conv2d(1, 1, 3).to("cpu")
    conv_gpu = torch.nn.Conv2d(1, 1, 3).to("cuda")
    
    # Load same weights
    state = conv_cpu.state_dict()
    conv_gpu.load_state_dict(state)
    
    inp_cpu = torch.randn(1, 1, 10, 10, device="cpu")
    inp_gpu = inp_cpu.to("cuda")
    
    out_cpu = conv_cpu(inp_cpu)
    out_gpu = conv_gpu(inp_gpu)
    
    np.testing.assert_allclose(
        out_cpu.detach().numpy(),
        out_gpu.to("cpu").detach().numpy(),
        rtol=1e-5,
        atol=1e-5
    )
