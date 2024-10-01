import pytest
import torch
import cusfft

def test_fft1d():
    """
    Test the 1D FFT of torch.fftn and cuFFT
    This test is mainly for learning how to use cuFFT
    """
    din = torch.randn(16)
    din_cuda = din.cuda()
    dout_cus = cusfft.fft(din_cuda)
    dout_torch = torch.fft.rfftn(din_cuda)
    print(dout_cus)
    print(dout_torch)
    assert torch.allclose(dout_torch, dout_cus)

def test_rfftn():
    """
    Test the rfftn function with a random input tensor
    Compare the parameters of torch.fft.rfftn and your rfftn:
    - Tensor size
    - Precision
    """
    # FFTs to be used
    torch_rfftn = torch.fft.rfftn
    # TODO: import your fft
    cus = __import__('torch')
    cus_rfftn = cus.fft.rfftn 
    # Create a random input tensor
    din = torch.randn(1, 1, 8, 8)
    # Compute the output using the rfftn function
    dout_torch = torch_rfftn(din, dim = tuple(range(2, din.ndim)))
    dout_cus = cus_rfftn(din, dim = tuple(range(2, din.ndim)))
    # Check the parameters of the output tensors
    assert dout_torch.size() == dout_cus.size()
    assert dout_torch.dtype == dout_cus.dtype
    assert torch.allclose(dout_torch, dout_cus)
