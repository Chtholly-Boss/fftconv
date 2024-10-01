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
    dout_cus = cusfft.rfft(din_cuda)
    dout_torch = torch.fft.rfftn(din_cuda)
    print(dout_cus)
    print(dout_torch)
    assert torch.allclose(dout_torch, dout_cus)
