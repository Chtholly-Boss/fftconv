import torch
import fftconv2d
from helpers import *

LENGTH = 1 << 10

def test_fft1d():
    """
    Test the 1D FFT of torch.fftn and cuFFT
    This test is mainly for learning how to use cuFFT
    """
    din = torch.randn(LENGTH,device='cuda')
    dout_cus = fftconv2d.rfft(din)
    dout_torch = torch.fft.rfftn(din)
    assert torch.allclose(dout_torch, dout_cus)

def test_ifft1d():
    """
    Test the 1D IFFT of torch.ifftn and cuFFT
    """
    din = torch.randn(LENGTH,device='cuda')
    # Do FFT
    dout = fftconv2d.rfft(din)
    # Do IFFT
    dout_inv = fftconv2d.irfft(dout)
    assert torch.allclose(din, dout_inv, atol=1e-5, rtol=1e-5)

# def test_perf_1dfft():
#     """
#     Test the performance of 1D FFT
#     """
#     din = torch.randn(LENGTH, device='cuda')

#     time_torch = perf_wrapper(fft_wrapper(torch.fft.rfftn, torch.fft.irfftn, din))
#     time_cus = perf_wrapper(fft_wrapper(fftconv2d.rfft, fftconv2d.irfft, din))
#     assert time_cus < time_torch, f"fftconv2d is slower than torch.fft by {time_torch/time_cus:.2f}x"