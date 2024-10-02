import torch
import cusfft
from helpers import *

HEIGTH = 1 << 10
WIDTH = 1 << 10

def test_rfftn():
    din = torch.randn(HEIGTH, WIDTH, device='cuda')

    dout_torch = torch.fft.rfftn(din)
    dout_cusfft = cusfft.rfftn(din)

    assert torch.allclose(dout_torch, dout_cusfft)

def test_irfftn():
    din = torch.randn(HEIGTH, WIDTH, device='cuda')

    dout = cusfft.rfftn(din)
    dout_inv = cusfft.irfftn(dout)
    
    assert torch.allclose(din, dout_inv, rtol=1e-3, atol=1e-3)

def test_perf_2dfft():
    din = torch.randn(HEIGTH, WIDTH, device='cuda')

    time_torch = perf_wrapper(fft_wrapper(torch.fft.rfftn, torch.fft.irfftn, din))
    time_cus = perf_wrapper(fft_wrapper(cusfft.rfft, cusfft.irfft, din))

    assert time_cus < time_torch, f"cusFFT is slower than torch.fft by {time_torch/time_cus:.2f}x"