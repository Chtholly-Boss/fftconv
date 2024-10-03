import torch
import cusfft
from helpers import *

HEIGTH = 1 << 10
WIDTH = 1 << 10

def test_rfft2():
    din = torch.randn(HEIGTH, WIDTH, device='cuda')

    dout_torch = torch.fft.rfftn(din)
    dout_cusfft = cusfft.rfft2(din)

    assert torch.allclose(dout_torch, dout_cusfft)

def test_irfft2():
    din = torch.randn(HEIGTH, WIDTH, device='cuda')

    dout = cusfft.rfft2(din)
    dout_inv = cusfft.irfft2(dout)
    
    assert torch.allclose(din, dout_inv, rtol=1e-3, atol=1e-3)

def test_rfftn():
    din = torch.randn(1, 3, HEIGTH, WIDTH, device='cuda')

    dout_torch = torch.fft.rfftn(din, dim=(2,3))
    dout_cusfft = cusfft.rfftn(din)

    assert torch.allclose(dout_torch, dout_cusfft, rtol=1e-3, atol=1e-3)