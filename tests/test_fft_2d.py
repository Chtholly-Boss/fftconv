import torch
import fftconv2d
from helpers import *

HEIGTH = 1 << 2
WIDTH = 1 << 2

def test_rfft2():
    din = torch.randn(HEIGTH, WIDTH, device='cuda')
    dout_torch = torch.fft.rfftn(din)
    dout_fftconv2d = fftconv2d.rfftn(din)

    assert torch.allclose(dout_torch, dout_fftconv2d)

def test_irfft2():
    din = torch.randn(4, 3, device='cuda', dtype=torch.complex64)
    print(din)
    dout_fftconv2d = fftconv2d.irfftn(din)
    print(din)
    
    dout_torch = torch.fft.irfftn(din)
    assert torch.allclose(dout_torch, dout_fftconv2d, rtol=1e-3, atol=1e-3)

def test_rfftn():
    din = torch.randn(1, 3, HEIGTH, WIDTH, device='cuda')

    dout_torch = torch.fft.rfftn(din, dim=(2,3))
    dout_fftconv2d = fftconv2d.rfftn(din)

    assert torch.allclose(dout_torch, dout_fftconv2d, rtol=1e-3, atol=1e-3)

if __name__ == '__main__':
    # din = torch.randn(1, 2, 4, 3, device='cuda', dtype=torch.complex64)
    # print(f'prev: {din}')
    # dout_fftconv2d = fftconv2d.irfftn(din)
    # print(f'after: {din}')
    test_irfft2()