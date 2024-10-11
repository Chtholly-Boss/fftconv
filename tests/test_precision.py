import torch
from conv import MyConv
from torch import tensor, randn
import fftconv2d

from functools import partial

from helper import *

def benchmark(din: tensor, base: callable, other: callable):
    # check din on cuda, if not, convert to cuda
    dout_base = base(din)
    dout_other = other(din)
    assert torch.allclose(dout_base, dout_other, atol=1e-3, rtol=1e-3)

def test_precision_ffts():
    rand_cuda = partial(randn, device='cuda')
    benchmark(rand_cuda(LENGTH), torch.fft.rfft, fftconv2d.fft.rfft)
    benchmark(rand_cuda(LENGTH, dtype=torch.complex64), torch.fft.irfft, fftconv2d.fft.irfft)
    benchmark(rand_cuda(HEIGHT, WIDTH), torch.fft.rfft2, fftconv2d.fft.rfft2)
    benchmark(rand_cuda(HEIGHT, WIDTH, dtype=torch.complex64), torch.fft.irfft2, fftconv2d.fft.irfft2)
    benchmark(
        rand_cuda(BATCH, IN_CHANNELS, HEIGHT, WIDTH),
        partial(torch.fft.rfftn, dim=(2,3)),
        fftconv2d.fft.rfftn)
    benchmark(
        rand_cuda(BATCH, IN_CHANNELS, HEIGHT, WIDTH, dtype=torch.complex64),
        partial(torch.fft.irfftn, dim=(2,3)),
        fftconv2d.fft.irfftn)
    
def test_precision_conv():
    std_conv, fft_conv = conv_wrap(MyConv)
    benchmark(
        randn(BATCH, IN_CHANNELS, HEIGHT, WIDTH),
        std_conv,
        fft_conv
    )