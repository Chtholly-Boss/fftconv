import torch

import fftconv2d
from fft_conv_pytorch import FFTConv2d

from torch import tensor, randn
from functools import partial

from helper import *

def measure_time(func:callable, iterations:int=ITERATIONS, din:tensor=None) -> float:
    total_time = 0
    for _ in range(iterations):
        din_rand = torch.randn(*din.size(), dtype=din.dtype, device=din.device)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        func(din_rand)
        end.record()
        torch.cuda.synchronize()
        total_time += start.elapsed_time(end)
    return total_time / iterations

def benchmark(din: torch.Tensor, base: callable, other: callable):
    measure = partial(measure_time, din=din)
    base_time = measure(base)
    other_time = measure(other)
    assert base_time > other_time

def test_speed_conv():
    std_conv, fftconv = conv_wrap(FFTConv2d)
    benchmark(
        din= randn(BATCH, IN_CHANNELS, HEIGHT, WIDTH),
        base= std_conv,
        other= fftconv
    )