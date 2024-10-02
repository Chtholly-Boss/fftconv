import pytest
import torch
import cusfft

ONE_SIZE = 1 << 10
ITERATION = 100

def perf_wrapper(func:callable, iter=1):
    total_time = 0
    for _ in range(iter):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        func()
        end.record()
        torch.cuda.synchronize()
        total_time += start.elapsed_time(end)
    return total_time / iter

def test_fft1d():
    """
    Test the 1D FFT of torch.fftn and cuFFT
    This test is mainly for learning how to use cuFFT
    """
    din = torch.randn(ONE_SIZE)
    din_cuda = din.cuda()
    dout_cus = cusfft.rfft(din_cuda)
    dout_torch = torch.fft.rfftn(din_cuda)
    print(dout_cus)
    print(dout_torch)
    assert torch.allclose(dout_torch, dout_cus)

def test_ifft1d():
    """
    Test the 1D IFFT of torch.ifftn and cuFFT
    """
    din = torch.randn(ONE_SIZE)
    din_cuda = din.cuda()
    # Do FFT
    dout_cus = cusfft.rfft(din_cuda)
    # Do IFFT
    din_cus = cusfft.irfft(dout_cus)
    assert torch.allclose(din_cuda, din_cus, atol=1e-5, rtol=1e-5)

def test_perf_1dfft():
    """
    Test the performance of 1D FFT
    """
    din = torch.randn(ONE_SIZE)
    din_cuda = din.cuda()

    def fft_wrapper(rfft, irfft):
        def wrapper():
            dout = rfft(din_cuda)
            din = irfft(dout)
        return wrapper
    
    time_torch = perf_wrapper(fft_wrapper(torch.fft.rfftn, torch.fft.irfftn),iter=ITERATION)
    time_cus = perf_wrapper(fft_wrapper(cusfft.rfft, cusfft.irfft),iter=ITERATION)
    assert time_cus < time_torch