import torch

ITERATIONS = 100

def perf_wrapper(func:callable):
    total_time = 0
    for _ in range(ITERATIONS):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        func()
        end.record()
        torch.cuda.synchronize()
        total_time += start.elapsed_time(end)
    return total_time / ITERATIONS

def fft_wrapper(fft:callable, ifft:callable, din: torch.Tensor) -> callable:
    def wrapper():
        dout = fft(din)
        dout_inv = ifft(dout)
    return wrapper