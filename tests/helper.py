import torch

ITERATIONS = 10

BATCH  = 3
GROUPS = 1
IN_CHANNELS = 3
OUT_CHANNELS = 1
KERNEL_SIZE = 10
LENGTH = 1 << 4
HEIGHT = 1 << 10
WIDTH  = 1 << 10

def conv_wrap(other: callable) -> tuple:
    params = (IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE)
    std_conv = torch.nn.Conv2d(*params, padding=1)
    other_conv = other(*params,padding=1)

    with torch.no_grad():
        other_conv.weight = std_conv.weight
        other_conv.bias = std_conv.bias
    
    return std_conv, other_conv