import torch
import fft_conv_pytorch
from conv import MyConv

BATCH = 1
IN_CHANNELS = 1
OUT_CHANNELS = 2
KERNEL_SIZE = 3
HEIGHT = 2
WIDTH = 2

def test_conv_precision():
    params = (IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE)
    std_conv = torch.nn.Conv2d(*params, padding=1)
    pkg_fft_conv = fft_conv_pytorch.FFTConv2d(*params,padding=1)
    my_fft_conv = MyConv(*params, padding=1)

    # din = torch.randn(BATCH, IN_CHANNELS, HEIGHT, WIDTH)
    din = torch.ones(BATCH, IN_CHANNELS, HEIGHT, WIDTH)
    with torch.no_grad():
        pkg_fft_conv.weight = std_conv.weight
        pkg_fft_conv.bias = std_conv.bias
        my_fft_conv.weight = std_conv.weight
        my_fft_conv.bias = std_conv.bias
    
    std_out = std_conv(din)
    pkg_fft_out = pkg_fft_conv(din)
    my_fft_out = my_fft_conv(din)

    print(std_out)
    print(my_fft_out)

    assert torch.allclose(std_out, pkg_fft_out, atol=1e-5, rtol=1e-5)
    assert torch.allclose(std_out, my_fft_out, atol=1e-3, rtol=1e-3)

if __name__ == '__main__':
    test_conv_precision()