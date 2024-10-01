# README
This is the repository for the project "Optimizing the Performance of CNN Using FFT".
Follows are the development steps, the result of the project will be put into other documents.
## Env
We will use `pytorch` and `fft_conv_pytorch` as the baseline.
And to do tests, we will use `pytest`

Follows are the packages we will use:
```py
torch
fft_conv_pytorch
pytest
setuptools
```

## Stage 0
To see what we will accomplish in this stage, let's take a look at the following snippet from `fft_conv_pytorch` package:
```python
# in fft_conv.py: 126,132
# Perform fourier convolution -- FFT, matrix multiply, then IFFT
signal_fr = rfftn(signal.float(), dim=tuple(range(2, signal.ndim)))
kernel_fr = rfftn(padded_kernel.float(), dim=tuple(range(2, signal.ndim)))

kernel_fr.imag *= -1
output_fr = complex_matmul(signal_fr, kernel_fr, groups=groups)
output = irfftn(output_fr, dim=tuple(range(2, signal.ndim)))
```

We can see how the naive FFT convolution is implemented.
The core function is `rfftn` and `irfftn`, which are the real-valued FFT and inverse FFT functions.
So in this stage, we will try to implement our `rfftn` and `irfftn` functions using `cuFFT`.
This requires us to construct a framework to call `cuFFT` functions and compare the result with the `torch.fft`

Let's first take a look at the `rfftn` function, which is taken from [torch.fft](https://pytorch.org/docs/stable/generated/torch.fft.rfftn.html#torch.fft.rfftn)

From the code above, we can see that only `input` and `dim` parameters are needed.

However, you should notice that the `dim` start from 2, this is because `dim[0] is batch-size` and `dim[1] is channels`.

We will do FFT on the last 2 dimensions, so we can assume that:

!!! note the input is a tensor with shape `(batch, channels, height, width)`

In this stage, we assume that `batch=channels=1` and focus on 2D FFT implementation. 

We will refer to [CPP Custom Ops Tutorial](https://pytorch.org/tutorials/advanced/cpp_custom_ops.html#cpp-custom-ops-tutorial) to do extensions.

When you write an extension, you should import like this:
```py
import torch
import your.package
```
Otherwise, you will get an error like this:
```shell
Traceback (most recent call last):
  File "/root/cuda_programming/nms/temp1.py", line 2, in <module>
    from nms_cuda import nms
ImportError: libc10.so: cannot open shared object file: No such file or directory
```
This is because your package is based on `torch`, so you should import `torch` first.

Now you should successfully import your package.

Let's take a look at the `cuFFT` and see how can we use it as our `rfftn` function.