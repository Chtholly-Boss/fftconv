# README
This is the repository for the project "Optimizing the Performance of CNN Using FFT".
Follows are the development steps, the result of the project will be put into other documents.
## Env
We will use `pytorch` and `fft_conv_pytorch` as the baseline.
And to do tests, we will use `pytest`

Follows are the packages we will use:
```py
torch
setuptools
pytest
fft_conv_pytorch
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

From the code above, we can see that only `input` and `dim` parameters are needed.

You should notice that the `dim` start from 2, this is because `dim[0] is batch-size` and `dim[1] is channels`.

We will do FFT on the last 2 dimensions, so we can assume that:

!!! note the input is a tensor with shape `(batch, channels, height, width)`

In this stage, we will focus on **Extension**. So we will implement **1D FFT** first.

### Extension
Refer to [CPP Custom Ops Tutorial](https://pytorch.org/tutorials/advanced/cpp_custom_ops.html#cpp-custom-ops-tutorial) to learn how to start.

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

### 1D FFT
Now we will implement the 1D FFT function using **cuFFT**.

```cpp
at::Tensor rfft(at::Tensor signal) {
  // Prerequistes Checking
  ...
	int n = signal.size(0);
	// create empty output tensor
	int out_size = n / 2 + 1;
  // torch::Complex64 corresponds to at::complex, so we do conversion
	auto dout_real = torch::empty({out_size}, signal.options());
	auto dout_imag = torch::empty({out_size}, signal.options());
	auto dout = at::complex(dout_real, dout_imag);
	
	// create 1D cuFFT plan
	cufftHandle plan;
	cufftPlan1d(&plan, n, CUFFT_R2C, 1);

	// cast signal to cuFFT type
  // reinterpret_cast is needed, C-like cast will not work when compiling
	cufftReal *din_fft = reinterpret_cast<cufftReal *>(signal.data_ptr());
	cufftComplex *dout_fft = reinterpret_cast<cufftComplex *>(dout.data_ptr());

	// execute cuFFT plan
	cufftExecR2C(plan, din_fft, dout_fft);

	// destroy cuFFT plan
	cufftDestroy(plan);
	return dout;
}
```

### Stage 1
After implementing the 1D FFT function, we are now familiar with how to write an extension using `pytorch` and how to use `cuFFT`.

It's easy to extend to 2D FFT, just change the `cufftPlan1d` to `cufftPlan2d` and the `dim` to `(height, width)`.

So in this stage we will focus on integrate `FFT` into `Convolution`, which is the core of this project.
