# README
This repository is a toy project to familiar myself with PyTorch C++ extension, aiming to implement FFT convolution in PyTorch.

## C++ Extension
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

## Convolution
We can learn how to do convolution throught FFT from [fft-conv-pytorch](https://github.com/fkodom/fft-conv-pytorch).
```python
# in fft_conv.py: 126,132
# Perform fourier convolution -- FFT, matrix multiply, then IFFT
signal_fr = rfftn(signal.float(), dim=tuple(range(2, signal.ndim)))
kernel_fr = rfftn(padded_kernel.float(), dim=tuple(range(2, signal.ndim)))

kernel_fr.imag *= -1
output_fr = complex_matmul(signal_fr, kernel_fr, groups=groups)
output = irfftn(output_fr, dim=tuple(range(2, signal.ndim)))
```

Refer to [Torch.Conv2d](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html), we can learn that:
* `signal` is a 4D tensor with shape `(batch, in_channels, height, width)`
* `kernel` is a 4D tensor with shape `(out_channels, in_channels // groups, kernel_height, kernel_width)`

Now we describe the logic of the code above:
* Perform FFT on the last 2 dimensions, treat other dimensions as batches, we get `signal_fr` and `kernel_fr`.
* `kernel_fr.imag *= -1` is a conjegation operation.
* `complex_matmul` is to perform matrix multiplication on the last 2 dimesions of 2 tensors, treat other dimension as batches.

To learn more about `complex_matmul` in the code, we should learn how to do convolution on multiple channels.

To put it simple, assume we have the following parameters:
```js
batch: int = 1
in_channels: int = 2
out_channels: int = 2
groups: int = 1
signal: [batch, in_channels, in_height, in_width]
kernel: [out_channels, in_channels / groups, out_height, out_width]
```

Follow the convolution fomula in [Conv2d](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html), we should apply the kernel to each in_channel, and then sum the results of them.

So our method is to:
* divide channels into groups when needed
* transform signal to `[batch, groups, in_height, in_width, in_channels / groups]`
* transform signal to `[groups, out_height, out_width, in_channels / groups, out_channels / groups]`
* perform matrix multiplication on the last 2 dimensions
* transform the result back to `[batch, out_channels, out_height, out_width]`

`fft-conv-pytorch` does this in the way described in the following picture:

![fft-conv-pytorch-matmul](./assets/conv2d_matmul.png)

Operations on dimensions include:
* [Tensor.view](https://pytorch.org/docs/stable/generated/torch.Tensor.view.html#torch.Tensor.view)
* [Tensor.movedim](https://pytorch.org/docs/stable/generated/torch.movedim.html)
* [Tensor.squeeze](https://pytorch.org/docs/stable/generated/torch.squeeze.html)

!!! warning `torch.matmul` only supports `torch.kFloat32`, to achieve complex multiplication, do it seperately.