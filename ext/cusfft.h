#ifndef CUSFFT
#define CUSFFT
// Header Files
#include <torch/extension.h>
#include <iostream>

// macros
#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_REAL(x) AT_ASSERTM(x.dtype() == torch::kFloat32, #x " must be a float tensor")
#define CHECK_COMPLEX(x) AT_ASSERTM(x.is_complex(), #x " must be a complex float tensor")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Functions
// fft.cu
at::Tensor rfft(at::Tensor signal);
at::Tensor irfft(at::Tensor signal);
at::Tensor rfft2(at::Tensor signal);
at::Tensor irfft2(at::Tensor signal);
at::Tensor rfftn(at::Tensor signal);
at::Tensor irfftn(at::Tensor signal);
// conv.cu
at::Tensor conv_forward(at::Tensor signal, at::Tensor kernel);

#endif // CUSFFT