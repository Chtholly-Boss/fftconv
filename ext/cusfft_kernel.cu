#include <torch/extension.h>
#include <vector>
#include <cufft.h>
#include <iostream>

at::Tensor fft(at::Tensor signal) {
	// check if signal is torch::kFloat32
	if (signal.type().scalarType() != torch::kFloat32) {
		throw std::runtime_error("signal must be torch::kFloat32");
	}
	// check if signal is on cuda
	if (!signal.is_cuda()) {
		throw std::runtime_error("signal must be on cuda");
	}

	int n = signal.size(0);
	// create empty output tensor
	int out_size = n / 2 + 1;
	auto dout_real = torch::empty({out_size}, signal.options());
	auto dout_imag = torch::empty({out_size}, signal.options());
	auto dout = at::complex(dout_real, dout_imag);
	
	// create 1D cuFFT plan
	cufftHandle plan;
	cufftPlan1d(&plan, n, CUFFT_R2C, 1);

	// cast signal to cuFFT type
	cufftReal *din_fft = reinterpret_cast<cufftReal *>(signal.data_ptr());
	cufftComplex *dout_fft = reinterpret_cast<cufftComplex *>(dout.data_ptr());

	// execute cuFFT plan
	cufftExecR2C(plan, din_fft, dout_fft);

	// destroy cuFFT plan
	cufftDestroy(plan);
	return dout;
}

at::Tensor rfftn(at::Tensor signal, std::vector<int64_t> dims) {
	// TODO: You will only perform FFT on the specified dim
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("fft", &fft, "1D FFT(CUDA)");
}
