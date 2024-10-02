#include <torch/extension.h>
#include <vector>
#include <cufft.h>
#include <iostream>

at::Tensor rfft(at::Tensor signal) {
	TORCH_CHECK(signal.dtype() == torch::kFloat32, "signal must be torch::kFloat32");
	TORCH_CHECK(signal.is_cuda(), "signal must be on cuda");

	int rSize = signal.size(0);
	// create empty output tensor
	int cSize = rSize / 2 + 1;
	auto dout_real = torch::empty({cSize}, signal.options());
	auto dout_imag = torch::empty({cSize}, signal.options());
	auto dout = at::complex(dout_real, dout_imag);
	
	// create 1D cuFFT plan
	int n = rSize;
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

at::Tensor irfft(at::Tensor signal) {
	TORCH_CHECK(signal.is_complex(), "signal must be at::complex");
	TORCH_CHECK(signal.is_cuda(), "signal must be on cuda");

	int cSize = signal.size(0);

	// create output tensor
	int rSize = (cSize - 1) * 2;
	auto dout = torch::empty({rSize}, signal.options().dtype(torch::kFloat32));

	// create 1D cuFFT plan
	int n = rSize;
	cufftHandle plan;
	cufftPlan1d(&plan, n, CUFFT_C2R, 1);

	// cast signal to cuFFT type
	cufftComplex *din_fft = reinterpret_cast<cufftComplex *>(signal.data_ptr());
	cufftReal *dout_fft = reinterpret_cast<cufftReal *>(dout.data_ptr());

	// execute cuFFT plan
	cufftExecC2R(plan, din_fft, dout_fft);

	// destroy cuFFT plan
	cufftDestroy(plan);

	// normalize output tensor
	dout /= n;
	return dout;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("rfft", &rfft, "1D FFT(CUDA)");
	m.def("irfft", &irfft, "Inverse 1D FFT(CUDA)");
}
