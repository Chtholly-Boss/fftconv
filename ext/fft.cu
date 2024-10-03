#include "cusfft.h"
#include <cufft.h>

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

at::Tensor rfft2(at::Tensor signal) {
	// TODO: implement 2D FFT
	// Check input type is kFloat32
	CHECK_INPUT(signal);
	CHECK_REAL(signal);
	TORCH_CHECK(signal.dim() == 2, "signal must be 2D");
	// Get Height and Width
	int rHeight = signal.size(0);
	int rWidth = signal.size(1);
	// create empty output tensor
	int cHeight = rHeight;
	int cWidth = rWidth / 2 + 1;

	auto dout_real = torch::empty({cHeight, cWidth}, signal.options());
	auto dout_imag = torch::empty({cHeight, cWidth}, signal.options());
	auto dout = at::complex(dout_real, dout_imag);

	// create 2D cuFFT plan
	int x = rHeight, y = rWidth;
	cufftHandle plan;
	cufftPlan2d(&plan, x, y, CUFFT_R2C);

	// cast signal to cuFFT type
	cufftReal *din_fft = reinterpret_cast<cufftReal *>(signal.data_ptr());
	cufftComplex *dout_fft = reinterpret_cast<cufftComplex *>(dout.data_ptr());

	// execute cuFFT plan
	cufftExecR2C(plan, din_fft, dout_fft);

	// destroy cuFFT plan
	cufftDestroy(plan);
	return dout;
}

at::Tensor irfft2(at::Tensor signal) {
    // TODO: implement Inverse 2D FFT
	// Check input type is complex
	TORCH_CHECK(signal.is_complex(), "signal must be at::complex");
	TORCH_CHECK(signal.is_cuda(), "signal must be on cuda");
	TORCH_CHECK(signal.dim() == 2, "signal must be 2D");

	// Get Height and Width
	int cHeight = signal.size(0);
	int cWidth = signal.size(1);

	// create output tensor
	int rHeight = cHeight;
	int rWidth = (cWidth - 1) * 2;

	auto dout = torch::empty({rHeight, rWidth}, signal.options().dtype(torch::kFloat32));

	// create 2D cuFFT plan
	int x = rHeight, y = rWidth;
	cufftHandle plan;

	cufftPlan2d(&plan, x, y, CUFFT_C2R);

	// cast signal to cuFFT type
	cufftComplex *din_fft = reinterpret_cast<cufftComplex *>(signal.data_ptr());
	cufftReal *dout_fft = reinterpret_cast<cufftReal *>(dout.data_ptr());

	// execute cuFFT plan
	cufftExecC2R(plan, din_fft, dout_fft);

	// destroy cuFFT plan
	cufftDestroy(plan);

	// normalize output tensor
	dout /= (rHeight * rWidth);
	return dout;
}

// Declare rfftn takes a Tensor and a vector of ints that specify the dimension to do fft
at::Tensor rfftn(at::Tensor signal) {
	// Check input type is kFloat32
	TORCH_CHECK(signal.dtype() == torch::kFloat32, "signal must be torch::kFloat32");
	TORCH_CHECK(signal.is_cuda(), "signal must be on cuda");
	// Get the parameters
	int batch = signal.size(0);
	int channel = signal.size(1);
	int height = signal.size(2);
	int width = signal.size(3);

	// create empty output tensor
	int cHeight = height;
	int cWidth = width / 2 + 1;

	auto dout_real = torch::empty({batch, channel, cHeight, cWidth}, signal.options());
	auto dout_imag = torch::empty({batch, channel, cHeight, cWidth}, signal.options());
	auto dout = at::complex(dout_real, dout_imag);

	// create 2D cuFFT plan
	int x = height, y = width;
	cufftHandle plan;
	cufftPlan2d(&plan, x, y, CUFFT_R2C);
	// cast signal to cuFFT type
	cufftReal *din_fft = reinterpret_cast<cufftReal *>(signal.data_ptr());
	cufftComplex *dout_fft = reinterpret_cast<cufftComplex *>(dout.data_ptr());
	// execute cuFFT plan
	// loop on batch and channel
	for (int i = 0; i < batch; i++) {
		for (int j = 0; j < channel; j++) {
			cufftExecR2C(
				plan, 
				din_fft + i * channel * height * width + j * height * width, 
				dout_fft + i * channel * cHeight * cWidth + j * cHeight * cWidth
			);
		}
	}
	
	// destroy cuFFT plan
	cufftDestroy(plan);
	return dout;
}

at::Tensor irfftn(at::Tensor signal) {
	// Check input type is complex
	TORCH_CHECK(signal.is_complex(), "signal must be at::complex");
	TORCH_CHECK(signal.is_cuda(), "signal must be on cuda");
	// Get the parameters
	int batch = signal.size(0);
	int channel = signal.size(1);
	int cHeight = signal.size(2);
	int cWidth = signal.size(3);

	// create empty output tensor
	int rHeight = cHeight;
	int rWidth = (cWidth - 1) * 2;
	auto dout = torch::empty({batch, channel, rHeight, rWidth}, signal.options().dtype(torch::kFloat32));

	// create 2D cuFFT plan
	int x = rHeight, y = rWidth;
	cufftHandle plan;
	cufftPlan2d(&plan, x, y, CUFFT_C2R);

	// cast signal to cuFFT type
	cufftComplex *din_fft = reinterpret_cast<cufftComplex *>(signal.data_ptr());
	cufftReal *dout_fft = reinterpret_cast<cufftReal *>(dout.data_ptr());
	// execute cuFFT plan
	// loop on batch and channel
	for (int i = 0; i < batch; i++) {
		for (int j = 0; j < channel; j++) {
			cufftExecC2R(
				plan, 
				din_fft + i * channel * cHeight * cWidth + j * cHeight * cWidth, 
				dout_fft + i * channel * rHeight * rWidth + j * rHeight * rWidth
			);
			// normalize output tensor
			dout[i][j] /= (rHeight * rWidth);
		}
	}
	// destroy cuFFT plan
	cufftDestroy(plan);
	return dout;
}