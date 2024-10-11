#include "fftconv2d.h"

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
	// do fft on the last 2 dimensions, treat other dimensino as batch dimension
	int ndim = signal.dim();
	int width = signal.size(ndim - 1);
	int height = signal.size(ndim - 2);

	int cHeight = height;
	int cWidth = width / 2 + 1;
	// Get signal.size() and replace last 2 dimension with cHeight and cWidth
	auto size_tuple = signal.sizes();
	std::vector<int64_t> sizes(size_tuple.begin(), size_tuple.end());

	sizes[ndim - 1] = cWidth;
	sizes[ndim - 2] = cHeight;
	auto dout_real = torch::empty(sizes, signal.options());
	auto dout_imag = torch::empty(sizes, signal.options());
	auto dout = at::complex(dout_real, dout_imag);

	// create 2D cuFFT plan
	int x = height, y = width;
	cufftHandle plan;
	cufftPlan2d(&plan, x, y, CUFFT_R2C);
	// cast signal to cuFFT type
	cufftReal *din_fft = reinterpret_cast<cufftReal *>(signal.data_ptr());
	cufftComplex *dout_fft = reinterpret_cast<cufftComplex *>(dout.data_ptr());

	// treat other dimensions as batch dimension, compute the batch size
	int batch = 1;
	for (int i = 0; i < ndim - 2; i++) {
		batch *= signal.size(i);
	}
	// perform 2D fft on the last 2 dimension
	for (int i = 0; i < batch; i++) {
		cufftExecR2C(
			plan, 
			din_fft + i * height * width, 
			dout_fft + i * cHeight * cWidth
		);
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
	int ndim = signal.dim();
	int cWidth = signal.size(ndim-1);
	int cHeight = signal.size(ndim-2);

	int rHeight = cHeight;
	int rWidth = (cWidth - 1) * 2;

	auto size_tuple = signal.sizes();
	std::vector<int64_t> sizes(size_tuple.begin(), size_tuple.end());

	sizes[ndim - 1] = rWidth;
	sizes[ndim - 2] = rHeight;
	auto dout = torch::empty(sizes, signal.options().dtype(torch::kFloat32));

	// compute the batch size
	int batch = 1;
	for (int i = 0; i < ndim - 2; i++) {
		batch *= signal.size(i);
	}
	// create 2D cuFFT plan
	int x = rHeight, y = rWidth;
	cufftHandle plan;
	cufftPlan2d(&plan, x, y, CUFFT_C2R);

	// cast signal to cuFFT type
	// dup the signal to avoid modifying the original signal
	at::Tensor signal_dup = signal.clone();
	cufftComplex *din_fft = reinterpret_cast<cufftComplex *>(signal_dup.data_ptr());
	cufftReal *dout_fft = reinterpret_cast<cufftReal *>(dout.data_ptr());

	// perform 2D ifft on the last 2 dimension
	for (int i = 0; i < batch; i++) {
		cufftExecC2R(
			plan,
			din_fft + i * cHeight * cWidth,
			dout_fft + i * rHeight * rWidth
		);
	}
	// destroy cuFFT plan
	cufftDestroy(plan);
	// normalize
	dout /= (rHeight * rWidth);
	return dout;
}