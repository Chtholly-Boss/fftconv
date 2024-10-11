#include "fftconv2d.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("rfft", &rfft, "1D FFT(CUDA)");
	m.def("irfft", &irfft, "Inverse 1D FFT(CUDA)");
	m.def("rfft2", &rfft2, "2-D FFT(CUDA)");
	m.def("irfft2", &irfft2, "Inverse 2-D FFT(CUDA)");
	m.def("rfftn", &rfftn, "2-D FFT with Batch and Channel(CUDA)");
	m.def("irfftn", &irfftn, "Inverse 2-D FFT with Batch and Channel(CUDA)");
    m.def("conv2d", &conv2d, "2D Convolution using cusFFT (CUDA)");
}