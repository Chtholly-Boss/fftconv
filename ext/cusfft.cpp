#include "cusfft.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("rfft", &rfft, "1D FFT(CUDA)");
	m.def("irfft", &irfft, "Inverse 1D FFT(CUDA)");
	m.def("rfft2", &rfft2, "2-D FFT(CUDA)");
	m.def("irfft2", &irfft2, "Inverse 2-D FFT(CUDA)");
	m.def("rfftn", &rfftn, "2-D FFT with Batch and Channel(CUDA)");
	m.def("irfftn", &irfftn, "Inverse 2-D FFT with Batch and Channel(CUDA)");
    m.def("conv_forward", &conv_forward, "Forward Convolution using cusFFT (CUDA)");
}