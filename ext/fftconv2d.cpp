#include "fftconv2d.h"

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.doc() = "Convolution using FFT (CUDA)";

	py::module fft = m.def_submodule("fft", "Modules contain FFT functions");
	fft.def("rfft", &rfft, "1D R2C FFT(CUDA)");
	fft.def("irfft", &irfft, "Inverse 1D FFT(CUDA)");
	fft.def("rfft2", &rfft2, "2-D FFT(CUDA)");
	fft.def("irfft2", &irfft2, "Inverse 2-D FFT(CUDA)");
	fft.def("rfftn", &rfftn, "2-D FFT with Batch and Channel(CUDA)");
	fft.def("irfftn", &irfftn, "Inverse 2-D FFT with Batch and Channel(CUDA)");

	py::module conv = m.def_submodule("conv", "Modules contain Convolution functions");
    conv.def("conv2d", &conv2d, "2D Convolution using cusFFT (CUDA)");
}
