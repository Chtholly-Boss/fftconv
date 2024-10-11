from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='fftconv2d',
    version='0.1.0',
    ext_modules=[
        CUDAExtension('fftconv2d', [
            'fft.cu',
            'conv.cu',
            'cusfft.cpp',
        ])
    ],
    cmdclass={'build_ext': BuildExtension}
)