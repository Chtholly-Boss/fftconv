from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

EXT_NAME = 'fftconv2d'
SRC = [
    'fft.cu',
    'conv.cu',
    'fftconv2d.cpp',
]

setup(
    name= EXT_NAME,
    version='0.2.0',
    ext_modules=[
        CUDAExtension(EXT_NAME, SRC)
    ],
    cmdclass={'build_ext': BuildExtension}
)