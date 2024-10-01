from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='cusfft',
    version='0.0.1',
    ext_modules=[
        CUDAExtension('cusfft', [
            'cusfft_kernel.cu',
        ])
    ],
    cmdclass={'build_ext': BuildExtension}
)