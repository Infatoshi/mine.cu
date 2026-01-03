from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="minecu",
    version="0.1.0",
    author="Infatoshi",
    description="High-performance batched voxel RL environment with custom CUDA kernels",
    packages=["minecu"],
    ext_modules=[
        CUDAExtension(
            name="minecu._C",
            sources=[
                "src/kernels.cu",
                "src/bindings.cpp",
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "--use_fast_math"],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0",
        "numpy",
    ],
    extras_require={
        "dev": ["matplotlib", "imageio"],
    },
)
