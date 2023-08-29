from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='gelu_cpp',
      ext_modules=[cpp_extension.CppExtension('gelu_cpp', ['gelu.cpp'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})

Extension(
   name='gelu_cpp',
   sources=['gelu.cpp'],
   include_dirs=cpp_extension.include_paths(),
   language='c++')