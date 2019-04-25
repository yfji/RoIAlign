from __future__ import print_function
import os
import torch
from torch.utils.ffi import create_extension
import sys
#this_file = os.path.dirname(__file__)

sources = []
headers = []
defines = []
with_cuda = False

if torch.cuda.is_available():
    print('Including CUDA code.')
    sources += ['src/nms_cuda.cpp']
    headers += ['src/nms_cuda.h']
    defines += [('WITH_CUDA', None)]
    with_cuda = True

this_file = os.path.dirname(os.path.realpath(__file__))
print(this_file)
extra_objects = ['src/nms_cuda_kernel.o']
extra_objects = [os.path.join(this_file, fname) for fname in extra_objects]
print(extra_objects)

extra_compile_args=None
if sys.platform=='win32':
    extra_compile_args=['/openmp','/MD']
    if torch.__version__=='0.4.0':
        libraries=['ATen','_C','cudart']
    else:
        libraries=['caffe2','caffe2_gpu','_C','cudart'] #check the libs in your site-packages/torch/lib
elif sys.platform=='linux':
    extra_compile_args=['-fopenmp','-std=c99']
else:
    raise NotImplementedError

ffi = create_extension(
    '_ext.nms',
    headers=headers,
    sources=sources,
    define_macros=defines,
    relative_to=__file__,
    with_cuda=with_cuda,
    extra_objects=extra_objects,
    extra_compile_args=extra_compile_args,
    libraries=libraries
)

if __name__ == '__main__':
    ffi.build()
