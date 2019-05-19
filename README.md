# PyNUFFT: Python non-uniform fast Fourier transform


The fast Fourier transform (FFT) is the standard method that estimates the frequency components at equispaced locations. For non-equispaced locations, FFT is not useful and the discrete Fourier transform 
(DFT) is required. 

Alternatively, NUFFT is the fast algorithm for calculating the frequency components at non-equispaced locations.

A minimal "getting start" tutorial is available at http://jyhmiinlin.github.io/pynufft/ .

Please use Python3. PyNUFFT has been tested with Python 3.4 and 3.6. However, it should work with Python 3.5. 


### New in version 2019.1.1

Version 2019.1.1 is a beta version. 

Installation in Windows 10 has been tested. 


### Summary

PyNUFFT implements the min-max NUFFT of Fessler and Sutton, with the following features:

- Written in pure Python.
- Based on Python numerical libraries, such as Numpy, Scipy (matplotlib for displaying examples).
- Provides the Python interface including forward transform, adjoint transform and other routines.
- Provides 1D/2D/3D examples for further developments.
- (Experimental) support of NVIDIA's graphic processing units (GPUs) and opencl devices (GPUs or a multi-core CPU)

### Bugs

- Experimental support for Python2. Some tests pass Python2.7.15 but the full support for Python2 is still pending (especially the GPU part).

- Kernel size of 5-7 has been tested. The numerical accuracy is limited to single-precision.  

- I am still writing examples. However tests/test_init.py might give you an indication whether NUFFT_cpu and NUFFT_hsa are working properly.

### Other nufft implementations in Python:

- Python-nufft: Python bindings to Fortran nufft. (https://github.com/dfm/Python-nufft/), MIT license

- pynfft: Python bindings around the NFFT C-library, which uses the speed of FFTW, (https://github.com/ghisvail/pyNFFT), GPL v3

- nfft: Pure Python implementation of 1D nfft (https://github.com/jakevdp/nfft). 

- nufftpy: Pure Python NUFFT of Python-nufft (https://github.com/jakevdp/nufftpy). 

- mripy: A Python based MRI package (https://github.com/peng-cao/mripy), which combines Numba and NUFFT.

- BART provides a Python wrapper.

### Acknowledgements

PyNUFFT was funded by the Cambridge Commonwealth, European and International Trust (Cambridge, UK) and Ministry of Education, Taiwan. 

I acknowledge the NVIDIA Corporation with the donation of a Titan X Pascal and a Quadro P6000 GPU used for developing the GPU code. Thanks to the authors of Michigan Image 
Reconstruction Toolbox (MIRT) for releasing the original min-max interpolator code. However, errors in PyNUFFT are not related to MIRT and please contact me at 
jyhmiinlin@gmail.com or open an issue. 


The interpolator is designed using the Fessler and Sutton's min-max NUFFT algorithm:
Fessler JA, Sutton BP. Nonuniform fast Fourier transforms using min-max interpolation. IEEE Trans Signal Process 2003;51(2):560-574.

If you find PyNUFFT useful, please cite:

Lin, Jyh-Miin. "Python Non-Uniform Fast Fourier Transform (PyNUFFT): An Accelerated Non-Cartesian MRI Package on a Heterogeneous Platform (CPU/GPU)." Journal of Imaging 4.3 (2018): 51.

@article{lin2018python,
  title={Python Non-Uniform Fast {F}ourier Transform ({PyNUFFT}): An Accelerated Non-{C}artesian {MRI} Package on a Heterogeneous Platform ({CPU/GPU})},
  author={Lin, Jyh-Miin},
  journal={Journal of Imaging},
  volume={4},
  number={3},
  pages={51},
  year={2018},
  publisher={Multidisciplinary Digital Publishing Institute}
}
