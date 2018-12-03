# pynufft: Python non-uniform fast Fourier transform


FFT is the standard method that estimates the frequency components at equispaced locations.

NUFFT can calculate the frequency components at non-equispaced locations.

A minimal "getting start" tutorial is available at http://jyhmiinlin.github.io/pynufft/ .

Please use Python3. PyNUFFT has been tested with Python3.4. However, it should work with Python3.5-3.6. 


### New in version 0.3.3.10

Version 0.3.3.10 is a bug-fixed version. 

Remove the keyword async for compatibility reasons because Reikna has changed the keyword to async_.


### Summary

Pynufft implements Fessler's min-max NUFFT, with the following features:

- Written in pure Python.
- Based on numerical libraries, such as Numpy, Scipy (matplotlib for displaying examples).
- Provides the python interface including forward transform, adjoint transform and other routines.
- Provides 1D/2D/3D examples for further developments.
- (Experimental) support of NUFFT on NVIDIA's graphic processing units (GPUs) and opencl devices (GPUs or a multi-core CPU)

### Bugs

Experimental support for Python2. Some tests pass Python2.7.15 but the full support for Python2 is still pending.



### Other nufft implementations in Python:

Python-nufft: Python bindings to Fortran nufft. (https://github.com/dfm/Python-nufft/), MIT license

pynfft: Python bindings around the NFFT C-library, which uses the speed of FFTW, (https://github.com/ghisvail/pyNFFT), GPL v3

nfft: Pure Python implementation of 1D nfft (https://github.com/jakevdp/nfft). 

nufftpy: Pure Python NUFFT of Python-nufft (https://github.com/jakevdp/nufftpy). 

mripy: A Python based MRI package (https://github.com/peng-cao/mripy), which combines Numba and NUFFT.

BART provides a Python wrapper.

### Acknowledgements

pynufft was funded by the Ministry of Science and Technology, Taiwan, Cambridge Commonwealth, European and International Trust (Cambridge, UK), and Ministry of Education, Taiwan.  

If you find pynufft useful, please cite:

Jyh-Miin Lin and Hsiao-Wen Chung, Pynufft: python non-uniform fast Fourier transform for MRI, Building Bridges in Medical Sciences 2017, St John's College, CB2 1TP Cambridge, UK, 2017

Lin, Jyh-Miin. "Python Non-Uniform Fast Fourier Transform (PyNUFFT): An Accelerated Non-Cartesian MRI Package on a Heterogeneous Platform (CPU/GPU)." Journal of Imaging 4.3 (2018): 51.

And Fessler and Sutton's min-max NUFFT algorithm:
Fessler JA, Sutton BP. Nonuniform fast Fourier transforms using min-max interpolation. IEEE Trans Signal Process 2003;51(2):560-574.

Please open an issue if you have any question related to pynufft.

### Cite pynufft

@article{lin2018python,
  title={Python Non-Uniform Fast Fourier Transform (PyNUFFT): An Accelerated Non-Cartesian MRI Package on a Heterogeneous Platform (CPU/GPU)},
  author={Lin, Jyh-Miin},
  journal={Journal of Imaging},
  volume={4},
  number={3},
  pages={51},
  year={2018},
  publisher={Multidisciplinary Digital Publishing Institute}
}
