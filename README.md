# pynufft: Python non-uniform fast Fourier transform


FFT is the standard method that estimates the frequency components at equispaced locations.

NUFFT can calculate the frequency components at non-equispaced locations.

A minimal "getting start" tutorial is available at http://jyhmiinlin.github.io/pynufft/ .


### Installation:

From pypi:

$ pip install pynufft

From github:

$ git clone https://github.com/jyhmiinlin/pynufft

$ python setup.py install

### Example:

Inside the Python environment, type:


>>> import pynufft.pynufft as pnft

>>> pnft.test_installation() # test required files

>>> pnft.test_2D() # test the 2D case


### Summary

Pynufft implements Fessler's min-max NUFFT, with the following features:

- Written in pure Python.
- Based on numerical libraries, such as Numpy, Scipy (matplotlib for displaying examples).
- Provides the python interface including forward transform, adjoint transform and other routines.
- Provides 1D/2D/3D examples for further developments.
- (Experimental) Supporting NUFFT on NVIDIA's graphic processing units (GPUs).

### Limitations

In Numpy, the default fft library is fftpack, so the speed of NUFFT transform may be suboptimal.

Python was limited by Global Interpret Lock (GIL). So you would need cython to release GIL and speed up for loops.

However, Anaconda Python environment and Intel's Python seems to provide openmp support for many critical computations.

### Other nufft implementations in Python:

Python-nufft: Python bindings to Fortran nufft. (https://github.com/dfm/Python-nufft/), MIT license

pynfft: Python bindings around the NFFT C-library, which uses the speed of FFTW, (https://github.com/ghisvail/pyNFFT), GPL v3

nufftpy: Pure Python NUFFT of Python-nufft (https://github.com/jakevdp/nufftpy). 

### Acknowledgements

pynufft was funded by the Ministry of Science and Technology, Cambridge Overseas Trust and Ministry of Education.  

If you find pynufft useful, please cite Fessler's min-max NUFFT paper. 
Fessler JA, Sutton BP. Nonuniform fast Fourier transforms using min-max interpolation. IEEE Trans Signal Process 2003;51(2):560-574.

Please open an issue if you have any question related to pynufft.

### Cite pynufft

@Misc{pynufft, author = {Jyh-Miin Lin}, title = {{Pynufft}: {Python} non-uniform fast {F}ourier transform}, year = {2013--}, url = "https://github.com/jyhmiinlin/pynufft", note = {Online; https://github.com/jyhmiinlin/pynufft; Dec 2016} }

[![DOI](https://zenodo.org/badge/49985083.svg)](https://zenodo.org/badge/latestdoi/49985083)
