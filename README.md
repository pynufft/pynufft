# pynufft: Python non-uniform fast Fourier transform


FFT is the standard method that estimates the frequency components on equispaced grids.

NUFFT can calculate the frequency components outside grids.


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


### Features

pynufft is written in Python, using the standard Numpy/Scipy packages. Numpy, Scipy, Matplotlib are prerequisites.

### Summary

A "getting start" tutorial will become availabe in the near future (Early ). 

Currently, please find the example in test_2D().

The forward transform (the forward() method) involves the following steps:

1. Scaling (the x2xx() method)

2. FFT (the xx2k() method)

3. Convert spectrum from array to vector: (the k2vec() method)

4. Interpolation (the vec2y() method)


The adjoint transform (the adjoint() method) involves the following steps:

1. Adjoint interpolation (the y2vec() method)

2. Convert kspectrum from vector to array: (the vec2k() method)

3. IFFT (the k2xx() method)

4. Rescaling (the x2xx() method)


If y is the data from the forward transform:
>>>> y=pynufft.forward(image)

The inverse transform (the inverse_DC() method) implemented the density compensation method of J. Pipe, Magnetic Resonance in Medicine, 1999
>>>>image=pynufft.inverse_DC(y)

k-space spectrum can be obtained from the data (y):
>>>>kspectrum = pynufft.y2k_DC(y)

### Limitations

In Numpy, the default fft library is fftpack, so the speed of NUFFT transform may be suboptimal.
However, pynufft can run with the fast FFT inside the Anaconda Python environment (which is based on Intel's Math Kernel library (MKL)).



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
