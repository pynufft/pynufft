### pynufft 0.3: A Pythonic non-uniform fast Fourier transform

## Pythonic Non-uniform fast Fourier transform (NUFFT)

FFT is the standard method for calculating the frequency components of a signal. However, FFT only applies to equispaced grids. 

NUFFT can calculate the frequency components at non-equispaced locations.


## Installation:

from pypi:

pip install pynufft

from github:

git clone https://github.com/jyhmiinlin/pynufft

python setup.py install

## Example:

Inside the Python environment, type:

>>>import pynufft.pynufft

>>>pynufft.pynufft.test_2D() # test the 2D case


## Features

pynufft is written in Python, using the standard Numpy/Scipy packages. Therefore, the external dependency has been avoided. 

Numpy, Scipy, Matplotlib are prerequisites.

## Summary

Please find the example in test_2D().

The forward transform (the foward() method) involves the following steps:

1. Scaling (the x2xx() method)

2. FFT (the xx2k() method)

3. Convert kspectrum from array to vector: (the k2vec() method)

4. Interpolation (the vec2y() method)


The adjoint transform (the bacward() method) involves the following steps:

1. Adjoint interpolation (the y2vec() method)

2. Convert kspectrum from vector to array: (the vec2k() method)

3. IFFT (the k2xx() method)

4. Rescaling (the x22x() method)


If y is the data from the forward transform:

y=pynufft.forward(image)


The inverse transform (the inverse_DC() method) implemented the density compensation method of J. Pipe, Magnetic Resonance in Medicine, 1999

image=pynufft.inverse_DC(y)


k-space spectrum can be obtained from the data (y):

kspectrum = pynufft.y2k_DC(y)

## Limitations

The speed of pynufft is suboptimal, because FFTW is currently unsupported in Numpy/Scipy. 

However, pynufft can enjoy the full speed of MKL FFT inside the Anaconda Python environment.

## Other nufft implementations in Python:

Python-nufft: Python bindings to Fortran nufft. (https://github.com/dfm/Python-nufft/), MIT license

pynfft: Python bindings around the NFFT C-library, which uses the speed of FFTW, (https://github.com/ghisvail/pyNFFT), GPL v3

nufftpy: Pure Python NUFFT of Python-nufft (https://github.com/jakevdp/nufftpy). 

## Acknowlegements

pynufft received financial supports from the Ministry of Science and Technology under grant MOST 105-2221-E-002-142-MY3.

If you find pynufft useful, please cite:

Lin J-M, Patterson AJ, Chang H-C, Gillard JH, Graves MJ. An iterative reduced field-of-view reconstruction for periodically rotated overlapping parallel lines with enhanced reconstruction 
(http://www.ncbi.nlm.nih.gov/pubmed/26429249)

pynufft was based on the min-max MATLAB NUFFT program, described in the following paper:
Fessler JA, Sutton BP. Nonuniform fast Fourier transforms using min-max interpolation. IEEE Trans Signal Process 2003;51(2):560-574.

