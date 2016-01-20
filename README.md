pynufft: A pythonic non-uniform fast Fourier transform (NUFFT)

## What is non-uniform fast Fourier transform (NUFFT)?

FFT is the standard method for calculating the frequency components of a signal. Normally, FFT only applies to equispaced grids. 

NUFFT can calculate the frequency components of non-equispaced locations.

## Installation:
pip install pynufft


## Example:

Inside the python environment, type:

>>>import pynufft.pynufft

>>>pynufft.pynufft.test_1D() # test an 1-D case

>>>pynufft.pynufft.test_2D() # test the 2D case


## Features

The pynufft is written in python, using the standard numpy/scipy packages. 

Therefore, the external dependency has been avoided. 

if you find this python program useful, please cite:

Lin J-M, Patterson AJ, Chang H-C, Gillard JH, Graves MJ. An iterative reduced field-of-view reconstruction for periodically rotated overlapping parallel lines with enhanced reconstruction (PROPELLER) MRI. Med Phys 2015;42(10):5757-5767.
(http://www.ncbi.nlm.nih.gov/pubmed/26429249)

The algorithm was modified from the MATLAB NUFFT program described in the following paper:
Fessler JA, Sutton BP. Nonuniform fast Fourier transforms using min-max interpolation. IEEE Trans Signal Process 2003;51(2):560-574.

## Other nufft implementations in Python language

python-nufft: Python bindings to Fortran nufft. (https://github.com/dfm/python-nufft/)

pynfft: Python bindings of the C-library NFFT which is built on top of FFTW (https://github.com/ghisvail/pyNFFT)


