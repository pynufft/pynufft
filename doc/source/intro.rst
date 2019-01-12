Introduction
============

**Purpose**

The PyNUFFT user manual documents *Python non-uniform fast Fourier transform*, a Python package for non-uniform fast Fourier transform.

PyNUFFT reimplements the MATLAB version of min-max NUFFT_, with the following features:  

.. _NUFFT: http://web.eecs.umich.edu/~fessler/irt/irt/nufft/

- Based on numerical libraries, such as Numpy, Scipy (matplotlib for displaying examples).

- CPU class is written in pure Python. HSA class uses JIT (just-in-time) compilation provided by Reikna/PyCUDA/PyOpenCL.  

- Provides the python interface including forward transform, adjoint transform and other routines.

- Provides 1D/2D/3D examples for further developments.

- (New in 0.4.0) Support batch mode and split-radix technique. 

If you find PyNUFFT useful, please cite:

*Lin, Jyh-Miin. "Python Non-Uniform Fast Fourier Transform (PyNUFFT): An Accelerated Non-Cartesian MRI Package on a Heterogeneous Platform (CPU/GPU)." Journal of Imaging 4.3 (2018): 51.*

Users of PyNUFFT should be familiar with discrete Fourier transform (DFT). 

.. toctree::
   :maxdepth: 2
   
   introduction/nufft
