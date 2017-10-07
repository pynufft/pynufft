Introduction
============

**Purpose**

The pynufft user manual documents *Python non-uniform fast Fourier transform*, a Python program for non-uniform fast Fourier transform.

Pynufft reimplements the MATLAB version of min-max NUFFT_, with the following features:  

.. _NUFFT: http://web.eecs.umich.edu/~fessler/irt/irt/nufft/

- Written in pure Python.

- Based on numerical libraries, such as Numpy, Scipy (matplotlib for displaying examples).

- Provides the python interface including forward transform, adjoint transform and other routines.

- Provides 1D/2D/3D examples for further developments.

If you find pynufft useful, please cite:

*Jyh-Miin Lin, Hsiao-Wen Chung, Pynufft: python non-uniform fast Fourier transform for MRI. Building Bridges in Medical Sciences Conference 2017 (BBMS17), Mar, 2017, St John's College, Cambridge, UK*

Users of pynufft should be familiar with discrete Fourier transform (DFT). 

.. toctree::
   :maxdepth: 2
   
   introduction/nufft
