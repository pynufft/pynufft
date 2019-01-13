Python Non-uniform fast Fourier transform (PyNUFFT)
===================================================

-------------------
Background of NUFFT
-------------------


Fast Fourier transform (FFT) has become one of the most important algorithms in signal processing. 
FFT delivers a fast and exact discrete Fourier transform (DFT) in a much shorter computation time than direct DFT does.
However, FFT does not allow for non-Cartesian DFT. 

That is where non-uniform fast Fourier transform (NUFFT) attempts to fill the gaps. 
The basic idea of NUFFT is to compute the spectra on the grid, by leveraging the speed of FFT. 
Then the non-Cartesian data are interpolated from these grid data. 

The flow diagram of pynufft can be found in :numref:`flow-diagram`, which includes the following three steps:

1. Scaling to extend the range of interpolation.

2. Oversampled FFT.

3. Interpolation (gridding). 


.. _flow-diagram:

.. figure:: ../figure/flow_diagram.png
   :width: 30%

   Flow diagram of pynufft

**PyNUFFT**

PyNUFFT attemps to bring the MATLAB implementation to Python. 
PyNUFFT provides two NUFFT classes: 
(1) NUFFT_cpu and (2) NUFFT_hsa (HSA: heterogeneous system architecture). 

NUFFT_cpu is the generic NUFFT class built on Numpy/Scipy. 
The algorithm is translated from MATLAB to Python and I have checked the digits during this translation. 

NUFFT_hsa transplants the NUFFT class to  PyCUDA/PyOpenCL, using the Reikna package to support both platforms. 
The full toolchain is open-source. 
The FFT kernel is from Reikna, which is independent of CUDA. 
PyNUFFT has its own multi-dimensional interpolator and scaling factor, which are also independent of CUDA. 
 
**Current status of PyNUFFT**

PyNUFFT is dependent on Numpy/Scipy (NUFFT_cpu) and Reikan/PyCUDA/PyOpenCL (NUFFT_hsa). 

NUFFT_cpu is beta thanks to the stability of Numpy/Scipy.  
NUFFT_cpu has been tested in Python 3 and Python 2. 

However, Reikna/PyCUDA/PyOpenCL are under development and may be influenced by changes from upstream.  
This is due to new drivers from manufacturers, changes of OpenCL standards, and new operating systems/compilers. 
For example, installation of PyCUDA on Windows 10 may be different from previous Windows 7.  

Up to the present, NUFFT_hsa is functional in Ubuntu 16.04, Ubuntu 18.04 and Gentoo Linux 2019. 
My experience with Reikna/PyCUDA/PyOpencl is positive recently. 
However, there is a warning that the support might change in the future.  

--------------------
Variables of PyNUFFT
--------------------

FFT is simple to use because it assumes the data is regularly sampled and the input and output have the same size. 
However, this is not the case for NUFFT as NUFFT is non-uniform.  
To use NUFFT, users need to know the geometry of NUFFT, 
including grid size (Nd), oversampled grid size (Kd), interpolator size (Jd) and non-Cartesian coordinates (om).


Here we summarize the required variables which the users need to be familiar with.



**Non-Cartesian coordinates (om)**

om is an numpy array with a shape of (M, dim). 
M is  the number of non-Cartesian samples, dim is the dimensionality. 
The dtype of om is float.  om is normalized between :math:`[-\pi, \pi]`. 

**Image grid size (Nd)**

The image grid determines the size of the image. 

**Oversampled Fourier grid (Kd)**

The oversampled Fourier grid determines the size of the frequency. 
Normally the Kd is 2 × Nd[d] for d-th axis. 

**Interpolator size (Jd)**

The interpolator computes the Jd[d] adjacent weighted sum of the oversampled Fourier grid.
A normal choice of Jd is 6 for all axes.  
 
Optionally, user can provide additional variables:
 
**ft_axes (default = None (all axes))**

ft_axes the NUFFT_cpu to operate the FFT on the given axes.
 
**batch (default = None)**

Batch mode allows NUFFT to operate on the additional axis. 
 
 
:numref:`anatomy_nufft` illustrates the variables for 1D, 2D, 3D NUFFT.




.. _anatomy_nufft:

.. figure:: ../figure/anatomy_nufft.png
   :width: 60%
   
   The anatomy of 1D, 2D, and 3D NUFFT. 
   (A) 1D NUFFT: om is a numpy.array of the shape (M,1). 
   M is the number of non-Cartesian points. 
   Nd = (8, ) is the image domain grid size and Kd = (16, ) is the oversampled grid size. 
   Jd = (6, ) is the interpolator size.
   (B) 2D NUFFT: om is a numpy.array of the shape (M,2). 
   M is the number of non-Cartesian points. 
   Nd = (8, 8 ) is the image domain grid size and Kd = (16, 16 ) is the oversampled grid size. 
   Jd = (6, 6 ) is the interpolator size.   
   (C) 3D NUFFT: om is a numpy.array of the shape (M,3). 
   M is the number of non-Cartesian points. 
   Nd = (8, 8, 8 ) is the image domain grid size and Kd = (16, 16, 16 ) is the oversampled grid size. 
   Jd = (6, 6, 6 ) is the interpolator size.      


**The NUFFT using min-max interpolator**
   
More information about min-max interpolator can be found in the literature:

*Fessler JA, Sutton BP. Nonuniform fast Fourier transforms using min-max interpolation. IEEE Trans Signal Process 2003;51(2):560-574.*

which details the min-max interpolator for NUFFT. 