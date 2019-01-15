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
