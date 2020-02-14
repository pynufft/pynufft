The Python Non-uniform fast Fourier transform (PyNUFFT)
=======================================================



**Purpose**

The PyNUFFT user manual documents the *Python non-uniform fast Fourier transform*, a Python package for non-uniform fast Fourier transform.

If you find PyNUFFT useful, please cite:

*Lin, Jyh-Miin. "Python Non-Uniform Fast Fourier Transform (PyNUFFT): An Accelerated Non-Cartesian MRI Package on a Heterogeneous Platform (CPU/GPU)." Journal of Imaging 4.3 (2018): 51.*

Users of PyNUFFT should be familiar with discrete Fourier transform (DFT). 


**The min-max interpolator**

- PyNUFFT reimplements the min-max interpolator, which is described in the literature:

*Fessler JA, Sutton BP. Nonuniform fast Fourier transforms using min-max interpolation. IEEE Trans Signal Process 2003;51(2):560-574.*

**Background**

- Fast Fourier transform (FFT) is one of the most important algorithms in signal processing. FFT delivers a fast and exact discrete Fourier transform (DFT) in a much shorter computation time than direct DFT does.

- However, FFT does not handle non-Cartesian DFT. 

- Thus, NUFFT is proposed as a way to compute the spectrum, by leveraging the speed of FFT and fast interpolation. 

 
**Current status of PyNUFFT**

- The current PyNUFFT relies on Numpy/Scipy (NUFFT_cpu) and Reikan/PyCUDA/PyOpenCL (NUFFT_hsa). 

- PyNUFFT provides two NUFFT classes: (1) NUFFT_cpu and (2) NUFFT_hsa (HSA: heterogeneous system architecture). 

- Unlike TensorFlow for AI, the current PyNUFFT does NOT recommend any single solver for a wide range of reconstruction problems, especially in medical imaging applications.

- However, it does provide some referenced solvers for a limited number of problems, but without any warranty.

**Technology overview**

- NUFFT_cpu is the generic NUFFT class built on Numpy/Scipy. 

- NUFFT_hsa transplants the NUFFT class to  PyCUDA/PyOpenCL, using the Reikna package to support both platforms. 
The full toolchain is open-source. 
The FFT kernel is from Reikna, which is independent of CUDA. 
PyNUFFT has its own multi-dimensional interpolator and scaling factor, which are also independent of CUDA. 
  
