The Python Non-uniform fast Fourier transform (PyNUFFT)
=======================================================



**Purpose**

The PyNUFFT user manual documents the *Python non-uniform fast Fourier transform*, a Python package for non-uniform fast Fourier transform.

PyNUFFT was created for fun. The content may not reflect the views of funding bodies, former or current partners, and contributors.

If you find PyNUFFT useful, please cite:

*Lin, Jyh-Miin. "Python Non-Uniform Fast Fourier Transform (PyNUFFT): An Accelerated Non-Cartesian MRI Package on a Heterogeneous Platform (CPU/GPU)." Journal of Imaging 4.3 (2018): 51.*

or

*J.-M. Lin and H.-W. Chung, Pynufft: python non-uniform fast Fourier transform for MRI Building Bridges in Medical Sciences 2017, St John's College, CB2 1TP Cambridge, UK*

Users of PyNUFFT should be familiar with discrete Fourier transform (DFT). 


**The min-max interpolator**

- PyNUFFT translates the min-max interpolator to Python. The min-max interpolator is described in the literature:

*Fessler JA, Sutton BP. Nonuniform fast Fourier transforms using min-max interpolation. IEEE Trans Signal Process 2003;51(2):560-574.*

**Background**

- Fast Fourier transform (FFT) is one of the most important algorithms in signal processing. FFT delivers a fast and exact discrete Fourier transform (DFT) in a much shorter computation time than direct DFT does.

- However, FFT does not handle non-Cartesian DFT. 

- Thus, NUFFT is proposed as a way to compute the spectrum, by leveraging the speed of FFT and fast interpolation. 

 
**Current status of PyNUFFT**

- The current PyNUFFT relies on Numpy/Scipy (NUFFT_cpu) and Reikan/PyCUDA/PyOpenCL (NUFFT_hsa). 

- PyNUFFT provides 3 NUFFT classes: (1) NUFFT_cpu, (2) NUFFT_hsa (HSA: heterogeneous system architecture), (3) NUFFT (mixed CPU/HSA classes without warranty)

- LGPLv3.
 

