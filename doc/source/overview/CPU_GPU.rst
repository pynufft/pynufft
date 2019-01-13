CPU and GPU (HSA)
=================

The PyNUFFT was originally designed for CPU, by leveraging the Numpy/Scipy. Later it was transplated to work with GPU and multi-core CPU.

NUFFT_cpu is planned and executed on the CPU, using Numpy/Scipy. Thus, the speed is dependent on the implemented Numpy/Scipy.

For heterogeneous accelerators such as CPU and GPU, NUFFT_hsa provides a route to accelerate the PyNUFFT.  The transformation part of NUFFT_hsa class is running entirely on the selected accelerator.  

Mixing NUFFT_cpu and NUFFT_hsa, or multiple NUFFT_cpu or multiple NUFFT_hsa is possible but no warranty. 

